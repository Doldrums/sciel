"""Portable native oracle evidence. Locations never supply evaluator authority.

Checkpoint bytes and the exact committed journal prefix bind the public launch,
the unique evaluation decision and its saved-record/sidecar join. This remains
unsigned host evidence, not scientific or Docker qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from ..eval import oracle_evidence as O
from ..methods.models import content_hash as native_hash
from .storage import read_json
from .types import sha256_bytes

DIRECTORY = "evaluation_evidence"
SCHEMA = "sciel-retained-evaluations/1"
ROW = "sciel-oracle-episode-binding/1"


class RunContext:
    def __init__(self, root, checkpoint_path=None):
        self.root = O.ordinary(root).resolve(strict=True)
        self.checkpoint_path = O.ordinary(
            checkpoint_path or self.root / "checkpoint.json"
        )
        self.checkpoint_bytes = self.checkpoint_path.read_bytes()
        self.checkpoint_sha256 = sha256_bytes(self.checkpoint_bytes)
        self.checkpoint = json.loads(self.checkpoint_bytes)
        self.anchor = self.checkpoint.get("provenance", {}).get("journal_anchor", {})
        offset = self.anchor.get("byte_offset")
        O.need(
            type(offset) is int and offset > 0,
            "oracle context requires committed journal prefix",
        )
        with O.ordinary(self.root / "sciel_events.jsonl").open("rb") as stream:
            self.prefix = stream.read(offset)
        O.need(
            len(self.prefix) == offset
            and self.prefix.endswith(b"\n")
            and sha256_bytes(self.prefix) == self.anchor.get("prefix_sha256"),
            "oracle journal anchor differs",
        )
        self.run_id = self.checkpoint.get("provenance", {}).get("run_id")
        O.need(
            type(self.run_id) is str and self.run_id,
            "oracle context has no run identity",
        )
        self.events = [json.loads(line) for line in self.prefix.splitlines()]
        self.events = [e for e in self.events if e.get("run_id") == self.run_id]
        identity = self.checkpoint.get("identity", {})
        O.need(
            identity.get("task") == O.TASK, "context is not the allowlisted oracle task"
        )
        self.launch = O.validate_retained_launch(identity.get("oracle_evaluation"))
        O.need(
            identity.get("task_hash") == self.launch["task_hash"],
            "checkpoint oracle task hash differs",
        )
        config = O.validate_config(
            self.checkpoint.get("config", {}).get("oracle_evaluation")
        )
        O.need(
            config["manifest_sha256"] == self.launch["study_manifest_sha256"]
            and config["panel"] == self.launch["panel_name"]
            and config["panel_sha256"] == self.launch["panel"]["sha256"]
            and config["image_id"] == self.launch["runtime"]["image_id"],
            "checkpoint launch configuration differs",
        )

    def decision(self, evaluation_id):
        found = [
            e
            for e in self.events
            if e.get("event") == "decision"
            and e.get("policy") == "evaluation"
            and e.get("action") == "result"
            and e.get("evaluation_id") == evaluation_id
        ]
        O.need(
            len(found) == 1, "oracle evaluation requires a unique committed decision"
        )
        return found[0]

    def resolve(self, record):
        eid, source = record.get("evaluation_id"), record.get("source_sha256")
        key = native_hash(record)
        O.need(
            record.get("schema") == "sciel-saved-evaluation/1"
            and record.get("run_id") == self.run_id
            and record.get("task_hash") == self.launch["task_hash"],
            "oracle saved evaluation identity differs",
        )
        decision = self.decision(eid)
        O.need(
            decision.get("source_sha256") == source
            and decision.get("reached") == record.get("reached")
            and decision.get("confirmed") == record.get("confirmed")
            and all(
                decision.get(k) == record["result"].get(k)
                for k in ("ok", "score", "cases", "artifacts", "confirmed_sampled")
            ),
            "oracle committed result differs from saved record",
        )
        from .export import _receipt

        O.need(
            O.canonical(_receipt(self.root, eid, source, self.run_id))
            == O.canonical(record),
            "oracle saved-record index differs",
        )
        sidecar = O.read(self.root / O.relative_evaluation(eid) / "sidecar.json")
        O.need(
            type(sidecar) is dict
            and set(sidecar)
            == {
                "schema",
                "run_id",
                "evaluation_id",
                "source_sha256",
                "task_hash",
                "launch_binding",
                "saved_record_sha256",
                "terminal_stage",
                "stages",
            }
            and sidecar["schema"] == O.EVIDENCE
            and O.digest(sidecar) == decision.get("oracle_evidence_sha256")
            and sidecar["run_id"] == self.run_id
            and sidecar["evaluation_id"] == eid
            and sidecar["source_sha256"] == source
            and sidecar["task_hash"] == self.launch["task_hash"]
            and sidecar["saved_record_sha256"] == key
            and O.canonical(sidecar["launch_binding"]) == O.canonical(self.launch)
            and sidecar["terminal_stage"] == record["reached"],
            "oracle sidecar commitment differs",
        )
        stages = sidecar["stages"]
        O.need(
            type(stages) is list
            and len(stages) in (2, 3)
            and record["reached"] in ("full", "confirm")
            and record["result"]["ok"] is True,
            "oracle learning result requires successful full/confirm",
        )
        wanted = ["smoke", "full"] + (
            ["confirm"] if record["reached"] == "confirm" else []
        )
        O.need(len(stages) == len(wanted), "oracle attempted stage sequence differs")
        terminal = None
        for entry, stage in zip(stages, wanted):
            O.need(
                type(entry) is dict
                and set(entry)
                == {
                    "invocation",
                    "invocation_sha256",
                    "state",
                    "report_sha256",
                    "runner_result",
                    "error",
                    "files",
                },
                "invalid oracle stage entry",
            )
            inv = entry["invocation"]
            limits = record["conditions"]
            relative = O.relative_evaluation(eid) + "/" + stage
            expected_inv = dict(
                schema=O.INVOCATION,
                run_id=self.run_id,
                evaluation_id=eid,
                stage=stage,
                source_sha256=source,
                task_hash=self.launch["task_hash"],
                launch_binding_sha256=O.digest(self.launch),
                expected=O.retained_expected(self.launch, source, stage),
                relative_directory=relative,
                limits=dict(
                    timeout=limits["stage_timeouts_seconds"][stage],
                    confirm_n=limits["confirm_n"] if stage == "confirm" else None,
                    cpu_seconds=limits["cpu_limit_seconds"],
                    memory_bytes=limits["memory_limit_bytes"],
                ),
            )
            O.need(
                O.canonical(inv) == O.canonical(expected_inv)
                and entry["invocation_sha256"] == O.digest(inv)
                and entry["state"] == "complete"
                and entry["error"] is None,
                "oracle stage invocation differs/incomplete",
            )
            directory = O.ordinary(self.root / relative)
            O.need(
                O.files(directory) == entry["files"]
                and O.canonical(O.read(directory / "invocation.json"))
                == O.canonical(inv)
                and O.canonical(O.read(directory / "runner.json"))
                == O.canonical(entry["runner_result"]),
                "oracle retained stage inventory differs",
            )
            result = entry["runner_result"]
            ref = json.loads(result["artifacts"]["oracle_evaluation"])
            envelope = (
                json.loads(result["artifacts"]["evaluation_receipt"])
                if stage != "smoke"
                else None
            )
            terminal = O.task_module().receipts.verify_retained(
                directory / "raw", ref, envelope=envelope, **inv["expected"]
            )
            O.need(
                terminal["sha256"] == entry["report_sha256"]
                and terminal["result"]["ok"] is True
                and O.canonical(terminal["result"])
                == O.canonical({k: result[k] for k in ("ok", "score", "cases")}),
                "oracle stage result differs",
            )
            if stage == "confirm":
                O.need(
                    result["confirmed_sampled"] is True and record["confirmed"] is True,
                    "oracle confirmation lacks actual sampling admission",
                )
        O.need(
            O.canonical(stages[-1]["runner_result"]) == O.canonical(record["result"]),
            "oracle terminal runner result differs from native admission",
        )
        resolved = dict(
            envelope=O.task_module().receipts.training_envelope(terminal),
            report=terminal,
            evidence_sha256=O.digest(sidecar),
            sidecar=sidecar,
        )
        return resolved

    def row_binding(self, row):
        child = row["evaluation_receipt"]
        parent = row["provenance"]["parent_evaluation_receipt"]
        a, b = self.resolve(child), self.resolve(parent)
        prov = row["provenance"]
        O.need(
            prov.get("run_id") == self.run_id
            and prov.get("checkpoint_sha256") == self.checkpoint_sha256
            and prov.get("journal_anchor") == self.anchor,
            "episode native checkpoint binding differs",
        )
        iterations = [
            e
            for e in self.events
            if e.get("event") == "iteration"
            and e.get("segment_id") == prov.get("segment_id")
            and e.get("event_seq") == prov.get("event_seq")
        ]
        O.need(len(iterations) == 1, "oracle episode has no unique committed iteration")
        meta = iterations[0]["meta"]
        O.need(
            self.parent_evaluation_id(iterations[0]) == parent["evaluation_id"]
            and self.events.index(self.decision(child["evaluation_id"]))
            < self.events.index(iterations[0]),
            "oracle paired evaluations were not available at admission",
        )
        parent_spec = meta["adaptation_parent"]
        parent_binding = self.checkpoint.get("research_evaluation_bindings", {}).get(
            parent_spec["program_id"]
        )
        O.need(
            meta.get("evaluation_id") == child["evaluation_id"]
            and parent_binding is not None
            and parent_binding["evaluation_id"] == parent["evaluation_id"]
            and parent_binding["source_sha256"]
            == parent["source_sha256"]
            == parent_spec["source_sha256"]
            and parent_spec["score"] == row["parent_score"],
            "oracle exact committed parent/child differs",
        )
        for field in ("panel", "runtime", "task_hash", "evaluator_version"):
            O.need(
                a["report"][field] == b["report"][field],
                "oracle parent/child pairing differs",
            )
        return dict(
            schema=ROW,
            task=O.TASK,
            launch_binding_sha256=O.digest(self.launch),
            child_evidence_sha256=a["evidence_sha256"],
            parent_evidence_sha256=b["evidence_sha256"],
        )

    def parent_evaluation_id(self, event):
        parent = event["meta"]["adaptation_parent"]
        binding = self.checkpoint.get("research_evaluation_bindings", {}).get(
            parent["program_id"]
        )
        O.need(
            binding is not None and binding["source_sha256"] == parent["source_sha256"],
            "oracle parent has no checkpoint-authenticated evaluation",
        )
        decision = self.decision(binding["evaluation_id"])
        O.need(
            decision.get("source_sha256") == parent["source_sha256"]
            and self.events.index(decision) < self.events.index(event),
            "oracle parent evaluation is not available",
        )
        return binding["evaluation_id"]

    def for_row(self, row):
        wanted = self.row_binding(row)
        O.need(
            row["provenance"].get("oracle_evaluation") == wanted,
            "oracle row lost its typed native binding",
        )
        return self


def native_context(root, checkpoint_path=None):
    value = read_json(checkpoint_path or Path(root) / "checkpoint.json")
    identity = value.get("identity", {})
    if identity.get("task") == O.TASK:
        return RunContext(root, checkpoint_path)
    O.need(
        identity.get("oracle_evaluation") is None
        and value.get("config", {}).get("oracle_evaluation") is None,
        "oracle binding appears on another native task",
    )
    return None


class Contexts:
    def __init__(self, runs=()):
        self.runs = {}
        for run in runs:
            previous = self.runs.get(run.checkpoint_sha256)
            O.need(
                previous is None or previous.prefix == run.prefix,
                "conflicting oracle run context",
            )
            self.runs[run.checkpoint_sha256] = run

    def for_row(self, row):
        prov = row.get("provenance", {})
        run = self.runs.get(prov.get("checkpoint_sha256"))
        if run is not None:
            return run.for_row(row)
        O.need(
            prov.get("oracle_evaluation") is None
            and not any(
                r.run_id == prov.get("run_id")
                or r.launch["task_hash"]
                == row.get("evaluation_receipt", {}).get("task_hash")
                for r in self.runs.values()
            ),
            "oracle row requires its retained checkpoint context",
        )
        return None


def context(value):
    if value is None or isinstance(value, (RunContext, Contexts)):
        return value
    raise ValueError(
        "evaluation context must be reconstructed from retained native evidence"
    )
