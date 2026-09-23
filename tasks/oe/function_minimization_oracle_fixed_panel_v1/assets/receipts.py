"""Pure fixed-panel receipt replay. Integrity is not evaluator provenance.

Objective values are observations of the trusted host broker. Arithmetic replay
cannot prove that arbitrary caller-authored values came from a hidden objective.
All expected identities/stage below must come from the trusted invocation.
"""

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
import re
import os
import stat

from . import oracle_runtime as runtime
from .panels import (
    PANEL_SIZE,
    SCORER,
    canonical,
    digest,
    need,
    require_hash,
    validate_panel,
)

SCHEMA = "sciel-fixed-oracle-evaluation/1"
from .receipt_trials import STATUSES, INFRASTRUCTURE, number, trial, replay_journal


def evaluator_version(task_root, binding):
    """Stage/source/panel independent, but binds every scorer/broker/image asset."""
    root = Path(task_root)
    files = [root / "evaluator.py", root / "task.yaml"] + [
        p
        for p in (root / "assets").rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in (".pyc", ".pyo")
    ]
    import hashlib

    return digest(
        dict(
            protocol=runtime.PROTOCOL,
            scorer=SCORER,
            runtime=binding,
            files={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(files)
            },
        )
    )




def aggregate(stage, trials):
    need(stage in ("smoke", "full", "confirm"), "unknown stage")
    n = 1 if stage == "smoke" else PANEL_SIZE
    need(len(trials) == n, "stage trial count differs")
    successful = sum(t["status"] == "ok" for t in trials)
    raw = sum(t["score"] for t in trials) / n
    infrastructure = any(t["status"] in INFRASTRUCTURE for t in trials)
    valid = (
        successful > 0
        and not infrastructure
        and (stage != "confirm" or successful == n)
    )
    score = raw if valid else 0.0
    return dict(
        ok=valid,
        score=score,
        cases=dict(
            combined_score=score,
            mean_best_query_score=score,
            validity=float(valid),
            n_trials=float(n),
            reliability_score=successful / n,
            objective_queries=float(sum(t["queries"] for t in trials)),
            objective_query_budget=50.0,
        ),
    ), raw




def failure_artifacts(report, journals):
    """Bounded public feedback from an already verified, complete stage.

    In particular, a valid full stage can include failed trials. Adding an
    ``error`` artifact there would change the runner's success decision.
    """
    need(type(report["result"]["ok"]) is bool, "invalid stage success flag")
    if report["result"]["ok"]:
        return {}
    trials = report["trials"]
    result, raw = aggregate(report["stage"], trials)
    need(
        result == report["result"] and raw == report["raw_trial_mean"]
        and len(journals) == len(trials),
        "failure feedback requires a complete stage",
    )
    protocol_causes = {
        "query budget exceeded": "query_budget_exceeded",
        "coordinate must be a number": "coordinate_domain",
        "invalid coordinate": "coordinate_domain",
        "coordinate outside finite domain": "coordinate_domain",
        "point must have two coordinates": "coordinate_domain",
    }
    status_causes = {
        "protocol_error": "other_protocol_error",
        "timeout": "trial_timeout",
        "candidate_error": "candidate_exit",
        "startup_timeout": "startup_timeout",
        "infrastructure_error": "infrastructure_error",
    }
    counts, charged_queries = {}, 0
    for value, journal in zip(trials, journals):
        if value["status"] == "ok":
            continue
        cause = status_causes[value["status"]]
        replayed_cause = replay_journal(journal, value)
        if value["status"] == "protocol_error":
            cause = protocol_causes.get(replayed_cause, cause)
        counts[cause] = counts.get(cause, 0) + 1
        charged_queries += value["queries"]
    message = (
        f"{report['stage']} failed: {sum(counts.values())}/{len(trials)} trials failed; "
        + ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
        + f"; charged_queries_in_failed_trials={charged_queries}; "
        + f"query_budget_per_trial={runtime.QUERY_BUDGET}."
    )
    if "coordinate_domain" in counts:
        message += " Each query point must have two finite coordinates in [-1,1]."
    need(len(message) <= 600, "failure feedback exceeded public bound")
    return {"error_type": "OracleTrialFailure", "error": message}


def build(
    *, stage, source_sha256, panel, binding, task_hash, version, trials, journals
):
    result, raw = aggregate(stage, trials)
    body = dict(
        schema=SCHEMA,
        stage=stage,
        source_sha256=source_sha256,
        panel=panel,
        runtime=binding,
        task_hash=task_hash,
        evaluator_version=version,
        trials=trials,
        result=result,
        raw_trial_mean=raw,
    )
    receipt = {**body, "sha256": digest(body)}
    return verify(
        receipt,
        expected_stage=stage,
        source_sha256=source_sha256,
        panel=panel,
        binding=binding,
        task_hash=task_hash,
        version=version,
        journals=journals,
    )


def verify(
    value,
    *,
    expected_stage,
    source_sha256,
    panel,
    binding,
    task_hash,
    version,
    journals,
):
    validate_panel(panel, expected_split="train")
    for h in (source_sha256, task_hash, version):
        require_hash(h)
    need(
        type(value) is dict
        and set(value)
        == {
            "schema",
            "stage",
            "source_sha256",
            "panel",
            "runtime",
            "task_hash",
            "evaluator_version",
            "trials",
            "result",
            "raw_trial_mean",
            "sha256",
        },
        "invalid oracle evaluation fields",
    )
    need(
        value["schema"] == SCHEMA
        and value["stage"] == expected_stage
        and value["source_sha256"] == source_sha256
        and value["panel"] == panel
        and value["runtime"] == binding
        and value["task_hash"] == task_hash
        and value["evaluator_version"] == version,
        "oracle invocation identity/stage differs",
    )
    need(
        type(binding) is dict
        and set(binding) == {"image_id", "assets_sha256", "base_image"}
        and type(binding["image_id"]) is str
        and re.fullmatch(r"sha256:[0-9a-f]{64}", binding["image_id"])
        and binding["base_image"] == runtime.BASE_IMAGE,
        "invalid runtime binding",
    )
    require_hash(binding["assets_sha256"])
    need(type(value["trials"]) is list, "invalid stage trials")
    expected_n = 1 if expected_stage == "smoke" else PANEL_SIZE
    need(
        expected_stage in ("smoke", "full", "confirm")
        and len(value["trials"]) == expected_n,
        "stage trial count differs",
    )
    need(
        type(journals) is list and len(journals) == expected_n,
        "journal inventory differs",
    )
    for position, row in enumerate(value["trials"]):
        trial(
            row,
            source_sha256=source_sha256,
            panel=panel,
            position=position,
            binding=binding,
        )
        replay_journal(journals[position], row)
    names = [t["container_name"] for t in value["trials"]]
    need(len(set(names)) == len(names), "reused physical trial/container identity")
    result, raw = aggregate(expected_stage, value["trials"])
    number(value["raw_trial_mean"], nonnegative=True)
    need(
        canonical(value["result"]) == canonical(result)
        and value["raw_trial_mean"] == raw,
        "aggregate/status arithmetic differs",
    )
    need(
        value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"}),
        "oracle evaluation hash differs",
    )
    return value


def training_envelope(verified):
    need(verified["stage"] in ("full", "confirm"), "smoke is not a training evaluation")
    return dict(
        schema="sciel-lora-evaluation/1",
        source_sha256=verified["source_sha256"],
        split="train",
        stage=verified["stage"],
        instance_set_sha256=verified["panel"]["sha256"],
        bank_sha256=verified["panel"]["bank_sha256"],
        evaluator_version=verified["evaluator_version"],
        task_hash=verified["task_hash"],
        result=verified["result"],
        oracle_evaluation_schema=SCHEMA,
        oracle_evaluation_sha256=verified["sha256"],
    )


def verify_pair(parent, child, *, parent_expected, child_expected):
    """Verify both using separately trusted invocation inputs, then compare."""
    verify(parent, **parent_expected)
    verify(child, **child_expected)
    need(
        parent["stage"] in ("full", "confirm")
        and child["stage"] in ("full", "confirm"),
        "paired evaluations must be full/confirm",
    )
    for field in ("panel", "runtime", "task_hash", "evaluator_version"):
        need(parent[field] == child[field], "paired oracle " + field + " differs")
    need(
        parent["result"]["ok"] is True and child["result"]["ok"] is True,
        "paired oracle evaluations must succeed",
    )
    return child["result"]["score"] - parent["result"]["score"]


def reference(verified):
    return dict(
        schema="sciel-fixed-oracle-reference/1",
        sha256=verified["sha256"],
        stage=verified["stage"],
        report_file="evaluation.json",
        trial_count=len(verified["trials"]),
        panel_sha256=verified["panel"]["sha256"],
    )


def verify_retained(directory, ref, *, envelope=None, **expected):
    """Open one canonical owned stage directory, then replay the full evidence.

    The caller supplies trusted invocation identities and an owned retained path;
    neither is inferred from the report or the wire's absolute display path.
    """
    from .panels import _private_root

    root = _private_root(directory)

    def read(name, limit):
        fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            need(
                stat.S_ISREG(info.st_mode)
                and stat.S_IMODE(info.st_mode) == 0o600
                and info.st_size <= limit,
                "invalid retained evidence file",
            )
            return stream.read()

    def decode(raw):
        return json.loads(
            raw,
            object_pairs_hook=runtime._unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ValueError("nonfinite retained JSON")
            ),
        )

    report = decode(read("evaluation.json", 16 * 1024 * 1024))
    n = 1 if expected["expected_stage"] == "smoke" else PANEL_SIZE
    journals = [read(f"trial-{i:02d}.jsonl", 8 * 1024 * 1024) for i in range(n)]
    verify(report, journals=journals, **expected)
    need(
        canonical(ref) == canonical(reference(report)),
        "compact reference differs from retained report",
    )
    if envelope is not None:
        need(
            canonical(envelope) == canonical(training_envelope(report)),
            "wire envelope differs from retained report",
        )
    intent = decode(read("intent.json", 1024 * 1024))
    need(
        intent
        == dict(
            schema="sciel-fixed-oracle-intent/1",
            stage=report["stage"],
            source_sha256=report["source_sha256"],
            panel=report["panel"],
            runtime=report["runtime"],
            task_hash=report["task_hash"],
            evaluator_version=report["evaluator_version"],
            container_names=[t["container_name"] for t in report["trials"]],
            trial_count=n,
        ),
        "retained stage intent differs",
    )
    files = {"intent.json", "evaluation.json"}
    for i, row in enumerate(report["trials"]):
        need(
            canonical(decode(read(f"trial-{i:02d}.json", 2 * 1024 * 1024)))
            == canonical(row),
            "retained per-trial record differs",
        )
        files.update((f"trial-{i:02d}.json", f"trial-{i:02d}.jsonl"))
    need(
        {p.name for p in root.iterdir()} == files,
        "retained stage file inventory differs",
    )
    return report
