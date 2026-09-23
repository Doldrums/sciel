"""Versioned fixed TRAIN panel. No implicit study creation or candidate import."""

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parent
_name = "sciel_fixed_oracle_" + hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
_spec = importlib.util.spec_from_file_location(
    _name,
    ROOT / "assets" / "__init__.py",
    submodule_search_locations=[str(ROOT / "assets")],
)
_package = importlib.util.module_from_spec(_spec)
sys.modules[_name] = _package
_spec.loader.exec_module(_package)
runtime = importlib.import_module(_name + ".oracle_runtime")
panels = importlib.import_module(_name + ".panels")
receipts = importlib.import_module(_name + ".receipts")
failure_feedback = importlib.import_module(_name + ".failure_feedback")


def _evaluate(program_path, stage):
    from sciel.eval.registry import load_task

    panels.need(
        stage in ("smoke", "full", "confirm"), "invalid trusted entrypoint stage"
    )
    # Mandatory trusted settings. No private split path or panel from candidate RPC.
    instances, panel = panels.load_panel(
        os.environ["SCIEL_ORACLE_FIXED_STUDY"],
        os.environ["SCIEL_ORACLE_FIXED_MANIFEST_SHA256"],
        os.environ["SCIEL_ORACLE_FIXED_PANEL"],
        os.environ["SCIEL_ORACLE_FIXED_PANEL_SHA256"],
        expected_split="train",
    )
    receipt_root = panels._private_root(os.environ["SCIEL_ORACLE_FIXED_RECEIPTS"])
    binding = runtime.verify_runtime_image(os.environ["SCIEL_ORACLE_IMAGE"])
    with open(program_path, "rb") as stream:
        source = stream.read(runtime.MAX_SOURCE_BYTES + 1)
    panels.need(
        0 < len(source) <= runtime.MAX_SOURCE_BYTES,
        "candidate size outside allowed range",
    )
    source_hash = hashlib.sha256(source).hexdigest()
    task_hash = load_task(str(ROOT)).task_hash
    version = receipts.evaluator_version(ROOT, binding)
    invocation_path = os.environ.get("SCIEL_ORACLE_FIXED_INVOCATION")
    if invocation_path is not None:
        # Native host allocated this exact empty leaf before dispatch. Display
        # paths in returned artifacts never authorize a different directory.
        from sciel.eval.oracle_evidence import INVOCATION
        invocation_path = Path(invocation_path)
        panels.need(invocation_path.resolve(strict=True) == invocation_path
            and invocation_path.name == "invocation.json"
            and invocation_path.parent / "raw" == receipt_root, "native oracle invocation path differs")
        invocation = panels._read(invocation_path)
        actual = dict(expected_stage=stage, source_sha256=source_hash, panel=panel,
                      binding=binding, task_hash=task_hash, version=version)
        panels.need(invocation.get("schema") == INVOCATION and invocation.get("stage") == stage
            and invocation.get("source_sha256") == source_hash and invocation.get("task_hash") == task_hash
            and panels.canonical(invocation.get("expected")) == panels.canonical(actual),
            "native oracle invocation inputs differ")
        panels.need(not any(receipt_root.iterdir()), "native oracle stage directory is not fresh")
        directory = receipt_root
    else:
        directory = receipt_root / (stage + "-" + uuid.uuid4().hex)
        directory.mkdir(mode=0o700)
    panels._sync(receipt_root)
    trial_count = 1 if stage == "smoke" else panels.PANEL_SIZE
    names = ["sciel-oracle-" + uuid.uuid4().hex for _ in range(trial_count)]
    intent = dict(
        schema="sciel-fixed-oracle-intent/1",
        stage=stage,
        source_sha256=source_hash,
        panel=panel,
        runtime=binding,
        task_hash=task_hash,
        evaluator_version=version,
        container_names=names,
        trial_count=trial_count,
    )
    panels._write(directory / "intent.json", intent)
    panels._sync(directory)
    print(
        json.dumps({"oracle_intent": intent, "receipt_directory": str(directory)}),
        flush=True,
    )
    trials, journals = [], []
    try:
        for position in range(trial_count):
            journal = directory / f"trial-{position:02d}.jsonl"
            value = runtime.run_container_trial(
                source,
                instances[position],
                binding["image_id"],
                algorithm_seed=panel["algorithm_seeds"][position],
                container_name=names[position],
                journal_path=journal,
            )
            value.update(
                panel_position=position,
                instance_index=panel["indices"][position],
                instance_sha256=panel["instance_sha256"][position],
            )
            raw = journal.read_bytes()
            receipts.trial(
                value,
                source_sha256=source_hash,
                panel=panel,
                position=position,
                binding=binding,
            )
            receipts.replay_journal(raw, value)
            panels._write(directory / f"trial-{position:02d}.json", value)
            panels._sync(directory)
            trials.append(value)
            journals.append(raw)
        report = receipts.build(
            stage=stage,
            source_sha256=source_hash,
            panel=panel,
            binding=binding,
            task_hash=task_hash,
            version=version,
            trials=trials,
            journals=journals,
        )
        panels._write(directory / "evaluation.json", report)
        panels._sync(directory)
    except BaseException as exc:
        # The intent and any in-flight raw journal remain; no invented zero-cost trial.
        panels._write(
            directory / "incomplete.json",
            dict(
                schema="sciel-fixed-oracle-incomplete/1",
                error_type=type(exc).__name__,
                committed_trials=len(trials),
                trial_count=trial_count,
            ),
        )
        panels._sync(directory)
        raise
    artifacts = dict(
        oracle_contract=runtime.PROTOCOL,
        oracle_evaluation=json.dumps(receipts.reference(report), sort_keys=True),
        oracle_receipt_directory=str(directory),
        raw_trial_mean=report["raw_trial_mean"],
    )
    artifacts.update(failure_feedback.append_hint(
        report, source, receipts.failure_artifacts(report, journals)
    ))
    if stage != "smoke":
        artifacts["evaluation_receipt"] = json.dumps(
            receipts.training_envelope(report), sort_keys=True
        )
    return {**report["result"], "artifacts": artifacts}


def smoke(program_path):
    return _evaluate(program_path, "smoke")


def full(program_path):
    return _evaluate(program_path, "full")


def confirm(program_path, n=20):
    panels.need(
        type(n) is int and n == 20,
        "fixed-panel confirmation requires exactly 20 trials",
    )
    return _evaluate(program_path, "confirm")
