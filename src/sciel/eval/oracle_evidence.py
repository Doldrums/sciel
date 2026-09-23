"""Owned native oracle invocations and retained host evidence, without qualification.

Only the allowlisted task is loaded. Expected inputs come from the launch and
host dispatch, never from a candidate result or its display-path artifact.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re

from .facade import EvalResult

TASK = "oe/function_minimization_oracle_fixed_panel_v1"
LAUNCH = "sciel-oracle-run-binding/1"
INVOCATION = "sciel-oracle-invocation/1"
EVIDENCE = "sciel-oracle-evidence/1"
# Recognition only: never authorize an old launch, receipt or scorer version.
# Exact evaluator.py at 7b10373 and the frozen acquisition source ec9448c.
HISTORICAL_EVALUATOR_SHA256 = frozenset({
    "ec90389cb82df477b6ebacf35f2f1f4b953946383dfb612a2792d35d68d22899",
    # NumPy/scientific75 evaluator before host exception feedback.
    "933cc4205b9e9865e3aceacb9f7965e475ca9590ef7d45dc263bd686e90b78bd",
})


@lru_cache(maxsize=1)
def task_module():
    path = Path(__file__).resolve().parents[3] / "tasks" / TASK / "evaluator.py"
    spec = importlib.util.spec_from_file_location("sciel_owned_oracle_task", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def known_evaluator(record):
    """Recognize the supported producer even after optional hints are stripped.

    This source identity is independent of public initial-program/task hashes.
    It is not authentication against a wholly rewritten native history.
    """
    return (
        record.get("schema") == "sciel-saved-evaluation/1"
        and type(record.get("conditions", {}).get("evaluator_sha256")) is str
        and record.get("conditions", {}).get("evaluator_sha256")
        in HISTORICAL_EVALUATOR_SHA256 | {
            hashlib.sha256((task_module().ROOT / "evaluator.py").read_bytes()).hexdigest()
        }
    )


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    need(
        ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "oracle evidence cannot traverse symlinks or parent paths",
    )
    return path


def write(path, value):
    path = ordinary(path)
    raw = canonical(value) + b"\n"
    if path.exists():
        need(path.read_bytes() == raw, "immutable oracle evidence changed on replay")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    task_module().panels._sync(path.parent)


def read(path):
    return task_module().panels._read(ordinary(path))


def validate_config(value):
    need(
        type(value) is dict
        and set(value)
        == {
            "schema",
            "study_root",
            "manifest_sha256",
            "panel",
            "panel_sha256",
            "image_id",
        }
        and value["schema"] == "sciel-oracle-launch/1",
        "invalid explicit oracle launch configuration",
    )
    module = task_module()
    for key in ("manifest_sha256", "panel_sha256"):
        module.panels.require_hash(value[key])
    need(
        type(value["study_root"]) is str
        and value["study_root"]
        and "\x00" not in value["study_root"],
        "invalid oracle study locator",
    )
    need(
        type(value["panel"]) is str and re.fullmatch(r"train-[0-3]", value["panel"]),
        "oracle requires a TRAIN panel",
    )
    need(
        type(value["image_id"]) is str
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value["image_id"]),
        "oracle requires an exact local image ID",
    )
    return deepcopy(value)


def launch(config, task):
    if getattr(task, "qualified_name", None) != TASK:
        need(
            config is None, "oracle launch applies only to the fixed-panel oracle task"
        )
        return None
    config = validate_config(config)
    module = task_module()
    manifest, panel = module.panels.load_contract(
        config["study_root"],
        config["manifest_sha256"],
        config["panel"],
        config["panel_sha256"],
        expected_split="train",
    )
    # Support public initial-program variants, but not an arbitrary task evaluator.
    need(
        Path(task.evaluator_path).read_bytes()
        == (module.ROOT / "evaluator.py").read_bytes(),
        "oracle evaluator differs from supported implementation",
    )
    binding = dict(
        image_id=config["image_id"],
        assets_sha256=module.runtime.runtime_assets_sha256(),
        base_image=module.runtime.BASE_IMAGE,
    )
    value = dict(
        schema=LAUNCH,
        task_name=TASK,
        task_hash=task.task_hash,
        study_manifest_sha256=config["manifest_sha256"],
        public_manifest=manifest,
        panel_name=config["panel"],
        panel=panel,
        runtime=binding,
        evaluator_version=module.receipts.evaluator_version(
            Path(task.evaluator_path).parent, binding
        ),
    )
    validate_launch(value)
    return value


def _launch_contract(value):
    module = task_module()
    need(
        type(value) is dict
        and set(value)
        == {
            "schema",
            "task_name",
            "task_hash",
            "study_manifest_sha256",
            "public_manifest",
            "panel_name",
            "panel",
            "runtime",
            "evaluator_version",
        }
        and value["schema"] == LAUNCH
        and value["task_name"] == TASK,
        "invalid oracle launch binding",
    )
    module.panels.require_hash(value["task_hash"])
    panel = module.panels.validate_contract(
        value["public_manifest"],
        value["study_manifest_sha256"],
        value["panel_name"],
        value["panel"]["sha256"],
        expected_split="train",
    )
    need(canonical(panel) == canonical(value["panel"]), "oracle launch panel differs")
    binding = value["runtime"]
    need(
        type(binding) is dict
        and set(binding) == {"image_id", "assets_sha256", "base_image"}
        and type(binding["image_id"]) is str
        and re.fullmatch(r"sha256:[0-9a-f]{64}", binding["image_id"])
        and binding["base_image"] == module.runtime.BASE_IMAGE,
        "oracle runtime pin differs",
    )
    module.panels.require_hash(binding["assets_sha256"])
    module.panels.require_hash(value["evaluator_version"])
    return value


def validate_launch(value):
    """New dispatch requires the currently installed task/image assets."""
    _launch_contract(value)
    module = task_module()
    binding = value["runtime"]
    need(
        binding["assets_sha256"] == module.runtime.runtime_assets_sha256(),
        "oracle runtime pin differs",
    )
    need(
        value["evaluator_version"]
        == module.receipts.evaluator_version(module.ROOT, binding),
        "unsupported oracle evaluator version",
    )
    return value


def validate_retained_launch(value):
    """Replay current evidence or an explicitly supported historical full tuple.

    This is not used by launch(), invocation() or prepare_stage(). Old evidence
    keeps its exact runtime/version; it never authorizes new work on that image.
    """
    _launch_contract(value)
    module = task_module()
    if (
        value["runtime"]["assets_sha256"] == module.runtime.runtime_assets_sha256()
        and value["evaluator_version"]
        == module.receipts.evaluator_version(module.ROOT, value["runtime"])
    ):
        return validate_launch(value)
    from .oracle_runtime_profiles import validate_historical_retained

    validate_historical_retained(module.ROOT, value["runtime"], value["evaluator_version"])
    return value


def expected(launch_binding, source_sha256, stage):
    validate_launch(launch_binding)
    return _expected(launch_binding, source_sha256, stage)


def retained_expected(launch_binding, source_sha256, stage):
    validate_retained_launch(launch_binding)
    return _expected(launch_binding, source_sha256, stage)


def _expected(launch_binding, source_sha256, stage):
    need(stage in ("smoke", "full", "confirm"), "invalid oracle dispatch stage")
    task_module().panels.require_hash(source_sha256)
    return dict(
        expected_stage=stage,
        source_sha256=source_sha256,
        panel=deepcopy(launch_binding["panel"]),
        binding=deepcopy(launch_binding["runtime"]),
        task_hash=launch_binding["task_hash"],
        version=launch_binding["evaluator_version"],
    )


def relative_evaluation(evaluation_id):
    need(
        type(evaluation_id) is str and evaluation_id,
        "missing native evaluation identity",
    )
    return (
        "research_evidence/oracle/" + hashlib.sha256(evaluation_id.encode()).hexdigest()
    )


def context(engine, evaluation_id, source_sha256):
    binding = getattr(engine, "oracle_evaluation", None)
    if binding is None:
        return None
    need(
        type(engine.log.run_id) is str and engine.log.run_id,
        "oracle requires an identified native run",
    )
    return dict(
        run_root=str(Path(engine.run_dir).resolve(strict=True)),
        run_id=engine.log.run_id,
        evaluation_id=evaluation_id,
        source_sha256=source_sha256,
        launch_binding=deepcopy(binding),
        study_root=engine.cfg["oracle_evaluation"]["study_root"],
    )


def invocation(ctx, task, stage):
    binding = ctx["launch_binding"]
    want = expected(binding, ctx["source_sha256"], stage)
    need(
        task.task_hash == want["task_hash"] and task.qualified_name == TASK,
        "oracle dispatch task differs",
    )
    return dict(
        schema=INVOCATION,
        run_id=ctx["run_id"],
        evaluation_id=ctx["evaluation_id"],
        stage=stage,
        source_sha256=ctx["source_sha256"],
        task_hash=task.task_hash,
        launch_binding_sha256=digest(binding),
        expected=want,
        limits=dict(
            timeout=getattr(task, stage + "_timeout"),
            confirm_n=task.confirm_n if stage == "confirm" else None,
            cpu_seconds=task.cpu_seconds,
            memory_bytes=task.mem_bytes,
        ),
        relative_directory=relative_evaluation(ctx["evaluation_id"]) + "/" + stage,
    )


def prepare_stage(ctx, task, stage):
    binding = ctx["launch_binding"]
    value = invocation(ctx, task, stage)
    directory = ordinary(Path(ctx["run_root"]) / value["relative_directory"])
    # An existing dispatched attempt is not an invitation to retry paid work.
    need(
        not directory.exists(),
        "oracle stage already reserved; explicit recovery required",
    )
    evaluation = directory.parent
    evidence_root = evaluation.parent
    evidence_root.parent.mkdir(exist_ok=True)
    evidence_root.mkdir(exist_ok=True, mode=0o700)
    evaluation.mkdir(exist_ok=True, mode=0o700)
    directory.mkdir(mode=0o700)
    raw = directory / "raw"
    raw.mkdir(mode=0o700)
    write(directory / "invocation.json", value)
    env = dict(
        SCIEL_ORACLE_FIXED_STUDY=ctx["study_root"],
        SCIEL_ORACLE_FIXED_MANIFEST_SHA256=binding["study_manifest_sha256"],
        SCIEL_ORACLE_FIXED_PANEL=binding["panel_name"],
        SCIEL_ORACLE_FIXED_PANEL_SHA256=binding["panel"]["sha256"],
        SCIEL_ORACLE_IMAGE=binding["runtime"]["image_id"],
        SCIEL_ORACLE_FIXED_RECEIPTS=str(raw),
        SCIEL_ORACLE_FIXED_INVOCATION=str(directory / "invocation.json"),
    )
    return directory, value, env






from .oracle_stage_evidence import files, inspect_stage


def collect_stage(directory, invocation, result):
    write(directory / "runner.json", result.to_json())
    return inspect_stage(directory, invocation, result)


def validate_outcome(engine, outcome, source_sha256):
    """Reopen live/WAL files before admitting a returned cascade result."""
    stages = outcome.oracle_evidence
    ctx = context(engine, outcome.evaluation_id, source_sha256)
    need(
        ctx is not None and type(stages) is list and stages,
        "oracle requires stage evidence",
    )
    wanted = ["smoke", "full", "confirm"]
    need(
        outcome.reached in wanted and len(stages) == wanted.index(outcome.reached) + 1,
        "oracle attempted stage sequence differs",
    )
    results = []
    for entry, stage in zip(stages, wanted):
        inv = invocation(ctx, engine.task, stage)
        directory = ordinary(Path(ctx["run_root"]) / inv["relative_directory"])
        need(
            type(entry) is dict
            and canonical(entry.get("invocation")) == canonical(inv)
            and canonical(read(directory / "invocation.json")) == canonical(inv),
            "oracle replay invocation differs from actual dispatch",
        )
        original = EvalResult.from_json(read(directory / "runner.json"))
        result, actual = inspect_stage(directory, inv, original)
        need(
            canonical(actual) == canonical(entry),
            "oracle retained stage changed before commitment",
        )
        results.append(result)
    need(all(r.ok for r in results[:-1]), "oracle continued after a failed stage")
    terminal = results[-1]
    confirmed = False
    if outcome.reached == "confirm":
        need(
            engine.cfg.get("confirm_records", False) and engine.task.stochastic,
            "unexpected oracle confirmation",
        )
        trials = terminal.cases.get("n_trials", 0.0)
        enough = trials >= max(2, engine.task.confirm_n) and float(trials).is_integer()
        confirmed = bool(terminal.ok and terminal.confirmed_sampled and enough)
        if terminal.ok and not confirmed:
            terminal = replace(
                terminal,
                ok=False,
                score=0.0,
                artifacts={
                    **terminal.artifacts,
                    "error_type": "InsufficientConfirmation",
                    "error": "confirm did not provide all requested repeated samples",
                },
            )
    need(
        canonical(terminal.to_json()) == canonical(outcome.result.to_json())
        and outcome.confirmed is confirmed
        and outcome.raw_first_score == results[min(1, len(results) - 1)].score,
        "oracle cascade admission differs from retained stages",
    )


def capture(engine, outcome, saved_reference):
    stages = getattr(outcome, "oracle_evidence", None)
    if stages is None:
        need(
            getattr(engine, "oracle_evaluation", None) is None,
            "oracle evaluation lost stage evidence",
        )
        return None
    need(stages and isinstance(stages, list), "oracle requires stage evidence")
    validate_outcome(engine, outcome, saved_reference["source_sha256"])
    first = stages[0]["invocation"]
    sidecar = dict(
        schema=EVIDENCE,
        run_id=engine.log.run_id,
        evaluation_id=outcome.evaluation_id,
        source_sha256=saved_reference["source_sha256"],
        task_hash=engine.task.task_hash,
        launch_binding=deepcopy(engine.oracle_evaluation),
        saved_record_sha256=saved_reference["sha256"],
        terminal_stage=outcome.reached,
        stages=stages,
    )
    need(
        first["evaluation_id"] == sidecar["evaluation_id"]
        and first["run_id"] == sidecar["run_id"],
        "oracle stage belongs to another native evaluation",
    )
    directory = ordinary(
        Path(engine.run_dir).resolve(strict=True)
        / relative_evaluation(sidecar["evaluation_id"])
    )
    write(directory / "sidecar.json", sidecar)
    return digest(sidecar)
