"""Explicit known-defect acceptance fixture, not a scientific experiment."""
from __future__ import annotations

from copy import deepcopy
import fcntl
from pathlib import Path

from .budget import ResourceLedger
from .coordinator import Coordinator
from .executor import DockerExecutor
from .models import canonical_bytes, content_hash
from .proposals import ScriptedProposer
from .registry import Registry, _atomic
from .tasks import SortTask

RESOURCES = dict(wall_seconds=5, cpu_seconds=2, memory_bytes=128 * 1024 * 1024,
                 pids=32, max_output_bytes=65536, max_stderr_bytes=4096)
BUDGET = dict(dollars=0, cpu_seconds=1000, gpu_seconds=0, wall_seconds=1800,
              storage_bytes=100_000_000, steps=10)


def sort_bundle(image, resources, *, author="common-toolbox", repaired=True, parents=()):
    task = SortTask()
    files = {"method.py": "def run(values):\n    return sorted(values)\n" if repaired else
             "def run(values):\n    return sorted(set(values))\n"}
    manifest = dict(name="trusted_sort_control" if author == "common-toolbox" else "known_defect_sort_fixture",
        kind="tool", author_lab=author, entrypoint="method.py:run", input_schema=task.input_schema,
        output_schema=task.output_schema, protocol=dict(technical_fixture=True,
            description="Sort values while preserving duplicate multiplicities.",
            defect="none" if repaired else "deliberate removal of duplicates"),
        runtime_lock={"image_id": image}, dependencies=[], parents=list(parents),
        resources=deepcopy(resources), scope="Scripted engineering acceptance; no discovery or scientific benefit claim.", examples=[])
    return dict(id=content_hash(dict(schema="sciel-method/1", manifest=manifest, files=files)), manifest=manifest, files=files)


def experiment(method, control, inputs, *, question="Technical exact sorting check"):
    return dict(question=question, treatment=[method], control=[control], inputs=inputs,
                criterion=dict(metric="correct", direction="max", minimum_effect=0),
                scope="Assigned technical fixture inputs only; zero difference is permitted.")


def _artifact_rows(registry, kind):
    return [(event["data"]["id"], registry.get_artifact(kind, event["data"]["id"]))
            for event in registry.events() if event["kind"] == "artifact_registered" and event["data"]["kind"] == kind]


def technical(output, image, *, executor=None, capacity=None):
    """Run/recover the bounded scripted cycle; never call an LLM provider."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _technical(output, image, executor=executor, capacity=capacity)


def _technical(output, image, *, executor=None, capacity=None):
    runtime = {"image_id": image}
    common = sort_bundle(image, RESOURCES)
    broken = sort_bundle(image, RESOURCES, author="author", repaired=False)
    repaired = sort_bundle(image, RESOURCES, author="author", parents=[broken["id"]])
    local_inputs, recipient_inputs = [[3, 1, 3], [2, 0, 2, 1]], [[4, 2, 4, -1]]
    responses = {"author": [
        dict(kind="method", files=broken["files"], manifest=broken["manifest"],
             experiment=experiment("$new", common["id"], local_inputs)),
        dict(kind="method", files=repaired["files"], manifest=repaired["manifest"],
             experiment=experiment("$new", common["id"], local_inputs))],
        "recipient": [dict(kind="experiment", experiment=experiment(repaired["id"], common["id"], [[5, 1, 5, 0]],
            question="After adoption, invoke the exact accepted method in the next proposed experiment"))], "reviewer": []}
    spec = dict(schema="sciel-method-technical-fixture/1", runtime_lock=runtime, resources=RESOURCES,
                budget=BUDGET, responses_sha256=content_hash(responses),
                description="Scripted known-defect sort repair, independent replay, adoption and next-action invocation.")
    _atomic(output / "technical-config.json", canonical_bytes(spec), immutable=True)
    registry = Registry(output / "registry")
    ledger = ResourceLedger(output / "resources.jsonl", BUDGET)
    executor = executor or DockerExecutor(image)
    proposer = ScriptedProposer(responses)
    registry.register(common["files"], common["manifest"])
    kwargs = dict(resources=RESOURCES, common_methods=[common["id"]], max_steps=3, max_cases=4,
                  calendar_seconds=1800, storage_bytes=200_000_000, capacity=capacity, capacity_provider="local")
    coordinator = Coordinator(registry, ledger, executor, proposer, SortTask(), **kwargs)
    for lab, mode in (("author", "C"), ("reviewer", "B"), ("recipient", "C")):
        try:
            registry.lab(lab)
        except KeyError:
            coordinator.create_lab(lab, "Technical sorting fixture", lab, BUDGET, mode=mode)

    def step(lab, index):
        existing = [e["data"] for e in registry.events() if e["kind"] == "episode_step"
                    and e["data"]["lab_id"] == lab and e["data"]["index"] == index]
        value = existing[0] if existing else coordinator.step(lab)
        if value.get("refusal_id") or "experiment_id" not in value:
            raise RuntimeError("technical proposal was refused or did not execute; inspect preserved receipts")
        return value

    first, second = step("author", 0), step("author", 1)
    if first.get("method_id") != broken["id"] or second.get("method_id") != repaired["id"]:
        raise RuntimeError("scripted fixture method identity differs")
    bad_result = registry.get_artifact("experiment", first["experiment_id"])
    good_result = registry.get_artifact("experiment", second["experiment_id"])
    if bad_result["valid_treatment"] or not good_result["valid_treatment"] or good_result["mean_effect"] != 0:
        raise RuntimeError("known-defect and equal-control fixture conditions did not hold")
    replay_id = coordinator.replay(second["experiment_id"], "reviewer", assignment_id="technical-replay")
    release_id = coordinator.release(repaired["id"], second["experiment_id"], replay_id)
    plans = [(ident, plan) for ident, plan in _artifact_rows(registry, "experiment_plan")
             if plan["proposal_id"] == "technical-recipient-trial"]
    if plans:
        trial_plan_id, trial_plan = plans[0]
    else:
        trial_plan_id = coordinator.plan("recipient", "technical-recipient-trial",
            experiment(repaired["id"], common["id"], recipient_inputs), purpose="transfer",
            selected=[repaired["id"]], input_origin="fixed_technical_recipient_assignment")
        trial_plan = registry.get_artifact("experiment_plan", trial_plan_id)
    trial_id = coordinator.execute(trial_plan_id)
    transfer_result = registry.get_artifact("experiment", trial_id)
    if not transfer_result["valid_treatment"] or transfer_result["mean_effect"] != 0:
        raise RuntimeError("recipient fixture did not reproduce the zero-effect control comparison")
    adoption_id = coordinator.adopt("recipient", repaired["id"], trial_id,
                                   expected_revision=trial_plan["lab_revision_id"])
    adoption = registry.get_artifact("adoption", adoption_id)
    following = step("recipient", 0)
    following_plan = registry.get_artifact("experiment_plan", following["plan_id"])
    following_result = registry.get_artifact("experiment", following["experiment_id"])
    if (following_plan["lab_revision_id"] != adoption["new_revision"]
            or repaired["id"] not in following_result["invoked_methods"]):
        raise RuntimeError("adoption did not change the next proposed/executed action")
    proposal = registry.get_artifact("proposal_receipt", following["proposal_id"])
    if repaired["id"] not in proposal["method_refs"]:
        raise RuntimeError("accepted method was absent from the next proposal context")

    checkpoint_path = output / "checkpoint.json"
    checkpoint = coordinator.checkpoint(checkpoint_path)
    before = ledger.snapshot()
    revision_before = registry.lab("recipient")["active_revision"]
    reopened = Registry(output / "registry")
    reopened.verify_checkpoint(checkpoint_path)
    reopened_ledger = ResourceLedger(output / "resources.jsonl", BUDGET)
    restored = Coordinator(reopened, reopened_ledger, executor, ScriptedProposer(responses), SortTask(), **kwargs)
    original_execute = executor.execute
    def forbidden(*args, **kwargs):
        raise AssertionError("cached acceptance replay attempted another external invocation")
    executor.execute = forbidden
    try:
        cached_experiment = restored.execute(following["plan_id"])
        cached_call = restored._invoke(following["plan_id"], following_plan, "treatment", 0, 0,
                                      repaired["id"], following_plan["inputs"][0])
    finally:
        executor.execute = original_execute
    if (cached_experiment != following["experiment_id"] or not cached_call["replayed"]
            or reopened_ledger.snapshot() != before or reopened.lab("recipient")["active_revision"] != revision_before):
        raise RuntimeError("resume changed the receipt, budget or active revision")
    report = dict(schema="sciel-method-technical-acceptance/1", status="passed", technical_only=True,
        llm_provider_calls=0, scripted_proposals=3, known_defect="deduplication destroys multiplicities",
        common_control=common["id"], broken_method=broken["id"], repaired_method=repaired["id"],
        experiments=dict(broken=first["experiment_id"], repaired=second["experiment_id"], replay=replay_id,
                         recipient_trial=trial_id, subsequent=following["experiment_id"]),
        repaired_vs_control_effect=good_result["mean_effect"],
        transfer_vs_control_effect=transfer_result["mean_effect"], release_id=release_id,
        adoption_id=adoption_id, adopted_revision=adoption["new_revision"], final_revision=revision_before,
        next_proposal_received_method=True, next_action_invoked_method=True,
        checkpoint_id=checkpoint["id"], cached_receipt_reused=True, cached_replay_new_executions=0,
        resource_ledger=before, executor_identity=executor.identity,
        limitations=["Known scripted defect and supplied repair; no autonomous discovery claim.",
                     "Sorted control matches the repaired method: no positive scientific benefit was demonstrated.",
                     "Reviewer has a separate history; deterministic replay does not establish generalization."])
    # Repeating the entire command revalidates receipts, without replacing history.
    _atomic(output / "acceptance.json", canonical_bytes(report))
    return report
