"""Research CLI. Configuration is JSON; credentials come only from environment.

`technical` is a scripted engineering fixture. `run` is a bounded arm or an
explicit continuous run, not a statistical campaign comparison.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .budget import BudgetExceeded, BudgetUnknown, ResourceLedger
from .campaign import upfront_provenance
from .configuration import _no_credentials, validate_config
from .coordinator import Coordinator, Refusal, TrialQuotaExceeded
from .executor import DockerExecutor
from .models import canonical_bytes, content_hash, require_hash
from .proposals import BudgetedProposer
from .registry import Registry, _atomic
from .tasks import Circle26Task, SortTask
from .technical import _artifact_rows, sort_bundle, technical


def shared_capacity(provider="openrouter.ai"):
    """Use the engine's existing host root and immutable pool limits, no bypass."""
    from ..engine.capacity import HostCapacity, SCHEMA
    baseline = HostCapacity()
    def limit(pool, fallback):
        path = baseline.root / (hashlib.sha256(pool.encode()).hexdigest() + ".json")
        if path.is_symlink():
            raise ValueError("host capacity policy cannot be a symlink")
        if not path.exists():
            return fallback
        value = json.loads(path.read_bytes())
        if value.get("schema") != SCHEMA or value.get("pool") != pool or type(value.get("limit")) is not int or value["limit"] <= 0:
            raise ValueError("invalid existing host capacity policy")
        return value["limit"]
    llm, evaluator = limit("llm", baseline.llm_slots), limit("evaluator", baseline.evaluator_slots)
    providers = {provider: limit("provider:" + provider, llm)}
    return HostCapacity(baseline.root, llm_slots=llm, evaluator_slots=evaluator, provider_limits=providers)






def _initial_methods(registry, config):
    values = list(config["common_methods"])
    if config["common_toolbox"] == "sort":
        values.append(sort_bundle(config["image_id"], config["resources"]))
    elif config["common_toolbox"] == "circle26":
        from .examples import circle_toolbox
        for value in circle_toolbox({"image_id": config["image_id"]}, config["resources"]).values():
            values.append(dict(value, id=content_hash(dict(schema="sciel-method/1", **value))))
    by_id = {value["id"]: value for value in values}
    if len(by_id) != len(values):
        raise ValueError("duplicate common bundle ID")
    for ident, value in by_id.items():
        if set(value) != {"id", "manifest", "files"} or ident != content_hash(dict(
                schema="sciel-method/1", manifest=value["manifest"], files=value["files"])):
            raise ValueError("common bundle hash mismatch")
    done, pending = [], set(by_id)
    while pending:
        ready = [ident for ident in sorted(pending)
                 if not (set(by_id[ident]["manifest"]["dependencies"] + by_id[ident]["manifest"]["parents"]) - set(done))]
        if not ready:
            raise ValueError("common bundle dependency closure is missing or cyclic")
        for ident in ready:
            value = by_id[ident]
            registered = registry.register(value["files"], value["manifest"])
            if registered != ident:
                raise ValueError("common bundle hash mismatch")
            done.append(ident); pending.remove(ident)
    return done


def _assigned_plan(coordinator, lab, assignment, protocol, **kwargs):
    """A resumed routing operation keeps its originally pinned lab revision."""
    existing = [(ident, value) for ident, value in _artifact_rows(coordinator.registry, "experiment_plan")
                if value["proposal_id"] == assignment and value["lab_id"] == lab]
    if len(existing) > 1:
        raise ValueError("routing assignment has more than one frozen plan")
    if existing:
        return existing[0]
    ident = coordinator.plan(lab, assignment, protocol, **kwargs)
    return ident, coordinator.registry.get_artifact("experiment_plan", ident)


def _quality_report(registry):
    solutions = [dict(id=ident, lab_id=value["lab_id"], valid=value["evaluation"]["ok"],
                      score=value["evaluation"]["score"], cases=value["evaluation"]["cases"],
                      source_sha256=content_hash(value["program"]["code"]))
                 for ident, value in _artifact_rows(registry, "solution")]
    experiments = [dict(id=ident, plan_id=value["plan_id"], valid_treatment=value["valid_treatment"],
                        contract_passed=value["contract_passed"], mean_effect=value["mean_effect"],
                        rule_passed=value["rule_passed"])
                   for ident, value in _artifact_rows(registry, "experiment")]
    return dict(scope="Descriptive assigned-case observations; no held-out or independent-repeat aggregate.",
                solutions=solutions, experiments=experiments)


def _route(coordinator, outcome, config):
    routing = config.get("routing")
    if not routing or config["mode"] == "A" or outcome["lab_id"] != routing["author"]:
        return
    registry = coordinator.registry
    existing = [e["data"] for e in registry.events() if e["kind"] == "cli_routing"]
    if config["mode"] == "C" and config["transfer_policy"] == "solution_only" and "solution_id" in outcome:
        solution_id = outcome["solution_id"]
        solution_routes = [row for row in existing if row.get("kind") == "solution_only"]
        if any(row["solution_id"] == solution_id for row in solution_routes) or (routing["max_transfers"] is not None and len(solution_routes) >= routing["max_transfers"]):
            return
        record = dict(kind="solution_only", solution_id=solution_id,
                      recipient=routing["recipient"], status="started")
        try:
            transfer = coordinator.transfer_solution(routing["recipient"], solution_id)
            receipt = registry.get_artifact("solution_transfer", transfer)
            record.update(status="completed", transfer_id=transfer, source_sha256=receipt["source_sha256"],
                          attribution="Validated solution outputs made available to recipient context; no program source, method adoption or use claim.")
        except Refusal as exc:
            record.update(status="refused", reason=str(exc)[:1000])
        registry.append("cli_routing", record)
        return
    if "method_id" not in outcome or "experiment_id" not in outcome:
        return
    method, original = outcome["method_id"], outcome["experiment_id"]
    method_routes = [row for row in existing if row.get("kind") == "method"]
    if any(e["method_id"] == method for e in method_routes) or (routing["max_transfers"] is not None and len(method_routes) >= routing["max_transfers"]):
        return
    trial = registry.get_artifact("experiment", original)
    if not trial["contract_passed"] or not trial["rule_passed"]:
        return
    record = dict(kind="method", method_id=method, author_experiment=original, status="started")
    try:
        local = registry.get_artifact("experiment_plan", trial["plan_id"])
        protocol = {key: local[key] for key in ("question", "treatment", "control", "inputs", "criterion", "scope")}
        if config["library_policy"] == "verified":
            replay_plan, _ = _assigned_plan(coordinator, routing["reviewer"], "cli-replay:" + method,
                protocol, purpose="replay", selected=local["treatment"] + local["control"],
                source_plan=trial["plan_id"], input_origin="independent_coordinator_replay_assignment",
                history_id=routing["reviewer"])
            replay = coordinator.execute(replay_plan)
            record["replay_id"] = replay
            record["release_id"] = coordinator.release(method, original, replay)
        else:
            coordinator.publish_unreplayed(method, original)
            record["evidence_level"] = "unreplayed"
        # The author's local trial belongs to an earlier immutable revision;
        # run its frozen protocol against the current revision before adoption.
        local_plan, local_body = _assigned_plan(coordinator, routing["author"], "cli-own-adoption:" + method, protocol)
        local_trial = coordinator.execute(local_plan)
        record["author_adoption_id"] = coordinator.adopt(routing["author"], method, local_trial,
                                                        expected_revision=local_body["lab_revision_id"])
        if config["mode"] == "C" and config["transfer_policy"] == "methods":
            recipient = routing["recipient"]
            protocol.update(question="Preassigned recipient method trial", treatment=[method],
                            control=routing["control"], inputs=routing["transfer_inputs"])
            plan, plan_body = _assigned_plan(coordinator, recipient, "cli-transfer:" + method, protocol,
                                    purpose="transfer", selected=[method], input_origin="frozen_cli_routing_assignment")
            recipient_trial = coordinator.execute(plan)
            record["recipient_trial_id"] = recipient_trial
            record["recipient_adoption_id"] = coordinator.adopt(recipient, method, recipient_trial,
                                                               expected_revision=plan_body["lab_revision_id"])
        record["status"] = "completed"
    except (Refusal, TrialQuotaExceeded) as exc:
        record.update(status="refused", reason=str(exc)[:1000])
    registry.append("cli_routing", record)


@contextmanager
def _stop_scope(output):
    """Finish an in-flight operation before honouring a local stop request."""
    requested = threading.Event()
    marker = Path(output) / "stop.request"
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: requested.set())
    try:
        yield lambda: requested.is_set() or marker.exists()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _continue_lab(coordinator, lab_id, previous):
    """Audited operational renewal, with no credit or fresh resource account."""
    state = coordinator.lab_state(lab_id)
    if state['status'] == 'archived':
        return False
    if not previous:
        return True
    last = previous[-1]
    evidence = list(dict.fromkeys(ref for ref in (
        last.get('proposal_id'), last.get('experiment_id'), last.get('refusal_id')) if ref))
    question = (
        f"Continuous-run assignment after recorded step {last['index']}: use its evidence "
        "to propose a materially different mechanism or repair the observed failure. "
        "Do not repeat an exhausted identical-solution attempt. Queue waits are not "
        "algorithm runtime or evidence of ineffectiveness. This renews one research "
        "trial, not resources, efficacy evidence or release authority."
    )
    if last['status'] == 'stopped':
        coordinator.reopen_episode(lab_id, question=question, evidence_refs=evidence)
    if coordinator.lab_state(lab_id)['status'] == 'dormant':
        coordinator.renew_trial(lab_id, 1, question=question, evidence_refs=evidence)
    if last.get('refusal_id'):
        refusal = coordinator.registry.get_artifact('proposal_receipt', last['refusal_id'])
        lineage = refusal.get('funding_lineage')
        if refusal.get('failure_scope') == 'trial_policy' and lineage:
            policy = coordinator._funding().snapshot(lineage)
            if policy['status'] == 'dormant':
                coordinator.renew_trial(lab_id, 1, question=question, evidence_refs=evidence,
                                        lineage_id=lineage)
    return True


def run_config(config, output):
    """A real run. Only calling this function may reach the provider."""
    with _stop_scope(output) as should_stop:
        return _run_config(config, output, should_stop)


def _run_config(config, output, should_stop):
    from ..campaign.budget import BudgetedLLM, CampaignStopped, CostLedger
    from .provider_budget import UnlimitedCostLedger
    from .coordinator import StopRequested
    config = validate_config(config)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required in the environment")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _atomic(output / "run-config.json", canonical_bytes(config), immutable=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = Registry(output / "registry")
        ledger = ResourceLedger(output / "resources.jsonl", config["budget"])
        ledger.create_account("campaign.setup", {})
        setup = ledger.reserve("campaign.setup", "bootstrap:" + uuid.uuid4().hex,
            dict(dollars=0, cpu_seconds=30, gpu_seconds=0, wall_seconds=120, storage_bytes=10_000_000, steps=0), "bootstrap")
        def registry_size():
            return sum(p.stat().st_size for p in registry.root.rglob('*') if p.is_file())
        cpu, wall, size_before = time.process_time(), time.monotonic(), registry_size()
        try:
            task = SortTask() if config["task"] == "sort" else Circle26Task()
            executor = DockerExecutor(config["image_id"])
            common = _initial_methods(registry, config)
            if config.get("routing") and not set(config["routing"]["control"]) <= set(common):
                raise ValueError("routing controls must be pinned common toolbox versions")
            provider = urlsplit(config["llm"]["api_base"]).hostname
            capacity = shared_capacity(provider)
            money_ledger_type = UnlimitedCostLedger if config['budget']['dollars'] is None else CostLedger
            money_ledger = money_ledger_type(output / "provider-cost.json", config["budget"]["dollars"])
            llm = BudgetedLLM(config["llm"], config["pricing"], money_ledger, content_hash(config), key)
            proposer = BudgetedProposer(llm, config["model"], temperature=config["llm"].get("temperature", .7),
                                       timeout=config["llm"].get("timeout", 120))
            coordinator = Coordinator(registry, ledger, executor, proposer, task, resources=config["resources"],
                common_methods=common, max_steps=config["max_steps"], max_cases=config["max_cases"],
                calendar_seconds=config["calendar_seconds"], storage_bytes=config["storage_bytes"],
                library_policy=config["library_policy"], transfer_policy=config["transfer_policy"],
                capacity=capacity, capacity_provider=provider,
                capacity_wait_seconds=config.get('capacity_wait_seconds', 60), should_stop=should_stop)
            for lab in config["labs"]:
                try:
                    registry.lab(lab["id"])
                except KeyError:
                    coordinator.create_lab(lab["id"], lab["agenda"], lab["account_id"], lab["limits"], mode=config["mode"])
        except BaseException:
            ledger.settle(setup["id"], dict(dollars=0, cpu_seconds=time.process_time()-cpu, gpu_seconds=0,
                wall_seconds=time.monotonic()-wall, storage_bytes=max(0, registry_size()-size_before), steps=0), status="failed")
            raise
        ledger.settle(setup["id"], dict(dollars=0, cpu_seconds=time.process_time()-cpu, gpu_seconds=0,
            wall_seconds=time.monotonic()-wall,
            storage_bytes=max(0, registry_size()-size_before), steps=0))
        # Explicit historical allocations are not added a second time to the
        # provider money ledger, which solely controls actual transport spending.
        for entry in config.get("upfront_costs", []):
            reservation = ledger.reserve("campaign.setup", "upfront:" + require_hash(entry["id"]),
                                         entry["resources"], "upfront_creation")
            if reservation["dispatch_allowed"]:
                ledger.settle(reservation["id"], entry["resources"])
        status, reason, steps_executed = "completed", None, 0
        continuous = config['claim'] == 'continuous_run'
        try:
            index = 0
            if continuous:
                histories = {lab['id']: [e['data'] for e in registry.events()
                    if e['kind'] == 'episode_step' and e['data']['lab_id'] == lab['id']] for lab in config['labs']}
                # At most the last committed step of a lab can precede an
                # interrupted route. Earlier routes completed before later steps.
                for previous in histories.values():
                    if should_stop():
                        raise StopRequested('operator stop requested')
                    if previous:
                        _route(coordinator, previous[-1], config)
                index = min(map(len, histories.values()))
            while config['max_steps'] is None or index < config['max_steps']:
                available = False
                for lab in config["labs"]:
                    if should_stop():
                        raise StopRequested('operator stop requested')
                    previous = [e["data"] for e in registry.events() if e["kind"] == "episode_step" and e["data"]["lab_id"] == lab["id"]]
                    if not continuous and previous and previous[-1]["status"] == "stopped":
                        continue
                    if len(previous) > index:
                        # Route any persisted successful step left before a crash.
                        _route(coordinator, previous[index], config)
                        available = True
                        continue
                    if continuous and not _continue_lab(coordinator, lab['id'], previous):
                        continue
                    available = True
                    outcome = coordinator.step(lab["id"])
                    steps_executed += 1
                    if not should_stop():
                        _route(coordinator, outcome, config)
                    if steps_executed % config.get('checkpoint_interval', 1) == 0:
                        coordinator.checkpoint(output / 'checkpoint.json')
                if not available:
                    reason = 'all_labs_archived_or_stopped'
                    break
                index += 1
        except StopRequested:
            status, reason = 'stopped', 'operator_requested_stop'
        except (BudgetExceeded, BudgetUnknown, CampaignStopped) as exc:
            status = "accounting_stopped" if isinstance(exc, (BudgetUnknown, CampaignStopped)) else "budget_exhausted"
            reason = str(exc)[:1000]
        finally:
            checkpoint = coordinator.checkpoint(output / "checkpoint.json")
        accounting = ledger.snapshot()
        provenance = upfront_provenance(config)
        declared_complete = ("upfront_costs" in config and not accounting["halted"]
                             and not accounting["unknown_reservations"] and not accounting["pending_reservations"])
        report = dict(schema="sciel-method-run-report/1", status=status, reason=reason,
            classification=config["claim"], scientific_claims=False, mode=config["mode"],
            configuration_sha256=content_hash(config), coordinator_identity=coordinator.identity,
            checkpoint_id=checkpoint["id"], resource_ledger=accounting,
            provider_cost_guard=money_ledger.snapshot(),
            accounting_note="The resource ledger includes provider cost; never sum the two ledgers. It is a known lower bound: outer controller/serialization overhead and Docker daemon/VM CPU are not fully charged.",
            accounting_coverage=dict(full_runtime_cost=False, provider_charges=True,
                executor_cgroup_cpu=True, controller_callbacks=True,
                all_controller_overhead=False, filesystem_metadata=False, docker_daemon_cpu=False),
            upfront_declared="upfront_costs" in config,
            upfront_provenance=provenance, declared_accounting_complete=declared_complete,
            cost_status="known_lower_bound_or_unresolved",
            quality=_quality_report(registry),
            outcomes=[e["data"] for e in registry.events() if e["kind"] == "episode_step"],
            steps_executed_this_invocation=steps_executed,
            routing=[e["data"] for e in registry.events() if e["kind"] == "cli_routing"])
        _atomic(output / "run-report.json", canonical_bytes(report))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded or continuous research-practices execution.")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("technical", help="scripted known-defect sorting cycle in Docker; no provider calls")
    fixture.add_argument("--output", required=True, type=Path)
    fixture.add_argument("--image", required=True, help="already available immutable Docker sha256 image ID")
    run = sub.add_parser("run", help="execute a frozen JSON config; uses OPENROUTER_API_KEY")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    stop = sub.add_parser('stop', help='request a graceful stop after in-flight operations settle')
    stop.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'stop':
            if not (args.output / 'run-config.json').is_file():
                raise ValueError('stop requires an existing methods run')
            _atomic(args.output / 'stop.request', canonical_bytes(dict(reason='operator_requested_stop', at=time.time())))
            print(json.dumps(dict(status='stop_requested', output=str(args.output))))
            return 0
        if args.command == "technical":
            report = technical(args.output, args.image, capacity=shared_capacity("local"))
        else:
            report = run_config(json.loads(args.config.read_text()), args.output)
        print(json.dumps({"status": report["status"], "output": str(args.output), "scientific_claims": False}, ensure_ascii=True))
        return 0 if report["status"] in ("passed", "completed", "stopped") else 2
    except (ValueError, RuntimeError, OSError, NotImplementedError) as exc:
        # Provider exceptions must not expose credential-bearing SDK objects.
        print(json.dumps({"status": "refused", "error_type": type(exc).__name__,
                          "message": str(exc)[:1000] if isinstance(exc, (ValueError, BudgetExceeded, BudgetUnknown)) else "Inspect the preserved local artifacts; no automatic retry."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
