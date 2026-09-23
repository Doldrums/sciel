"""Read only checkpoint-committed native SCiEL episodes with exact provenance."""
from __future__ import annotations

import json
from pathlib import Path

from ..methods.models import content_hash as native_hash
from .dataset import make_episode, task_receipt
from .storage import read_json, write_json
from .types import InferenceSpec, canonical_bytes, content_hash, sha256_bytes
from .target_policy import LEGACY_POLICY, POLICY


def _receipt(root: Path, evaluation_id: str, source_hash: str, run_id: str) -> dict:
    directory = root / "research_evidence" / "evaluations"
    index = read_json(directory / ("evaluation-" + sha256_bytes(evaluation_id.encode()) + ".json"))
    if (index.get("evaluation_id") != evaluation_id or index.get("source_sha256") != source_hash
            or index.get("run_id") != run_id):
        raise ValueError("evaluation index differs from committed source/run")
    from .types import require_hash
    require_hash(index.get("sha256"), "native evaluation artifact hash")
    record = read_json(directory / (index["sha256"] + ".json"))
    if native_hash(record) != index["sha256"]:
        raise ValueError("native evaluation artifact hash mismatch")
    if any(record.get(k) != index.get(k) for k in
           ("evaluation_id", "source_sha256", "run_id", "task_hash")):
        raise ValueError("native evaluation receipt identity mismatch")
    return record


def _task_receipt(receipt: dict, *, evaluation_context=None) -> dict:
    return task_receipt(receipt, evaluation_context=evaluation_context)


def _isolated_bootstrap(config: dict) -> bool:
    """A narrow provable profile; broader contexts need explicit visibility IDs."""
    return (config.get("num_islands") == 1
        and all(config.get(key) is False for key in
                ("crossover", "portfolio_seed", "dynamic_islands", "idea_island", "kb_free", "c2_event_bets"))
        and config.get("research", {}).get("enabled") is False
        and config.get("research_tools", {}).get("enabled") is False
        and not config.get("warm_start"))


def export_native(run_dir: str | Path, output_dir: str | Path, *,
                  checkpoint_path: str | Path | None = None,
                  ancestry_sources: dict | None = None, target_policy: str = LEGACY_POLICY) -> dict:
    if target_policy not in (LEGACY_POLICY, POLICY):
        raise ValueError("unsupported native export target policy")
    root = Path(run_dir)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else root / "checkpoint.json"
    checkpoint = read_json(checkpoint_path)
    provenance = checkpoint.get("provenance", {})
    anchor = provenance.get("journal_anchor", {})
    offset = anchor.get("byte_offset")
    if type(offset) is not int or offset < 1 or not provenance.get("run_id"):
        raise ValueError("export requires a checkpoint with a committed journal anchor")
    with (root / "sciel_events.jsonl").open("rb") as stream:
        data = stream.read(offset)
    if len(data) != offset or not data.endswith(b"\n") or sha256_bytes(data) != anchor.get("prefix_sha256"):
        raise ValueError("journal prefix differs from checkpoint commitment")
    events = [json.loads(line) for line in data.splitlines()]
    run_id = provenance["run_id"]
    from .evaluation_evidence import native_context, retain as retain_evaluations
    evaluation_context = native_context(root, checkpoint_path)
    events = [e for e in events if e.get("run_id") == run_id]
    evaluations = [e for e in events if e.get("event") == "decision"
                   and e.get("policy") == "evaluation" and e.get("action") == "result"]
    config = checkpoint.get("config", {})
    isolated = _isolated_bootstrap(config)
    trace_records = []
    trace_path = root / "artifacts" / "llm_calls.jsonl"
    if trace_path.exists():
        for line in trace_path.read_bytes().splitlines():
            try:
                trace_records.append(json.loads(line))
            except json.JSONDecodeError:
                # An uncommitted trailing transcript cannot make earlier data eligible.
                continue
    transcripts = {}
    for event in trace_records:
        key = event.get("call_id"), event.get("event")
        if key in transcripts and transcripts[key] != event:
            raise ValueError("conflicting transcript call identity")
        transcripts[key] = event
    from .ancestry import NativeAncestry
    ancestry = NativeAncestry(root, checkpoint, events, transcripts, sources=ancestry_sources)
    rows, excluded = [], []
    for event in events:
        if event.get("event") != "iteration":
            continue
        episode_id = f"{event.get('segment_id')}-{event.get('event_seq')}"
        try:
            target = event["child_code"]
            source_hash = sha256_bytes(target.encode())
            meta = event["meta"]
            attempts = [a for a in meta.get("validation_trace", {}).get("attempts", [])
                        if a.get("disposition") == "evaluate"
                        and a.get("candidate_code_sha256") == source_hash
                        and a.get("extracted_code_sha256") == source_hash
                        and a.get("call_id") and a.get("inference_spec")]
            if len(attempts) != 1:
                raise ValueError("candidate has no unique unrepaired generated source")
            attempt = attempts[0]
            spec = InferenceSpec.from_dict(attempt["inference_spec"])
            if spec.role not in ("G", "repair"):
                raise ValueError("candidate did not come from an adapted generation role")
            local = config.get("llm", {}).get("local_inference", {})
            if local.get("learner_by_island", {}).get(str(event["island"])) != spec.owner:
                raise ValueError("producer owner differs from checkpoint learner mapping")
            for key in ("base_model", "base_revision", "tokenizer_revision", "chat_template_sha256"):
                if getattr(spec, key) != local.get(key):
                    raise ValueError("generation model identity differs from checkpoint")
            call = attempt["call_id"]
            request = transcripts[(call, "request")]
            response = transcripts[(call, "response")]
            if request.get("run_id") != run_id or response.get("run_id") != run_id:
                raise ValueError("generation transcript belongs to another run")
            from .types import require_identifier
            require_identifier(call, "call_id")
            prompt = (root / "artifacts" / "llm_calls" / f"{call}.prompt.txt").read_bytes().decode("utf-8")
            text = (root / "artifacts" / "llm_calls" / f"{call}.response.txt").read_bytes().decode("utf-8")
            if sha256_bytes(prompt.encode()) != request.get("prompt_sha256"):
                raise ValueError("generation prompt hash mismatch")
            if prompt != event.get("prompt"):
                raise ValueError("generation prompt differs from committed iteration")
            if (sha256_bytes(text.encode()) != response.get("response_sha256")
                    or response.get("response_sha256") != attempt.get("response_sha256")):
                raise ValueError("generation response hash mismatch")
            if request.get("inference_spec") != spec.to_dict() or response.get("inference_spec") != spec.to_dict():
                raise ValueError("generation revision differs across receipts")
            visibility = attempt.get('adaptation_visibility')
            resolution = None
            if visibility is not None or any('adaptation_visibility' in row for row in (request, response)):
                from .visibility import validate_capsule
                validate_capsule(visibility, prompt=prompt, inference_spec=spec.to_dict())
                if any(row.get('adaptation_visibility') != visibility for row in (request, response)):
                    raise ValueError('generation visibility differs across receipts')
                resolution = ancestry.resolve(event)
            eid = meta["evaluation_id"]
            matches = [e for e in meta.get("evaluations", []) if e.get("evaluation_id") == eid
                       and e.get("source_sha256") == source_hash and e.get("ok") is True]
            if len(matches) != 1:
                raise ValueError("candidate evaluation lacks a committed identity")
            receipt = _receipt(root, eid, source_hash, run_id)
            if receipt.get("task_hash") != checkpoint.get("identity", {}).get("task_hash"):
                raise ValueError("evaluation task differs from committed checkpoint")
            child_task = _task_receipt(receipt, evaluation_context=evaluation_context)
            parent = meta["adaptation_parent"]
            parent_receipt = None
            parent_evaluation_id = ancestry.parent_evaluation_id(event) if resolution is not None else None
            if evaluation_context is not None:
                parent_evaluation_id = evaluation_context.parent_evaluation_id(event)
            for e in evaluations:
                if (e.get("source_sha256") == parent["source_sha256"]
                        and e.get("score") == parent["score"] and e.get("ok") is True
                        and (parent_evaluation_id is None or e.get('evaluation_id') == parent_evaluation_id)):
                    candidate = _receipt(root, e["evaluation_id"], parent["source_sha256"], run_id)
                    try:
                        parent_task = _task_receipt(candidate, evaluation_context=evaluation_context)
                    except ValueError:
                        continue
                    if parent_task["instance_set_sha256"] == child_task["instance_set_sha256"]:
                        parent_receipt = candidate
                        break
            if parent_receipt is None:
                raise ValueError("parent score has no paired committed train evaluation")
            visible = meta.get("adaptation_visible_episode_ids")
            if resolution is not None:
                if visible is not None and visible != resolution['prompt_episode_ids']:
                    raise ValueError('legacy visibility differs from authenticated prompt episodes')
                visible = resolution['prompt_episode_ids']
            if visible is None:
                if not isolated or spec.adapter_id is not None:
                    raise ValueError("experience visibility is unknown outside isolated frozen bootstrap")
                # Legacy first-step pilots had no capsule. Own ownership alone
                # cannot certify empty delivery in later native generations.
                prior = events[:ancestry.positions[(event['segment_id'], event['event_seq'])]]
                admission = ancestry.programs.get(parent['program_id'])
                if (any(e.get('event') == 'iteration' for e in prior) or admission is None
                        or admission[1].get('action') != 'seed_program'
                        or admission[1].get('source_sha256') != parent['source_sha256']):
                    raise ValueError('legacy generation visibility is unknown beyond the public first step')
                visible = []
            target_options = {}
            if target_policy == POLICY:
                from .native_target import bind_native_response_target
                hypotheses = [e for e in events if e.get("event") == "research"
                    and e.get("phase") == "hypothesis" and e.get("segment_id") == event["segment_id"]
                    and e.get("experiment_id") == meta.get("research_id")]
                admitted_evaluations = [e for e in evaluations if e.get("evaluation_id") == eid
                    and e.get("segment_id") == event["segment_id"]]
                if len(hypotheses) != 1 or len(admitted_evaluations) != 1:
                    raise ValueError("native target requires unique hypothesis and evaluation admissions")
                evidence = dict(accepted_hypothesis=meta["hypothesis"],
                    validation_attempts=meta["validation_trace"]["attempts"], response_receipt=response,
                    hypothesis_event=hypotheses[0], evaluation_event=admitted_evaluations[0])
                bound = bind_native_response_target(raw_response=text, evaluated_source=target,
                    **{k: evidence[k] for k in ("accepted_hypothesis", "validation_attempts", "response_receipt")})
                target_options = dict(target_binding=bound, target_evidence=evidence)
                target = text
            oracle_binding = {}
            if evaluation_context is not None:
                oracle_binding = {'oracle_evaluation': evaluation_context.row_binding(dict(
                    evaluation_receipt=receipt, parent_score=parent['score'], provenance=dict(
                        run_id=run_id, segment_id=event['segment_id'], event_seq=event['event_seq'],
                        checkpoint_sha256=sha256_bytes(checkpoint_path.read_bytes()), journal_anchor=anchor,
                        parent_evaluation_receipt=parent_receipt)))}
            row = make_episode(episode_id=episode_id, owner=spec.owner, prompt=prompt,
                ancestry_sources=ancestry.adapter_context,
                evaluation_context=evaluation_context,
                **target_options,
                target=target, parent_score=parent["score"], evaluation_receipt=receipt,
                visible_episode_ids=visible, provenance=dict(
                    **oracle_binding,
                    run_id=run_id, segment_id=event['segment_id'], event_seq=event["event_seq"], call_id=call,
                    inference_spec=spec.to_dict(), parent_source_sha256=parent["source_sha256"],
                    parent_evaluation_receipt=parent_receipt,
                    evaluation_split="train", instance_set_sha256=child_task["instance_set_sha256"],
                    checkpoint_sha256=sha256_bytes(checkpoint_path.read_bytes()),
                    **({'adaptation_visibility': visibility} if visibility is not None else {}),
                    **({'ancestry_resolution': resolution} if resolution is not None else {}),
                    **({'producer_call_binding': ancestry.producer_binding(event, resolution)}
                       if resolution is not None and resolution.get('auxiliary_bindings') else {}),
                    journal_anchor=anchor, config_sha256=content_hash(config),
                    visibility_policy=('authenticated-native-ancestry' if resolution is not None
                        else "isolated-frozen-bootstrap" if isolated else "explicit-record-ids")))
            rows.append(row)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            excluded.append({"episode_id": episode_id, "reason": str(exc)})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    path = out / "episodes.jsonl"
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("episode output already contains a different export")
    else:
        path.write_bytes(payload)
    result = dict(schema="sciel-lora-native-export/1", run_id=run_id,
        checkpoint_sha256=sha256_bytes(checkpoint_path.read_bytes()), journal_anchor=anchor,
        episodes_sha256=sha256_bytes(payload), eligible_count=len(rows), exclusions=excluded,
        events_in_committed_prefix=len(events), isolated_frozen_bootstrap=isolated)
    if target_policy == POLICY:
        result.update(schema="sciel-lora-native-export/2", target_policy=POLICY)
    retained_evaluations = retain_evaluations(rows, evaluation_context, out)
    if retained_evaluations is not None:
        result['evaluation_evidence'] = retained_evaluations
    write_json(out / "export.json", result, immutable=True)
    return result
