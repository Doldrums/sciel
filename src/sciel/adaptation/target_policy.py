"""Versioned episode target semantics; source identity always names evaluated code."""
from __future__ import annotations

from pathlib import Path

from .native_target import POLICY, validate_native_response_target
from .types import content_hash

LEGACY_POLICY = "evaluated-code/1"
EPISODE_V2 = "sciel-lora-episode/2"
DATASET_V2 = "sciel-lora-dataset/2"


def policy(row):
    if row.get("schema") == "sciel-lora-episode/1":
        if any(k in row for k in ("evaluated_source", "target_sha256", "target_binding", "target_evidence")):
            raise ValueError("legacy episode cannot carry native target policy fields")
        return LEGACY_POLICY
    if row.get("schema") == EPISODE_V2:
        return POLICY
    raise ValueError("unsupported episode target schema")


def binding_inputs(row):
    evidence = row.get("target_evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
            "accepted_hypothesis", "validation_attempts", "response_receipt",
            "hypothesis_event", "evaluation_event"}:
        raise ValueError("native target evidence is missing or unsupported")
    return dict(raw_response=row["target"], evaluated_source=row["evaluated_source"],
        **{k: evidence[k] for k in ("accepted_hypothesis", "validation_attempts", "response_receipt")})


def verify_target(row):
    if policy(row) == LEGACY_POLICY:
        return row["target"]
    binding = row.get("target_binding")
    validate_native_response_target(binding, **binding_inputs(row))
    if any(row.get(key) != binding[key] for key in
           ("target", "target_sha256", "evaluated_source", "source_sha256")):
        raise ValueError("episode source/response differs from native target binding")
    provenance = row.get("provenance", {})
    response = row["target_evidence"]["response_receipt"]
    hypothesis = row["target_evidence"]["hypothesis_event"]
    evaluation = row["target_evidence"]["evaluation_event"]
    if type(hypothesis) is not dict or type(evaluation) is not dict:
        raise ValueError("native target admission events must be mappings")
    evidence = hypothesis.get("evidence", {})
    response_hashes = [v for v in (hypothesis.get("response_sha256"),
        evidence.get("generation_response_sha256")) if v is not None]
    if (hypothesis.get("event") != "research" or hypothesis.get("phase") != "hypothesis"
            or evidence.get("hypothesis") != row["target_evidence"]["accepted_hypothesis"]
            or evidence.get("source_sha256") != row["source_sha256"]
            or evidence.get("proposed_sha256") != row["source_sha256"]
            or evidence.get("source_repaired") is not False
            or not response_hashes or any(v != row["target_sha256"] for v in response_hashes)
            or evaluation.get("event") != "decision" or evaluation.get("policy") != "evaluation"
            or evaluation.get("evaluation_id") != row["evaluation_receipt"].get("evaluation_id")
            or any(type(e.get("event_seq")) is not int for e in (hypothesis, evaluation))
            or not hypothesis["event_seq"] < evaluation["event_seq"] < provenance.get("event_seq", -1)
            or any(e.get(k) != provenance.get(k) for e in (hypothesis, evaluation)
                   for k in ("run_id", "segment_id"))):
        raise ValueError("native target lacks the exact pre-evaluation hypothesis admission")
    if (binding["target_policy"]["call_id"] != provenance.get("call_id")
            or response.get("inference_spec") != provenance.get("inference_spec")
            or any(response.get(k) != provenance.get(k) for k in ("run_id", "segment_id"))):
        raise ValueError("native target call differs from episode provenance")
    producer = provenance.get("producer_call_binding")
    if producer is not None and (producer["identity"]["call_id"] != provenance["call_id"]
            or producer["proof"]["response_text_sha256"] != row["target_sha256"]):
        raise ValueError("native response differs from authenticated producer")
    return row["evaluated_source"]


def manifest_policy(manifest):
    if manifest.get("schema") == "sciel-lora-dataset/1":
        if "target_policy" in manifest or "ancestry_evidence" in manifest:
            raise ValueError("legacy manifest cannot carry a native target policy")
        return LEGACY_POLICY
    if manifest.get("schema") == DATASET_V2 and manifest.get("target_policy") == POLICY:
        return POLICY
    raise ValueError("unsupported dataset target policy")


def encode_episode(tokenizer, row, *, max_sequence_tokens, expected_chat_template_sha256):
    """Return the unchanged legacy example or exact native example plus counts."""
    from .tokenization import encode_native_response_example, encode_response_example
    if policy(row) == LEGACY_POLICY:
        return encode_response_example(tokenizer, row, max_sequence_tokens=max_sequence_tokens), None
    verify_target(row)
    value = encode_native_response_example(tokenizer, row, target_binding=row["target_binding"],
        binding_inputs=binding_inputs(row), expected_chat_template_sha256=expected_chat_template_sha256,
        max_sequence_tokens=max_sequence_tokens)
    return value.example, value.receipt()


def implementation_sha256(target_policy):
    from . import native_target, tokenization
    from .types import sha256_bytes
    if target_policy == LEGACY_POLICY:
        return sha256_bytes(Path(tokenization.__file__).read_bytes())
    if target_policy != POLICY:
        raise ValueError("unsupported tokenizer target policy")
    return content_hash({Path(path).name: sha256_bytes(Path(path).read_bytes())
                         for path in (__file__, native_target.__file__, tokenization.__file__)})


def verify_encoding(receipt, row, input_ids, labels, template_hash):
    """Check retained native mask arithmetic/bytes; actual tokenizer replay is separate."""
    from .tokenization import IGNORE_INDEX
    keys = {"schema", "policy", "target_binding_sha256", "chat_template_sha256",
        "rendered_prefix_sha256", "user_prefix_tokens", "masked_assistant_prefix_tokens",
        "supervised_assistant_tokens", "supervised_eos_tokens", "loss_tokens", "sequence_tokens",
        "input_ids_sha256", "labels_sha256"}
    if type(receipt) is not dict or set(receipt) != keys:
        raise ValueError("native tokenization receipt fields differ")
    verify_target(row)
    from .types import require_hash
    require_hash(receipt["rendered_prefix_sha256"], "rendered prefix")
    counts = [receipt[k] for k in ("user_prefix_tokens", "masked_assistant_prefix_tokens",
                                  "supervised_assistant_tokens", "supervised_eos_tokens")]
    if (any(type(n) is not int or n < 1 for n in counts + [receipt["loss_tokens"], receipt["sequence_tokens"]])
            or counts[-1] != 1):
        raise ValueError("native tokenization counts are invalid")
    start = counts[0] + counts[1]
    if (receipt["schema"] != "sciel-native-response-tokenization/1" or receipt["policy"] != POLICY
            or receipt["target_binding_sha256"] != row["target_binding"]["sha256"]
            or receipt["chat_template_sha256"] != template_hash
            or receipt["sequence_tokens"] != len(input_ids) or sum(counts) != len(input_ids)
            or receipt["loss_tokens"] != counts[2] + 1
            or labels != [IGNORE_INDEX] * start + input_ids[start:]
            or receipt["input_ids_sha256"] != content_hash(input_ids)
            or receipt["labels_sha256"] != content_hash(labels)):
        raise ValueError("native tokenization receipt differs from exact tokens/mask")
    return receipt
