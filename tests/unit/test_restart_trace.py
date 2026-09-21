"""Pure synthetic collector/codec tests: no helper, optimizer or provider calls."""
import base64
import copy
import hashlib
import json

import numpy as np
import pytest

from sciel.hypothesis.restart_trace import (
    LEGACY_TRACE_SCHEMA, TRACE_SCHEMA, MAX_TRACE_BYTES, PREFIX_LIMIT, RestartTrace,
    encode_trace, parse_trace, validate_trace,
)


IDENTITY = dict(source_sha256="a" * 64, scaffold_sha256="b" * 64,
                runtime_sha256="c" * 64, seed=17, invocation_id="d" * 32)


def args(offset=0):
    return {"best": np.arange(78, dtype=float) / 100 + offset,
            "initial": np.zeros(78), "noise": np.full(78, .01),
            "ordinary": np.arange(78, dtype=float) / 100 + offset + .01}


def seal(value):
    value = copy.deepcopy(value)
    value.pop("trace_sha256", None)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    value["trace_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    return value


def capture(kind="none", *, mutate=False, raw=None, error=None, stage=None,
            optimizer=True, returned=True, terminal="restart_limit", gain=0.0, result=None):
    collector = RestartTrace(IDENTITY)
    inputs = args()
    saved = inputs["ordinary"].copy()
    collector.begin_start(0, incumbent_sum=1.0, improvements=0)
    collector.begin_helper(inputs)
    if mutate:
        inputs["ordinary"][0] = 99
    if raw is None and kind == "value":
        raw = saved.copy()
    effective = None if error else (saved if kind == "none" else raw.astype(float).copy())
    collector.helper_end(raw_kind=kind, raw_value=raw, raw_type="ndarray" if kind == "value" else None,
        post_inputs=inputs, effective=effective,
        origin=None if error else ("saved_ordinary_fallback" if kind == "none" else "accepted_return"),
        error=error, error_stage=stage, incumbent_sum=1.0, improvements=0)
    if optimizer and not error:
        collector.optimizer_enter(incumbent_sum=1.0, improvements=0)
        if returned:
            collector.optimizer_exit(incumbent_sum=1.0, improvements=0, result=result)
    if terminal == "restart_limit":
        collector.end_start("completed", incumbent_sum=1.0 + gain, improvements=int(gain > 0))
    return collector.finish(terminal, incumbent_sum=1.0 + gain, improvements=int(gain > 0))


def test_empty_trace_roundtrip_binding_and_detachment():
    value = RestartTrace(IDENTITY).finish("restart_limit", incumbent_sum=0.0, improvements=0)
    restored = parse_trace(encode_trace(value), **IDENTITY)
    assert restored == value and restored is not value
    restored["identity"]["seed"] = 99
    assert value["identity"]["seed"] == 17
    assert value["calls"] == [] and value["vectors"] == {}


def test_none_and_equal_return_are_distinct_and_bytes_preserved():
    fallback, accepted = capture(), capture("value")
    a, b = fallback["calls"][0], accepted["calls"][0]
    assert a["effective"]["vector"] == b["effective"]["vector"]
    assert a["effective"]["origin"] == "saved_ordinary_fallback"
    assert b["effective"]["origin"] == "accepted_return"
    assert accepted["aggregates"]["accepted_return"]["starts"] == 1
    assert fallback["aggregates"]["saved_ordinary_fallback"]["starts"] == 1
    vector = accepted["vectors"][b["effective"]["vector"]["ref"]]
    assert base64.b64decode(vector["data_base64"]) == args()["ordinary"].tobytes()


def test_mutated_none_argument_does_not_change_saved_fallback():
    value = capture(mutate=True)
    row = value["calls"][0]
    assert row["post_inputs"]["ordinary"]["ref"] != row["inputs"]["ordinary"]["ref"]
    assert row["effective"]["vector"]["ref"] == row["inputs"]["ordinary"]["ref"]
    assert row["post_inputs"]["ordinary"]["ref"] not in value["vectors"]


@pytest.mark.parametrize("raw", [np.arange(78, dtype=np.int16), np.arange(78, dtype=np.float32),
                                np.full(78, -0.0), np.arange(78, dtype=">f8")])
def test_raw_dtype_and_normalized_binary64_join(raw):
    value = capture("value", raw=raw)
    row = value["calls"][0]
    vector = value["vectors"][row["raw_return"]["vector"]["ref"]]
    assert vector["dtype"] == raw.dtype.str
    assert base64.b64decode(vector["data_base64"]) == raw.tobytes()
    assert validate_trace(value) == value


@pytest.mark.parametrize("kind,raw,stage", [
    ("exception", None, "helper"),
    ("value", np.ones(77), "normalization"),
    ("value", np.full(78, np.nan), "normalization"),
    ("value", np.array([object()], dtype=object), "normalization"),
])
def test_errors_are_finished_handoffs_without_effective_or_optimizer(kind, raw, stage):
    value = capture(kind, raw=raw, error="ValueError", stage=stage, terminal="error")
    row = value["calls"][0]
    assert row["helper_finished"] is True and row["error"] == {"stage": stage, "code": "ValueError"}
    assert row["effective"]["origin"] is None and row["effective"]["vector"]["ref"] is None
    assert row["phases"]["optimizer_entered"] is None
    if raw is not None and raw.dtype.kind == "O":
        assert row["raw_return"]["vector"]["reason"] == "unsupported_dtype"
    assert value["aggregates"]["helper_error"]["error"] == 1


def test_prehelper_deadline_is_unresolved_start_not_helper_invocation():
    collector = RestartTrace(IDENTITY)
    collector.begin_start(0, incumbent_sum=1.0, improvements=0)
    value = collector.finish("deadline", incumbent_sum=1.0, improvements=0)
    assert value["calls"] == [] and value["total_scheduled_calls"] == 0
    assert value["aggregates"]["scheduled_unresolved"]["deadline"] == 1


def test_deadline_after_helper_can_admit_gain_without_optimizer():
    value = capture(optimizer=False, terminal="deadline", gain=.25)
    row = value["calls"][0]
    assert row["helper_finished"] and row["phases"]["optimizer_entered"] is None
    aggregate = value["aggregates"]["saved_ordinary_fallback"]
    assert aggregate["gain"] == .25 and aggregate["improvements"] == 1
    assert aggregate["optimizer_returned"] == 0 and aggregate["deadline"] == 1


@pytest.mark.parametrize("returned", [False, True])
def test_optimizer_return_is_separate_from_terminal_deadline(returned):
    value = capture(returned=returned, terminal="deadline", gain=.25)
    aggregate = value["aggregates"]["saved_ordinary_fallback"]
    assert aggregate["optimizer_entered"] == 1
    assert aggregate["optimizer_returned"] == int(returned)
    assert aggregate["completed"] == 0 and aggregate["deadline"] == 1


def many_calls(*, unique=False, result_for_index=None):
    collector = RestartTrace(IDENTITY)
    for index in range(81):
        collector.begin_start(index, incumbent_sum=1., improvements=0)
        if index % 4 == 0:
            inputs = args(index if unique else 0)
            if unique:
                inputs["initial"] += index * .015
                inputs["noise"] += index * .001
            collector.begin_helper(inputs)
            raw = (np.arange(78, dtype=np.float32) + index * .03125) if unique else None
            effective = raw.astype(float) if unique else inputs["ordinary"].copy()
            collector.helper_end(raw_kind="value" if unique else "none", raw_value=raw,
                raw_type="ndarray" if unique else None, post_inputs=inputs, effective=effective,
                origin="accepted_return" if unique else "saved_ordinary_fallback",
                incumbent_sum=1., improvements=0)
        collector.optimizer_enter(incumbent_sum=1., improvements=0)
        collector.optimizer_exit(incumbent_sum=1., improvements=0,
                                 result=result_for_index(index) if result_for_index else None)
        collector.end_start("completed", incumbent_sum=1., improvements=0)
    return collector.finish("restart_limit", incumbent_sum=1., improvements=0)


def test_first16_prefix_and_global_dedup_keep_whole_run_counts():
    value = many_calls()
    assert len(value["calls"]) == PREFIX_LIMIT
    assert [r["restart_index"] for r in value["calls"]] == list(range(0, 64, 4))
    assert value["total_scheduled_calls"] == 21 and value["total_starts"] == 81
    assert value["prefix"] == dict(limit=16, captured_count=16, omitted_count=5,
                                  first_omitted_ordinal=16, omission_reason="prefix_limit")
    assert len(value["vectors"]) == 4
    assert value["aggregates"]["ordinary"]["starts"] == 60


def test_byte_cap_removes_whole_latest_call_then_keeps_no_later_examples():
    value = many_calls(unique=True)
    assert 0 < len(value["calls"]) < PREFIX_LIMIT
    assert value["prefix"]["omission_reason"] == "byte_limit"
    assert value["prefix"]["omitted_count"] == 21 - len(value["calls"])
    assert len(encode_trace(value).encode()) <= MAX_TRACE_BYTES
    assert value["aggregates"]["accepted_return"]["starts"] == 21


@pytest.mark.parametrize("key,replacement", [("seed", True), ("source_sha256", "e" * 64),
                                             ("runtime_sha256", "e" * 64), ("invocation_id", "e" * 32)])
def test_external_binding_rejects_even_rehashed_identity(key, replacement):
    value = capture()
    value["identity"][key] = replacement
    with pytest.raises(ValueError):
        validate_trace(seal(value), **IDENTITY)


def test_rehashed_changed_effective_vector_does_not_bypass_raw_normalization():
    value = capture("value")
    row = value["calls"][0]
    row["effective"]["vector"] = copy.deepcopy(row["inputs"]["initial"])
    with pytest.raises(ValueError, match="normalization mismatch"):
        validate_trace(seal(value))


@pytest.mark.parametrize("raw", [np.ones(78, dtype=bool), np.ones(77), np.full(78, np.nan)])
def test_captured_invalid_raw_cannot_claim_acceptance_when_effective_unavailable(raw):
    value = capture("value")
    row = value["calls"][0]
    vector = dict(dtype=raw.dtype.str, shape=list(raw.shape), order="C",
                  data_base64=base64.b64encode(raw.tobytes()).decode(),
                  bytes_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    ref = hashlib.sha256(json.dumps(vector, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    value["vectors"][ref] = vector
    row["raw_return"]["vector"] = {"ref": ref, "reason": None}
    row["effective"]["vector"] = {"ref": None, "reason": "array_unavailable"}
    with pytest.raises(ValueError, match="accepted raw"):
        validate_trace(seal(value))


def test_value_raw_capture_cannot_be_relabelled_not_applicable():
    value = capture("value")
    value["calls"][0]["raw_return"]["vector"] = {"ref": None, "reason": "not_applicable"}
    with pytest.raises(ValueError, match="value raw capture"):
        validate_trace(seal(value))


def test_rehashed_counter_phase_and_schedule_changes_rejected():
    original = capture()
    cases = []
    item = copy.deepcopy(original)
    item["calls"][0]["restart_index"] = 4
    cases.append(item)
    item = copy.deepcopy(original)
    item["aggregates"]["saved_ordinary_fallback"]["gain"] = .125
    cases.append(item)
    item = copy.deepcopy(original)
    item["calls"][0]["helper_finished"] = 1
    cases.append(item)
    item = copy.deepcopy(original)
    item["calls"][0]["phases"]["optimizer_entered"] = None
    cases.append(item)
    for item in cases:
        with pytest.raises(ValueError):
            validate_trace(seal(item))


def test_bad_bytes_duplicates_and_nonfinite_or_huge_metadata_rejected():
    value = capture()
    ref = next(iter(value["vectors"]))
    value["vectors"][ref]["data_base64"] = "!"
    with pytest.raises(ValueError):
        validate_trace(seal(value))
    with pytest.raises(ValueError, match="duplicate"):
        parse_trace('{"schema":1,"schema":2}')
    for number in (True, float("nan"), 10 ** 1000):
        with pytest.raises(ValueError):
            RestartTrace(IDENTITY).finish("restart_limit", incumbent_sum=number, improvements=0)


def test_lifecycle_and_finished_collector_cannot_be_reused():
    collector = RestartTrace(IDENTITY)
    with pytest.raises(ValueError):
        collector.begin_start(True, incumbent_sum=0., improvements=0)
    collector.finish("restart_limit", incumbent_sum=0., improvements=0)
    with pytest.raises(ValueError, match="finished"):
        collector.begin_start(0, incumbent_sum=0., improvements=0)


def test_rehashed_restart_limit_cannot_hide_deadline():
    value = capture(terminal="deadline")
    value["status"] = "restart_limit"
    with pytest.raises(ValueError, match="restart limit contradicts"):
        validate_trace(seal(value))


def test_rehashed_error_requires_an_interrupted_start():
    value = capture()
    value["status"] = "error"
    with pytest.raises(ValueError, match="run status contradicts"):
        validate_trace(seal(value))


def test_rehashed_two_interrupted_starts_are_impossible():
    value = capture(terminal="deadline")
    value["total_starts"] = 2
    value["aggregates"]["ordinary"]["starts"] = 1
    value["aggregates"]["ordinary"]["error"] = 1
    with pytest.raises(ValueError, match="multiple interrupted starts"):
        validate_trace(seal(value))


def test_rehashed_interrupted_prefix_call_cannot_precede_later_starts():
    value = many_calls()
    value["status"] = "deadline"
    value["calls"][0]["terminal"] = "deadline"
    value["aggregates"]["saved_ordinary_fallback"]["completed"] -= 1
    value["aggregates"]["saved_ordinary_fallback"]["deadline"] += 1
    with pytest.raises(ValueError, match="interrupted call is not final"):
        validate_trace(seal(value))


def test_optimizer_result_snapshot_is_exact_detached_and_deduplicated():
    result = args()["ordinary"]
    before = result.tobytes()
    value = capture(result=result)
    assert value["schema"] == TRACE_SCHEMA == "sciel-restart-trace/2"
    row = value["calls"][0]
    assert row["optimizer_result"] == row["effective"]["vector"]
    assert len(value["vectors"]) == 4
    result.fill(-99)
    vector = value["vectors"][row["optimizer_result"]["ref"]]
    assert base64.b64decode(vector["data_base64"]) == before
    assert value["final_quality"] == {"sum": 1.0, "improvements": 0}


@pytest.mark.parametrize("result", [np.arange(156, dtype=">f8")[::2],
                                    np.array([np.nan, np.inf, -0.0]),
                                    np.array([[1, 2]], dtype=np.int16)])
def test_result_captures_observed_array_without_geometry_or_quality_claim(result):
    value = capture(result=result, terminal="deadline")
    descriptor = value["calls"][0]["optimizer_result"]
    vector = value["vectors"][descriptor["ref"]]
    assert vector["dtype"] == result.dtype.str and vector["shape"] == list(result.shape)
    assert base64.b64decode(vector["data_base64"]) == result.tobytes(order="C")
    assert validate_trace(value) == value
    assert value["aggregates"]["saved_ordinary_fallback"]["optimizer_returned"] == 1
    assert value["aggregates"]["saved_ordinary_fallback"]["completed"] == 0
    assert value["final_quality"] == {"sum": 1.0, "improvements": 0}


@pytest.mark.parametrize("result,reason", [
    (None, "array_unavailable"),
    (np.array([object()], dtype=object), "unsupported_dtype"),
    (np.zeros((1, 1, 1, 1, 1)), "unsupported_shape"),
    (np.zeros(4097), "oversized"),
])
def test_returned_optimizer_unavailable_snapshot_is_explicit(result, reason):
    value = capture(result=result)
    assert value["calls"][0]["optimizer_result"] == {"ref": None, "reason": reason}


def test_result_does_not_coerce_python_value():
    class ArrayLike:
        def __array__(self, *args, **kwargs):
            raise AssertionError("coercion must never run")
    value = capture(result=ArrayLike())
    assert value["calls"][0]["optimizer_result"] == {"ref": None, "reason": "array_unavailable"}


def test_no_optimizer_return_has_not_applicable_result_and_rejects_forged_capture():
    value = capture(returned=False, terminal="deadline")
    row = value["calls"][0]
    assert row["optimizer_result"] == {"ref": None, "reason": "not_applicable"}
    row["optimizer_result"] = copy.deepcopy(row["inputs"]["best"])
    with pytest.raises(ValueError, match="optimizer result without return"):
        validate_trace(seal(value))


def test_returned_optimizer_result_descriptor_is_required_and_strict():
    original = capture()
    for descriptor in (None, {"ref": None, "reason": "not_applicable"},
                       {"ref": "e" * 64, "reason": None},
                       {"ref": False, "reason": None},
                       {"ref": None, "reason": "array_unavailable", "valid": True}):
        value = copy.deepcopy(original)
        if descriptor is None:
            del value["calls"][0]["optimizer_result"]
        else:
            value["calls"][0]["optimizer_result"] = descriptor
        with pytest.raises(ValueError):
            validate_trace(seal(value))


def test_legacy_v1_roundtrip_does_not_invent_optimizer_result_and_stays_strict():
    legacy = capture()
    legacy["schema"] = LEGACY_TRACE_SCHEMA
    for row in legacy["calls"]:
        del row["optimizer_result"]
    legacy = seal(legacy)
    assert parse_trace(encode_trace(legacy), **IDENTITY) == legacy
    assert "optimizer_result" not in validate_trace(legacy)["calls"][0]
    hybrid = copy.deepcopy(legacy)
    hybrid["calls"][0]["optimizer_result"] = {"ref": None, "reason": "array_unavailable"}
    with pytest.raises(ValueError, match="call fields"):
        validate_trace(seal(hybrid))
    corrupt = copy.deepcopy(legacy)
    corrupt["calls"][0]["phases"]["optimizer_entered"] = None
    with pytest.raises(ValueError, match="phase order"):
        validate_trace(seal(corrupt))


def test_result_capture_obeys_whole_call_byte_cap_and_prunes_unused_vectors():
    value = many_calls(result_for_index=lambda index: np.full(4096, index, dtype=float))
    assert 0 < len(value["calls"]) < PREFIX_LIMIT
    assert value["prefix"]["omission_reason"] == "byte_limit"
    assert value["prefix"]["omitted_count"] == 21 - len(value["calls"])
    assert [c["ordinal"] for c in value["calls"]] == list(range(len(value["calls"])))
    assert len(value["vectors"]) == 4 + len(value["calls"])
    assert len(encode_trace(value).encode()) <= MAX_TRACE_BYTES
    assert value["aggregates"]["saved_ordinary_fallback"]["optimizer_returned"] == 21


def test_ordinary_and_prefix_omitted_results_are_never_inspected():
    inspected = []
    class Unobserved:
        @property
        def dtype(self):
            inspected.append(True)
            raise AssertionError("outside helper prefix")
    marker = Unobserved()
    value = many_calls(result_for_index=lambda index: args()["ordinary"]
                       if index % 4 == 0 and index < 64 else marker)
    assert len(value["calls"]) == 16 and inspected == []
    assert all(row["optimizer_result"]["ref"] is not None for row in value["calls"])


def test_two_step_result_attachment_occurs_once_after_return_before_consider():
    collector = RestartTrace(IDENTITY)
    inputs = args()
    collector.begin_start(0, incumbent_sum=1., improvements=0)
    collector.begin_helper(inputs)
    collector.helper_end(raw_kind="none", post_inputs=inputs, effective=inputs["ordinary"],
                         origin="saved_ordinary_fallback", incumbent_sum=1., improvements=0)
    with pytest.raises(ValueError, match="attachment order"):
        collector.optimizer_result(inputs["best"])
    collector.optimizer_enter(incumbent_sum=1., improvements=0)
    collector.optimizer_exit(incumbent_sum=1., improvements=0)
    result = np.full(78, .25)
    before = result.tobytes()
    collector.optimizer_result(result)
    result.fill(.5)
    with pytest.raises(ValueError, match="attachment order"):
        collector.optimizer_result(result)
    value = collector.finish("error", incumbent_sum=1., improvements=0)
    row = value["calls"][0]
    assert base64.b64decode(value["vectors"][row["optimizer_result"]["ref"]]["data_base64"]) == before
    assert row["phases"]["optimizer_returned"] is not None and row["terminal"] == "error"
