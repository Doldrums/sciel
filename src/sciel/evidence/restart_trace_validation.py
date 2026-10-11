"""Strict validation and byte encoding of bounded restart observations."""
from __future__ import annotations

from .restart_trace_values import (
    LEGACY_TRACE_SCHEMA,
    MAX_STARTS,
    MAX_TRACE_BYTES,
    PREFIX_LIMIT,
    TRACE_SCHEMA,
    _RT_COUNTS,
    _RT_INPUTS,
    _RT_ORIGINS,
    _RT_PHASES,
    _rt_check_quality,
    _rt_clone,
    _rt_descriptor,
    _rt_digest,
    _rt_empty_aggregate,
    _rt_hex,
    _rt_identity,
    _rt_int,
    _rt_json,
    _rt_json_text,
    _rt_keys,
    _rt_math,
    _rt_normalization,
    _rt_number,
    _rt_progress,
    _rt_re,
    _rt_record_refs,
    _rt_require,
    _rt_vector,
)

def validate_trace(value, *, source_sha256=None, seed=None, runtime_sha256=None,
                   scaffold_sha256=None, invocation_id=None):
    """Validate /1 or /2 without upgrading, coercing, or inferring feasibility."""
    value = _rt_clone(value)
    _rt_require(len(_rt_json_text(value).encode("ascii")) <= MAX_TRACE_BYTES, "payload byte cap")
    _rt_keys(value, ("schema", "identity", "status", "initial_quality", "final_quality", "total_starts",
                     "total_scheduled_calls", "aggregates", "prefix", "vectors", "calls", "trace_sha256"), "trace")
    _rt_require(value["schema"] in (LEGACY_TRACE_SCHEMA, TRACE_SCHEMA), "schema")
    has_result = value["schema"] == TRACE_SCHEMA
    identity = _rt_identity(value["identity"])
    for name, expected in (("source_sha256", source_sha256), ("seed", seed),
                           ("runtime_sha256", runtime_sha256), ("scaffold_sha256", scaffold_sha256),
                           ("invocation_id", invocation_id)):
        if expected is not None:
            _rt_require(type(identity[name]) is type(expected) and identity[name] == expected,
                        "external " + name + " mismatch")
    _rt_require(value["status"] in ("restart_limit", "deadline", "error"), "run status")
    for name in ("initial_quality", "final_quality"):
        _rt_check_quality(value[name])
    _rt_progress(value["initial_quality"], value["final_quality"])
    _rt_require(_rt_int(value["total_starts"]) and value["total_starts"] <= MAX_STARTS
                and _rt_int(value["total_scheduled_calls"]), "total counts")
    _rt_keys(value["aggregates"], _RT_ORIGINS, "aggregates")
    for origin, aggregate in value["aggregates"].items():
        _rt_keys(aggregate, _RT_COUNTS + ("gain",), "aggregate")
        _rt_require(all(_rt_int(aggregate[k]) for k in _RT_COUNTS)
                    and type(aggregate["gain"]) is float and _rt_number(aggregate["gain"])
                    and aggregate["gain"] >= 0, "aggregate values")
        _rt_require(sum(aggregate[k] for k in ("completed", "deadline", "error", "incomplete")) == aggregate["starts"]
                    and aggregate["completed"] <= aggregate["optimizer_returned"] <= aggregate["optimizer_entered"] <= aggregate["starts"]
                    and aggregate["helper_calls"] <= aggregate["starts"], "aggregate count relations")
        if origin == "ordinary":
            _rt_require(aggregate["helper_calls"] == 0, "ordinary helper count")
        elif origin != "scheduled_unresolved":
            _rt_require(aggregate["helper_calls"] == aggregate["starts"], "scheduled helper count")
        if origin in ("helper_error", "scheduled_unresolved"):
            _rt_require(aggregate["optimizer_entered"] == 0, "failed/unresolved optimizer entry")
    _rt_require(sum(a["starts"] for a in value["aggregates"].values()) == value["total_starts"]
                and sum(a["helper_calls"] for a in value["aggregates"].values()) == value["total_scheduled_calls"], "aggregate totals")
    terminal_counts = {name: sum(a[name] for a in value["aggregates"].values())
                       for name in ("deadline", "error", "incomplete")}
    _rt_require(sum(terminal_counts.values()) <= 1, "multiple interrupted starts")
    if value["status"] == "restart_limit":
        _rt_require(terminal_counts["deadline"] == terminal_counts["error"] == 0,
                    "restart limit contradicts terminal counts")
    else:
        _rt_require(terminal_counts[value["status"]] == int(value["total_starts"] > 0),
                    "run status contradicts terminal counts")
    scheduled_starts = (value["total_starts"] + 3) // 4
    _rt_require(value["aggregates"]["ordinary"]["starts"] == value["total_starts"] - scheduled_starts
                and 0 <= scheduled_starts - value["total_scheduled_calls"] <= 1
                and (scheduled_starts == value["total_scheduled_calls"] or value["total_starts"] % 4 == 1),
                "scheduled start/call totals")
    _rt_require(sum(a["improvements"] for a in value["aggregates"].values()) ==
                value["final_quality"]["improvements"] - value["initial_quality"]["improvements"],
                "improvement accounting")
    # Only this diagnostic floating-point summation uses a rounding allowance.
    # It is never a feasibility, packing-quality, or zero-loss comparison.
    gain = _rt_math.fsum(a["gain"] for a in value["aggregates"].values())
    delta = value["final_quality"]["sum"] - value["initial_quality"]["sum"]
    bound = 8 * max(1, value["total_starts"]) * _rt_math.ulp(max(1.0, abs(value["initial_quality"]["sum"]),
                                                              abs(value["final_quality"]["sum"])))
    _rt_require(abs(gain - delta) <= bound, "gain accounting")
    vectors, calls = value["vectors"], value["calls"]
    _rt_require(type(vectors) is dict and type(calls) is list and len(calls) <= PREFIX_LIMIT, "prefix/catalog")
    for ref, vector in vectors.items():
        _rt_require(_rt_hex(ref) and _rt_digest(vector) == ref, "vector catalog hash")
        _rt_vector(vector)
    prefix = value["prefix"]
    _rt_keys(prefix, ("limit", "captured_count", "omitted_count", "first_omitted_ordinal", "omission_reason"), "prefix")
    _rt_require(type(prefix["limit"]) is int and prefix["limit"] == PREFIX_LIMIT
                and _rt_int(prefix["captured_count"]) and prefix["captured_count"] == len(calls)
                and _rt_int(prefix["omitted_count"]) and prefix["omitted_count"] == value["total_scheduled_calls"] - len(calls), "prefix counts")
    if prefix["omitted_count"]:
        _rt_require(type(prefix["first_omitted_ordinal"]) is int and prefix["first_omitted_ordinal"] == len(calls)
                    and prefix["omission_reason"] in ("prefix_limit", "byte_limit"), "prefix omission")
        if prefix["omission_reason"] == "prefix_limit":
            _rt_require(len(calls) == PREFIX_LIMIT, "prefix limit omission")
    else:
        _rt_require(prefix["first_omitted_ordinal"] is None and prefix["omission_reason"] is None, "empty omission")
    used, observed = set(), {origin: _rt_empty_aggregate() for origin in _RT_ORIGINS}
    last_terminal = value["initial_quality"]
    for ordinal, call in enumerate(calls):
        call_fields = ("ordinal", "restart_index", "inputs", "post_inputs", "raw_return", "effective",
                       "helper_finished", "error", "phases", "terminal")
        _rt_keys(call, call_fields + (("optimizer_result",) if has_result else ()), "call")
        _rt_require(type(call["ordinal"]) is int and call["ordinal"] == ordinal
                    and type(call["restart_index"]) is int and call["restart_index"] == ordinal * 4
                    and call["restart_index"] < value["total_starts"], "call order")
        for field in ("inputs", "post_inputs"):
            _rt_keys(call[field], _RT_INPUTS, field)
            for desc in call[field].values():
                _rt_descriptor(desc, vectors, post=field == "post_inputs", normalized=field == "inputs")
        _rt_keys(call["raw_return"], ("kind", "type", "vector"), "raw return")
        raw = call["raw_return"]
        _rt_require(raw["kind"] in (None, "none", "value", "exception")
                    and (raw["type"] is None or (type(raw["type"]) is str and 0 < len(raw["type"]) <= 96)), "raw return metadata")
        _rt_descriptor(raw["vector"], vectors)
        if raw["kind"] != "value":
            _rt_require(raw["vector"] == {"ref": None, "reason": "not_applicable"}, "nonvalue raw bytes")
        else:
            _rt_require(raw["vector"]["reason"] != "not_applicable", "value raw capture unavailable reason")
        _rt_keys(call["effective"], ("origin", "vector"), "effective")
        effective = call["effective"]
        _rt_descriptor(effective["vector"], vectors, normalized=True)
        _rt_require(type(call["helper_finished"]) is bool, "helper finished flag")
        _rt_keys(call["phases"], _RT_PHASES, "phases")
        phases = call["phases"]
        previous = last_terminal
        for name in _RT_PHASES:
            if phases[name] is not None:
                _rt_check_quality(phases[name])
                _rt_progress(previous, phases[name])
                previous = phases[name]
        _rt_require(phases["start"] is not None and phases["terminal"] is not None
                    and (phases["helper_finished"] is not None) == call["helper_finished"]
                    and (phases["optimizer_returned"] is None or phases["optimizer_entered"] is not None), "phase order")
        if has_result:
            result = call["optimizer_result"]
            _rt_descriptor(result, vectors)
            if phases["optimizer_returned"] is None:
                _rt_require(result == {"ref": None, "reason": "not_applicable"},
                            "optimizer result without return")
            else:
                _rt_require(result["reason"] != "not_applicable",
                            "returned optimizer result capture must be explicit")
        _rt_progress(phases["terminal"], value["final_quality"])
        last_terminal = phases["terminal"]
        if not call["helper_finished"]:
            _rt_require(raw["kind"] is None and raw["type"] is None and call["error"] is None
                        and effective["origin"] is None and phases["optimizer_entered"] is None,
                        "unfinished helper fields")
            origin = "scheduled_unresolved"
        elif call["error"] is not None:
            _rt_keys(call["error"], ("stage", "code"), "helper error")
            _rt_require(call["error"]["stage"] in ("helper", "normalization")
                        and type(call["error"]["code"]) is str
                        and _rt_re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,95}", call["error"]["code"])
                        and effective["origin"] is None and phases["optimizer_entered"] is None,
                        "helper error fields")
            _rt_require((call["error"]["stage"] == "helper" and raw["kind"] == "exception")
                        or (call["error"]["stage"] == "normalization" and raw["kind"] in ("none", "value")),
                        "error/raw stage mismatch")
            origin = "helper_error"
        else:
            origin = effective["origin"]
            _rt_require((raw["kind"] == "none" and origin == "saved_ordinary_fallback")
                        or (raw["kind"] == "value" and origin == "accepted_return"), "origin/raw mismatch")
            _rt_require(effective["vector"]["reason"] != "not_applicable", "missing successful effective output")
            if origin == "accepted_return" and raw["vector"]["ref"] is not None:
                _rt_normalization(vectors[raw["vector"]["ref"]],
                                  vectors[effective["vector"]["ref"]] if effective["vector"]["ref"] is not None else None)
            if origin == "saved_ordinary_fallback" and effective["vector"]["ref"] is not None and call["inputs"]["ordinary"]["ref"] is not None:
                _rt_require(effective["vector"]["ref"] == call["inputs"]["ordinary"]["ref"], "fallback differs from saved ordinary")
        if origin in ("helper_error", "scheduled_unresolved"):
            _rt_require(effective["vector"] == {"ref": None, "reason": "not_applicable"}, "failed effective vector")
        terminal = call["terminal"]
        _rt_require(terminal in ("completed", "deadline", "error", "incomplete")
                    and (terminal != "completed" or phases["optimizer_returned"] is not None), "terminal phase")
        _rt_require(terminal == "completed" or call["restart_index"] == value["total_starts"] - 1,
                    "interrupted call is not final start")
        aggregate = observed[origin]
        for field, amount in (("starts", 1), ("helper_calls", 1), ("optimizer_entered", int(phases["optimizer_entered"] is not None)),
                               ("optimizer_returned", int(phases["optimizer_returned"] is not None)), (terminal, 1),
                               ("improvements", phases["terminal"]["improvements"] - phases["start"]["improvements"])):
            aggregate[field] += amount
        aggregate["gain"] += phases["terminal"]["sum"] - phases["start"]["sum"]
        used.update(_rt_record_refs(call))
    _rt_require(set(vectors) == used, "unused vector catalog entries")
    for origin, aggregate in observed.items():
        _rt_require(all(aggregate[k] <= value["aggregates"][origin][k] for k in _RT_COUNTS)
                    and aggregate["gain"] <= value["aggregates"][origin]["gain"], "prefix exceeds aggregates")
    expected_hash = value.pop("trace_sha256")
    _rt_require(_rt_hex(expected_hash) and _rt_digest(value) == expected_hash, "trace hash")
    value["trace_sha256"] = expected_hash
    return value


def parse_trace(text, **bindings):
    """Read a bounded JSON trace, rejecting duplicate object keys."""
    _rt_require(type(text) in (str, bytes), "JSON input type")
    try:
        raw = text.encode("utf-8") if type(text) is str else text
        _rt_require(len(raw) <= MAX_TRACE_BYTES, "payload byte cap")
        def pairs(items):
            result = {}
            for key, value in items:
                _rt_require(key not in result, "duplicate JSON key")
                result[key] = value
            return result
        value = _rt_json.loads(raw, object_pairs_hook=pairs,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, RecursionError, _rt_json.JSONDecodeError) as exc:
        raise ValueError("restart trace: invalid JSON") from exc
    return validate_trace(value, **bindings)


def encode_trace(value):
    """Return canonical ASCII JSON after full validation."""
    return _rt_json_text(validate_trace(value))
