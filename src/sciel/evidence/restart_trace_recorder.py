"""Record existing restart calls without invoking helpers or optimization."""
from __future__ import annotations

from .restart_trace_values import (
    MAX_STARTS,
    MAX_TRACE_BYTES,
    PREFIX_LIMIT,
    TRACE_SCHEMA,
    _RT_INPUTS,
    _RT_ORIGINS,
    _RT_PHASES,
    _rt_array,
    _rt_clone,
    _rt_digest,
    _rt_empty_aggregate,
    _rt_identity,
    _rt_int,
    _rt_json_text,
    _rt_keys,
    _rt_progress,
    _rt_quality,
    _rt_re,
    _rt_record_refs,
    _rt_require,
)
from .restart_trace_validation import (
    validate_trace,
)

class RestartTrace:
    """Collect calls already made by the scaffold, with fixed prefix/cap limits.

    ``begin_helper`` receives the four detached argument arrays before calling
    the helper. ``helper_end`` receives those same arrays afterward and, when
    available, the array from the scaffold's one existing ``asarray`` call.
    ``optimizer_exit(result=...)`` belongs immediately after the optimizer
    returns, before considering its result. It snapshots the existing array
    without coercion, only for retained helper calls; absence or unsupported
    capture is explicit, and captured bytes need not be valid geometry.
    Alternatively, mark the return without a result and attach it once with
    ``optimizer_result(result)`` at the original result.x access before consider.
    ``finish`` settles an open start even on deadline.
    No method performs optimization, invokes a helper, or changes an input.
    """

    def __init__(self, identity):
        self.identity = _rt_identity(_rt_clone(identity))
        self.calls, self.vectors = [], {}
        self.aggregates = {origin: _rt_empty_aggregate() for origin in _RT_ORIGINS}
        self.total_starts = self.total_scheduled_calls = 0
        self.initial_quality = None
        self.final_quality = None
        self.status = None
        self._active = None
        self._byte_stop = False
        self._first_omitted = None

    def _live(self):
        _rt_require(self.status is None, "collector is finished")

    def _snapshot(self, value, *, retain=True):
        vector, reason = _rt_array(value)
        if vector is None:
            return {"ref": None, "reason": reason}
        ref = _rt_digest(vector)
        if retain:
            self.vectors.setdefault(ref, vector)
        return {"ref": ref, "reason": None}

    def _prefix(self):
        captured = len(self.calls)
        omitted = self.total_scheduled_calls - captured
        return {"limit": PREFIX_LIMIT, "captured_count": captured,
                "omitted_count": omitted,
                "first_omitted_ordinal": captured if omitted else None,
                "omission_reason": ("byte_limit" if self._byte_stop else "prefix_limit")
                if omitted else None}

    def _wire(self):
        return {"schema": TRACE_SCHEMA, "identity": self.identity,
                "status": self.status, "initial_quality": self.initial_quality,
                "final_quality": self.final_quality, "total_starts": self.total_starts,
                "total_scheduled_calls": self.total_scheduled_calls,
                "aggregates": self.aggregates, "prefix": self._prefix(),
                "vectors": self.vectors, "calls": self.calls}

    def _trim(self):
        # Reserve final counters, quality and the terminal hash while collecting.
        if len(_rt_json_text(self._wire()).encode("ascii")) <= MAX_TRACE_BYTES - 2048:
            return
        record = self._active["record"] if self._active is not None else None
        _rt_require(record is not None and self.calls and self.calls[-1] is record,
                    "fixed metadata exceeds byte cap")
        self.calls.pop()
        self._active["record"] = None
        self._byte_stop = True
        used = {ref for item in self.calls for ref in _rt_record_refs(item)}
        self.vectors = {key: value for key, value in self.vectors.items() if key in used}

    def begin_start(self, restart_index, *, incumbent_sum, improvements):
        self._live()
        _rt_require(self._active is None and _rt_int(restart_index)
                    and restart_index == self.total_starts and restart_index < MAX_STARTS, "start order")
        quality = _rt_quality(incumbent_sum, improvements)
        if self.initial_quality is None:
            self.initial_quality = quality
        if self.final_quality is not None:
            _rt_progress(self.final_quality, quality)
        self.total_starts += 1
        self._active = {"restart_index": restart_index, "quality": quality,
                        "origin": "scheduled_unresolved" if restart_index % 4 == 0 else "ordinary",
                        "helper": False, "helper_finished": False, "entered": False,
                        "returned": False, "result_attached": False, "record": None}

    def begin_helper(self, inputs):
        self._live()
        active = self._active
        _rt_require(active is not None and active["restart_index"] % 4 == 0
                    and not active["helper"] and not active["entered"], "helper order")
        _rt_keys(inputs, _RT_INPUTS, "helper inputs")
        ordinal = self.total_scheduled_calls
        self.total_scheduled_calls += 1
        active["helper"] = True
        if ordinal >= PREFIX_LIMIT or self._byte_stop:
            return
        record = {"ordinal": ordinal, "restart_index": active["restart_index"],
                  "inputs": {k: self._snapshot(inputs[k]) for k in _RT_INPUTS},
                  "post_inputs": {k: {"ref": None, "reason": "not_applicable"} for k in _RT_INPUTS},
                  "raw_return": {"kind": None, "type": None,
                                 "vector": {"ref": None, "reason": "not_applicable"}},
                  "effective": {"origin": None, "vector": {"ref": None, "reason": "not_applicable"}},
                  "optimizer_result": {"ref": None, "reason": "not_applicable"},
                  "helper_finished": False, "error": None,
                  "phases": {key: active["quality"] if key == "start" else None for key in _RT_PHASES},
                  "terminal": None}
        self.calls.append(record)
        active["record"] = record
        self._trim()

    def helper_end(self, *, raw_kind, raw_value=None, raw_type=None, post_inputs,
                   effective=None, origin=None, error=None, error_stage=None,
                   incumbent_sum, improvements):
        self._live()
        active = self._active
        _rt_require(active is not None and active["helper"] and not active["helper_finished"]
                    and not active["entered"], "helper completion order")
        _rt_require(raw_kind in ("none", "value", "exception"), "raw kind")
        _rt_require(raw_type is None or (type(raw_type) is str and 0 < len(raw_type) <= 96), "raw type")
        _rt_keys(post_inputs, _RT_INPUTS, "post inputs")
        quality = _rt_quality(incumbent_sum, improvements)
        _rt_progress(active["quality"], quality)
        if error is None:
            _rt_require(error_stage is None and effective is not None
                        and ((raw_kind == "none" and origin == "saved_ordinary_fallback")
                             or (raw_kind == "value" and origin == "accepted_return")), "successful handoff")
            active["origin"] = origin
        else:
            _rt_require(type(error) is str and _rt_re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,95}", error)
                        and error_stage in ("helper", "normalization") and effective is None and origin is None,
                        "failed handoff")
            active["origin"] = "helper_error"
        active["helper_finished"] = True
        record = active["record"]
        if record is not None:
            record["helper_finished"] = True
            record["phases"]["helper_finished"] = quality
            record["post_inputs"] = {k: self._snapshot(post_inputs[k], retain=False) for k in _RT_INPUTS}
            record["raw_return"] = {"kind": raw_kind, "type": raw_type,
                                    "vector": self._snapshot(raw_value) if raw_kind == "value"
                                    else {"ref": None, "reason": "not_applicable"}}
            record["effective"] = {"origin": origin, "vector": self._snapshot(effective)
                                   if effective is not None else {"ref": None, "reason": "not_applicable"}}
            record["error"] = {"stage": error_stage, "code": error} if error is not None else None
            self._trim()

    def _optimizer(self, phase, incumbent_sum, improvements, result=None):
        self._live()
        active = self._active
        _rt_require(active is not None, "optimizer without start")
        quality = _rt_quality(incumbent_sum, improvements)
        _rt_progress(active["quality"], quality)
        if phase == "optimizer_entered":
            _rt_require(not active["entered"] and (active["origin"] == "ordinary" or
                        (active["helper_finished"] and active["origin"] in _RT_ORIGINS[1:3])),
                        "optimizer entry order")
            active["entered"] = True
        else:
            _rt_require(active["entered"] and not active["returned"], "optimizer return order")
            active["returned"] = True
            active["result_attached"] = result is not None
        if active["record"] is not None:
            active["record"]["phases"][phase] = quality
            if phase == "optimizer_returned":
                active["record"]["optimizer_result"] = self._snapshot(result)
            self._trim()

    def optimizer_enter(self, *, incumbent_sum, improvements):
        self._optimizer("optimizer_entered", incumbent_sum, improvements)

    def optimizer_exit(self, *, incumbent_sum, improvements, result=None):
        self._optimizer("optimizer_returned", incumbent_sum, improvements, result)

    def optimizer_result(self, result):
        """Attach the already-read result once, without reading or coercing it again."""
        self._live()
        active = self._active
        _rt_require(active is not None and active["returned"] and not active["result_attached"],
                    "optimizer result attachment order")
        active["result_attached"] = True
        if active["record"] is not None:
            active["record"]["optimizer_result"] = self._snapshot(result)
            self._trim()

    def end_start(self, status, *, incumbent_sum, improvements):
        self._live()
        active = self._active
        _rt_require(active is not None and status in ("completed", "deadline", "error", "incomplete"),
                    "terminal start status")
        _rt_require(status != "completed" or active["returned"], "completed start without optimizer return")
        quality = _rt_quality(incumbent_sum, improvements)
        _rt_progress(active["quality"], quality)
        aggregate = self.aggregates[active["origin"]]
        aggregate["starts"] += 1
        aggregate["helper_calls"] += int(active["helper"])
        aggregate["optimizer_entered"] += int(active["entered"])
        aggregate["optimizer_returned"] += int(active["returned"])
        aggregate[status] += 1
        aggregate["improvements"] += quality["improvements"] - active["quality"]["improvements"]
        aggregate["gain"] += quality["sum"] - active["quality"]["sum"]
        if active["record"] is not None:
            active["record"]["phases"]["terminal"] = quality
            active["record"]["terminal"] = status
            self._trim()
        self.final_quality = quality
        self._active = None

    def finish(self, status, *, incumbent_sum, improvements):
        self._live()
        _rt_require(status in ("restart_limit", "deadline", "error"), "run status")
        quality = _rt_quality(incumbent_sum, improvements)
        if self._active is not None:
            self.end_start("incomplete" if status == "restart_limit" else status,
                           incumbent_sum=incumbent_sum, improvements=improvements)
        if self.initial_quality is None:
            self.initial_quality = quality
        if self.final_quality is not None:
            _rt_progress(self.final_quality, quality)
        self.final_quality, self.status = quality, status
        value = self._wire()
        value["trace_sha256"] = _rt_digest(value)
        return validate_trace(value)


