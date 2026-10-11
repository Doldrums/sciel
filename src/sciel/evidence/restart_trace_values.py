"""Bounded observations of existing restart calls, never a helper executor.

This stdlib-only module is also embedded verbatim in a source-pinned scaffold.
Array bytes are C-order snapshots of arrays supplied by that scaffold. Raw
return bytes describe its existing ``asarray`` result, not Python container
storage. Hashes bind recorded bytes; they do not authenticate their origin.
Schema /2 also records the optimizer's returned array before incumbent
admission. That snapshot makes no feasibility or quality assertion. Legacy /1
traces remain readable without fabricating a missing optimizer result.
"""
from __future__ import annotations

import base64 as _rt_base64
import hashlib as _rt_hashlib
import json as _rt_json
import math as _rt_math
import re as _rt_re
import struct as _rt_struct

TRACE_SCHEMA = "sciel-restart-trace/2"
LEGACY_TRACE_SCHEMA = "sciel-restart-trace/1"
MAX_TRACE_BYTES = 98304 - 2048
PREFIX_LIMIT = 16
MAX_STARTS = 10000
_RT_ARRAY_BYTES = 32768
_RT_INPUTS = ("best", "initial", "noise", "ordinary")
_RT_ORIGINS = ("ordinary", "accepted_return", "saved_ordinary_fallback",
               "helper_error", "scheduled_unresolved")
_RT_COUNTS = ("starts", "helper_calls", "optimizer_entered", "optimizer_returned",
              "completed", "deadline", "error", "incomplete", "improvements")
_RT_PHASES = ("start", "helper_finished", "optimizer_entered", "optimizer_returned", "terminal")
_RT_REASONS = ("not_applicable", "array_unavailable", "unsupported_dtype",
               "unsupported_shape", "oversized")


def _rt_require(value, message):
    if not value:
        raise ValueError("restart trace: " + message)


def _rt_json_text(value):
    return _rt_json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)


def _rt_sha(raw):
    return _rt_hashlib.sha256(raw).hexdigest()


def _rt_digest(value):
    return _rt_sha(_rt_json_text(value).encode("ascii"))


def _rt_hex(value, size=64):
    return type(value) is str and _rt_re.fullmatch("[0-9a-f]{%d}" % size, value) is not None


def _rt_int(value):
    return type(value) is int and 0 <= value <= 2 ** 63 - 1


def _rt_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return _rt_math.isfinite(value)
    except OverflowError:
        return False


def _rt_keys(value, keys, where):
    _rt_require(type(value) is dict and set(value) == set(keys), where + " fields")


def _rt_clone(value, depth=0, budget=None):
    if budget is None:
        budget = [30000]
    budget[0] -= 1
    _rt_require(depth <= 32 and budget[0] >= 0, "JSON structure limit")
    if value is None or type(value) in (str, bool, int):
        if type(value) is str:
            _rt_require(len(value) <= MAX_TRACE_BYTES, "string limit")
        return value
    if type(value) is float:
        _rt_require(_rt_math.isfinite(value), "nonfinite JSON number")
        return value
    if type(value) is list:
        return [_rt_clone(v, depth + 1, budget) for v in value]
    if type(value) is dict:
        _rt_require(all(type(k) is str for k in value), "JSON object keys")
        return {k: _rt_clone(v, depth + 1, budget) for k, v in value.items()}
    raise ValueError("restart trace: non-JSON value")


def _rt_identity(value):
    _rt_keys(value, ("source_sha256", "scaffold_sha256", "runtime_sha256", "seed",
                     "invocation_id"), "identity")
    _rt_require(all(_rt_hex(value[k]) for k in ("source_sha256", "scaffold_sha256",
                                              "runtime_sha256")), "identity hashes")
    _rt_require(_rt_hex(value["invocation_id"], 32), "invocation id")
    _rt_require(_rt_int(value["seed"]) and value["seed"] < 2 ** 32, "seed")
    return dict(value)


def _rt_quality(incumbent_sum, improvements):
    _rt_require(_rt_number(incumbent_sum) and _rt_int(improvements), "quality")
    return {"sum": float(incumbent_sum), "improvements": improvements}


def _rt_check_quality(value):
    _rt_keys(value, ("sum", "improvements"), "quality")
    _rt_require(type(value["sum"]) is float and _rt_number(value["sum"])
                and _rt_int(value["improvements"]), "quality types")


def _rt_progress(before, after):
    _rt_require(after["sum"] >= before["sum"]
                and after["improvements"] >= before["improvements"], "incumbent regressed")


def _rt_dtype(dtype):
    if type(dtype) is not str:
        return None
    match = _rt_re.fullmatch(r"([<>|])([fiub])(1|2|4|8)", dtype)
    if not match:
        return None
    endian, kind, width = match.group(1), match.group(2), int(match.group(3))
    formats = {"f": {2: "e", 4: "f", 8: "d"},
               "i": {1: "b", 2: "h", 4: "i", 8: "q"},
               "u": {1: "B", 2: "H", 4: "I", 8: "Q"}, "b": {1: "?"}}
    if width not in formats[kind] or (endian == "|") != (width == 1):
        return None
    return ("<" if endian == "|" else endian) + formats[kind][width], width


def _rt_array(value):
    """Inspect only an already-converted array; never perform array coercion."""
    if value is None:
        return None, "array_unavailable"
    try:
        dtype = value.dtype
        if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
            return None, "unsupported_dtype"
        dtype_text = dtype.str
        spec = _rt_dtype(dtype_text)
        if spec is None:
            return None, "unsupported_dtype"
        shape = list(value.shape)
        if len(shape) > 4 or not all(_rt_int(n) for n in shape):
            return None, "unsupported_shape"
        count = _rt_math.prod(shape)
        size = count * spec[1]
        if size > _RT_ARRAY_BYTES:
            return None, "oversized"
        if value.nbytes != size:
            return None, "array_unavailable"
        raw = value.tobytes(order="C")
        if type(raw) is not bytes or len(raw) != size:
            return None, "array_unavailable"
        return {"dtype": dtype_text, "shape": shape, "order": "C",
                "data_base64": _rt_base64.b64encode(raw).decode("ascii"),
                "bytes_sha256": _rt_sha(raw)}, None
    except Exception:
        # An unsupported capture cannot silently become a normalized success.
        return None, "array_unavailable"


def _rt_vector(value):
    _rt_keys(value, ("dtype", "shape", "order", "data_base64", "bytes_sha256"), "vector")
    spec = _rt_dtype(value["dtype"])
    _rt_require(spec is not None and value["order"] == "C", "vector dtype/order")
    shape = value["shape"]
    _rt_require(type(shape) is list and len(shape) <= 4 and all(_rt_int(n) for n in shape),
                "vector shape")
    size = _rt_math.prod(shape) * spec[1]
    _rt_require(size <= _RT_ARRAY_BYTES and type(value["data_base64"]) is str,
                "vector byte limit")
    try:
        raw = _rt_base64.b64decode(value["data_base64"], validate=True)
    except Exception as exc:
        raise ValueError("restart trace: invalid base64") from exc
    _rt_require(len(raw) == size and _rt_base64.b64encode(raw).decode("ascii") == value["data_base64"]
                and _rt_hex(value["bytes_sha256"]) and _rt_sha(raw) == value["bytes_sha256"],
                "vector bytes/hash")
    return raw, spec


def _rt_normalized(value):
    raw, spec = _rt_vector(value)
    _rt_require(value["shape"] == [78] and value["dtype"] in ("<f8", ">f8")
                and all(_rt_math.isfinite(x[0]) for x in _rt_struct.iter_unpack(spec[0], raw)),
                "effective/input vector must be finite binary64 shape78")


def _rt_normalization(raw_value, effective_value=None):
    """Bind captured real array values to the scaffold's binary64 conversion."""
    raw, spec = _rt_vector(raw_value)
    _rt_require(raw_value["shape"] == [78] and raw_value["dtype"][1] in "fiu",
                "accepted raw shape/dtype")
    numbers = [float(item[0]) for item in _rt_struct.iter_unpack(spec[0], raw)]
    _rt_require(all(_rt_math.isfinite(x) for x in numbers), "accepted raw nonfinite")
    if effective_value is not None:
        effective, _ = _rt_vector(effective_value)
        converted = _rt_struct.pack(effective_value["dtype"][0] + "78d", *numbers)
        _rt_require(converted == effective, "raw/effective normalization mismatch")


def _rt_descriptor(value, vectors, *, post=False, normalized=False):
    _rt_keys(value, ("ref", "reason"), "snapshot descriptor")
    if value["ref"] is None:
        _rt_require(value["reason"] in _RT_REASONS, "snapshot omission reason")
    else:
        _rt_require(_rt_hex(value["ref"]) and value["reason"] is None, "snapshot reference")
        if not post:
            _rt_require(value["ref"] in vectors, "missing vector")
            if normalized:
                _rt_normalized(vectors[value["ref"]])


def _rt_empty_aggregate():
    return {**{k: 0 for k in _RT_COUNTS}, "gain": 0.0}


def _rt_record_refs(record):
    descriptors = list(record["inputs"].values()) + [record["raw_return"]["vector"],
                                                   record["effective"]["vector"]]
    if "optimizer_result" in record:
        descriptors.append(record["optimizer_result"])
    return [v["ref"] for v in descriptors if v["ref"] is not None]


