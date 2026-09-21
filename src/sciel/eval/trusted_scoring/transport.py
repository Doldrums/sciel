"""Strict primitive-value decoder; it does not authenticate an execution."""
import json
import math
import warnings

import numpy as np

MAX_BYTES = 128 * 1024
MAX_DEPTH = 5
SCHEMA = "sciel-candidate-values/1"
KINDS = {"circle", "heil", "beat"}
DTYPES = {"float16", "float32", "float64"}


def _reject(message):
    raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _reject("duplicate JSON key")
        result[key] = value
    return result


def _depth(text):
    depth = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                _reject("JSON nesting limit exceeded")
        elif char in "]}":
            depth -= 1


def _fields(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        _reject("unexpected output fields")


def _numbers(values, count):
    if type(values) is not list or len(values) != count:
        _reject("invalid numeric vector shape")
    if any(type(v) is not float or not math.isfinite(v) for v in values):
        _reject("numeric values must be finite binary64 JSON floats")


def _matrix(values, rows):
    if type(values) is not list or len(values) != rows:
        _reject("invalid numeric matrix shape")
    for row in values:
        _numbers(row, 2)


def _array(values, dtype):
    if type(dtype) is not str or dtype not in DTYPES:
        _reject("unsupported floating dtype")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            result = np.array(values, dtype=dtype, copy=True)
        except (ValueError, TypeError, OverflowError, Warning) as exc:
            raise ValueError("values do not reconstruct the declared dtype") from exc
    original = [v for row in values for v in row] if type(values[0]) is list else values
    if any(v.hex() != float(restored).hex() for v, restored in zip(original, result.flat)):
        _reject("inexact dtype round trip")
    return result


def decode(kind, payload):
    """Accept bytes only. No coercion, callback, path or candidate score is used."""
    if type(kind) is not str or kind not in KINDS:
        _reject("unsupported adapter")
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_BYTES:
        _reject("output must be bounded bytes")
    try:
        text = payload.decode("utf-8", errors="strict")
        _depth(text)
        data = json.loads(text, object_pairs_hook=_object,
                          parse_constant=lambda _: _reject("nonfinite JSON number"))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid UTF-8 JSON output") from exc
    _fields(data, {"schema", "kind", "values"})
    if data["schema"] != SCHEMA or data["kind"] != kind:
        _reject("output schema/adapter mismatch")
    values = data["values"]
    if kind == "circle":
        _fields(values, {"centers", "radii", "centers_dtype", "radii_dtype", "reported_sum"})
        _matrix(values["centers"], 26)
        _numbers(values["radii"], 26)
        if type(values["reported_sum"]) is not str or len(values["reported_sum"]) > 320:
            _reject("reported sum must be bounded diagnostic text")
        return {"centers": _array(values["centers"], values["centers_dtype"]),
                "radii": _array(values["radii"], values["radii_dtype"]),
                "reported_sum": values["reported_sum"]}
    if kind == "heil":
        _fields(values, {"points"})
        _matrix(values["points"], 11)
        return {"points": _array(values["points"], "float64")}
    _fields(values, {"support", "weights"})
    if type(values["support"]) is not list or not 1 <= len(values["support"]) <= 2000:
        _reject("support count must be between 1 and 2000")
    count = len(values["support"])
    _numbers(values["support"], count)
    _numbers(values["weights"], count)
    return {"support": _array(values["support"], "float64"),
            "weights": _array(values["weights"], "float64")}
