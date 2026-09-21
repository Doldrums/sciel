"""Validated resource vectors, balances and evidence-bound resolutions."""
from __future__ import annotations

import json
from copy import deepcopy
from decimal import Decimal, InvalidOperation, localcontext


RESOURCES = ("dollars", "cpu_seconds", "gpu_seconds", "wall_seconds", "storage_bytes", "steps")


_INTEGER = {"storage_bytes", "steps"}


_SCHEMA = "sciel-resource-ledger/1"


_ZERO_HASH = "0" * 64


class BudgetExceeded(RuntimeError):
    """Admission was denied, or an observed overrun was durably recorded."""

    def __init__(self, message, *, record=None):
        super().__init__(message)
        self.record = deepcopy(record)


class BudgetUnknown(RuntimeError):
    """An unresolved reservation must not be dispatched or spent again."""

    def __init__(self, message, *, record=None):
        super().__init__(message)
        self.record = deepcopy(record)


def _text(value, name, limit=256):
    if type(value) is not str or not value.strip() or len(value) > limit or any(ord(c) < 32 for c in value):
        raise ValueError(f"invalid {name}")
    return value


def _number(value, key):
    if type(value) not in (str, int, float, Decimal):
        raise ValueError(f"{key} must be a finite nonnegative number")
    if len(str(value)) > 128:
        raise ValueError(f"invalid {key} precision")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid {key}") from exc
    # Bounded input precision also prevents pathological exponent allocations.
    if (not number.is_finite() or number < 0 or number > Decimal("1e30")
            or number.as_tuple().exponent < -100):
        raise ValueError(f"{key} must be finite, nonnegative and bounded in precision")
    if key in _INTEGER and number != number.to_integral_value():
        raise ValueError(f"{key} must be integral")
    return Decimal(0) if number == 0 else number


def _wire(number, key):
    if key in _INTEGER:
        return int(number)
    return format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else format(number, "f")


def _vector(values, *, defaults=None, partial=False):
    if type(values) is not dict or set(values) - set(RESOURCES):
        raise ValueError("resource vector contains unknown keys or is not an object")
    keys = values if partial else RESOURCES
    return {key: _wire(_number(values.get(key, (defaults or {}).get(key, 0)), key), key) for key in keys}


def _limit_vector(values, *, defaults=None):
    """Normalize immutable ceilings; unlike usage, an explicit null is allowed."""
    if type(values) is not dict or set(values) - set(RESOURCES):
        raise ValueError("resource limits contain unknown keys or are not an object")
    result = {}
    for key in RESOURCES:
        value = values.get(key, (defaults or {}).get(key, 0))
        result[key] = None if value is None else _wire(_number(value, key), key)
    return result


def _cap_exceeds(cap, ceiling):
    return ceiling is not None and (cap is None or Decimal(str(cap)) > Decimal(str(ceiling)))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _balance(state, account_id=None):
    spent, held = {k: Decimal(0) for k in RESOURCES}, {k: Decimal(0) for k in RESOURCES}
    with localcontext() as ctx:
        ctx.prec = 160
        for row in state["reservations"].values():
            if account_id is not None and row["account_id"] != account_id:
                continue
            for key in RESOURCES:
                actual = Decimal(str(row["actual"].get(key, 0)))
                spent[key] += actual
                if row["status"] in ("reserved", "unknown"):
                    held[key] += max(Decimal(0), Decimal(str(row["bounds"][key])) - actual)
    return spent, held


def _violations(state, row):
    errors = [f"reservation bound exceeded: {key}" for key in RESOURCES
              if Decimal(str(row["actual"].get(key, 0))) > Decimal(str(row["bounds"][key]))]
    with localcontext() as ctx:
        ctx.prec = 160
        for scope, limits in ((None, state["limits"]),
                              (row["account_id"], state["accounts"][row["account_id"]])):
            spent, held = _balance(state, scope)
            errors.extend(f"{'campaign' if scope is None else 'account'} ceiling exceeded: {k}"
                          for k in RESOURCES if limits[k] is not None
                          and spent[k] + held[k] > Decimal(str(limits[k])))
    return errors


def resolution(reason, evidence_refs):
    reason = _text(reason, "resolution reason", 2048)
    if type(evidence_refs) is not list or not 1 <= len(evidence_refs) <= 32:
        raise ValueError("unknown resolution requires 1–32 evidence references")
    refs = [_text(ref, "evidence reference", 512) for ref in evidence_refs]
    if len(set(refs)) != len(refs):
        raise ValueError("duplicate resolution evidence references")
    return {"reason": reason, "evidence_refs": refs}
