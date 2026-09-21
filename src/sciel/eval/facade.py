"""Evaluator contract and EvalResult: the only interface between engine and task."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Protocol


MAX_RESULT_BYTES = 4 * 1024 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024
MAX_ARTIFACT_TOTAL_BYTES = 2 * 1024 * 1024
MAX_ARTIFACTS = 256
MAX_CASES = 4096
MAX_STDERR_BYTES = 65536


def _number(value, field, *, nonnegative=False):
    if type(value) not in (int,float):
        raise ValueError(f'{field} must be a finite JSON number')
    try:
        value=float(value)
    except (OverflowError,ValueError):
        raise ValueError(f'{field} must be a finite JSON number') from None
    if not math.isfinite(value) or nonnegative and value<0:
        raise ValueError(f'{field} must be finite' + (' and nonnegative' if nonnegative else ''))
    return value


def _string(value, field, limit):
    if type(value) is not str or len(value.encode('utf-8'))>limit:
        raise ValueError(f'{field} must be text within {limit} UTF-8 bytes')
    return value


SETUP_FAILURE_SCHEMA = "sciel-native-evaluator-setup-failure/1"
_SETUP_CODES = frozenset({"unsupported_platform", "resource_unavailable",
    "setrlimit_failed", "setup_exception", "malformed_report", "request_mismatch"})
_SETUP_RESOURCES = frozenset({"cpu_seconds", "mem_bytes", "platform", "unknown"})


def setup_failure(code, resource, cpu_seconds=None, mem_bytes=None, reported_wall_seconds=None):
    """Conservative diagnostic only; never proof of zero execution or resources."""
    return {"schema": SETUP_FAILURE_SCHEMA, "code": code, "resource": resource,
        "requested_limits": {"cpu_seconds": cpu_seconds, "mem_bytes": mem_bytes},
        "execution": "unknown", "resource_usage": "unknown",
        "reported_wall_seconds": reported_wall_seconds}


def validate_setup_failure(value):
    import sys
    fields = {"schema", "code", "resource", "requested_limits", "execution",
              "resource_usage", "reported_wall_seconds"}
    if (type(value) is not dict or set(value) != fields
            or value["schema"] != SETUP_FAILURE_SCHEMA
            or type(value["code"]) is not str or value["code"] not in _SETUP_CODES
            or type(value["resource"]) is not str or value["resource"] not in _SETUP_RESOURCES
            or value["execution"] != "unknown" or value["resource_usage"] != "unknown"):
        raise ValueError("invalid native evaluator setup diagnostic")
    limits = value["requested_limits"]
    if type(limits) is not dict or set(limits) != {"cpu_seconds", "mem_bytes"}:
        raise ValueError("invalid native evaluator setup limits")
    for name, cap in limits.items():
        if cap is not None and (type(cap) is not int or cap <= 0
                or cap + (name == "cpu_seconds") >= sys.maxsize):
            raise ValueError("invalid native evaluator setup cap")
    elapsed = value["reported_wall_seconds"]
    if elapsed is not None:
        elapsed = _number(elapsed, "reported setup wall time", nonnegative=True)
    return setup_failure(value["code"], value["resource"], limits["cpu_seconds"],
                         limits["mem_bytes"], elapsed)


class EvaluationInfrastructureError(RuntimeError):
    """Fatal evaluator infrastructure fault; never a candidate quality result.

    Subclasses own strict diagnostics and explicit WAL families. Unknown
    execution keeps the source reservation pending. Only an independently
    host-proven predispatch failure may override execution_status.
    """
    execution_status = "unknown"


class EvaluationSetupError(EvaluationInfrastructureError):
    """Stop on reported setup failure without assigning candidate quality.

    Native children share the host trust domain. Their marker can justify a
    conservative stop, but cannot prove that code did not run or resources were
    zero. Physical dispatch/elapsed time and prior paid calls remain separate.
    """
    def __init__(self, diagnostic):
        self.diagnostic = validate_setup_failure(diagnostic)
        super().__init__("Native evaluator setup failed: " + self.diagnostic["code"]
                         + " (" + self.diagnostic["resource"] + ")")


@dataclass
class EvalResult:
    ok: bool
    score: float
    cases: dict[str, float] = field(default_factory=dict)
    stderr: str = ""
    timing: float = 0.0
    cost: float = 0.0
    artifacts: dict[str, str] = field(default_factory=dict)
    confirmed_sampled: bool = False

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "score": self.score,
            "cases": self.cases,
            "stderr": self.stderr,
            "timing": self.timing,
            "cost": self.cost,
            "artifacts": self.artifacts,
            "confirmed_sampled": self.confirmed_sampled,
        }

    @classmethod
    def from_json(cls, d: dict) -> "EvalResult":
        fields={'ok','score','cases','stderr','timing','cost','artifacts','confirmed_sampled'}
        if type(d) is not dict or set(d)-fields or not {'ok','score'}<=set(d):
            raise ValueError('invalid evaluator result object/fields')
        if type(d['ok']) is not bool or type(d.get('confirmed_sampled',False)) is not bool:
            raise ValueError('evaluator result status flags must be boolean')
        cases,artifacts=d.get('cases',{}),d.get('artifacts',{})
        if type(cases) is not dict or len(cases)>MAX_CASES:
            raise ValueError('invalid/bounded evaluator cases object')
        if type(artifacts) is not dict or len(artifacts)>MAX_ARTIFACTS:
            raise ValueError('invalid/bounded evaluator artifacts object')
        validated_cases={_string(k,'case key',512):_number(v,'case value') for k,v in cases.items()}
        validated_artifacts={_string(k,'artifact key',512):_string(v,'artifact value',MAX_ARTIFACT_BYTES)
                             for k,v in artifacts.items()}
        if sum(len(k.encode('utf-8'))+len(v.encode('utf-8')) for k,v in validated_artifacts.items())>MAX_ARTIFACT_TOTAL_BYTES:
            raise ValueError('evaluator artifacts exceed aggregate byte bound')
        return cls(
            ok=d['ok'], score=_number(d['score'],'score'), cases=validated_cases,
            stderr=_string(d.get('stderr',''),'stderr',MAX_STDERR_BYTES),
            timing=_number(d.get('timing',0.),'timing',nonnegative=True),
            cost=_number(d.get('cost',0.),'cost',nonnegative=True),
            artifacts=validated_artifacts, confirmed_sampled=d.get('confirmed_sampled',False),
        )


class Evaluator(Protocol):
    def smoke(self, program_path: str) -> EvalResult: ...
    def full(self, program_path: str) -> EvalResult: ...
    def confirm(self, program_path: str, n: int) -> EvalResult: ...
