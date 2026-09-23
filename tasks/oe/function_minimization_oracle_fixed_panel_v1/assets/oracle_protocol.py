"""Trusted host protocol for the opt-in hidden-oracle task.

The candidate and its client are untrusted. Only this process evaluates the
objective, owns the query count and selects the best observed point. Resource
isolation alone is insufficient: production trials require the container backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random


PROTOCOL = "sciel-hidden-oracle-fixed-panel/1"
BOOTSTRAP = "sciel-oracle-public-seed/1"
QUERY_BUDGET = 50
MAX_FRAME_BYTES = 4096
MAX_SOURCE_BYTES = 262144
STARTUP_TIMEOUT = 10.0
TRIAL_TIMEOUT = 5.0
BASE_IMAGE = (
    "python@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff"
)


class ProtocolError(ValueError):
    pass


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate JSON key")
        result[key] = value
    return result


def decode_frame(data: bytes) -> dict:
    if not data or len(data) > MAX_FRAME_BYTES:
        raise ProtocolError("invalid frame length")
    try:
        value = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ProtocolError("nonfinite JSON number")
            ),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ProtocolError("invalid JSON frame") from exc
    if not isinstance(value, dict):
        raise ProtocolError("frame must be an object")
    return value


def _coordinate(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("coordinate must be a number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ProtocolError("invalid coordinate") from exc
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ProtocolError("coordinate outside finite domain")
    return value


@dataclass
class OracleSession:
    """A single non-resettable trial, never shared with candidate memory.

    Repeated coordinates consume repeated queries. Malformed input, excess
    requests and objective failures latch failure; there is no recovery opcode.
    A finish request performs no evaluation and cannot override the best point.
    """

    objective: object = field(repr=False)
    algorithm_seed: int = 13007
    initialized: bool = False
    queries: int = 0
    best_point: tuple[float, float] | None = None
    best_value: float | None = None
    finished: bool = False
    failed: bool = False
    failure: str | None = None
    objective_failed: bool = False
    transcript: list = field(default_factory=list)

    def hello(self):
        if type(self.algorithm_seed) is not int or not 0 <= self.algorithm_seed < 2**32:
            raise ValueError("invalid public algorithm seed")
        return {
            "protocol": PROTOCOL,
            "bounds": [-1.0, 1.0],
            "dimensions": 2,
            "query_budget": QUERY_BUDGET,
            "bootstrap": BOOTSTRAP,
            "algorithm_seed": self.algorithm_seed,
        }

    def handle(self, data: bytes) -> dict:
        if self.finished or self.failed:
            raise ProtocolError("trial already closed")
        try:
            frame = decode_frame(data)
            if not self.initialized:
                if (
                    frame
                    != {
                        "op": "initialized",
                        "bootstrap": BOOTSTRAP,
                        "algorithm_seed": self.algorithm_seed,
                    }
                    or type(frame.get("algorithm_seed")) is not int
                ):
                    raise ProtocolError("invalid pre-import public seed initialization")
                self.initialized = True
                return {"status": "initialized"}
            if frame.get("op") == "finish":
                if set(frame) != {"op"} or not self.queries:
                    raise ProtocolError("invalid finish")
                self.finished = True
                return {"status": "finished", "queries": self.queries}
            if set(frame) != {"op", "point"} or frame["op"] != "query":
                raise ProtocolError("unsupported request")
            point = frame["point"]
            if not isinstance(point, list) or len(point) != 2:
                raise ProtocolError("point must have two coordinates")
            x, y = map(_coordinate, point)
            if self.queries >= QUERY_BUDGET:
                raise ProtocolError("query budget exceeded")
            # Charge before computation, even if the objective itself fails.
            self.queries += 1
            try:
                value = float(self.objective(x, y))
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid trusted objective value")
            except Exception as exc:
                self.objective_failed = True
                # Do not expose the objective's repr, arguments or traceback.
                raise ProtocolError("trusted objective failed") from exc
            if self.best_value is None or value < self.best_value:
                self.best_point, self.best_value = (x, y), value
            self.transcript.append(
                {"query": self.queries, "point": [x, y], "value": value}
            )
            return {
                "status": "ok",
                "query": self.queries,
                "value": value,
                "remaining": QUERY_BUDGET - self.queries,
            }
        except ProtocolError as exc:
            self.failed, self.failure = True, str(exc)
            raise


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def instance_from_seed(seed: bytes):
    """Public family; private parameters. The unique analytic minimum is zero."""
    rng = random.Random(int.from_bytes(seed, "big"))
    return {
        "centre": [rng.uniform(-0.75, 0.75) for _ in range(2)],
        "angle": rng.uniform(0, 2 * math.pi),
        "curvature": [rng.uniform(0.2, 1.5) for _ in range(2)],
        "waves": [
            {
                "weight": rng.uniform(0.1, 0.7),
                "frequency": [rng.uniform(-8.0, 8.0) for _ in range(2)],
            }
            for _ in range(4)
        ],
    }


def objective_for(instance):
    # Validate private manifests too: malformed trusted input is an infra failure.
    if set(instance) != {"centre", "angle", "curvature", "waves"}:
        raise ValueError("invalid oracle instance schema")
    c = instance["centre"]
    q = instance["curvature"]
    waves = instance["waves"]
    if len(c) != 2 or len(q) != 2 or len(waves) != 4:
        raise ValueError("invalid oracle instance shape")
    values = [*c, instance["angle"], *q]
    for wave in waves:
        if set(wave) != {"weight", "frequency"} or len(wave["frequency"]) != 2:
            raise ValueError("invalid wave")
        values += [wave["weight"], *wave["frequency"]]
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values
    ):
        raise ValueError("invalid oracle parameter")
    if (
        any(not -0.75 <= v <= 0.75 for v in c)
        or any(not 0.2 <= v <= 1.5 for v in q)
        or not 0 <= instance["angle"] <= 2 * math.pi
        or any(
            not 0.1 <= w["weight"] <= 0.7
            or any(not -8 <= z <= 8 for z in w["frequency"])
            for w in waves
        )
    ):
        raise ValueError("oracle parameter outside declared family")
    ca, sa = math.cos(instance["angle"]), math.sin(instance["angle"])

    def objective(x, y):
        dx, dy = x - c[0], y - c[1]
        u, v = ca * dx - sa * dy, sa * dx + ca * dy
        return (
            q[0] * u * u
            + q[1] * v * v
            + sum(
                w["weight"]
                * math.sin(w["frequency"][0] * dx + w["frequency"][1] * dy) ** 2
                for w in waves
            )
        )

    return objective


