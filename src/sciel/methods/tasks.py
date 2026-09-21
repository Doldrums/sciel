"""Trusted task measurements for externally executed JSON solution bundles.

Nothing here imports submitted source. These adapters are separate protocols
from the ordinary Engine cascade; a single receipt is not a stochastic
confirmation, independent replication, or proof of transfer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from pathlib import Path

from .models import content_hash, detached, validate_manifest, validate_value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest(name, author_lab, input_schema, output_schema, runtime_lock, resources, identity):
    return dict(name=name, kind="adapter", author_lab=author_lab, entrypoint="adapter.py:run",
        input_schema=detached(input_schema), output_schema=detached(output_schema),
        protocol={"purpose": "solution_execution_adapter", "permanent_toolbox_method": False,
                  "task_identity": identity, "execution": "fresh external JSON invocation"},
        runtime_lock=detached(runtime_lock), dependencies=[], parents=[], resources=detached(resources),
        scope={"measurement": "one assigned input; no replication or transfer claim"}, examples=[])


class TaskAdapter(ABC):
    metric: str
    direction: str = "max"

    @property
    def identity(self):
        return detached(self._identity)

    @property
    def input_schema(self):
        return detached(self._input_schema)

    @property
    def output_schema(self):
        return detached(self._output_schema)

    @abstractmethod
    def measure(self, input_value, output_value):
        """Measure already-returned JSON, never execute the candidate."""

    def transfer_payload(self, input_value, output_value):
        """Export checked ready output only; unsupported tasks fail closed."""
        raise ValueError('this task has no trusted ready-output transfer adapter')

    @abstractmethod
    def solution_bundle(self, code, author_lab, runtime_lock, resources):
        """Wrap source for external execution; do not register it as a method."""


class SortTask(TaskAdapter):
    """A task-general fixture: preserve the multiset and sort it ascending."""
    metric = "correct"

    def __init__(self):
        self._input_schema = {"type": "array", "items": {"type": "number"}, "maxItems": 4096}
        self._output_schema = detached(self._input_schema)
        policy = {"schema": "sciel-json-sort/1", "metric": self.metric,
                  "input_schema": self.input_schema, "output_schema": self.output_schema,
                  "criterion": "exact ascending values and preserved multiplicities"}
        self._identity = {**policy, "task_hash": content_hash(policy),
                          "adapter_sha256": _sha(__file__), "evaluator_sha256": _sha(__file__)}

    def measure(self, input_value, output_value):
        validate_value(input_value, self.input_schema)
        try:
            validate_value(output_value, self.output_schema)
        except ValueError:
            return {"valid": False, "metrics": {self.metric: 0.0},
                    "observables": {"reason": "invalid_output_schema"}}
        correct = output_value == sorted(input_value)
        return {"valid": correct, "metrics": {self.metric: float(correct)},
                "observables": {"input_count": len(input_value), "output_count": len(output_value),
                                "comparison": "exact_values_and_multiplicities",
                                "output_sha256": content_hash(output_value)}}

    def transfer_payload(self, input_value, output_value):
        if not self.measure(input_value, output_value)['valid']:
            raise ValueError('invalid sorting output cannot be transferred')
        return {'input':detached(input_value),'output':detached(output_value)}

    def solution_bundle(self, code, author_lab, runtime_lock, resources):
        files = {"candidate.py": code, "adapter.py":
                 "def run(values):\n    import candidate\n    return candidate.solve(values)\n"}
        manifest = _manifest("sort_solution", author_lab, self.input_schema, self.output_schema,
                             runtime_lock, resources, self.identity)
        validate_manifest(manifest, files)
        return {"files": files, "manifest": manifest}


def circle_geometry_schema(*, extra=True):
    number = {"type": "number"}
    return {"type": "object", "properties": {
        "centers": {"type": "array", "minItems": 26, "maxItems": 26,
                    "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": number}},
        "radii": {"type": "array", "minItems": 26, "maxItems": 26, "items": number},
        "centers_dtype": {"type": "string", "enum": ["float16", "float32", "float64"]},
        "radii_dtype": {"type": "string", "enum": ["float16", "float32", "float64"]},
    }, "required": ["centers", "radii"], "additionalProperties": extra}


class Circle26Task(TaskAdapter):
    """Canonical task binding with data-only native-precision measurement.

JSON-only tools default to binary64. Solution wrappers preserve native floating
array dtypes explicitly. Validity arithmetic uses binary64, as in the current
canonical evaluator; native score reduction retains the declared exact dtype.
"""
    metric = "combined_score"

    def __init__(self):
        from ..eval.registry import load_task
        root = Path(__file__).resolve().parents[3] / "tasks" / "oe" / "circle_packing"
        self.task = load_task(str(root))
        self._evaluator_source = Path(self.task.evaluator_path).read_text()
        self._seed_source = Path(self.task.initial_program_path).read_text()
        self._input_schema = {"type": "object", "properties": {
            "seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1}}, "additionalProperties": False}
        self._output_schema = circle_geometry_schema()
        self._identity = {"schema": "sciel-method-circle26-task/1", "task": self.task.qualified_name,
            "task_hash": self.task.task_hash, "evaluator_sha256": _sha(self.task.evaluator_path),
            "canonical_seed_sha256": _sha(self.task.initial_program_path), "adapter_sha256": _sha(__file__),
            "input_schema": self.input_schema, "output_schema": self.output_schema,
            "metric": self.metric, "direction": self.direction, "normalizer": 2.635,
            "validity_tolerance": 1e-6, "arithmetic": "binary64 validity; exact declared native dtype sum",
            "repetition": "one external invocation; plan assigns every seed and replication"}

    @property
    def canonical_seed_source(self):
        return self._seed_source

    def measure(self, input_value, output_value):
        import numpy as np
        # Tools can condition on a complete packing state, so their own frozen
        # manifest validates input shape. This task measures their returned data.
        detached(input_value)
        invalid = {"valid": False, "metrics": {self.metric: 0.0},
                   "observables": {"validity": 0.0, "sum_radii": 0.0}}
        try:
            validate_value(output_value, self.output_schema)
            arrays = []
            for key in ("centers", "radii"):
                encoded = np.asarray(output_value[key], dtype=np.float64)
                array = np.asarray(output_value[key], dtype=output_value.get(key + "_dtype", "float64"))
                if not np.isfinite(array).all() or not np.array_equal(array.astype(np.float64), encoded):
                    raise ValueError("geometry values do not round-trip through declared dtype")
                arrays.append(array)
            centers, radii = arrays
            c, r = centers.astype(np.float64), radii.astype(np.float64)
            valid = bool(np.all(r >= 0))
            for i in range(26):
                x, y = c[i]
                radius = r[i]
                if x - radius < -1e-6 or x + radius > 1 + 1e-6 or y - radius < -1e-6 or y + radius > 1 + 1e-6:
                    valid = False
                for j in range(i + 1, 26):
                    if np.sqrt(np.sum((c[i] - c[j]) ** 2)) < r[i] + r[j] - 1e-6:
                        valid = False
            total = float(np.sum(radii)) if valid else 0.0
            numeric = {"centers": c.tolist(), "radii": r.tolist()}
            return {"valid": valid, "metrics": {self.metric: total / 2.635},
                    "observables": {"validity": float(valid), "sum_radii": total,
                                    "numeric_sha256": content_hash(numeric),
                                    "centers_dtype": centers.dtype.name, "radii_dtype": radii.dtype.name,
                                    "validity_tolerance": 1e-6}}
        except (ValueError, TypeError, OverflowError):
            return invalid

    def transfer_payload(self, input_value, output_value):
        if not self.measure(input_value, output_value)['valid']:
            raise ValueError('invalid packing output cannot be transferred')
        return {
            'centers':detached(output_value['centers']),'radii':detached(output_value['radii']),
            'centers_dtype':output_value.get('centers_dtype','float64'),
            'radii_dtype':output_value.get('radii_dtype','float64')}

    def solution_bundle(self, code, author_lab, runtime_lock, resources):
        wrapper = '''def run(inputs):
    import random
    import numpy as np
    seed = inputs.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    import candidate
    from canonical_evaluator import _snapshot
    centers, radii, reported_sum = candidate.run_packing()
    centers, radii = _snapshot(centers, radii)
    return {"centers": centers.astype(float).tolist(), "radii": radii.astype(float).tolist(),
            "centers_dtype": centers.dtype.name, "radii_dtype": radii.dtype.name}
'''
        files = {"candidate.py": code, "adapter.py": wrapper,
                 "canonical_evaluator.py": self._evaluator_source}
        manifest = _manifest("circle26_solution", author_lab, self.input_schema, self.output_schema,
                             runtime_lock, resources, self.identity)
        validate_manifest(manifest, files)
        return {"files": files, "manifest": manifest}
