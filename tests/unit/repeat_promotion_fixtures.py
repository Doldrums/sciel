"""Deterministic receipts and coordinator setup for repeat promotion tests."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import random
import pytest
from sciel.engine.repeat_promotion import (
    CONTRACT, RepeatPromotion, RepeatPromotionError, bootstrap_decision,
    paired_dominance, validate_config,
)
from sciel.engine.repeat_journal import digest


INITIAL = "# initial\r\npass\r\n"


CHALLENGER = "# challenger\npass\n"


def sha(code):
    return hashlib.sha256(code.encode()).hexdigest()


def config(**changes):
    return dict(enabled=True, seed_master=31, max_comparisons=2,
                max_draw_attempts=15, max_wall_seconds=150,
                timeout_seconds=8, certification_seconds=2,
                cpu_seconds=10, mem_bytes=1024**3, **changes)


def identity():
    return {"contract": CONTRACT, "task": "oe/circle_packing", "task_hash": "a" * 64,
            "canonical_initial_source_sha256": sha(INITIAL), "task_evaluator_sha256": "b" * 64,
            "adapter": {"schema": "sciel-repeat-adapter/1", "identity_sha256": "c" * 64},
            "integration_sha256": {"loop": "d" * 64}}


def value(raw=2, certified=None, status="valid"):
    cert = Fraction(raw if certified is None else certified)
    return {"status": status, "raw_sum": raw if status == "valid" else None,
            "certified_sum": {"numerator": cert.numerator, "denominator": cert.denominator} if status == "valid" else None}


def receipt(source, seed, limits, raw=2, certified=None, status="valid"):
    metrics = value(raw, certified, status)
    centers = [[0.1 + i * 0.2, 0.1 + j * 0.2] for i in range(5) for j in range(5)] + [[0.0, 0.0]]
    radii = [float(raw) / 25] * 25 + [0.0]
    certified_radii = [float(raw if certified is None else certified) / 25] * 25 + [0.0]
    cert_sum = sum(map(Fraction, certified_radii), Fraction())
    if status == "valid":
        metrics["certified_sum"] = {"numerator": cert_sum.numerator, "denominator": cert_sum.denominator}
    return {"schema": "sciel-repeat-draw/1", "source_sha256": sha(source),
            "adapter_identity": identity()["adapter"], "seed": seed,
            "seed_applied": {"python_random": seed, "numpy_legacy": seed, "python_hash_seed": str(seed),
                             "before_candidate_import": True, "determinism_proven": False},
            "limits": {"candidate_wall_seconds": limits["timeout"], "certification_wall_seconds": limits["certification_seconds"],
                       "candidate_cpu_seconds": limits["cpu_seconds"], "candidate_mem_bytes": limits["mem_bytes"],
                       "certification_cpu_seconds": limits["cpu_seconds"], "certification_mem_bytes": limits["mem_bytes"]},
            **metrics, "reason": None if status == "valid" else "fixture failure",
            "capture": {"valid": status == "valid", "sum_radii": metrics["raw_sum"], "centers": centers, "radii": radii},
            "certificate": {"schema": "sciel-circle-packing-certificate/1", "status": "certified",
                "sum_exact": metrics["certified_sum"], "centers": centers, "radii": certified_radii,
                "input_numeric_sha256": digest({"centers": centers, "radii": radii}),
                "certified_numeric_sha256": digest({"centers": centers, "radii": certified_radii})} if status == "valid" else None,
            "started_at": "2026-09-16T00:00:00+00:00", "completed_at": "2026-09-16T00:00:01+00:00",
            "candidate_elapsed_seconds": 0.8, "certification_elapsed_seconds": 0.2,
            "identity_elapsed_seconds": 0.0, "elapsed_seconds": 1.0}


class FakeDraw:
    def __init__(self, effects=None):
        self.calls = []
        self.effects = effects or {}

    def __call__(self, source, *, seed, **limits):
        self.calls.append((source, seed, deepcopy(limits)))
        effect = self.effects.get(len(self.calls), {})
        if isinstance(effect, BaseException):
            raise effect
        return receipt(source, seed, limits, raw=2 if source == INITIAL else 2.1, **effect)


@pytest.fixture
def coordinator(tmp_path):
    instances = []

    def make(draw=None, cfg=None, ident=None):
        controller = RepeatPromotion(tmp_path, cfg or config(), ident or identity(), draw or FakeDraw())
        instances.append(controller)
        return controller

    yield make
    for item in instances:
        item.close()


def bootstrap(controller):
    controller.start("run-fixture")
    assert controller.plan_bootstrap(INITIAL)
    saved = controller.state()
    controller.checkpoint_ack(saved["ledger_anchor"])
    controller.run_pending()
    return saved


def plan_comparison(controller):
    assert controller.plan_candidate(CHALLENGER, {"sum_radii": 2.01, "validity": 1}, "p2")
    saved = controller.state()
    controller.checkpoint_ack(saved["ledger_anchor"])
    return saved


class Crash(BaseException):
    pass
