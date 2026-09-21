"""P6 contract property tests, run across ALL selection kernels (I-14)."""

import random

import pytest

from sciel.components.selection.novelty_chain import (IslandView, NoveltyChainKernel)
from sciel.components.selection.kernels import (StockSelectionKernel, UCB1Kernel,
                                                UCBSlidingWindowKernel)


class _Ev:
    def __init__(self, r_norm, is_island_event=True):
        self.r_norm = r_norm
        self.is_island_event = is_island_event


def _kernels():
    return [
        ("novelty_chain", NoveltyChainKernel(floor=1e-3)),
        ("stock", StockSelectionKernel(floor=1e-3)),
        ("ucb1", UCB1Kernel(floor=1e-3)),
        ("ucb_sw", UCBSlidingWindowKernel(floor=1e-3, window=5)),
    ]


def _view(n=5):
    ids = [f"p{i}" for i in range(n)]
    scores = {pid: 0.5 + 0.1 * i for i, pid in enumerate(ids)}
    return IslandView(ids=ids, scores=scores, inspiration_ids={})


@pytest.mark.parametrize("name,kernel", _kernels())
def test_floor_and_propensity_range(name, kernel):
    view = _view()
    rng = random.Random(0)
    for _ in range(50):
        sel = kernel.select(view, tilt=None, rng=rng)
        assert 0.0 < sel.propensity <= 1.0, name
        assert sel.debug["rho_min"] >= kernel.floor - 1e-12, name
        assert abs(sel.debug["rho_sum"] - 1.0) < 1e-9, name
        assert sel.parent_id in view.ids


@pytest.mark.parametrize("name,kernel", _kernels())
def test_determinism_same_seed(name, kernel):
    view = _view()
    a = [kernel.select(view, None, random.Random(7)).parent_id for _ in range(1)]
    b = [type(kernel)(floor=1e-3).select(view, None, random.Random(7)).parent_id]
    assert a == b, name


@pytest.mark.parametrize("name,kernel", _kernels())
def test_observe_then_serialize_roundtrip(name, kernel):
    view = _view()
    rng = random.Random(3)
    for _ in range(10):
        sel = kernel.select(view, None, rng)
        kernel.observe(sel.parent_id, _Ev(0.4))
    d = kernel.state()
    clone = type(kernel)(floor=1e-3)
    clone.load_state(d)
    assert clone.state() == d, name


@pytest.mark.parametrize("name,kernel", _kernels())
def test_resume_reproduces_next_choice(name, kernel):
    view = _view()
    rng = random.Random(11)
    for _ in range(5):
        sel = kernel.select(view, None, rng)
        kernel.observe(sel.parent_id, _Ev(0.3))
    d = kernel.state()
    clone = type(kernel)(floor=1e-3)
    clone.load_state(d)
    r1 = random.Random(99)
    r2 = random.Random(99)
    assert (kernel.select(view, None, r1).parent_id
            == clone.select(view, None, r2).parent_id), name


def test_single_program_view():
    view = IslandView(ids=["p0"], scores={"p0": 0.9}, inspiration_ids={})
    for _name, kernel in _kernels():
        sel = kernel.select(view, None, random.Random(0))
        assert sel.parent_id == "p0"
        assert sel.propensity == pytest.approx(1.0)
