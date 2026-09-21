"""PlateauMonitor e-detector: false-alarm control, detection, phase reset, serialize.

OpportunityCostMonitor: forecast-based plateau (expected yield), not anomaly.
"""

import math
import random

from sciel.stats import OpportunityCostMonitor, PlateauMonitor


def _stream_detect(seq):
    m = PlateauMonitor(alpha=0.1, threshold=0.05, min_t=15)
    for i, x in enumerate(seq):
        m.observe(bool(x), phase=0)
        if m.newly_declared():
            return i
    return None


def test_detects_drop_to_zero():
    rng = random.Random(0)
    high = [rng.random() < 0.4 for _ in range(30)]
    low = [0] * 150
    at = _stream_detect(high + low)
    assert at is not None and at > 30


def test_no_detect_on_steady_high_rate():
    rng = random.Random(1)
    fired = sum(_stream_detect([rng.random() < 0.3 for _ in range(200)]) is not None
                for _ in range(1))
    assert fired == 0


def test_false_alarm_controlled_above_threshold():
    alarms = 0
    trials = 300
    for s in range(trials):
        rng = random.Random(1000 + s)
        if _stream_detect([rng.random() < 0.12 for _ in range(200)]) is not None:
            alarms += 1
    assert alarms / trials <= 0.1


def test_phase_change_resets_segment():
    m = PlateauMonitor()
    for _ in range(30):
        m.observe(False, phase=0)
    assert m.t_seg > 0 and m.wealth > 1.0
    m.observe(False, phase=1)
    assert m.t_seg == 1 and m.t == 1 and m.wealth < 1.1


def test_state_roundtrip():
    m = PlateauMonitor()
    for x in (0, 0, 1, 0, 0):
        m.observe(bool(x), phase=0)
    d = m.state()
    m2 = PlateauMonitor()
    m2.load_state(d)
    assert m2.state() == d


# --- OpportunityCostMonitor (§6.3 step 3) ---

def test_oc_zero_events_never_declares():
    # k=0 must NOT declare a plateau: c_hat=0 is "never started", not diminishing
    # returns. A barren island stays in base rather than being shifted to explore.
    m = OpportunityCostMonitor(threshold=0.5, horizon=50, min_n=15)
    declared = False
    for _ in range(200):
        m.observe(False, phase=0)
        declared = declared or m.newly_declared()
    assert not declared


def test_oc_steady_events_no_plateau():
    m = OpportunityCostMonitor(threshold=0.5, horizon=50, min_n=15)
    fired = False
    for i in range(200):
        m.observe(i % 2 == 0, phase=0)  # ~0.5 event rate, no diminishing returns
        fired = fired or m.newly_declared()
    assert not fired


def test_oc_flat_tail_declares():
    m = OpportunityCostMonitor(threshold=0.5, horizon=50, min_n=15)
    for i in range(20):
        m.observe(True, phase=0)  # dense early events
    at = None
    for i in range(400):  # long flat tail: c_hat/ln(n) collapses the forecast
        m.observe(False, phase=0)
        if m.newly_declared():
            at = i
            break
    assert at is not None


def test_oc_forecast_formula():
    m = OpportunityCostMonitor(horizon=50)
    m.n, m.k = 100, 10
    c_hat = 10 / math.log(100)
    assert m.forecast() == c_hat * math.log(150 / 100)


def test_oc_phase_reset():
    m = OpportunityCostMonitor()
    for _ in range(10):
        m.observe(True, phase=0)   # k>=1: productive, so a plateau can be declared
    for _ in range(300):           # flat tail collapses the forecast -> declares
        m.observe(False, phase=0)
        if m.newly_declared():
            break
    assert m.declared and m.k >= 1  # latch tripped
    m.observe(True, phase=1)        # phase change resets the segment
    assert m.n == 1 and m.k == 1 and not m.declared


def test_oc_state_roundtrip():
    m = OpportunityCostMonitor()
    for x in (1, 0, 1, 0, 0):
        m.observe(bool(x), phase=0)
    d = m.state()
    m2 = OpportunityCostMonitor()
    m2.load_state(d)
    assert m2.state() == d
