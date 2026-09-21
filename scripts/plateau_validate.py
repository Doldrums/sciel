"""Validate the plateau monitors: e-detector (futility) + opportunity-cost forecast.

Usage: plateau_validate.py [run_dir ...]
"""

import json
import math
import sys

import numpy as np

sys.path.insert(0, "src")
from sciel.stats import OpportunityCostMonitor, PlateauMonitor


def false_alarm_rate(true_p, n_streams=2000, horizon=200, seed=0):
    rng = np.random.default_rng(seed)
    alarms = 0
    for _ in range(n_streams):
        m = PlateauMonitor(alpha=0.1, threshold=0.05, min_t=15)
        stream = rng.random(horizon) < true_p
        fired = False
        for x in stream:
            m.observe(bool(x), phase=0)
            if m.newly_declared():
                fired = True
                break
        alarms += 1 if fired else 0
    return alarms / n_streams


def detection(p_high, p_low, n_high=40, n_low=160, n_streams=2000, seed=1):
    rng = np.random.default_rng(seed)
    detected = 0
    delays = []
    for _ in range(n_streams):
        m = PlateauMonitor(alpha=0.1, threshold=0.05, min_t=15)
        high = rng.random(n_high) < p_high
        low = rng.random(n_low) < p_low
        stream = np.concatenate([high, low])
        for i, x in enumerate(stream):
            m.observe(bool(x), phase=0)
            if m.newly_declared():
                if i >= n_high:
                    detected += 1
                    delays.append(i - n_high)
                break
    med_delay = float(np.median(delays)) if delays else None
    return detected / n_streams, med_delay


def real_journal(run_dir):
    events = [json.loads(l) for l in open(f"{run_dir}/sciel_events.jsonl")]
    monitors = {}
    detections = []
    phase_of = _phase_reader(events)
    for e in events:
        phase = phase_of(e)
        if e.get("event") != "iteration" or e.get("child_score") is None:
            continue
        isl = e["island"]
        m = monitors.setdefault(isl, PlateauMonitor(alpha=0.1, threshold=0.05, min_t=15))
        m.observe(bool(e.get("is_island_event")), phase=phase)
        if m.newly_declared():
            detections.append((isl, e["iteration_no"], round(m.wealth, 2)))
    return detections, {i: (m.t, int(m.s), round(m.wealth, 2)) for i, m in monitors.items()}


def _phase_reader(events):
    """Replay the SH phase the way the engine saw it.

    `iteration` rows carry no `phase` field -- it lives on `decision` and `population`
    rows -- so reading e.get("phase", 0) was a constant 0 and the monitors never
    segmented, unlike the live engine, which resets wealth and counts at every
    milestone. This walks the journal in order and remembers the last phase any event
    announced, so an iteration inherits the phase in force when it ran."""
    current = {"phase": 0}

    def phase_of(event):
        p = event.get("phase")
        if p is None and event.get("event") == "sh_cut":
            p = event.get("phase")
        if isinstance(p, int):
            current["phase"] = p
        elif isinstance(p, str):
            current["phase"] = 0 if p == "uniform" else (-1 if p == "final"
                                                         else int(p.split("_")[1]))
        return current["phase"]

    return phase_of


def oc_synthetic(c, horizon_evals=400, threshold=0.5, horizon=50, min_n=15, seed=2):
    """Log-improvement stream: P(event at t) = c/t so K(n) ~ c*ln(n). Report the eval at
    which the forecast crosses the opportunity-cost threshold and the c_hat vs true c."""
    rng = np.random.default_rng(seed)
    m = OpportunityCostMonitor(threshold=threshold, horizon=horizon, min_n=min_n)
    crossed = None
    for t in range(1, horizon_evals + 1):
        m.observe(bool(rng.random() < c / t), phase=0)
        if crossed is None and m.newly_declared():
            crossed = t
    c_hat = m.k / math.log(m.n) if m.n >= 2 else float("nan")
    return crossed, m.k, round(c_hat, 3)


def oc_real_journal(run_dir, threshold=0.5, horizon=50, min_n=15):
    events = [json.loads(l) for l in open(f"{run_dir}/sciel_events.jsonl")]
    monitors = {}
    declarations = []
    phase_of = _phase_reader(events)
    for e in events:
        phase = phase_of(e)
        if e.get("event") != "iteration" or e.get("child_score") is None:
            continue
        isl = e["island"]
        m = monitors.setdefault(isl, OpportunityCostMonitor(threshold, horizon, min_n))
        m.observe(bool(e.get("is_island_event")), phase=phase)
        if m.newly_declared():
            declarations.append((isl, e["iteration_no"], round(m.forecast(), 3)))
    return declarations, {i: (m.n, m.k, round(m.forecast(), 3)) for i, m in monitors.items()}


def main(argv):
    print("== e-detector false-alarm rate (futility; should be <= alpha=0.1) ==")
    for p in (0.05, 0.08, 0.12):
        print(f"  true_p={p}: FA={false_alarm_rate(p):.3f}")
    print("== e-detector detection (rate drops to p_low) ==")
    for ph, pl in ((0.3, 0.0), (0.2, 0.01), (0.15, 0.02)):
        rate, delay = detection(ph, pl)
        print(f"  p_high={ph} p_low={pl}: detect={rate:.3f} median_delay={delay}")
    print("== opportunity-cost forecast (log-improvement K(n)~c*ln(n)) ==")
    for c in (0.5, 1.0, 3.0):
        crossed, k, c_hat = oc_synthetic(c)
        print(f"  true_c={c}: plateau_at_eval={crossed} events={k} c_hat={c_hat}")
    for run_dir in argv[1:]:
        det, final = real_journal(run_dir)
        print(f"== {run_dir} [e-detector]: detections={det} final_per_island={final}")
        ocd, ocf = oc_real_journal(run_dir)
        print(f"== {run_dir} [opportunity-cost]: declarations={ocd} final_per_island={ocf}")


if __name__ == "__main__":
    main(sys.argv)
