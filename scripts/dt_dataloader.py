"""Decision-Transformer / return-conditioned trajectory dataloader.

Builds per-island lineage trajectories from run journals for return-conditioned
learning. Return-to-go RTG_t = V_n - V_t ("improvement still ahead", Lemma 1: raw
per-iteration gains telescope to the final best). Skeleton no longer -- Stage A
(return-conditioned mode MLP) trains on these; Stage B (LaMo online-DT) will too.

Hygiene (Stage 0):
- REQUIRE run_end: a killed run has no final V_n; using 0.0 made RTG negative and
  poisoned the corpus. Such runs are skipped (and reported, not dropped silently).
- RTG from RAW gains (r_raw), never r_norm -- only raw telescopes.
- Feature parity: context uses ONLY pre-step state known at controller.pick_mode time
  (before parent/op are chosen). See mode_features() -- the SAME function feeds
  training and live inference, so train/serve features can't drift.

Trains ONLY on v1 (new-engine) journals; old-repo journals are forbidden.
"""

import os
import sys
import math
import hashlib
import json
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from sciel.components.policy.mode_features import MODES, mode_features  # noqa: E402
from journal_data import DirtyRun, completed_events  # noqa: E402


def _load_journal(path, *, trajectory=None):
    events, end = completed_events(path) if trajectory is None else trajectory
    start = next((e for e in events if e["event"] == "run_start"), None)
    if start is None or (start.get("non_comparable") is None
                         and start.get("schema_version") is None):
        raise DirtyRun(f"{path}: not a v1 journal (old-repo data forbidden)")
    budget = start.get("budget", start["config"].get("budget"))
    # Mirror Engine._schedule_horizon, not the stopping counter. Unlimited
    # search still uses a finite recorded horizon for SH and progress features.
    horizon = start["config"].get("schedule_horizon") if budget is None else budget
    if type(horizon) is not int or horizon < 1:
        raise DirtyRun(f"{path}: missing or invalid recorded scheduling horizon")
    fracs = start["config"].get("milestone_fracs", [0.5, 0.75])
    milestones = [max(1, round(f * horizon)) for f in fracs]
    policy = start.get("schedule_policy") or start["config"].get("schedule_policy", "sh")
    return events, end["pop_best"], horizon, milestones, len(fracs), policy, budget is None


def _phase_at(b_evals_pre, milestones, n_fracs, policy="sh"):
    """Mirror of Schedule.phase_code.

    The live scheduler checks `policy` FIRST and reports phase 0 for the whole run
    under round_robin. Re-deriving the phase from milestones alone labelled those same
    steps 1 and -1, so every arm-B run in the corpus poisoned the phase one-hot for
    about half its samples."""
    if policy == "round_robin":
        return 0
    passed = sum(1 for m in milestones if b_evals_pre >= m)
    return passed if passed < n_fracs else -1  # 0=uniform, 1..=sh_k, -1=final


def build_samples(path, *, trajectory=None):
    """Yield per-step training records with pre-step (feature-parity) context."""
    events, final_v, horizon, milestones, n_fracs, policy, unlimited = _load_journal(
        path, trajectory=trajectory)
    succ = [e for e in events
            if e["event"] == "iteration" and e.get("child_score") is not None
            and e.get('reason') != 'remeasurement' and not (e.get('meta') or {}).get('remeasurement')]
    parallel = any((e.get('config',{}).get('parallel_search') or {}).get('enabled') is True
                   for e in events if e.get('event')=='run_start')
    samples = []
    prev_pop, island_bests = None, {}
    # Journal order is global execution order, including resumed segments. Keeping
    # one prev_pop per island lost every intervening improvement on other islands.
    for e in succ:
        island = e["island"]
        decision = e.get('decision_context')
        if decision is not None:
            if (type(decision) is not dict or decision.get('schema')!='sciel-mode-decision/1'
                    or type(decision.get('phase')) is not int or decision['phase'] < -1
                    or any(type(decision.get(k)) not in (int,float) or not math.isfinite(decision[k])
                           for k in ('progress','pop_best','island_best'))
                    or decision['progress'] < 0):
                raise DirtyRun(f'{path}: invalid action-selection feature context')
        elif parallel:
            raise DirtyRun(f'{path}: parallel action lacks recorded decision-time context')
        # Post-best minus the raw jump recovers the exact historical pre-best,
        # including portfolio/spawn/migration changes absent from iteration rows.
        pop_pre = e.get("pop_best_pre", e.get("v_t"))
        if pop_pre is None and "d_pop" in e and "pop_best" in e:
            pop_pre = e["pop_best"] - e["d_pop"]
        isl_pre = e.get("island_best_pre")
        if isl_pre is None and "d_isl" in e and "island_best" in e:
            isl_pre = e["island_best"] - e["d_isl"]
        if pop_pre is None:
            pop_pre = prev_pop
        if isl_pre is None:
            isl_pre = island_bests.get(island)
        if decision is not None:
            pop_pre, isl_pre = decision['pop_best'], decision['island_best']
        # Advance even when this row's mode is unsuitable for imitation.
        prev_pop = e.get("pop_best", pop_pre)
        island_bests[island] = e.get("island_best", isl_pre)
        if pop_pre is None or isl_pre is None:
            continue  # historical row lacks enough evidence; never invent a seed
        pop_pre, isl_pre = float(pop_pre), float(isl_pre)
        mode = e.get("effective_mode", e.get("mode"))
        if mode not in MODES:
            continue
        b_pre = max(0, e["B_evals"] - 1)
        phase = e.get("phase_pre", _phase_at(b_pre, milestones, n_fracs, policy))
        progress = e.get("progress_pre")
        if progress is None:
            progress = b_pre / horizon
            if unlimited:
                progress = min(1.0, progress)
        if decision is not None:
            phase, progress = decision['phase'], decision['progress']
        feats = mode_features(phase, progress, pop_pre, isl_pre)
        samples.append({
            "features": feats, "mode": mode, "r_raw": e.get("r_raw", 0.0),
            "rtg": max(0.0, final_v - pop_pre),
            "rtg_raw": float(final_v - pop_pre), "v_t": pop_pre,
            "island": island, "run": path,
        })
    return samples, final_v


def _unique_runs(run_dirs):
    """Validate once, then deduplicate physical/selected trajectory identities.

    Modern run IDs cover copied and resumed versions even when their final
    checkpoint differs. Legacy selected-event hashes conservatively group
    byte-equivalent evidence without pretending it proves independent runs.
    """
    runs, seen = [], {}
    report = {"loaded": [], "skipped": []}
    for d in run_dirs:
        try:
            path = (Path(d).expanduser() / "sciel_events.jsonl").resolve()
            info = path.stat()
            trajectory = completed_events(str(path))
            events, _ = trajectory
            run_ids = {e['run_id'] for e in events if e.get('event') == 'run_start'
                       and type(e.get('run_id')) is str and e['run_id']}
            if len(run_ids) > 1:
                raise DirtyRun(f'{path}: selected trajectory has multiple run identities')
            encoded = json.dumps(events, sort_keys=True, separators=(',', ':'),
                                 allow_nan=False).encode('utf-8')
            keys = {('path', str(path)), ('inode', info.st_dev, info.st_ino),
                    ('selected_events', hashlib.sha256(encoded).hexdigest())}
            keys.update(('run_id', ident) for ident in run_ids)
            duplicates = [seen[key] for key in keys if key in seen]
            if duplicates:
                report['skipped'].append((d, f'duplicate run identity; already loaded {min(duplicates)}'))
                continue
            s, _ = build_samples(str(path), trajectory=trajectory)
            seen.update((key, str(path)) for key in keys)
            runs.append((s, keys))
            report["loaded"].append((d, len(s)))
        except (DirtyRun, OSError, ValueError) as exc:
            report["skipped"].append((d, str(exc)))
    return runs, report


def load_corpus(run_dirs):
    """Load unique runs and report dirty inputs/aliases instead of weighting them twice."""
    runs, report = _unique_runs(run_dirs)
    return [sample for samples, _ in runs for sample in samples], report


def held_out_split(run_dirs, holdout_frac=0.2):
    """Split unique validated runs, keeping copied/resumed identities together."""
    if type(holdout_frac) not in (int, float) or not 0 < holdout_frac < 1:
        raise ValueError('holdout_frac must lie strictly between zero and one')
    runs, _ = _unique_runs(run_dirs)
    n_hold = max(1, int(round(holdout_frac * len(runs))))
    train, hold = runs[n_hold:], runs[:n_hold]
    train_keys = {key for _, keys in train for key in keys}
    hold_keys = {key for _, keys in hold for key in keys}
    if train_keys & hold_keys:
        raise DirtyRun('training and holdout have overlapping run identities')
    return ([sample for samples, _ in train for sample in samples],
            [sample for samples, _ in hold for sample in samples])


def summarize(run_dirs):
    samples, report = load_corpus(run_dirs)
    rtgs = [s["rtg"] for s in samples]
    modes = {m: sum(1 for s in samples if s["mode"] == m) for m in MODES}
    print(f"runs: loaded={len(report['loaded'])} skipped={len(report['skipped'])}")
    for d, why in report["skipped"]:
        print(f"  SKIP {d}: {why}")
    print(f"samples: {len(samples)}  mode counts: {modes}")
    if rtgs:
        raw = [s.get("rtg_raw", s["rtg"]) for s in samples]
        neg = sum(1 for r in raw if r < 0)
        print(f"RTG range: [{min(rtgs):.4f}, {max(rtgs):.4f}]  negatives before "
              f"clamping: {neg} (must be 0 after Stage-0 hygiene)")
        if neg:
            print(f"  WARNING: {neg} samples had V_n < V_t and were clamped to 0. "
                  f"The old report counted them AFTER the clamp, so it always said 0.")


if __name__ == "__main__":
    summarize(sys.argv[1:])
