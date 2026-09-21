"""Evaluator for function_minimization; smoke/full/confirm on the sciel contract.

The score formula matches the openevolve reference for honest, successful fixture
programs within the task budget. This evaluator recomputes the objective from the
returned coordinates, bounds each trial and observable objective calls, and rejects
confirmation failures. Task hashes distinguish these evaluation contracts.
"""

from sciel.candidate_modules import load_candidate
import json
import os
from pathlib import Path
import random
import select
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

GLOBAL_MIN_X = -1.704
GLOBAL_MIN_Y = 0.678
GLOBAL_MIN_VALUE = -1.519
# Owned by the evaluator, so changing the candidate's globals cannot change the
# budget used to compare programs. Inlined objectives remain unobservable.
OBJECTIVE_QUERY_BUDGET = 50


def _invoke_trial(program_path, module_name, module_dir, sender, receiver):
    """Fresh-interpreter entry point; keep imports available to spawn children."""
    try:
        # A spawned multiprocessing child must be able to import the candidate
        # by name. The shim preserves its original __file__ and code filenames;
        # it lives until the complete trial tree has been removed by the caller.
        original = str(Path(program_path).resolve())
        shim = Path(module_dir) / (module_name + ".py")
        shim.write_text(
            f"__file__ = {original!r}\n"
            "with open(__file__, 'rb') as _sciel_source:\n"
            "    _sciel_code = _sciel_source.read()\n"
            "exec(compile(_sciel_code, __file__, 'exec'), globals())\n")
        sys.path.insert(0, module_dir)
        # A candidate can still choose a deterministic seed after import.
        random.seed()
        np.random.seed()
        value = _trial_body(original, module_name)
        payload = {"ok": True, "value": value}
    except BaseException as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        data = json.dumps(payload, allow_nan=False).encode() + b"\n"
        with os.fdopen(sender, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
    finally:
        # Keep the process alive until its caller has removed its descendants.
        # Exiting immediately would reparent them before the caller can discover
        # their relationship to this trial. EOF releases us if the caller exits.
        try:
            os.read(receiver, 1)
        except (EOFError, OSError):
            pass
        os.close(receiver)


def _trial_descendants(pid):
    """Snapshot descendants without changing the outer sandbox's process group."""
    result = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True,
                            text=True, check=True, timeout=1)
    children = {}
    for line in result.stdout.splitlines():
        child, parent = map(int, line.split())
        children.setdefault(parent, []).append(child)
    descendants, pending = [], list(children.get(pid, []))
    while pending:
        child = pending.pop()
        descendants.append(child)
        pending.extend(children.get(child, []))
    return descendants


def _signal_if_alive(pid, sig):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _kill_trial_tree(proc):
    """Stop further forks, then kill descendants before reaping the trial worker.

    No setsid/setpgid here: a stage timeout must still kill the entire tree through
    sandbox.killpg, including a trial that is currently stopped for this cleanup.
    """
    if proc.poll() is not None:
        proc.wait()
        return
    _signal_if_alive(proc.pid, signal.SIGSTOP)
    stopped = set()
    try:
        # A child may fork before its SIGSTOP arrives. Resnapshot until the
        # discovered tree is stable; stopped members cannot add further children.
        while True:
            descendants = _trial_descendants(proc.pid)
            fresh = set(descendants) - stopped
            if not fresh:
                break
            for pid in fresh:
                _signal_if_alive(pid, signal.SIGSTOP)
            stopped.update(fresh)
        for pid in reversed(descendants):
            _signal_if_alive(pid, signal.SIGKILL)
    finally:
        # Even a failed ps must not leave our stopped worker behind. The stage
        # sandbox remains responsible for any descendants discovery could not see.
        for pid in stopped:
            _signal_if_alive(pid, signal.SIGKILL)
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def _run_with_timeout(program_path, module_name="program", timeout_seconds=5):
    """Run by path after exec, including startup/imports in the trial deadline.

    No Python callable or inherited interpreter state crosses the process boundary.
    The trial inherits the stage's process group: an outer timeout can still kill
    its whole tree, even during inner cleanup. This remains a POSIX/ps helper.
    """
    if os.name != "posix":
        raise RuntimeError("trial isolation requires POSIX process-tree cleanup")
    if not module_name.isidentifier():
        raise ValueError("trial module name must be an identifier")
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("trial timeout must be positive and finite")
    started = time.monotonic()
    owned_fds = set()
    proc = None
    with tempfile.TemporaryDirectory(prefix="sciel_trial_") as directory:
        try:
            receiver, sender = os.pipe()
            owned_fds.update((receiver, sender))
            release_reader, release_writer = os.pipe()
            owned_fds.update((release_reader, release_writer))
            proc = subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--_trial",
                 str(Path(program_path).resolve()), module_name, directory,
                 str(sender), str(release_reader)],
                pass_fds=(sender, release_reader),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            for fd in (sender, release_reader):
                os.close(fd)
                owned_fds.remove(fd)
            data = bytearray()
            while b"\n" not in data:
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0 or not select.select([receiver], [], [], remaining)[0]:
                    raise TimeoutError(f"trial exceeded {timeout_seconds}s")
                chunk = os.read(receiver, 4096)
                if not chunk:
                    raise RuntimeError("trial exited without a result")
                data.extend(chunk)
                if len(data) > 65536:
                    raise RuntimeError("trial result exceeds size limit")
            payload = json.loads(data)
            if not payload["ok"]:
                raise RuntimeError(payload["error"])
            return tuple(payload["value"])
        finally:
            try:
                if proc is not None:
                    _kill_trial_tree(proc)
            finally:
                for fd in owned_fds:
                    os.close(fd)


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _objective(x, y):
    """The true objective, owned by the evaluator."""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x ** 2 + y ** 2) / 20


def _coerce(result):
    """(x, y, value), with `value` ALWAYS recomputed from the coordinates.

    The 3-tuple path used to return the program's own third element unchecked, so a
    program could report any value it liked for the point it returned. The 2-tuple
    path already recomputed, which is the behaviour both should have had."""
    if isinstance(result, (tuple, list)):
        if len(result) in (2, 3):
            x, y = result[0], result[1]
            return x, y, _objective(x, y)
    return None


def _budget_probe(program):
    """Count calls to the candidate's objective against the evaluator's budget.

    Inlined formulas cannot be observed by this hook. The count and whether the
    hook exists are reported, so zero calls need not imply zero objective work.
    """
    seen = {"n": 0, "observed": False}
    target = "_objective" if hasattr(program, "_objective") else None
    if target is None:
        return seen
    seen["observed"] = True
    original = getattr(program, target)

    def counted(x, y):
        seen["n"] += 1
        if seen["n"] > OBJECTIVE_QUERY_BUDGET:
            raise ValueError(f"objective budget exceeded: {seen['n']} queries, "
                             f"limit {OBJECTIVE_QUERY_BUDGET}")
        return original(x, y)

    setattr(program, target, counted)
    return seen


def _trial_body(program_path, module_name):
    program = _load(program_path, module_name)
    if not hasattr(program, "run_search"):
        raise AttributeError("missing run_search")
    seen = _budget_probe(program)
    result = program.run_search()
    coerced = _coerce(result)
    if coerced is None:
        raise ValueError("invalid result format")
    x, y, value = (_safe_float(v) for v in coerced)
    if any(np.isnan(v) or np.isinf(v) for v in (x, y, value)):
        raise ValueError("nan/inf result")
    if seen["n"] > OBJECTIVE_QUERY_BUDGET:
        raise ValueError(f"objective budget exceeded: {seen['n']} queries, "
                         f"limit {OBJECTIVE_QUERY_BUDGET}")
    return x, y, value, seen["n"], seen["observed"]


def _trial(program_path, module_name):
    # Loading and validating the candidate is part of the bounded trial too.
    return _run_with_timeout(program_path, module_name, 5)


# --- behavior readout -------------------------------------------------------
# The score mixes how low the optimum found is with how close it is. These describe
# the SEARCH instead: how widely the trials land, how many distinct basins they find,
# how concentrated they are. Two searches with the same score can differ on all of
# them -- a lucky single-basin hill climb and a broad multi-start look identical to
# the score and nothing alike here. Keys are the engine's task-agnostic slots.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}
_BASIN_R = 0.5           # two landings within this distance count as the same basin
_DOMAIN_R = 5.0          # the region the reference program searches


def _behavior(points, values):
    """points: [(x, y), ...] landings of the successful trials; values: their f."""
    try:
        p = np.asarray(points, dtype=float)
        v = np.asarray(values, dtype=float)
        if p.ndim != 2 or p.shape[0] == 0 or not np.isfinite(p).all():
            return dict(_BEHAVIOR_ZERO)
        n = p.shape[0]

        # spread of the landings, normalised by the search region
        dispersion = float(np.clip(p.std(axis=0).mean() / _DOMAIN_R, 0.0, 1.0))

        outside = float((np.abs(p) > _DOMAIN_R).any(axis=1).mean())

        # greedy basin count: how many distinct attractors the trials reached
        centres = []
        labels = []
        for q in p:
            hit = next((i for i, c in enumerate(centres)
                        if np.linalg.norm(q - c) < _BASIN_R), None)
            if hit is None:
                centres.append(q)
                hit = len(centres) - 1
            labels.append(hit)
        labels = np.asarray(labels)
        structure = float(len(centres) / n)
        counts = np.bincount(labels)
        concentration = float(counts.max() / n)

        # spread of the objective values reached, squashed to [0,1]
        spread_v = float(v.std()) if v.size and np.isfinite(v).all() else 0.0
        contact = float(spread_v / (1.0 + spread_v))

        # displacement of the landing cloud's centre from its own modal basin:
        # a search that wanders far from where it concentrates scores high here
        offset = float(np.clip(
            np.linalg.norm(p.mean(axis=0) - centres[int(counts.argmax())]) / _DOMAIN_R,
            0.0, 1.0))

        out = {"b_dispersion": dispersion, "b_boundary_frac": outside,
               "b_contact": contact, "b_offset": offset,
               "b_concentration": concentration, "b_structure": structure}
        return {k: (float(val) if np.isfinite(val) else 0.0) for k, val in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def smoke(program_path):
    try:
        x, y, value, queries, observed = _trial(program_path, "program_stage1")
    except TimeoutError:
        return _fail("Timeout")
    except Exception as e:
        return _fail(str(e))
    distance = float(np.sqrt((x - GLOBAL_MIN_X) ** 2 + (y - GLOBAL_MIN_Y) ** 2))
    value_score = float(1.0 / (1.0 + abs(value - GLOBAL_MIN_VALUE)))
    distance_score = float(1.0 / (1.0 + distance))
    mult = _quality_multiplier(distance, close=1.4, region=1.15, far=0.8)
    combined = float((0.6 * value_score + 0.4 * distance_score) * mult)
    cases = {
        "runs_successfully": 1.0,
        "n_trials": 1.0,
        "value_score": value_score,
        "distance_score": distance_score,
        "combined_score": combined,
        "objective_queries": float(queries),
        "objective_counted_trials": float(observed),
        "objective_query_budget": float(OBJECTIVE_QUERY_BUDGET),
    }
    cases.update(_behavior([(x, y)], [value]))
    return {
        "ok": True,
        "score": combined,
        "cases": cases,
        "artifacts": {"distance_to_global": f"{distance:.4f}"},
    }


def full(program_path, num_trials=10):
    points, values, distances = [], [], []
    success = 0
    queries_total, counted_trials = 0, 0
    for trial in range(num_trials):
        try:
            x, y, value, queries, observed = _trial(program_path, f"program_trial_{trial}")
        except Exception:
            continue
        distance = float(np.sqrt((x - GLOBAL_MIN_X) ** 2 + (y - GLOBAL_MIN_Y) ** 2))
        points.append((x, y))
        values.append(value)
        distances.append(distance)
        success += 1
        queries_total += queries
        counted_trials += observed
    if success == 0:
        failed = _fail("All trials failed")
        failed["cases"]["n_trials"] = float(num_trials)
        return failed
    avg_value = float(np.mean(values))
    avg_distance = float(np.mean(distances))
    value_score = float(1.0 / (1.0 + abs(avg_value - GLOBAL_MIN_VALUE)))
    distance_score = float(1.0 / (1.0 + avg_distance))
    reliability_score = float(success / num_trials)
    mult = _quality_multiplier(avg_distance, close=1.5, region=1.2, far=0.7)
    base = 0.5 * value_score + 0.3 * distance_score + 0.2 * reliability_score
    combined = float(base * mult)
    cases = {
        "value_score": value_score,
        "distance_score": distance_score,
        "reliability_score": reliability_score,
        "combined_score": combined,
        # how many trials actually ran. The engine treats a confirm as re-sampled
        # only when the evaluator says so, instead of inferring it from a signature.
        "n_trials": float(num_trials),
        # Counts cover successful trials with an observable objective hook;
        # crashed/timed-out trials may have performed additional unknown work.
        "objective_queries": float(queries_total),
        "objective_counted_trials": float(counted_trials),
        "objective_query_budget": float(OBJECTIVE_QUERY_BUDGET),
    }
    cases.update(_behavior(points, values))
    return {
        "ok": True,
        "score": combined,
        "cases": cases,
        "artifacts": {"average_distance_to_global": f"{avg_distance:.4f}"},
    }


def confirm(program_path, n=20):
    result = full(program_path, num_trials=max(1, int(n or 20)))
    if result["ok"] and result["cases"]["reliability_score"] < 1.0:
        return {**result, "ok": False, "score": 0.0,
                "artifacts": {**result["artifacts"],
                              "error": "one or more confirmation trials failed"}}
    return result


def _quality_multiplier(distance, close, region, far):
    if distance < 0.5:
        return close
    if distance < 1.5:
        return region
    if distance < 3.0:
        return 1.0
    return far


def _fail(msg):
    return {
        "ok": False,
        "score": 0.0,
        "cases": {"combined_score": 0.0, "value_score": 0.0,
                  "distance_score": 0.0, "reliability_score": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }


if __name__ == "__main__":
    if len(sys.argv) != 7 or sys.argv[1] != "--_trial":
        raise SystemExit("This evaluator's private CLI requires --_trial and a program path")
    _invoke_trial(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]))
