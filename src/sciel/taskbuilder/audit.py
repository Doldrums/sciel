"""Is a drafted task worth searching? The seed must run, repeat measurements set the noise floor,
and a reference score must leave headroom above the seed that the noise cannot explain."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

RUNNER = r"""
import importlib.util, json, sys, time
from pathlib import Path
task, stage = Path(sys.argv[1]), sys.argv[2]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
start = time.monotonic()
if stage == "upper":
    problem = load("problem_upper", task / "problem.py")
    value = problem.upper() if hasattr(problem, "upper") else None
    print(json.dumps({"upper": None if value is None else float(value), "seconds": time.monotonic() - start}))
else:
    result = load("evaluator", task / "evaluator.py").full(str(task / "initial_program.py"))
    result["seconds"] = time.monotonic() - start
    print(json.dumps(result))
"""


def _run(task_dir: Path, stage: str, timeout: float) -> dict:
    try:
        done = subprocess.run([sys.executable, "-c", RUNNER, str(task_dir), stage], capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "artifacts": {"error": f"timed out after {timeout:g} s"}}
    if done.returncode:
        return {"ok": False, "artifacts": {"error": (done.stderr or done.stdout)[-3000:]}}
    try:
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"ok": False, "artifacts": {"error": "no result printed:\n" + done.stdout[-2000:]}}


def audit(task_dir, *, timeout=300.0, repeats=3) -> dict:
    """Checks as (mark, name, detail) rows, a verdict, and the first error if the seed fails."""
    task_dir = Path(task_dir)
    rows = []
    first = _run(task_dir, "full", timeout)
    if not first.get("ok"):
        error = first.get("artifacts", {}).get("error", "unknown failure")
        rows.append(("✗", "seed runs", error.strip().splitlines()[-1][:120] if error.strip() else "failed"))
        return {"rows": rows, "verdict": "seed fails", "admitted": False, "error": error}
    seed = first["score"]
    if seed <= 0:
        error = (f"score(solve()) is {seed:.6g}; valid results must score above 0, because a crashing "
                 "candidate counts as 0. Rescale score(), for an error E use 1 / (1 + E).")
        rows.append(("✗", "seed scores above 0", f"score {seed:.6g}"))
        return {"rows": rows, "verdict": "score must be positive", "admitted": False, "error": error}
    rows.append(("✓", "seed runs", f"score {seed:.6g} in {first['seconds']:.1f} s"))
    repeats_run = [_run(task_dir, "full", timeout) for _ in range(repeats - 1)]
    broken = next((r for r in repeats_run if not r.get("ok")), None)
    if broken is not None:
        error = broken.get("artifacts", {}).get("error", "a repeat run failed")
        rows.append(("✗", "repeatable", "a repeat run failed: " + error.strip().splitlines()[-1][:100]))
        return {"rows": rows, "verdict": "unstable seed", "admitted": False, "error": error}
    scores = [seed] + [r["score"] for r in repeats_run]
    spread = statistics.pstdev(scores)
    eps = 2 * spread
    rows.append(("✓", "deterministic", f"{repeats} of {repeats} runs equal") if spread == 0 else
                ("!", "stochastic", f"scores vary by {spread:.3g} (sd over {repeats} runs)"))
    rows.append(("✓", "noise floor", f"ε = {eps:.3g}"))
    reference = _run(task_dir, "upper", timeout)
    upper = reference.get("upper")
    if upper is None:
        note = reference.get("artifacts", {}).get("error")
        rows.append(("!", "ceiling", "upper() failed: " + note.strip().splitlines()[-1][:100] if note
                     else "no upper(): headroom unknown"))
        verdict, admitted = "undetermined: no reference score", True
    else:
        headroom = upper - seed
        rows.append(("✓" if headroom > 0 else "✗", "ceiling", f"reference {upper:.6g}, headroom {headroom:.3g}"))
        if headroom > max(3 * eps, 1e-12):
            ratio = f" ({headroom / eps:.0f}× the noise floor)" if eps > 0 else ""
            verdict, admitted = f"admitted: headroom {headroom:.3g}{ratio}", True
        else:
            verdict, admitted = "no room: the seed already reaches the reference within noise", False
    rows.append(("!", "isolation", "candidates run in the scorer's process and could read its files"))
    return {"rows": rows, "verdict": verdict, "admitted": admitted, "error": None, "seed": seed, "upper": upper,
            "eps": eps, "stochastic": spread > 0}
