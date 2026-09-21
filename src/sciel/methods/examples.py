"""Authored common toolbox packages, not automatically discovered methods.

Builders read and package trusted source; they never execute a candidate or
optimizer on the host. Every numerical helper is included byte-for-byte.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .models import content_hash, detached, validate_manifest
from .tasks import Circle26Task, circle_geometry_schema

_SCIEL = Path(__file__).resolve().parents[1]

_SUPPORT = '''import copy
import math
import time
import numpy as np

def begin(state):
    original = copy.deepcopy(state)
    started = time.monotonic()
    seconds = state["budget_seconds"]
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid remaining cooperative budget")
    centers = np.asarray(state["centers"], dtype=float)
    radii = np.asarray(state["radii"], dtype=float)
    if centers.shape != (26, 2) or radii.shape != (26,) or not np.isfinite(centers).all() or not np.isfinite(radii).all():
        raise ValueError("invalid Circle geometry")
    return original, np.concatenate((centers.ravel(), radii)), started, started + seconds

def finish(state, params, started, label, stats):
    result = copy.deepcopy(state)
    result.pop("certificate", None)
    result["centers"] = params[:52].reshape(26, 2).tolist()
    result["radii"] = params[52:].tolist()
    result["centers_dtype"] = "float64"
    result["radii_dtype"] = "float64"
    result["budget_seconds"] = max(0.0, state["budget_seconds"] - (time.monotonic() - started))
    # Diagnostics are method output, never an independent invocation receipt.
    result["diagnostics"] = {"method": label, "details": stats,
        "resource_note": "cooperative solver time only; executor receipt charges full invocation"}
    return result
'''


def circle_state_schema():
    schema = circle_geometry_schema(extra=False)
    schema["properties"].update({
        "budget_seconds": {"type": "number", "minimum": 0, "maximum": 3600},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2**32 - 1},
        "restart_index": {"type": "integer", "minimum": 0, "maximum": 1000000},
        "maxiter": {"type": "integer", "minimum": 1, "maximum": 15000},
        "noise": {"type": "array", "minItems": 78, "maxItems": 78,
                  "items": {"type": "number", "minimum": -.06, "maximum": .06}},
        "diagnostics": {"type": "object"}, "certificate": {"type": "object"},
    })
    schema["required"] = ["centers", "radii", "budget_seconds"]
    return schema


def _source(path):
    return Path(path).read_text()


def _module(source, role):
    digest = hashlib.sha256(source.encode()).hexdigest()
    name = f"common_circle_{role}_{digest[:16]}"
    return name, {name + ".py": source}, digest


def _bundle(label, code, files, sources, runtime_lock, resources, author_lab, *, dependencies=(), scope=None):
    files = {**files, "method.py": code}
    manifest = {"name": label, "kind": "tool", "author_lab": author_lab,
        "entrypoint": "method.py:run", "input_schema": circle_state_schema(),
        "output_schema": circle_state_schema(), "runtime_lock": detached(runtime_lock),
        "dependencies": list(dependencies), "parents": [], "resources": detached(resources),
        "protocol": {"provenance": "authored_common_toolbox", "automatically_discovered": False,
                     "sources": sources, "budget": "remaining cooperative solver budget; external limits still authoritative"},
        "scope": {"task": "Circle26", "geometric_contract": "represented finite binary64; tolerance 1e-6",
                  "no_claim": "No measured utility, independent replication or transfer is asserted by packaging.",
                  **(scope or {})},
        "examples": [], "preconditions": {"state": "26 centers, 26 radii; assigned remaining budget"}}
    validate_manifest(manifest, files)
    return {"files": files, "manifest": manifest}


def circle_toolbox(runtime_lock, resources, author_lab="common-toolbox"):
    """Return LP/relocation/SLSQP/greedy-control/certifier unregistered bundles.

The common initial toolbox is available equally to experimental arms. These
packages carry no released status or fabricated observations.
"""
    support, common_files, support_sha = _module(_SUPPORT, "support")
    sources = {"wrapper_policy": "sciel-authored-circle-toolbox/1", "support_sha256": support_sha}
    bundles = {}

    path = _SCIEL / "solvers" / "circle_lp_initializer.py"
    helper, files, sha = _module(_source(path), "lp")
    code = f'''from {support} import begin, finish
from {helper} import initialize_radii, new_lp_stats

def run(state):
    original, params, started, deadline = begin(state)
    stats = new_lp_stats()
    result = initialize_radii(params, deadline=deadline, stats=stats)
    return finish(original, result, started, "fixed_centers_lp", stats)
'''
    bundles["lp"] = _bundle("fixed_centers_lp", code, {**common_files, **files},
        {**sources, "sciel/solvers/circle_lp_initializer.py": sha}, runtime_lock, resources, author_lab,
        scope={"role": "fixed-center radius maximization; bounded HiGHS call, not joint center optimization"})

    path = _SCIEL / "solvers" / "circle_relocation.py"
    helper, files, sha = _module(_source(path), "relocation")
    code = f'''import numpy as np
from {support} import begin, finish
from {helper} import relocate_circle, new_relocation_stats

def run(state):
    original, params, started, deadline = begin(state)
    noise = np.asarray(state["noise"], dtype=float) if "noise" in state else np.random.default_rng(state.get("seed", 0)).uniform(-.06, .06, 78)
    stats = new_relocation_stats()
    result = relocate_circle(params, noise, params, state.get("restart_index", 0), deadline=deadline, stats=stats)
    return finish(original, result, started, "single_circle_relocation", stats)
'''
    bundles["relocation"] = _bundle("single_circle_relocation", code, {**common_files, **files},
        {**sources, "sciel/solvers/circle_relocation.py": sha}, runtime_lock, resources, author_lab,
        scope={"role": "feasible structural proposal every fourth index; may reduce sum; no optimizer"})

    path = _SCIEL / "solvers" / "circle_anytime.py"
    helper, files, sha = _module(_source(path), "slsqp")
    code = f'''import time
import numpy as np
from scipy.optimize import minimize
from {support} import begin, finish
from {helper} import Incumbent, Deadline, objective, objective_jac, packing_constraints, packing_jacobian

def run(state):
    original, params, started, deadline = begin(state)
    incumbent = Incumbent(params)
    stats = {{"completed": False, "deadline_reached": False, "solver": "one supplied-state SLSQP solve"}}
    def check(values=None):
        if time.monotonic() >= deadline:
            if values is not None:
                incumbent.consider(values)
            raise Deadline("cooperative Circle method deadline")
    def timed(function):
        def call(values):
            check(values)
            return function(values)
        return call
    def callback(values):
        incumbent.consider(values)
        check()
    try:
        check()
        result = minimize(timed(objective), params, method="SLSQP", jac=timed(objective_jac),
            bounds=[(None, None)] * 52 + [(0.0, None)] * 26,
            constraints={{"type": "ineq", "fun": timed(packing_constraints), "jac": timed(packing_jacobian)}},
            callback=callback, options={{"maxiter": state.get("maxiter", 200), "ftol": 1e-7, "disp": False}})
        incumbent.consider(result.x)
        stats.update(completed=True, solver_success=bool(result.success))
    except Deadline:
        stats["deadline_reached"] = True
    centers, radii, total = incumbent.result()
    stats["best_sum"] = total
    return finish(original, np.concatenate((centers.ravel(), radii)), started, "supplied_state_slsqp", stats)
'''
    bundles["slsqp"] = _bundle("supplied_state_slsqp", code, {**common_files, **files},
        {**sources, "sciel/solvers/circle_anytime.py": sha}, runtime_lock, resources, author_lab,
        scope={"role": "binary64 joint local refinement using existing analytic Jacobians",
               "wrapper_change": "One solve starts from supplied state; archived solver restart loop is not invoked.",
               "not_high_precision": True})

    task = Circle26Task()
    helper, files, sha = _module(task.canonical_seed_source, "greedy")
    code = f'''import time
import numpy as np
from {support} import begin, finish
from {helper} import compute_max_radii

def run(state):
    original, params, started, deadline = begin(state)
    skipped = time.monotonic() >= deadline
    if not skipped:
        params[52:] = compute_max_radii(params[:52].reshape(26, 2))
    return finish(original, params, started, "fixed_centers_greedy_control", {{"deadline_skip": skipped}})
'''
    bundles["greedy_control"] = _bundle("fixed_centers_greedy_control", code, {**common_files, **files},
        {**sources, "tasks/oe/circle_packing/initial_program.py": sha}, runtime_lock, resources, author_lab,
        scope={"role": "canonical seed's fixed-center pairwise greedy radius rule; assigned LP control"})

    path = _SCIEL / "eval" / "packing_geometry.py"
    helper, files, sha = _module(_source(path), "certificate")
    code = f'''import time
import numpy as np
from {support} import begin, finish
from {helper} import certify_packing

def run(state):
    original, params, started, deadline = begin(state)
    if time.monotonic() >= deadline:
        return finish(original, params, started, "strict_feasibility_certificate", {{"deadline_skip": True}})
    certificate = certify_packing({{"centers": params[:52].reshape(26, 2).tolist(), "radii": params[52:].tolist()}})
    if certificate["status"] != "certified":
        raise ValueError("packing cannot receive strict radius-shrink certificate")
    result = np.concatenate((np.asarray(certificate["centers"]).ravel(), np.asarray(certificate["radii"])))
    output = finish(original, result, started, "strict_feasibility_certificate", {{"deadline_skip": False}})
    output["certificate"] = certificate
    return output
'''
    bundles["certifier"] = _bundle("strict_feasibility_certificate", code, {**common_files, **files},
        {**sources, "sciel/eval/packing_geometry.py": sha}, runtime_lock, resources, author_lab,
        scope={"role": "exact feasibility of represented geometry after conservative radius shrink",
               "not_optimizer": True, "not_high_precision_refinement": True,
               "no_global_optimality_bound": True})
    return bundles


def circle_pipeline(methods, runtime_lock, resources, author_lab="common-toolbox"):
    """Compose registered tools in an explicit order or refuse before execution.

Nested dependency calls are not independently attested receipts. For method-use
credit the coordinator must arrange separate external invocations/comparisons.
"""
    if not isinstance(methods, list) or not 1 <= len(methods) <= 16:
        raise ValueError("pipeline requires 1..16 pinned methods")
    ids = []
    for method in methods:
        if not isinstance(method, dict) or set(method) != {"id", "manifest", "files"}:
            raise ValueError("pipeline needs registered immutable bundles")
        validate_manifest(method["manifest"], method["files"])
        if method["id"] != content_hash({"schema": "sciel-method/1", "manifest": method["manifest"], "files": method["files"]}):
            raise ValueError("pipeline method content hash differs")
        if (method["manifest"]["input_schema"] != circle_state_schema()
                or method["manifest"]["output_schema"] != circle_state_schema()
                or method["manifest"]["runtime_lock"] != runtime_lock):
            raise ValueError("pipeline requires compatible state schemas and exact runtime locks")
        ids.append(method["id"])
    code = f'''from sciel_method_runtime import invoke_dependency

def run(state):
    for method_id in {ids!r}:
        if state["budget_seconds"] <= 0:
            break
        state = invoke_dependency(method_id, state)
    return state
'''
    bundle = _bundle("explicit_circle_pipeline", code, {}, {"ordered_methods": ids}, runtime_lock,
        resources, author_lab, dependencies=list(dict.fromkeys(ids)),
        scope={"role": "explicit composition; compatibility is structural, not proof of utility",
               "nested_invocations": "unattested; separate top-level trials required for method-use credit"})
    bundle["manifest"]["kind"] = "search"
    return bundle
