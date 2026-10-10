"""Prompt assembly from templates and research context."""

from __future__ import annotations

import ast
import os

from .sources import source_bundle, source_catalog_block, source_reference

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

EDIT_OPS = ("diff", "error_repair")

_FULL_CONTRACT = {
    "diff": "Output the COMPLETE improved program in a single\n```python code block.",
    "error_repair": "Output the COMPLETE fixed program in a single ```python code block.",
}

EDIT_CONTRACT = (
    "Reply with one or more SEARCH/REPLACE edits to the current program, each exactly:\n"
    "<<<<<<< SEARCH\n"
    "lines copied verbatim from the current program\n"
    "=======\n"
    "replacement lines\n"
    ">>>>>>> REPLACE\n"
    "Each SEARCH must match the current program exactly and occur once. "
    "Prefer several small edits that copy only the lines you change. "
    "Do not output the whole program.")

_FIX_CONTEXT = 8

_BET_BLOCK = (
    "\nAfter the code block, state a wager on one line exactly as:\n"
    'BET: {mechanism: "<what and why this may cause a jump>", q: <0..1>}\n'
    "where q is your probability the child beats the island best."
)


def _load_template(op: str) -> str:
    path = os.path.join(_TEMPLATE_DIR, f"{op}.txt")
    if not os.path.exists(path):
        path = os.path.join(_TEMPLATE_DIR, "full_rewrite.txt")
    with open(path) as f:
        return f.read()


def _format_inspirations(inspirations: list) -> str:
    if not inspirations:
        return "(none)"
    parts = []
    for i, insp in enumerate(inspirations):
        card = f" -- strategy: {insp['card']}" if insp.get("card") else ""
        parts.append(f"# inspiration {i + 1} (score {insp['score']:.4f}){card}\n"
                     f"{insp['code']}")
    return "\n\n".join(parts)


def _code_outline(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    doc = (ast.get_docstring(tree) or "").strip().split("\n")[0][:160]
    imports = sorted({a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
                     | {(n.module or "").split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)} - {""})
    defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    parts = [doc] if doc else []
    if imports:
        parts.append("uses " + ", ".join(imports))
    if defs:
        parts.append("defines " + ", ".join(defs[:12]))
    return "; ".join(parts)


def _summarize_inspirations(inspirations: list) -> str:
    if not inspirations:
        return "(none)"
    lines = []
    for i, insp in enumerate(inspirations):
        card = (insp.get("card") or "").strip()
        outline = _code_outline(insp["code"])
        lines.append(f"- inspiration {i + 1} (score {insp['score']:.4f}): "
                     + (" | ".join(part for part in (card, outline) if part) or "no description"))
    return "\n".join(lines)


def _time_share(frac: float) -> str:
    if frac < 0.25:
        return "under a quarter"
    if frac < 0.5:
        return "under half"
    if frac < 0.8:
        return "most"
    return "nearly all"


def compose_directive(effective_timeout: float, parent_timing: float | None,
                      plateau: bool, parent_noop: bool = False,
                      refine: bool = False, *, population_stalled: bool = False,
                      parent_timing_scope: str = 'candidate_run') -> str:
    """Engine-side directive (espalier-2): the compute-budget signal, plus a
    structural-novelty push when the island is in a plateau hold.

    `effective_timeout` must be the BINDING wall-clock cap -- the smoke stage of
    the cascade runs the same program under `smoke_timeout`, so quoting
    `full_timeout` here overstated the budget 4x and drove ~75% of evals into
    timeouts (trace audit 2026-08-14).

    The parent's measured wall-time enters only as a coarse share of the limit
    (quantized buckets), never as a raw number -- journals embed prompts, and a
    raw wall-clock reading would break same-seed journal identity."""
    if parent_timing is None:
        used = ""
    elif parent_timing_scope == 'evaluation_stage_mean':
        used = (" Recorded parent timing is an evaluation-stage average over reported trials, "
                "including evaluator overhead. It does not measure candidate-only runtime "
                "or a comparable fraction of this per-run limit.")
    elif parent_timing_scope != 'candidate_run':
        used = " Parent timing scope is unavailable; per-run utilization is unknown."
    else:
        share = _time_share(parent_timing / max(effective_timeout, 1e-9))
        used = f" The recorded parent run used {share} of that budget."
    parts = [
        f"COMPUTE BUDGET: the evaluator KILLS the program after {effective_timeout:.0f}s "
        f"wall-clock.{used} Keep an EXPLICIT deadline guard: track elapsed time and "
        "return the best result found with a safety margin before the limit. "
        "Spare time does not establish a bottleneck or automatically increase fixed "
        "restart or step counts. If claiming that a speedup enables more search, "
        "identify the additional work and the loop or budget condition that permits it. "
        "Keep that claim conditional until the speedup and additional work are measured."]
    if parent_noop:
        parts.append(
            "SCORE OBSERVATION: the previous child's primary score differed from this "
            "parent's recorded score by less than 1e-9. A tied score does not establish "
            "equivalent behavior. Consider a different mechanism or search scope and "
            "state what could affect the measured objective.")
    if refine:
        parts.append(
            "REFINEMENT: for this proposal, KEEP this construction and investigate "
            "improvements within its family. This is a search instruction; the cause "
            "of stalled progress has not been established. Where applicable, consider "
            "a smooth surrogate for a nonsmooth min/max objective (soft-min/soft-max "
            "via log-sum-exp), or an auxiliary variable t and an epigraph formulation "
            "with smooth constraints. Other possibilities are a change of variables "
            "that removes constraints or a final high-precision polish pass (tighter "
            "tolerance, more iterations, an exact local solve). Treat these as "
            "hypotheses to test. Do NOT switch to a different construction family.")
    elif plateau:
        observation = (
            "the population best has not improved within the configured evaluation window. "
            "This is a population-wide signal"
            if population_stalled else
            "the island's progress detector is active. This is an island-level signal")
        parts.append(
            f"STAGNATION: {observation}; it does not establish stagnation in this "
            "particular lineage. For this proposal, do not micro-tune constants. "
            "Make a STRUCTURAL change: a different "
            "construction/initialization family, a different optimizer class "
            "(joint or global optimization -- basin hopping, simulated annealing, "
            "multi-start), or a different move/neighborhood design. A first try "
            "in a new basin may score worse; that is acceptable.")
    return "\n".join(parts)


def source_bundle_for_prompt(op: str, snapshot):
    """The same source roles as the resolved template, for trace diagnostics."""
    return source_bundle(snapshot, include_inspirations='{inspirations}' in _load_template(op))


def build_fix(code: str, error: str, lineno: int | None) -> str:
    """Short syntax-repair turn: the error and the lines around it, not the whole task."""
    lines = code.split("\n")
    center = min(max((lineno or 1) - 1, 0), max(len(lines) - 1, 0))
    first, last = max(center - _FIX_CONTEXT, 0), min(center + _FIX_CONTEXT + 1, len(lines))
    return _load_template("fix").format(
        error=error, first=first + 1, last=last, snippet="\n".join(lines[first:last]),
        output_contract=EDIT_CONTRACT)


def build(op: str, snapshot, *, edits: bool = False) -> str:
    template = _load_template(op)
    lessons = "\n".join(f"- {ls}" for ls in snapshot.kb_lessons) or "(none)"
    artifacts = "\n".join(f"{k}: {v}" for k, v in snapshot.artifacts.items()) or "(none)"
    research = getattr(snapshot, 'research_context', None) is not None
    bundle = (source_bundle(snapshot, include_inspirations='{inspirations}' in template)
              if research else None)
    parent_code = snapshot.parent_code
    inspirations = snapshot.inspirations
    if bundle is not None:
        # Change only the trusted template BEFORE inserting arbitrary task/code/
        # artifact text. A reference is prose, not an executable Python fence.
        template = template.replace('```python\n{parent_code}\n```', '{parent_code}')
        parent_code = source_reference(bundle['target_ref'])
        inspirations = [{**row, 'code': source_reference(row['source_ref'])}
                        for row in bundle['inspirations']]
    prompt = template.format(
        task_prompt=snapshot.task_prompt,
        parent_code=parent_code,
        parent_score=f"{snapshot.parent_score:.4f}",
        inspirations=(_summarize_inspirations(inspirations) if bundle is None and op != "cross"
                      else _format_inspirations(inspirations)),
        directive=snapshot.directive or "(none)",
        lessons=lessons,
        artifacts=artifacts,
        bet_block=_BET_BLOCK if snapshot.bet_enabled else "",
        output_contract=EDIT_CONTRACT if edits and bundle is None else _FULL_CONTRACT.get(op, ""),
    )
    return prompt + '\n' + source_catalog_block(bundle) if bundle is not None else prompt
