# Task formats SCiEL accepts

Read the current SCiEL code (a source checkout or the installed package) before relying on details here.

## 1. One file: problem.py (local task)

`src/sciel/local_task.py` `build(problem, tasks_root)` turns one file into `tasks/local/<stem>_<hash>/`:

- `solve()` — no arguments, returns a result. The seed; the search rewrites it.
- `score(result) -> float` — higher is better, every valid result > 0 (a crash counts as 0; for an error E use
  1 / (1 + E)). Deterministic, frozen: the evaluator loads its own copy, so a candidate editing score changes nothing.
- `metrics(result) -> dict` — optional, 2–4 numbers on HOW a result behaves (behaviour axes for maps), not quality.
- `upper() -> float` — optional reference/oracle; stripped from the seed, may use data solve() must not see.
- The module docstring becomes the start of `task_prompt`.
- Defaults: smoke 30 s, full 120 s, `stochastic` false (set true if repeats differ; confirm then keeps the worst of 3).

`sciel task FOLDER --note "..."` does read → draft (LLM) → audit → search interactively; its built-in audit
(`src/sciel/taskbuilder/audit.py`) is weaker than the admission rule (ε = 2·sd of 3 seed runs, headroom > 3ε, no
π_ε, no leak probes). Use the skill's audit instead.

## 2. Full task directory (suite task)

`tasks/<suite>/<name>/` with `task.yaml`, `evaluator.py`, `initial_program.py`, optional `assets/` (modules
candidates import via `sciel.task.asset('name')`). Copy the closest existing task as a template:

- exact math constructions: `tasks/math/third_autocorrelation_absolute`, `tasks/math/erdos_minimum_overlap`
- geometry: `tasks/oe/circle_packing`, `tasks/oe/heilbronn_triangle`
- data / statistical: no public example yet; follow the data rules at the end of this file

`evaluator.py` defines `smoke(path)`, `full(path)`, `confirm(path, n)`, each returning
`{"ok", "score", "cases": {...}, "artifacts": {...}}`. task.yaml keys to set deliberately: `task_prompt` (the
whole contract the LLM sees: interface, objective, scoring exactness, tolerances, wall limits, available modules),
`primary_metric`, `case_metrics` (include `b_*` behaviour metrics), `tiers`, `stochastic`, `confirm_n`,
`*_timeout`, `paradigms` (2–3 genuinely different approach families), `license`/provenance.

Validate: `sciel tasks validate <suite>/<name>` and `sciel tasks hash <suite>/<name>`.

## Things that broke before (check each)

- Tolerance exploits: circle_s2 2.635988 passed by using the 1e-6 tolerance; strict LP gave 2.635977. Score exactly
  or repair, never accept within tolerance.
- task_prompt mentioned a full-stage limit while smoke timed out at 30 s → ~20% eval failures. Every wall limit in
  the prompt must match task.yaml, and programs can read their stage's wall seconds.
- Children misread the output structure (dict vs array) and fell back to the seed: the prompt must state the
  output structure exactly.
- A seed stuck at a flat start (third autocorrelation): the seed must leave symmetric/stationary points.
- Data tasks: solve() may fit on train only; held-out targets never readable (research tools see development data read-only at /data
  via `research_tools.data_env`). Leak probes are mandatory.
