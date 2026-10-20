# What tasks SCiEL solves (condensed)

Source: the CPIL definition summarized in the SCiEL README, section "What tasks SCiEL is for".

## Tuple ⟨X, F, h, h*, M, B⟩

max_{x ∈ F} h(x),  F = { x ∈ X : c_j(x) ≤ 0, t(x) ≤ T, inputs(x) ⊆ I }

- **X**: what the LLM writes and edits. The solution itself or a program that generates it (the engine uses a
  Python module). If generation is random, h is taken over its distribution.
- **F**: constraints on the solution, wall time, legitimate inputs.
- **h**: objective the search sees; **h\***: true objective. Well posed if h*(argmax h) ≥ max h* − γ, γ small.
- **M(x′ | x, ctx)**: the LLM edit operator; depends on prompt, history, findings, lessons. One operator, one
  landscape (Jones 1995), so the landscape moves during a run. Cultural transmission acts through M.
- **B**: evaluation budget.

## Five axes

| Axis | What it is | Pre-run measure |
|---|---|---|
| A. Solution space | combinatorial; continuous with constraints; algebraic; functional over data; statements/hypotheses; images; algorithm over an instance distribution; bilevel; multi-objective | from the formulation |
| B. Feasible set | explicit constraints; optimum on the boundary or inside; penalty / repair / feasibility rules | share of feasible proposals from M; exact check with no exploitable tolerance |
| C. Determinism and noise | h deterministic or sampled; σ_d = SE of the paired difference candidate − reference | re-measurements (σ_e), cluster bootstrap (σ_d) |
| D. Landscape relative to M | π_ε(x \| ctx) = P_{x′∼M}[h(x′) > h(x) + ε]; parent–child ρ(1); plateau if π_ε ≲ 1/B | share of improving edits at the frontier, with its upper confidence bound |
| E. Verifiability | h = h* (exact check) or a proxy | leak probes no higher than seed; for mechanism tasks, a reference mechanism beats a strong table on a held-out slice |

## Bounds

- Adaptive selection: select scores feed the next proposals, so a union bound over selected candidates is invalid
  (Dwork et al. 2015; Blum–Hardt 2015). In-run records use an anytime-valid e-process with e-LOND/e-BH
  (`causal/sequential.py`, `stats.PlateauMonitor`).
- Pre-run radius (conservative): **ε = σ_d · z_{1−δ/(2B)} + σ_e**, correcting for all select evaluations in the run.
  Hoeffding form for metrics bounded per cluster; bootstrap form for correlations and geometric means.
- Final check: one measurement on a never-seen slice.
- Lower bound: best reference, re-measured with the same evaluator. Upper bound: known optimum, proven bound or
  oracle; if none, it is not declared.
- Pace: on a plateau the expected evaluations to escape ≈ 1/π_ε (Levin 1973; Orseau et al. 2018).
- Attainability: for deterministic h = h* and for statistical tasks over data (CSQ/SQ, Feldman 2008, 2009) the
  working condition is the same: π_ε ≥ 1/B. Cells with raw data give sample access; the bottleneck is what of
  the findings passes through h.

## Admission rule

A task is admitted if:

1. axis E passes;
2. upper − lower > 4ε at the planned B;
3. the upper confidence bound of π_ε at the frontier is ≥ 5/B.

Then the ladder: a short run, then a full one.

For solving a task: M is what we can change from the task side (prompt, seed, importable helpers, paradigms,
data access). Raising π_ε at the frontier by changing M is how a stalled task is unblocked.

## Worked example: research-9 (order-book target)

| Axis | Value |
|---|---|
| A | functional over data: distribution over (t0, t1) given the order-book sequence |
| B | legitimate inputs, time limit; 40 accepted, 5 outside F |
| C | statistical; ε ≈ 11% of the h level (rough, select–holdout gap) |
| D | 12 of 37 edits above noise, π_ε ≈ 0.32; frontier 0 of 10, upper bound π_ε ≤ 0.31 |
| E | proxy: the table beats the formula on select and holdout, so the metric cannot distinguish mechanism |

Lesson: a holdout with the same metric tests noise, not meaning; distinguishing mechanism needs a held-out regime.
