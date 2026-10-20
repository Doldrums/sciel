---
name: sciel-task
description: Solve a user's problem well with SCiEL. Set it up as a task SCiEL can make real progress on, check it against SCiEL's task definition (axes A–E, noise radius ε, headroom > 4ε, frontier improvement probability π_ε ≥ 5/B) to find what would stop the search, fix that, run, and deliver a verified best solution. Use when the user brings a problem, paper, dataset, folder of materials or an existing task and wants it solved or pushed with SCiEL, or asks whether SCiEL can solve it, e.g. "solve this with SCiEL", "turn this into a SCiEL task", "is this a good task for SCiEL", "audit this task", "why does SCiEL not move on this task".
---

# Solve a problem with SCiEL

The goal is the best verified solution to the user's problem. SCiEL is the tool; the task we hand it decides
whether it gets there. Work in this loop: **formulate → build → diagnose → fix → run → verify and deliver**.

The definition in `references/definition.md` is the diagnostic: each axis that fails is a concrete reason SCiEL
would optimise the wrong thing, fail to see its own progress, or stall. Read it first. Read
`references/task-format.md` before writing task files. Read the current SCiEL code (a source checkout, or the
installed `sciel` package) rather than trusting names here blindly.

## Language and style

Talk with the user in their language, concisely, idea first, each problem stated plainly. Write task files,
prompts and code in English, in the style of the SCiEL repository. Never reuse old runs or old audit outputs for a
new version of a task.

## 1. Formulate

Pin down what "solved well" means for the user: the objective h* they actually care about, the best known result
(literature record, baseline, oracle) and its source, and what would count as a win. Then write the tuple in one
line each: X (what the LLM edits), F (constraints, wall time T, legitimate inputs), h and why it tracks h*, M (what
context and tools help: data access, importable helpers, findings), B. Ask the user only for choices that change
the answer (objective variant, data split, budget, compute).

## 2. Build for the search, not just for scoring

Pick the form (`references/task-format.md`): `problem.py` for a quick local task, a suite directory for a serious
run. Beyond a correct evaluator, give the LLM leverage:
- **an exact objective it can import**: put the scorer, a fast bound, smooth surrogates with gradients, active-set
  or diagnostic helpers, refine/resample and repair functions in `assets/`, so candidates build their search on
  them instead of rewriting them (see `tasks/math/third_autocorrelation_absolute`);
- **a seed that searches**: it continues the best construction in its population view (`sciel.population`),
  leaves symmetric or stationary points, uses its stage's wall time (`SCIEL_STAGE_SECONDS`), and seeds its
  randomness;
- **a prompt that leaves nothing to guess**: output structure, exactness and tolerances, every wall limit equal to
  `task.yaml`, available modules, the record to beat;
- **2–3 real `paradigms`** and `b_*` behaviour metrics, so islands spread over different approaches;
- **stage times** that let a real search finish (full ≫ smoke; confirm keeps the worst of its repeats when the task
  is stochastic).

## 3. Diagnose with the definition

Write audit programs in a directory of your choice, for example `./sciel-audit/<task>/programs/`, with the role in
the file name: `expert*` (best known references, reproduced: the bar to beat), `cheat*` (leak and exploit probes:
tolerance, held-out reads, editing the scorer, degenerate outputs), `strawman*` (mechanism-free lookups, for data
tasks). Then run, with the Python environment that has `sciel` installed:

```
python <this skill>/scripts/audit_task.py TASK_DIR --programs PROGRAMS --budget B \
  [--upper U] [--repeats 5] [--run RUN_DIR] --json audit.json
```

Set `SCIEL_REPO=<checkout>` to audit against a source checkout instead of the installed package. `--run` takes a
short ladder run (B ≈ 40–60) to measure how often SCiEL's edits improve the frontier.

Read each failure as a fix, not a verdict:

| Signal | What SCiEL would do | Fix |
|---|---|---|
| E: cheat above seed + ε, or strawman ≈ expert | win by exploit; the "best" result is not a solution | exact check or repair, hide held-out data, a held-out regime instead of held-out rows |
| upper − lower ≤ 4ε | cannot tell real gains from noise; records are luck | more cases, paired evaluation against the reference, deterministic seeds; or the problem is already at its bound, so say so |
| no upper bound | fine to run, with no stopping point | state the record to beat instead |
| low feasible share (B) | most edits wasted | state the structure in the prompt, give repair and clean helpers |
| frontier π_ε upper < 5/B | stalls near the seed | importable objective and gradients, a better seed, refine-to-finer-grid helpers, a longer full stage, different paradigms |
| π_ε high at the frontier | will move fast | run with the full budget |

Fix, rebuild, re-audit; usually one or two rounds.

## 4. Run

Choose where the model runs with the user: a local OpenAI-compatible server (for example vLLM, `llm.api_base:
http://127.0.0.1:8000/v1`), a hosted API (Anthropic, OpenRouter), or a subscription CLI (`provider: claude_code`,
`provider: codex_responses`). Before launching on a shared machine, check what is already running (processes, GPU
load) and ask the user.

Run full SCiEL: research cells on (`research_tools.enabled: true` with an immutable Docker image), everything else
at its defaults, a fresh run id, and several seeds when the result matters. Switch components off only for an
ablation the user asked for, and say so. Never kill processes by a pattern that also matches your own command
line.

While it runs, read the trace (events, curves, failures, research cells), not just counters, and fix the task if
the trace shows a systematic problem: the same failure repeating, a misread output, timeouts. With a subscription
provider, a usage limit pauses the run until the reset; `scripts/continue_live_run.py` resumes it.

## 5. Verify and deliver

The deliverable is the solution, checked independently of the search:
- re-score the best program exactly (no tolerance), on a never-seen slice for data tasks, repeated if stochastic;
- compare with the best known result and its source, and say plainly whether it is a record, a match, or below;
- give the user the solution itself (construction, program, parameters) and where it is saved, plus one line on
  what limited it and the next lever if they want more.

Save the audit outputs and a short `report.md` next to the audit programs.
