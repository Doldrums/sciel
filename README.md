# SCiEL

SCiEL is an evolutionary program-search engine. It generates candidate programs,
evaluates them against a task contract, and uses recorded hypotheses, measurements,
and lessons to guide subsequent proposals.

The dashboard lives in the separate Arena repository. This package writes run
journals, checkpoints, model-call records, and artifacts for external consumers.

## Installation

Python 3.11–3.14 is supported on Linux and macOS.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[tasks,dev]'
```

Optional dependencies:

- `learning`: PyTorch models and offline training.
- `analysis`: dataframe and process-metric utilities.
- `board`: the optional Weights & Biases sink.

Learned storage profiles require `.[learning]` and external weights (default:
`../sciel-artifacts/`). Set `embed_grid_path` plus `embed_axis_path`, or
`embed_niche_path`, in a local YAML overlay extending the desired profile. Missing
artifacts stop startup; `embed_axis_path: null` explicitly selects legacy projection.
Portable defaults and the mock smoke profile do not enable learned storage.

Task bundles are separate from the installed wheel. A source checkout includes
them under `tasks/`; pass an explicit task root when using an installed release.

```sh
.venv/bin/sciel tasks list --tasks-root ./tasks
.venv/bin/sciel run configs/smoke_mock.yaml --tasks-root ./tasks \
  --outputs ../sciel-runs --run-id smoke
```

The smoke profile uses a mock model. Live profiles require provider credentials
in the environment variable named by `llm.api_key_env`. Keep tokens and local
configuration outside Git.

## Search and research

The engine coordinates two related histories:

- **Programs:** parent selection, mutation or crossover, repair, novelty checks,
  evaluation, and bounded population storage.
- **Research methods:** questions, tool revisions, bounded execution, saved
  evidence, independent replay, adoption, and subsequent use.

Research cells persist independently of individual programs. Their results enter
later generation context; saved candidate outputs still require normal evaluator
validation. A released or adopted tool is not evidence that it improves search.

Population storage, selection policies, parallel workers, and research tools are
configured separately. Configuration validation rejects unsupported controller
combinations. See the maintained profiles in `configs/` and defaults in
`src/sciel/config.py`.

Each run owns its directory. Existing run data is never silently overwritten.
Checkpoints bind execution and task identities; use the matching engine version
to resume a stopped run.

```sh
.venv/bin/sciel stop ../sciel-runs/example
.venv/bin/sciel resume ../sciel-runs/example --tasks-root ./tasks
```

Provider requests have durable records. An unknown request outcome is retained
as unknown and is not automatically replayed. Accounting distinguishes known
charges from completed calls whose monetary cost is unavailable.

## Execution boundaries

Native evaluator subprocesses apply resource limits but do not isolate filesystem
or network access. Container execution requires Docker and an explicit configured
image. Task and research-tool limits remain bounded even in runs without a total
iteration limit.

## Code layout

| Path | Responsibility |
| --- | --- |
| `src/sciel/engine/` | Scheduling, population, generation, checkpoint and replay |
| `src/sciel/eval/` | Task loading, evaluation, result certification |
| `src/sciel/hypothesis/` | Proposal and conclusion protocols, evidence validation |
| `src/sciel/knowledge_base/` | Experiment records, retrieval, bounded prompt views |
| `src/sciel/research_tools/` | Research cells, tool execution and result delivery |
| `src/sciel/methods/` | Immutable tool registry, resource budgets and method exchange |
| `src/sciel/components/` | Selection, storage, learned policies and operators |
| `tasks/` | Task contracts, evaluators and initial programs |
| `scripts/` | Operational commands and offline training |
| `tests/` | Unit, integration, identity and resume contracts |

Development and verification instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).
