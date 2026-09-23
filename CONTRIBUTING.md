# Contributing

Use a separate environment and install the dependencies needed by your change:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[tasks,dev,learning,analysis]'
```

## Repository boundaries

- Keep maintained code files at or below 500 physical lines. Split by
  responsibility; do not compress code or use generated execution to bypass the
  limit.
- `README.md` and `CONTRIBUTING.md` are the only Markdown files in this repository.
- Keep experiment outputs, reports, working notes, model weights, credentials and
  build artifacts outside the source tree.
- The Git ignore policy denies files by default. Add an allowlist entry only for
  a maintained source, configuration or test format.
- UI changes belong in Arena. Preserve its engine-facing journals and artifacts
  when changing serialization.

## Implementation

Keep configuration validation, side effects and pure transformations separate.
Preserve public imports when moving code, and update execution identities to bind
all extracted implementation files.

Research claims must refer to recorded measurements. Do not turn missing costs
into zero, retries into new scientific evidence, or repeated measurements into
novel candidates. Resume must preserve both provenance and accounting.

Tests should use authored, bounded fixtures. A unit test must not require an old
experiment directory, an external model request or an unpublished local file.
Use isolated integration tests when subprocess, container or installed-wheel
behavior is part of the contract.

## Verification

Run the focused tests for changed behavior, then the full checkout suite before
merging a refactor:

```sh
.venv/bin/python scripts/check_repository.py
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python -m pytest tests -q
.venv/bin/python -m build
git diff --check
```

CI also installs the wheel in a separate environment and verifies task and resume
contracts on supported Python versions. Docker-specific tests require Docker;
report skips separately from passing checks.

Use concise Conventional Commit messages, for example
`fix: retain unknown provider charges` or `refactor: separate tool execution`.
Describe the resulting behavior and relevant verification in pull requests.

## Local inference and adaptation

The local provider port starts from upstream v0.2 (`espalier-3.64`). Its LoRA
reference used `espalier-3.44` with unrelated Git history. Preserve the new
engine's lifecycle, evaluator backend, resolved-model fields, subscription
accounting and durable call journal when porting individual capabilities.
Existing experiments retain their original frozen sources and controls.

Local requests bind a base identity, sampling seed, chat-template behavior and
signed deployment receipt. A generation snapshot must freeze its specification
before queueing; the worker, transcript and call journal must preserve the same
binding. Replaying a returned durable call must not send another request. A zero
external API charge does not imply zero GPU, elapsed-time or rental cost.

Checkpoint restore must reject a changed local/deployment or oracle identity
even when forced. The port supports fresh 3.64 runs and same-identity restarts;
it does not migrate a 3.44 checkpoint by clearing routes, pending requests or
paid outcomes. Continuous adaptation and configured adapters remain disabled
until their route, publication, owned execution and continuation hooks have
been ported and verified together.

TRAIN export must bind the final delivered prompt, selected memory and tool
cards to native calls and full/confirmation evaluation evidence. The current
tool protocol is `/34`: verify the complete installed renderer and engine source
maps, including extracted modules. Retained source bytes are evidence, never
executable renderers. Keep the finite legacy `/24`–`/26` readers separate. A
changed or missing prompt, card, invocation, owner or source binding must prevent
an unsupported eligibility claim.

Context limits are explicit opt-in controls. Count the complete prompt and its
reserved output before dispatch; omit only whole optional inputs with retained
selection evidence. Keep default upstream rendering unchanged when limits are
absent. Exercise generation, repair, planner, conclusion and reflection paths
separately; a successful ordinary generation does not cover them all.

Use a real native engine and journal in integration fixtures, with any synthetic
model, tokenizer, tool or numerical leaves clearly identified. Verify export
reopening without additional calls and reject tampered evidence. These tests
establish software behavior, not model quality, GPU speed or continuous LoRA
compatibility. Record actual experiment results and validation logs outside the
maintained source tree.
