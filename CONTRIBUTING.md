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

Use concise Conventional Commit messages, for example
`fix: retain unknown provider charges` or `refactor: separate tool execution`.
Describe the resulting behavior and relevant verification in pull requests.
