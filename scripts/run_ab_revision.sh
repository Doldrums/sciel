#!/usr/bin/env bash
# Compare two engine revisions with isolated imports and the same task definitions.
# Usage: run_ab_revision.sh [--dry-run] BASE_REF [CONFIG] [SEEDS...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_ab_revision.py" "$@"
