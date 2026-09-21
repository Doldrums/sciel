import os
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

TASKS_ROOT = os.path.join(ROOT, "tasks")
PYTHON = sys.executable

# Archived test sources are data for explicit frozen-runtime harnesses. Do not
# import their conftest modules into the live collection (even when their own
# tests are ignored): bare `import conftest` must resolve to this live helper.
collect_ignore = ['fixtures']


@pytest.fixture(autouse=True)
def isolate_candidate_import_namespace():
    # In-process evaluator tests also retain candidate modules for pickle/spawn.
    # Isolate their temporary paths from later source-provenance tests without
    # changing the production loader or importing SCiEL into unrelated tests.
    package = 'sciel.candidate_modules'
    prefix = package + '.'
    modules_before = set(sys.modules)
    namespace = sys.modules.get(package)
    attributes_before = set(vars(namespace)) if namespace is not None else set()
    try:
        yield
    finally:
        for name in set(sys.modules) - modules_before:
            if name.startswith(prefix):
                del sys.modules[name]
        namespace = sys.modules.get(package)
        if namespace is not None:
            for name in set(vars(namespace)) - attributes_before:
                child = getattr(namespace, name)
                if isinstance(child, types.ModuleType) and child.__name__.startswith(prefix):
                    delattr(namespace, name)


def mock_config(budget=30, seed=42, policy="sh", task="test/toy_quadratic"):
    return {
        "task": task,
        "num_islands": 3, "budget": budget, "pop_cap": 50, "archive_size": 20,
        "grid_bins": 10, "inflight_cap": 4, "dup_cap": 4, "backstop_k": 10,
        "random_seed": seed, "schedule_policy": policy,
        "milestone_fracs": [0.5, 0.75], "op_temperature": 0.5, "op_floor": 0.05,
        "op_window": 50, "selection_floor": 0.001, "tau_floor": 0.001,
        "reward_w_pop": 1.0, "reward_w_isl": 1.0, "base_temp": 0.7,
        "bet_enabled": False, "confirm_records": False,
        "checkpoint_on_milestone": True,
        "llm": {"mock": True, "system_message": "minimize"},
    }


VOLATILE = {"timing", "path", "timestamp", "segment_elapsed_seconds"}
PROVENANCE_IDS = {"run_id", "segment_id", "checkpoint_id",
                  "checkpoint_sha256", "prefix_sha256", "evaluation_id"}


def strip_volatile(line, provenance_ids=None):
    """Compare search semantics while retaining provenance links and counters.

    UUIDs and byte hashes differ between equivalent runs; canonical IDs preserve
    their equality relationships. Physical byte offsets also depend on stripped
    timing/path fields. Costs (including unknown costs), metrics and nested
    metadata remain evidence. Exact provenance is tested against raw journals
    elsewhere.
    """
    import json
    if provenance_ids is None:
        provenance_ids = {}

    def identity(key, item):
        known = provenance_ids.setdefault(key, {})
        return known.setdefault(item, f"<{key}:{len(known) + 1}>")

    def provenance(value, *, anchor=False):
        if isinstance(value, list):
            return [provenance(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key in PROVENANCE_IDS and isinstance(item, str) and item:
                result[key] = identity(key, item)
            elif key == "evaluation_ids" and isinstance(item, list):
                # Worker metadata refers to the same scalar evaluation IDs
                # emitted by evaluation events. Keep list order/multiplicity,
                # unknown references and malformed entries visible.
                result[key] = [identity("evaluation_id", entry)
                               if isinstance(entry, str) and entry else provenance(entry)
                               for entry in item]
            elif key == "byte_offset" and anchor and type(item) is int:
                result[key] = "<physical journal offset>"
            elif key in ("provenance", "journal_anchor"):
                result[key] = provenance(item, anchor=key == "journal_anchor")
            else:
                result[key] = provenance(item)
        return result

    e = json.loads(line)
    return provenance({k: v for k, v in e.items() if k not in VOLATILE})


def read_journal(run_dir):
    provenance_ids = {}
    with open(os.path.join(run_dir, "sciel_events.jsonl")) as f:
        return [strip_volatile(ln, provenance_ids) for ln in f]
