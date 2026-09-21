"""Unit tests for engine invariants (§8, §9)."""

import random

import pytest

from sciel.components.policy.base import boltzmann, floor_clip, sample
from sciel.config import _merge, validate
from sciel.engine.loop import _changed_region
from sciel.engine.operators import OpBandit
from sciel.engine.schedule import Schedule
from sciel.reward import RewardBus
from sciel.rng import RngHub


class _Ev:
    def __init__(self, r_norm, is_island_event=True):
        self.r_norm = r_norm
        self.is_island_event = is_island_event


# --- reward (I-1, I-8, Lemma 1) ---

def test_reward_only_positive_increments():
    bus = RewardBus()
    ev = bus.improvement_event(0.5, {"pop_best": 0.8, "island_best": 0.9}, "gen", 1)
    assert ev.r_raw == 0.0 and ev.d_pop == 0.0 and ev.d_isl == 0.0
    assert not ev.is_island_event and not ev.is_pop_event


def test_reward_normalizer_predictable_excludes_current():
    bus = RewardBus()
    e1 = bus.improvement_event(1.0, {"pop_best": 0.0, "island_best": 0.0}, "gen", 1)
    assert e1.med == 1.0
    e2 = bus.improvement_event(3.0, {"pop_best": 1.0, "island_best": 1.0}, "gen", 2)
    assert e2.med == 2.0
    assert e2.r_norm == pytest.approx(min(1.0, e2.r_raw / 2.0))


def test_reward_telescopes_to_final_max():
    bus = RewardBus(w_pop=1.0, w_isl=0.0)
    scores = [0.2, 0.5, 0.4, 0.9, 0.7, 1.1]
    running = 0.0
    total = 0.0
    for s in scores:
        ev = bus.improvement_event(s, {"pop_best": running, "island_best": running},
                                   "gen", 1)
        total += ev.d_pop
        running = max(running, s)
    assert total == pytest.approx(max(scores) - 0.0)


def test_reward_state_roundtrip():
    bus = RewardBus()
    for s in (1.0, 2.0, 1.5):
        bus.improvement_event(s, {"pop_best": 0.0, "island_best": 0.0}, "gen", 1)
    d = bus.state()
    bus2 = RewardBus()
    bus2.load_state(d)
    assert bus2.state() == d


# --- floor / boltzmann (I-5) ---

def test_floor_clip_enforces_floor():
    out = floor_clip({"a": 0.999, "b": 0.001}, 0.05)
    assert min(out.values()) >= 0.05 - 1e-9
    assert sum(out.values()) == pytest.approx(1.0)


def test_floor_clip_rejects_nonpositive_floor():
    with pytest.raises(ValueError):
        floor_clip({"a": 1.0}, 0.0)


def test_boltzmann_sums_to_one():
    p = boltzmann({"a": 0.0, "b": 10.0, "c": 5.0}, 1.0)
    assert sum(p.values()) == pytest.approx(1.0)
    assert p["b"] > p["c"] > p["a"]


# --- op bandit (I-4, I-5, I-10) ---

def test_op_bandit_error_repair_gated():
    b = OpBandit()
    rng = random.Random(0)
    for _ in range(200):
        op, prop, stat = b.pick(fresh_failure=False, rng=rng)
        assert op in ("diff", "full_rewrite")
        assert prop >= b.floor - 1e-9
    seen = {b.pick(fresh_failure=True, rng=rng)[0] for _ in range(200)}
    assert "error_repair" in seen


def test_op_bandit_window_max_and_serialization():
    b = OpBandit(window=3)
    for r in (0.1, 0.5, 0.2, 0.05):
        b.observe("diff", _Ev(r))
    assert b._window_max("diff") == 0.5
    d = b.state()
    b2 = OpBandit()
    b2.load_state(d)
    assert b2._window_max("diff") == 0.5
    assert list(b2.events["diff"]) == [0.5, 0.2, 0.05]


def test_op_bandit_window_counts_events_not_evals():
    b = OpBandit(window=3)
    b.observe("diff", _Ev(0.9, is_island_event=True))
    for _ in range(5):
        b.observe("diff", _Ev(0.0, is_island_event=False))
    assert b._window_max("diff") == 0.9
    assert list(b.events["diff"]) == [0.9]


def test_op_bandit_intensify_biases_diff():
    # with equal (empty) stats the arms are uniform; intensify must swing sampling
    # decisively toward diff (the local-refinement arm) without becoming argmax.
    b = OpBandit(arms=["diff", "full_rewrite", "cross"])
    rng = random.Random(0)
    base = sum(b.pick(False, rng)[0] == "diff" for _ in range(400))
    rng = random.Random(0)
    hot = sum(b.pick(False, rng, intensify=True)[0] == "diff" for _ in range(400))
    assert hot > base
    assert hot > 200                       # diff becomes the majority pick
    # other arms still reachable (floor respected, no argmax collapse)
    rng = random.Random(0)
    seen = {b.pick(False, rng, intensify=True)[0] for _ in range(400)}
    assert "full_rewrite" in seen and "cross" in seen


def test_op_bandit_intensify_noop_without_diff_arm():
    # a bandit whose arms exclude diff must not crash when asked to intensify
    b = OpBandit(arms=["full_rewrite", "cross"])
    rng = random.Random(0)
    op, prop, stat = b.pick(False, rng, intensify=True)
    assert op in ("full_rewrite", "cross") and prop >= b.floor - 1e-9


def test_changed_region_captures_modified_excludes_unchanged():
    parent = "a = 1\nrestarts = 5\nc = 3\n"
    child = "a = 1\nrestarts = 20\nc = 3\n"
    region = _changed_region(parent, child)
    assert "restarts = 20" in region  # modified preexisting line captured (recall)
    assert "a = 1" not in region and "c = 3" not in region  # unchanged excluded


# --- schedule (SH, §4.4) ---

class _FakeDB:
    def __init__(self, bests):
        self.bests = bests
        self.migrations = []

    def island_best(self, i):
        return self.bests[i]

    def migrate_unique_elites(self, src, survivors, origin="migrant"):
        self.migrations.append((src, tuple(survivors)))
        return 1


class _FakeLog:
    def __init__(self):
        self.cuts = []

    def sh_cut(self, **f):
        self.cuts.append(f)


def test_sh_cuts_bottom_half():
    s = Schedule(4, policy="sh")
    db = _FakeDB({0: 0.9, 1: 0.1, 2: 0.5, 3: 0.2})
    log = _FakeLog()
    s.on_milestone(10, db, log)
    assert set(s.survivors) == {0, 2}
    assert log.cuts[0]["cut"] == [3, 1] or set(log.cuts[0]["cut"]) == {1, 3}
    assert db.migrations


def test_graft_entrypoints_recovers_dropped_function():
    from sciel.engine.worker import _graft_entrypoints
    import ast
    parent = "def helper():\n    return 5\ndef run_packing():\n    return helper(), [1], 2"
    child = "def helper():\n    return 9\ndef construct():\n    return helper()"   # dropped run_packing
    out, grafted = _graft_entrypoints(child, parent, ["run_packing"])
    have = {n.name for n in ast.walk(ast.parse(out)) if isinstance(n, ast.FunctionDef)}
    assert "run_packing" in have                                   # recovered from parent
    assert grafted == ["run_packing"]
    assert _graft_entrypoints(child, parent, []) == (child, [])    # no-op: flag off
    assert _graft_entrypoints("def run_packing(): pass", parent, ["run_packing"]) \
        == ("def run_packing(): pass", [])                         # no-op: already present
    assert _graft_entrypoints("def x(:", parent, ["run_packing"]) == ("def x(:", [])  # unparseable


def test_graft_skips_child_without_own_functions():
    # empty / prose-only responses must NOT be "rescued" by stitching the parent's
    # entry-points onto them: that builds a franken-child that NameErrors on the
    # parent's imports and helpers (the dominant eval-fail of circle_fr_graft_s42)
    from sciel.engine.worker import _graft_entrypoints
    parent = "import numpy as np\ndef run_packing():\n    return np.zeros(3)"
    assert _graft_entrypoints("\n", parent, ["run_packing"]) == ("\n", [])
    assert _graft_entrypoints("x = 1\n", parent, ["run_packing"]) == ("x = 1\n", [])


def test_graft_carries_imports_helpers_and_constants():
    from sciel.engine.worker import _graft_entrypoints
    parent = (
        "import math\n"
        "from os.path import join\n"
        "N = 3\n"
        "def _helper(v):\n"
        "    return math.sqrt(v) + N\n"
        "def run_packing():\n"
        "    return _helper(4.0)\n"
    )
    child = "def construct():\n    return 1\n"          # dropped run_packing entirely
    out, grafted = _graft_entrypoints(child, parent, ["run_packing"])
    assert grafted == ["run_packing"]
    ns = {}
    exec(out, ns)                                        # grafted module must actually run
    assert ns["run_packing"]() == 5.0                    # sqrt(4)+3: helper+import+const carried
    # transitive deps present, unrelated parent bindings ("join") not dragged in
    assert "import math" in out and "N = 3" in out and "_helper" in out
    assert "join" not in out


def test_graft_module_level_contract():
    # "present" must mean bound at MODULE level -- that is what the evaluator's
    # hasattr(module, name) checks
    from sciel.engine.worker import _graft_entrypoints
    parent = "def solve():\n    return 1\n"
    # (a) entry-point hidden in a nested scope does not satisfy the contract -> graft
    child = "def outer():\n    def solve():\n        return 2\n    return solve()\n"
    out, g = _graft_entrypoints(child, parent, ["solve"])
    assert g == ["solve"]
    ns = {}
    exec(out, ns)
    assert ns["solve"]() == 1
    # (b) an async-only child is real code, not prose -> graft still fires
    out, g = _graft_entrypoints("async def helper():\n    return 2\n", parent, ["solve"])
    assert g == ["solve"] and "def solve" in out
    # (c) a top-level class binding satisfies hasattr -> counts as present
    child = "class solve:\n    pass\n"
    assert _graft_entrypoints(child, parent, ["solve"]) == (child, [])


def test_graft_closure_sees_guarded_imports_and_all_constant_forms():
    from sciel.engine.worker import _graft_entrypoints
    parent = (
        "try:\n    import math\nexcept ImportError:\n    math = None\n"
        "W, H = 4.0, 3\n"
        "X: int = 2\n"
        "def solve():\n    return math.sqrt(W) + H + X\n"
    )
    out, g = _graft_entrypoints("def other():\n    return 1\n", parent, ["solve"])
    assert g == ["solve"]
    ns = {}
    exec(out, ns)
    assert ns["solve"]() == 7.0   # try-guarded import + tuple-unpacked + annotated consts carried


def test_graft_reports_only_names_it_actually_attached():
    from sciel.engine.worker import _graft_entrypoints
    parent = "def solve():\n    return 1\n"       # run_packing absent from parent too
    out, g = _graft_entrypoints("def construct():\n    return 1\n", parent,
                                ["solve", "run_packing"])
    assert g == ["solve"]
    assert "run_packing" not in out


def test_graft_does_not_shadow_child_bindings():
    from sciel.engine.worker import _graft_entrypoints
    parent = ("N = 3\n"
              "def _helper():\n    return N\n"
              "def run_packing():\n    return _helper()\n")
    child = ("N = 10\n"
             "def construct():\n    return N\n")         # child redefines N; graft must reuse it
    out, _ = _graft_entrypoints(child, parent, ["run_packing"])
    ns = {}
    exec(out, ns)
    assert ns["N"] == 10                                 # child's binding survives
    assert ns["run_packing"]() == 10                     # grafted helper sees the child's N


def test_run_worker_escalates_on_empty_llm_response():
    from types import SimpleNamespace
    from sciel.engine.types import NoChild, Snapshot
    from sciel.engine.worker import run_worker

    snap = Snapshot(island=0, op="diff", model="m", iteration_no=1, parent_id="p1",
                    parent_code="def f():\n    return 1\n", parent_score=0.5)
    calls = []

    class _LLM:
        def __init__(self, texts):
            self.texts = list(texts)

        def generate(self, prompt, model, temp):
            return SimpleNamespace(text=self.texts.pop(0), cost=0.0)

    def _eval(code):
        calls.append(code)
        res = SimpleNamespace(score=0.7, cases={}, ok=True, timing=0.0, artifacts={})
        return SimpleNamespace(result=res, raw_first_score=0.7, reached="full",
                               confirmed=False)

    # all attempts empty: nothing is ever evaluated, honest reason (not dup_cap)
    out = run_worker("diff", snap, _LLM(["", "", ""]), _eval, dup_cap=3)
    assert isinstance(out, NoChild) and out.reason == "empty_response"
    assert out.meta["empty_responses"] == 3 and calls == []

    # empty then real: retries and evaluates the real child, journals the empty
    out = run_worker("diff", snap, _LLM(["", "```python\ndef f():\n    return 2\n```"]),
                     _eval, dup_cap=3)
    assert out.ok and len(calls) == 1
    assert out.meta["empty_responses"] == 1 and out.meta["grafted"] == []


def test_config_defaults_deprecated_and_merge_semantics():
    from sciel.config import apply_defaults, validate
    # deprecated key from historical run snapshots is dropped, never an error
    cfg = validate({"task": "oe/x", "inject_entrypoints": True})
    assert "inject_entrypoints" not in cfg
    # yaml `policy:` with no value (None) normalizes instead of crashing
    cfg = validate({"task": "oe/x", "policy": None})
    assert cfg["policy"]["enabled"] is True
    # espalier defaults deep-merge scalars, but a mode mix is a complete
    # distribution: a user mix REPLACES the default (no resurrected modes)
    cfg = validate({"task": "oe/x", "policy": {"base_mix": {"base": 1.0}}})
    assert cfg["policy"]["base_mix"] == {"base": 1.0}
    assert cfg["policy"]["plateau_hold"] == 40
    cfg = validate({"task": "oe/x", "llm": {"max_tokens": 16000}})
    assert cfg["llm"]["max_tokens"] == 16000
    # idempotent: the engine re-applies defaults on top of the CLI path
    once = apply_defaults({"task": "oe/x"})
    assert apply_defaults(once) == once
    assert once["sh_min_fair_evals"] == 16 and once["log_population"] is True


def test_run_worker_mixed_exhaustion_reason_follows_majority():
    from types import SimpleNamespace
    from sciel.engine.db import normalized_hash
    from sciel.engine.types import NoChild, Snapshot
    from sciel.engine.worker import run_worker

    dup_code = "def f():\n    return 1\n"
    snap = Snapshot(island=0, op="diff", model="m", iteration_no=1, parent_id="p1",
                    parent_code=dup_code, parent_score=0.5,
                    code_hashes={normalized_hash(dup_code)})

    class _LLM:
        def __init__(self, texts):
            self.texts = list(texts)

        def generate(self, prompt, model, temp):
            return SimpleNamespace(text=self.texts.pop(0), cost=0.0)

    def _eval(code):
        raise AssertionError("nothing should be evaluated")

    dup_resp = f"```python\n{dup_code}```"
    # empties dominate (3 of 4) -> empty_response: not an island-saturation signal
    out = run_worker("diff", snap, _LLM(["", "", "", dup_resp]), _eval, dup_cap=4)
    assert isinstance(out, NoChild) and out.reason == "empty_response"
    # dups dominate (3 of 4) -> dup_cap: genuine saturation
    out = run_worker("diff", snap, _LLM([dup_resp, dup_resp, "", dup_resp]), _eval, dup_cap=4)
    assert isinstance(out, NoChild) and out.reason == "dup_cap"


def test_internal_contract_detects_block_helper_called_by_fixed_wrapper():
    from sciel.engine.loop import _internal_contract
    prog = ("# EVOLVE-BLOCK-START\n"
            "def construct():\n    return helper()\n"
            "def helper():\n    return 1\n"
            "# EVOLVE-BLOCK-END\n"
            "def run():\n    return construct()\n")
    # construct is defined in-block and called by the fixed run() outside -> required
    assert _internal_contract(prog) == ["construct"]
    # helper is called only inside the block -> not part of the fixed contract
    assert "helper" not in _internal_contract(prog)
    assert _internal_contract("def a(): return b()\nx = a()") == []   # no EVOLVE-BLOCK
    assert _internal_contract("# EVOLVE-BLOCK-START\ndef x(:\n# EVOLVE-BLOCK-END") == []  # unparseable


def test_population_snapshot_is_compact_and_captures_lineage():
    from sciel.engine.db import Database
    import json
    db = Database(num_islands=2, pop_cap=50, grid_bins=10)
    p1 = db.add("x=1\n" * 3, 0.3, {}, island=0, origin="gen")
    db.add("x=2\n" * 10, 0.6, {}, island=0, origin="gen", parent_id=p1.id)  # child gen=1
    db.add("y=1", 0.1, {}, island=1, origin="gen")
    snap = db.population_snapshot()
    assert set(snap) == {"pop_best", "archive", "islands"}
    assert snap["pop_best"] == 0.6 and len(snap["islands"]) == 2
    isl0 = snap["islands"][0]
    assert isl0["best"] == 0.6 and isl0["n"] == 2 and isl0["max_gen"] == 1  # lineage depth
    assert isl0["cells"] >= 1 and isl0["top"][0] == 0.6
    assert "code" not in json.dumps(snap)                      # compact: no code strings
    assert all(isinstance(s, (int, float)) for s in snap["archive"])


def test_sh_fair_chance_defers_cut_of_underserved_island():
    # Island 1 is worst by best-score but has had few evals -> protected from cut.
    s = Schedule(3, policy="sh", min_fair_evals=10)
    s.dispatched = {0: 30, 1: 3, 2: 30}  # island 1 starved
    db = _FakeDB({0: 0.9, 1: 0.0, 2: 0.5})
    log = _FakeLog()
    s.on_milestone(10, db, log)
    assert 1 in s.survivors                 # deferred, not cut
    assert log.cuts[0]["protected"] == [1]
    assert (1,) not in [m[0:1] for m in db.migrations]  # no migration out of protected

    # Same ranking but island 1 has had a fair chance -> cut as usual.
    s2 = Schedule(3, policy="sh", min_fair_evals=10)
    s2.dispatched = {0: 30, 1: 20, 2: 30}
    db2 = _FakeDB({0: 0.9, 1: 0.0, 2: 0.5})
    s2.on_milestone(10, db2, _FakeLog())
    assert 1 not in s2.survivors

    # min_fair_evals=0 (default) -> behaviour identical to plain SH (regression guard).
    s3 = Schedule(3, policy="sh")
    s3.dispatched = {0: 0, 1: 0, 2: 0}
    s3.on_milestone(10, _FakeDB({0: 0.9, 1: 0.0, 2: 0.5}), _FakeLog())
    assert 1 not in s3.survivors


def test_sh_pick_island_counts_dispatch():
    s = Schedule(2, policy="round_robin")
    picks = [s.pick_island([0, 1], in_flight={}) for _ in range(6)]
    assert s.dispatched[picks[0]] + s.dispatched[picks[1]] == 6
    assert sum(s.dispatched.values()) == 6


def test_round_robin_never_cuts():
    s = Schedule(4, policy="round_robin")
    db = _FakeDB({0: 0.9, 1: 0.1, 2: 0.5, 3: 0.2})
    log = _FakeLog()
    s.on_milestone(10, db, log)
    assert set(s.survivors) == {0, 1, 2, 3}
    assert not db.migrations


def test_schedule_state_roundtrip():
    s = Schedule(3)
    s.note_saturation(1)
    s.note_saturation(1)
    d = s.state()
    s2 = Schedule(3)
    s2.load_state(d)
    assert s2.state() == d


# --- rng (§7.3) ---

def test_rng_same_seed_same_sequence():
    a = RngHub(42).stream("parent")
    b = RngHub(42).stream("parent")
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]


def test_rng_streams_independent():
    hub = RngHub(42)
    p = [hub.stream("parent").random() for _ in range(3)]
    hub2 = RngHub(42)
    _ = [hub2.stream("op").random() for _ in range(3)]
    p2 = [hub2.stream("parent").random() for _ in range(3)]
    assert p == p2


def test_rng_state_roundtrip():
    hub = RngHub(42)
    hub.stream("parent").random()
    d = hub.state()
    hub2 = RngHub(1)
    hub2.load_state(d)
    assert hub.stream("parent").random() == hub2.stream("parent").random()


# --- config (§7.2, I-3) ---

def test_config_merge_scalar_override_list_replace():
    base = {"a": 1, "b": [1, 2], "c": {"x": 1, "y": 2}}
    over = {"b": [9], "c": {"y": 3}}
    out = _merge(base, over)
    assert out == {"a": 1, "b": [9], "c": {"x": 1, "y": 3}}


def test_config_unknown_key_errors():
    with pytest.raises(ValueError):
        validate({"bogus_key": 1})


def test_config_seed_flag_rejected():
    with pytest.raises(ValueError):
        validate({"seed": 1})
