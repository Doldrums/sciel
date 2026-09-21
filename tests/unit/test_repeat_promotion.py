"""H15 state-machine tests: every draw is a pure fixture, never an evaluator."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import random
import pytest
from sciel.engine.repeat_promotion import (
    CONTRACT, RepeatPromotion, RepeatPromotionError, bootstrap_decision,
    paired_dominance, validate_config,
)
from sciel.engine.repeat_journal import digest
from repeat_promotion_fixtures import (
    INITIAL, CHALLENGER, sha, config,
    identity, value, receipt, FakeDraw,
    coordinator, bootstrap, plan_comparison, Crash,
)


@pytest.mark.parametrize("key,bad", [
    ("enabled", 1), ("seed_master", True), ("seed_master", -1), ("seed_master", 2**64),
    ("max_comparisons", -1), ("max_comparisons", 1.0), ("max_draw_attempts", 2),
    ("max_draw_attempts", 16), ("timeout_seconds", True), ("timeout_seconds", 0),
    ("max_wall_seconds", float("nan")), ("certification_seconds", float("inf")),
    ("mem_bytes", 0), ("cpu_seconds", False), ("draws_per_source", 4),
    ("rule", "mean"), ("quality_loss_tolerance", 1e-10), ("quality_loss_tolerance", False),
    ("require_confirmed", False), ("unknown", 1),
])
def test_strict_config(key, bad):
    cfg = config()
    cfg[key] = bad
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_config_no_implicit_budget_and_minimal_disabled():
    assert validate_config(None) == validate_config({}) == validate_config({"enabled": False}) == {"enabled": False}
    with pytest.raises(ValueError):
        validate_config({"enabled": False, "seed_master": 1})
    for key in ("timeout_seconds", "max_draw_attempts", "max_wall_seconds"):
        cfg = config()
        del cfg[key]
        with pytest.raises(ValueError):
            validate_config(cfg)
    assert validate_config(config())["draws_per_source"] == 3


@pytest.mark.parametrize("old,new,expected", [
    ([2, 2, 2], [2, 2.1, 2], "promote"),
    ([2, 2, 2], [1.9, 2.5, 2.5], "reject"),  # better mean, one loss
    ([1, 3, 4], [2, 2.9, 5], "reject"),       # better minimum, another loss
    ([2, 2, 2], [2, 2, 2], "reject"),
])
def test_paired_rule_not_mean_or_worst(old, new, expected):
    assert paired_dominance(list(map(value, old)), list(map(value, new)))["status"] == expected


@pytest.mark.parametrize("raw,cert", [(2.1, 1.9), (2.1, 2), (2, 2.1)])
def test_both_metrics_need_nonnegative_and_some_strict_gain(raw, cert):
    assert paired_dominance([value()] * 3, [value(raw, cert)] * 3)["status"] == "reject"


def test_exact_fraction_loss_is_not_rounded_away():
    new = value(2.1, Fraction(2) - Fraction(1, 10**30))
    result = paired_dominance([value()] * 3, [new] * 3)
    assert result["reason"] == "observed_loss"
    assert result["certified_deltas"][0] == {"numerator": -1, "denominator": 10**30}


@pytest.mark.parametrize("broken", [
    {"status": "unknown", "raw_sum": None, "certified_sum": None},
    {"status": "candidate_failure", "raw_sum": None, "certified_sum": None},
    {"status": "valid", "raw_sum": float("nan"), "certified_sum": {"numerator": 2, "denominator": 1}},
    {"status": "valid", "raw_sum": True, "certified_sum": {"numerator": 2, "denominator": 1}},
    {"status": "valid", "raw_sum": 2, "certified_sum": {"lower": 1.9, "upper": 2.1}},
])
def test_no_failure_zero_or_survivor_average(broken):
    result = paired_dominance([value(), broken, value()], [value(2.1)] * 3)
    assert result["status"] == "incomplete"
    assert result["aggregates"] is result["raw_deltas"] is result["certified_deltas"] is None
    assert bootstrap_decision([value(), broken, value()])["aggregates"] is None
    assert paired_dominance([value()] * 2, [value()] * 3)["aggregates"] is None


def test_constructor_no_io_bootstrap_binding_and_checkpoint_gate(coordinator, tmp_path):
    fake = FakeDraw()
    c = coordinator(fake)
    assert list(tmp_path.iterdir()) == []
    c.start("run-fixture")
    with pytest.raises(RepeatPromotionError, match="canonical"):
        c.plan_bootstrap(CHALLENGER)
    assert c.plan_bootstrap(INITIAL)
    assert c.summary()["counters"]["reserved_draw_attempts"] == 3
    with pytest.raises(RepeatPromotionError, match="checkpoint"):
        c.run_pending()
    assert fake.calls == []
    with pytest.raises(RepeatPromotionError):
        c.checkpoint_ack({"entries": 0, "sha256": "0" * 64})
    c.checkpoint_ack(c.state()["ledger_anchor"])
    c.run_pending()
    assert len(fake.calls) == 3
    assert c.export()["confirmed_output_incumbent"]["code"] == INITIAL  # CRLF preserved
    assert c.plan_bootstrap(INITIAL) is False


def test_dispatch_is_after_fsynced_intent(coordinator, monkeypatch):
    import sciel.engine.repeat_journal as journal
    real_fsync, syncs = journal.os.fsync, []
    monkeypatch.setattr(journal.os, "fsync", lambda fd: (syncs.append(fd), real_fsync(fd))[-1])
    c = coordinator()
    seen = []
    def draw(source, *, seed, **limits):
        row = c.journal.read()[-1]
        assert row["event"]["type"] == "intent"
        assert len(syncs) >= 2
        seen.append(row["event"]["draw_id"])
        syncs.clear()
        return receipt(source, seed, limits)
    c.draw_fn = draw
    bootstrap(c)
    assert len(set(seen)) == 3


def test_fresh_pair_seed_schedule_rng_and_retention(coordinator):
    fake, before = FakeDraw(), random.getstate()
    c = coordinator(fake)
    bootstrap(c)
    original_export = c.export()["confirmed_output_incumbent"]
    saved = plan_comparison(c)
    assert c.summary()["counters"]["reserved_draw_attempts"] == 9
    c.run_pending()
    calls = fake.calls[3:]
    assert len(calls) == 6
    for first, second in zip(calls[::2], calls[1::2]):
        assert first[1] == second[1]
        assert {first[0], second[0]} == {INITIAL, CHALLENGER}
    assert [p[0] for p in calls] == [CHALLENGER, INITIAL, INITIAL, CHALLENGER, CHALLENGER, INITIAL]
    assert not {s for _, s, _ in fake.calls[:3]} & {s for _, s, _ in calls}
    assert random.getstate() == before
    assert saved["data"]["confirmed"] == {k: v for k, v in original_export.items() if k != "code"}
    assert c.export({"id": "evicted-db-best", "code": INITIAL, "score": 0.99})["confirmed_output_incumbent"]["code"] == CHALLENGER
    assert c.export()["confirmed_output_incumbent"]["metrics"]["raw"]["values"] == [2.1] * 3


def test_eligibility_raw_not_normalized_and_claim_once(coordinator):
    c = coordinator()
    bootstrap(c)
    for cases in ({}, {"validity": 0, "sum_radii": 3}, {"validity": True, "sum_radii": 3},
                  {"validity": 1, "sum_radii": float("nan")}, {"validity": 1, "sum_radii": 2}):
        assert not c.plan_candidate(CHALLENGER, cases, "p2")
    plan_comparison(c)
    c.run_pending()
    assert not c.plan_candidate(CHALLENGER, {"sum_radii": 3, "validity": 1}, "p3")


def test_post_evaluation_stop_retains_unexamined_source_without_draw_or_promotion(coordinator):
    fake, c = FakeDraw(), None
    c = coordinator(fake)
    bootstrap(c)
    counters, confirmed = c.summary()["counters"], c.export()["confirmed_output_incumbent"]
    reason = "post_evaluation_research_stopped:RequestNotDispatched"
    assert c.note_unexamined_candidate(CHALLENGER, {"sum_radii": 3., "validity": 1.}, "p2", reason=reason)
    assert c.summary()["counters"] == counters and len(fake.calls) == 3
    assert not c.pending and not c.summary()["coverage_complete"]
    assert c.stopped_reason == "search_step_incomplete"
    assert c.export()["confirmed_output_incumbent"] == confirmed
    event, = c.summary()["unexamined_candidates"]
    assert event == {"type": "unexamined_candidate", "source_sha256": sha(CHALLENGER),
        "program_id": "p2", "screening": {"sum_radii": 3., "validity": 1.}, "reason": reason}
    assert (c.directory / "sources" / (sha(CHALLENGER) + ".py")).read_text() == CHALLENGER
    assert c.journal.read()[-1]["event"] == event
    saved = c.state(); c.close()
    restored = coordinator(fake); restored.load_state(saved)
    restored.run_pending()
    assert restored.state() == saved and restored.summary()["counters"] == counters
    assert not restored.plan_candidate("# another\npass\n", {"sum_radii": 4., "validity": 1.}, "p3")
    assert len(fake.calls) == 3


@pytest.mark.parametrize("code,cases", [
    (INITIAL, {"sum_radii": 3., "validity": 1.}),
    (CHALLENGER, {"sum_radii": 2., "validity": 1.}),
    (CHALLENGER, {"sum_radii": 3., "validity": 0.}),
    (CHALLENGER, {"sum_radii": float('nan'), "validity": 1.}),
])
def test_unexamined_marker_does_not_invent_eligibility(coordinator, code, cases):
    c = coordinator(); bootstrap(c)
    before = c.state()
    assert not c.note_unexamined_candidate(code, cases, "p2", reason="research_stopped")
    assert c.state() == before and c.summary()["coverage_complete"]


def test_unexamined_event_cannot_bypass_pending_batch_or_unknown_source_on_resume(coordinator):
    c = coordinator(); bootstrap(c)
    saved_before = c.state()
    assert c.note_unexamined_candidate(CHALLENGER, {"sum_radii": 3., "validity": 1.}, "p2", reason="research_stopped")
    saved_after = c.state(); c.close()
    with pytest.raises(RepeatPromotionError, match="uncheckpointed"):
        coordinator().load_state(saved_before)
    source_path = c.directory / "sources" / (sha(CHALLENGER) + ".py")
    source_path.write_text("altered")
    with pytest.raises(RepeatPromotionError, match="source hash mismatch"):
        coordinator().load_state(saved_after)


def test_unexamined_marker_rejects_pending_and_invalid_reason_atomically(coordinator):
    c = coordinator(); bootstrap(c)
    before = c.state()
    with pytest.raises(ValueError, match="stop reason"):
        c.note_unexamined_candidate(CHALLENGER, {"sum_radii": 3., "validity": 1.}, "p2", reason="")
    assert c.state() == before
    plan_comparison(c)
    before = c.state()
    with pytest.raises(RepeatPromotionError, match="already pending"):
        c.note_unexamined_candidate(CHALLENGER, {"sum_radii": 3., "validity": 1.}, "p2", reason="research_stopped")
    assert c.state() == before


def test_failed_comparison_finishes_all_six_and_keeps_source(coordinator):
    fake = FakeDraw({4: {"status": "candidate_failure"}})
    c = coordinator(fake)
    bootstrap(c)
    plan_comparison(c)
    c.run_pending()
    assert len(fake.calls) == 9
    assert c.stopped_reason is None
    assert c.summary()["decisions"][-1]["result"]["aggregates"] is None
    assert c.summary()["counters"]["failed"] == 1
    assert c.export()["confirmed_output_incumbent"]["code"] == INITIAL
    assert not c.plan_candidate(CHALLENGER, {"sum_radii": 3, "validity": 1}, "p2-again")


def test_bootstrap_failure_has_no_observed_best_fallback(coordinator):
    fake = FakeDraw({1: {"status": "candidate_failure"}})
    c = coordinator(fake)
    bootstrap(c)
    assert len(fake.calls) == 3 and c.stopped_reason == "bootstrap_failed"
    exported = c.export({"id": "best", "code": CHALLENGER, "score": 1})
    assert exported["confirmed_output_incumbent"] is None
    assert exported["observed_search_best"]["source_sha256"] == sha(CHALLENGER)
    assert not c.plan_bootstrap(INITIAL)


@pytest.mark.parametrize("limits", [{"max_draw_attempts": 8}, {"max_wall_seconds": 89}, {"max_comparisons": 0, "max_draw_attempts": 3}])
def test_full_batch_budget_before_dispatch(coordinator, limits):
    cfg = config(); cfg.update(limits)
    fake, c = FakeDraw(), None
    c = coordinator(fake, cfg)
    bootstrap(c)
    assert not c.plan_candidate(CHALLENGER, {"sum_radii": 3, "validity": 1}, "p2")
    assert len(fake.calls) == 3
    assert c.summary()["budget_unavailable"] and not c.summary()["coverage_complete"]
    assert c.summary()["counters"]["reserved_draw_attempts"] == 3


@pytest.mark.parametrize("point", ["before_dispatch", "after_intent", "after_artifact", "after_all_receipts", "after_decision"])
def test_crash_boundaries_no_replay(coordinator, point, monkeypatch):
    fake = FakeDraw()
    c = coordinator(fake)
    c.start("run-fixture"); c.plan_bootstrap(INITIAL)
    saved = c.state(); c.checkpoint_ack(saved["ledger_anchor"])
    if point == "after_intent":
        c.draw_fn = FakeDraw({1: Crash()})
    if point == "after_artifact":
        real_append = c._append
        def append(event):
            if event["type"] == "terminal":
                raise Crash()
            real_append(event)
        monkeypatch.setattr(c, "_append", append)
    if point == "after_all_receipts":
        monkeypatch.setattr(c, "_finish", lambda: (_ for _ in ()).throw(Crash()))
    if point not in ("before_dispatch", "after_decision"):
        with pytest.raises(Crash):
            c.run_pending()
    elif point == "after_decision":
        c.run_pending()
    c.close()
    resumed_fake = FakeDraw()
    resumed = coordinator(resumed_fake)
    resumed.load_state(saved)
    resumed.run_pending()
    if point in ("after_intent", "after_artifact"):
        assert resumed.stopped_reason == "unknown_dispatch"
        assert resumed.summary()["counters"]["unknown"] == 1
        assert resumed.summary()["counters"]["held_wall_seconds"] == 30
        assert resumed.export()["confirmed_output_incumbent"] is None
        assert resumed_fake.calls == []
    else:
        assert resumed.export()["confirmed_output_incumbent"]["code"] == INITIAL
        assert len(resumed_fake.calls) == (3 if point == "before_dispatch" else 0)
        assert len(resumed.summary()["decisions"]) == 1


def test_uncheckpointed_new_plan_tail_is_not_resumed(coordinator):
    c = coordinator(); bootstrap(c)
    saved = c.state()
    plan_comparison(c)
    c.close()
    other = coordinator()
    with pytest.raises(RepeatPromotionError, match="uncheckpointed"):
        other.load_state(saved)


@pytest.mark.parametrize("tamper", ["source", "receipt", "truncated_journal", "checkpoint_state", "identity", "config"])
def test_tampered_resume_refused_before_dispatch(coordinator, tamper):
    c = coordinator(); bootstrap(c)
    saved = c.state(); directory = c.directory
    c.close()
    cfg, ident = config(), identity()
    if tamper == "source":
        (directory / "sources" / (sha(INITIAL) + ".py")).write_text("tampered")
    elif tamper == "receipt":
        path = next((directory / "receipts").glob("*.json")); path.write_bytes(path.read_bytes() + b" ")
    elif tamper == "truncated_journal":
        with (directory / "journal.jsonl").open("ab") as stream:
            stream.write(b'{"incomplete":')
    elif tamper == "checkpoint_state":
        saved["data"]["confirmed"]["source_sha256"] = sha(CHALLENGER)
    elif tamper == "identity":
        ident["integration_sha256"]["loop"] = "e" * 64
    elif tamper == "config":
        cfg["seed_master"] += 1
    fake = FakeDraw(); other = coordinator(fake, cfg, ident)
    with pytest.raises(ValueError):
        other.load_state(saved)
    assert fake.calls == []


def test_active_coordinator_cannot_be_resumed_concurrently(coordinator):
    c = coordinator(); bootstrap(c)
    other = coordinator()
    with pytest.raises(RepeatPromotionError, match="already active"):
        other.load_state(c.state())


def test_malformed_adapter_return_retained_and_halts(coordinator):
    c = coordinator()
    def changed(source, *, seed, **limits):
        r = receipt(source, seed, limits)
        r["seed"] += 1
        return r
    c.draw_fn = changed
    bootstrap(c)
    assert c.stopped_reason == "infrastructure"
    assert c.summary()["counters"]["draw_attempts"] == 1
    assert c.summary()["decisions"][0]["result"]["aggregates"] is None
    paths = list((c.directory / "receipts").glob("*.rejected.json"))
    assert len(paths) == 1
    assert json.loads(paths[0].read_text())["status"] == "valid"


def test_partial_receipt_tail_completes_only_remaining_draws(coordinator, monkeypatch):
    c = coordinator(); c.start("run-fixture"); c.plan_bootstrap(INITIAL)
    saved = c.state(); c.checkpoint_ack(saved["ledger_anchor"])
    real_append = c._append
    def stop_after_terminal(event):
        real_append(event)
        if event["type"] == "terminal":
            raise Crash()
    monkeypatch.setattr(c, "_append", stop_after_terminal)
    with pytest.raises(Crash):
        c.run_pending()
    c.close()
    fake = FakeDraw(); other = coordinator(fake)
    other.load_state(saved); other.run_pending()
    assert len(fake.calls) == 2
    assert other.summary()["counters"]["draw_attempts"] == 3
    assert other.summary()["counters"]["actual_wall_seconds"] == 3
    assert other.summary()["counters"]["released_wall_seconds"] == 27


def test_failed_infrastructure_retains_all_planned_positions_and_stops(coordinator):
    fake = FakeDraw({4: RuntimeError("must not leak provider secret")})
    c = coordinator(fake); bootstrap(c); plan_comparison(c); c.run_pending()
    result = c.summary()
    assert c.stopped_reason == "infrastructure" and len(fake.calls) == 4
    assert len(result["decisions"][-1]["draw_statuses"]) == 6
    assert result["counters"]["reserved_draw_attempts"] == 9
    assert "must not leak" not in c.journal.path.read_text()
    assert all("must not leak" not in p.read_text() for p in (c.directory / "receipts").glob("*.json"))
    assert c.export()["confirmed_output_incumbent"]["code"] == INITIAL


@pytest.mark.parametrize("runtime", [{"backend": "docker"}, {"backend": "auto"},
    {"backend": "native", "image": "ignored"}, {"backend": "docker", "image": "python:latest"},
    {"backend": "docker", "image": "sha256:" + "G" * 64}])
def test_runtime_is_explicit_and_immutable(runtime):
    cfg = config(); cfg["runtime"] = runtime
    with pytest.raises(ValueError, match="runtime"):
        validate_config(cfg)


def test_docker_normalization_and_copy():
    cfg = config(); cfg["runtime"] = {"backend": "docker", "image": "sha256:" + "f" * 64}
    normalized = validate_config(cfg)
    cfg["runtime"]["image"] = "mutated"
    assert normalized["runtime"]["image"] == "sha256:" + "f" * 64
    assert validate_config(config())["runtime"] == {"backend": "native"}


def test_unfunded_bootstrap_stops_without_search_fallback(coordinator):
    cfg = config(); cfg["max_wall_seconds"] = 29
    fake = FakeDraw(); c = coordinator(fake, cfg)
    c.start("run-fixture")
    assert not c.plan_bootstrap(INITIAL)
    assert c.stopped_reason == "bootstrap_budget_unavailable"
    assert c.export()["confirmed_output_incumbent"] is None
    assert not c.plan_candidate(CHALLENGER, {"sum_radii": 3, "validity": 1}, "p2")
    assert not fake.calls


def test_closed_owner_cannot_dispatch_after_a_new_owner_resumes(coordinator):
    c = coordinator(); c.start("run-fixture"); c.plan_bootstrap(INITIAL)
    saved = c.state(); c.checkpoint_ack(saved["ledger_anchor"]); c.close()
    other = coordinator(); other.load_state(saved)
    with pytest.raises(RepeatPromotionError, match="not active"):
        c.run_pending()
    other.run_pending()


def test_fsync_error_before_dispatch_never_executes_draw(coordinator, monkeypatch):
    import sciel.engine.repeat_journal as journal
    fake = FakeDraw(); c = coordinator(fake)
    c.start("run-fixture"); c.plan_bootstrap(INITIAL)
    c.checkpoint_ack(c.state()["ledger_anchor"])
    monkeypatch.setattr(journal.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk failed")))
    with pytest.raises(OSError, match="disk failed"):
        c.run_pending()
    assert not fake.calls
    # A possibly persisted intent is conservatively unknown on a later resume,
    # never a reason to blindly retry within this in-memory coordinator.
    with pytest.raises(ValueError, match="changed outside"):
        c.run_pending()


def test_resume_crash_after_unknown_before_decision_remains_stopped(coordinator, monkeypatch):
    c = coordinator(FakeDraw({1: Crash()})); c.start("run-fixture"); c.plan_bootstrap(INITIAL)
    saved = c.state(); c.checkpoint_ack(saved["ledger_anchor"])
    with pytest.raises(Crash):
        c.run_pending()
    c.close()
    recovering = coordinator()
    monkeypatch.setattr(recovering, "_finish", lambda: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        recovering.load_state(saved)
    fake = FakeDraw(); final = coordinator(fake)
    final.load_state(saved); final.run_pending()
    assert final.stopped_reason == "unknown_dispatch" and not fake.calls


@pytest.mark.parametrize("field,changed", [
    ("schema", "future-schema"), ("source_sha256", "f" * 64), ("adapter_identity", {"wrong": "runtime"}),
    ("seed_applied", {"determinism_proven": True}), ("raw_sum", True),
    ("certificate", {"status": "certified", "sum_exact": {"numerator": 7, "denominator": 1}}),
    ("started_at", "not-a-timestamp"), ("elapsed_seconds", -1),
])
def test_mismatched_receipt_never_becomes_quality(field, changed, coordinator):
    c = coordinator()
    def bad(source, *, seed, **limits):
        r = receipt(source, seed, limits); r[field] = changed
        return r
    c.draw_fn = bad; bootstrap(c)
    assert c.stopped_reason == "infrastructure"
    assert c.summary()["decisions"][0]["result"]["aggregates"] is None
    assert c.export()["confirmed_output_incumbent"] is None


def test_live_configuration_cannot_drift_after_checkpoint(coordinator):
    fake = FakeDraw(); c = coordinator(fake)
    c.start("run-fixture"); c.plan_bootstrap(INITIAL)
    c.checkpoint_ack(c.state()["ledger_anchor"])
    c.identity["integration_sha256"]["loop"] = "e" * 64
    with pytest.raises(RepeatPromotionError, match="changed in memory"):
        c.run_pending()
    assert not fake.calls


def test_continuous_and_resumed_comparison_match_exact_state(tmp_path):
    first_dir, second_dir = tmp_path / "continuous", tmp_path / "resumed"
    first_dir.mkdir(); second_dir.mkdir()
    a = RepeatPromotion(first_dir, config(), identity(), FakeDraw())
    b = RepeatPromotion(second_dir, config(), identity(), FakeDraw())
    resumed = None
    try:
        bootstrap(a); plan_comparison(a); a.run_pending()
        bootstrap(b); saved = plan_comparison(b); b.close()
        resumed = RepeatPromotion(second_dir, config(), identity(), FakeDraw())
        resumed.load_state(saved); resumed.run_pending()
        assert resumed.state() == a.state()
        assert resumed.export() == a.export()
    finally:
        a.close(); b.close()
        if resumed:
            resumed.close()
