"""Future journal event clocks; no search, evaluator, or live directory access."""
from datetime import datetime, timezone
import hashlib
import json

import pytest

import sciel.run_log as module
from sciel.run_log import EVENT_TYPES, RunLog


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_every_event_has_logger_owned_utc_timestamp(tmp_path):
    path = tmp_path / "events.jsonl"
    log = RunLog(str(path))
    before = datetime.now(timezone.utc)
    try:
        log.run_start()
        for event in sorted(EVENT_TYPES - {"run_start"}):
            assert log.emit(event, timestamp="forged", segment_elapsed_seconds=-100)
    finally:
        log.close()
    after = datetime.now(timezone.utc)
    recorded = rows(path)
    assert len(recorded) == len(EVENT_TYPES)
    for event in recorded:
        value = datetime.fromisoformat(event["timestamp"])
        assert value.utcoffset().total_seconds() == 0
        assert before <= value <= after
        assert event["segment_elapsed_seconds"] >= 0
    assert [event["event_seq"] for event in recorded] == list(range(1, len(recorded) + 1))


def test_hypothesis_intervals_use_monotonic_clock_and_reset_on_resume(tmp_path, monkeypatch):
    ticks = iter([10, 10.1, 12.6, 15.6, 20, 40, 40.1, 40.2, 41])
    monkeypatch.setattr(module, "monotonic", lambda: next(ticks))
    path = tmp_path / "events.jsonl"
    log = RunLog(str(path))
    try:
        log.run_start()
        initial_run = log.run_id
        initial_segment = log.segment_id
        log.research(phase="hypothesis", experiment_id="first")
        # A wall-clock adjustment cannot make the measured interval negative.
        monkeypatch.setattr(module, "_timestamp", lambda: "2000-01-01T00:00:00.000000+00:00")
        log.research(phase="hypothesis", experiment_id="second")
        log.run_end(reason="paused")
        log.run_start(restored_checkpoint={"provenance": {"run_id": initial_run}})
        log.research(phase="hypothesis", experiment_id="third")
        assert log.run_id == initial_run
        assert log.segment_id != initial_segment
    finally:
        log.close()
    hypotheses = [row for row in rows(path) if row.get("phase") == "hypothesis"]
    assert hypotheses[1]["segment_elapsed_seconds"] - hypotheses[0]["segment_elapsed_seconds"] == pytest.approx(3)
    assert hypotheses[2]["segment_elapsed_seconds"] == pytest.approx(1)
    assert hypotheses[0]["segment_id"] == hypotheses[1]["segment_id"]
    assert hypotheses[2]["segment_id"] != hypotheses[1]["segment_id"]


def test_pre_segment_events_have_wall_time_without_claiming_segment_duration(tmp_path):
    path = tmp_path / "events.jsonl"
    log = RunLog(str(path))
    try:
        log.decision(action="initialization", segment_elapsed_seconds=999)
    finally:
        log.close()
    event, = rows(path)
    assert datetime.fromisoformat(event["timestamp"]).utcoffset().total_seconds() == 0
    assert "segment_elapsed_seconds" not in event
    assert "segment_id" not in event


def test_checkpoint_anchor_binds_timestamped_event_bytes(tmp_path):
    path = tmp_path / "events.jsonl"
    log = RunLog(str(path))
    try:
        log.run_start()
        log.research(phase="hypothesis", experiment_id="first")
        anchor = log.anchor()
        raw = path.read_bytes()
        assert anchor["event_seq"] == 2
        assert anchor["byte_offset"] == len(raw)
        assert anchor["prefix_sha256"] == hashlib.sha256(raw).hexdigest()
        assert all("timestamp" in row for row in rows(path))
    finally:
        log.close()
