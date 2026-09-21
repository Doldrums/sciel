"""Search identity strips physical provenance, not counters or link structure."""
import copy
import json

import conftest


def sample(suffix, offset):
    return [{"event": "run_start", "run_id": "run" + suffix,
             "segment_id": "segment" + suffix, "event_seq": 1,
             "counters": {"iteration_no": 0, "B_evals": 0}},
            {"event": "checkpoint", "run_id": "run" + suffix,
             "segment_id": "segment" + suffix, "event_seq": 3,
             "checkpoint_id": "checkpoint" + suffix, "checkpoint_sha256": "sha" + suffix,
             "provenance": {"checkpoint_id": "checkpoint" + suffix,
                 "run_id": "run" + suffix, "counters": {"iteration_no": 1, "B_evals": 1},
                 "journal_anchor": {"segment_id": "segment" + suffix, "event_seq": 2,
                                    "byte_offset": offset, "prefix_sha256": "prefix" + suffix}}},
            {"event": "run_end", "run_id": "run" + suffix,
             "segment_id": "segment" + suffix, "event_seq": 4,
             "checkpoint_id": "checkpoint" + suffix, "checkpoint_sha256": "sha" + suffix,
             "iteration_no": 1, "B_evals": 1, "pop_best": .5}]


def normalize(rows):
    identities = {}
    return [conftest.strip_volatile(json.dumps(row), identities) for row in rows]


def test_equivalent_run_metadata_normalizes_without_altering_raw_journal():
    first, second = sample("a", 123), sample("b", 987)
    original = copy.deepcopy(first)
    assert normalize(first) == normalize(second)
    assert first == original


def test_semantic_counters_event_positions_and_checkpoint_links_stay_visible():
    original = sample("a", 123)
    mutations = [
        lambda rows: rows[1]["provenance"]["counters"].update(B_evals=2),
        lambda rows: rows[1]["provenance"]["journal_anchor"].update(event_seq=1),
        lambda rows: rows[-1].update(checkpoint_id="different-checkpoint"),
        lambda rows: rows[-1].update(checkpoint_sha256="different-hash"),
        lambda rows: rows[-1].update(segment_id="different-segment"),
        lambda rows: rows[-1].update(pop_best=.9),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(original)
        mutate(changed)
        assert normalize(changed) != normalize(original)
