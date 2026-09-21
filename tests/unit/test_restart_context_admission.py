"""Authored dense traces and inert projections; no runtime/candidate/provider calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.knowledge_base import restart_context as context
from test_restart_context import traced_group
from restart_admission_fixture import dense_group

@pytest.fixture(scope='module')
def dense_trace_group():
    return dense_group()


def observation(group, seed, role):
    receipt = next(p for p in group['pairs'] if p['seed'] == seed)[role]
    return context.motion_observations.build_observation(receipt,
        source_sha256=group[role + '_source_sha256'], seed=seed,
        adapter_identity=group['adapter_identity'])


def test_dense_trace_zero_old_cards_becomes_one_exact_whole_card_without_cap_growth(dense_trace_group):
    before = context._wire(dense_trace_group)
    sizes = []
    for pair in dense_trace_group['pairs']:
        for role in ('comparator', 'challenger'):
            value = observation(dense_trace_group, pair['seed'], role)
            old = context.observations.observation_view(value, max_chars=8192)
            assert old['included'] is False and old['omission_reason'] == 'card_char_limit'
            whole = context.observations.observation_view(value, max_chars=131072)
            sizes.append(len(context._wire(whole)))
    assert sizes == [8918, 8910, 8916, 8914, 8912, 8924]

    wire = context.build_context([dense_trace_group])
    assert context.MAX_CARD_CHARS == 8192 and wire['max_chars'] == 12000
    assert wire['schema'] == context.SCHEMA == 'sciel-restart-observation-context/5'
    assert (wire['selected_count'], wire['presented_count'], wire['omitted_count']) == (6, 1, 5)
    assert wire['rendered_chars'] == len(context._wire(wire)) == 9658
    assert context._factored_card_chars(wire, 0) == 7352 <= 8192
    assert wire['omissions'] == [[i, 'memory_char_limit'] for i in range(1, 6)]
    expanded = context.expand_context(wire)
    card = expanded['cards'][0]
    assert (card['seed'], card['role']) == (11, 'comparator')
    expected = observation(dense_trace_group, card['seed'], card['role'])
    assert context._wire(card['observation_view']['observation']) == context._wire(expected)
    assert len(expected['cases']) == 16
    assert card['receipt_sha256'] == expected['receipt_sha256']
    assert card['source_sha256'] == expected['bindings']['source_sha256']
    assert wire['groups'][0]['group_sha256'] == context._digest(dense_trace_group)
    assert context._pack(expanded) == wire and context.validate_context(wire) == wire
    assert context._wire(dense_trace_group) == before


def test_per_card_limit_counts_every_referenced_field_even_when_shared(dense_trace_group, monkeypatch):
    wire = context.build_context([dense_trace_group], max_chars=64000)
    assert wire['presented_count'] == 6
    for index, card in enumerate(wire['cards']):
        exact = dict(card=card, observation_catalog=[wire['observation_catalog'][ref]
            for ref in card['observation_refs']])
        assert context._factored_card_chars(wire, index) == len(context._wire(exact))
        assert context._factored_card_chars(wire, index) <= 8192
    first_only = context.build_context([dense_trace_group], max_items=1)
    size = context._factored_card_chars(first_only, 0)
    monkeypatch.setattr(context, 'MAX_CARD_CHARS', size)
    assert context.validate_context(first_only) == first_only
    monkeypatch.setattr(context, 'MAX_CARD_CHARS', size - 1)
    with pytest.raises(ValueError, match='factored card character limit'):
        context.validate_context(first_only)
    rejected = context.build_context([dense_trace_group], max_items=1)
    assert rejected['omissions'][0] == [0, 'card_char_limit']
    assert rejected['rendered_chars'] <= 12000


def test_legacy_v4_reader_retains_its_original_whole_view_guard(dense_trace_group):
    current = context.build_context([dense_trace_group])
    restamped = context._seal(dict(current, schema=context.PREVIOUS_FACTORED_SCHEMA))
    with pytest.raises(ValueError, match='whole view required'):
        context.validate_context(restamped)
    small = context.build_context([traced_group()], max_chars=64000)
    legacy = context._seal(dict(small, schema=context.PREVIOUS_FACTORED_SCHEMA))
    assert context.validate_context(legacy) == legacy
    assert context.expand_context(legacy) == context.expand_context(small)


def test_total_wire_limit_is_checked_after_card_factoring(dense_trace_group):
    wire = context.build_context([dense_trace_group], max_items=1)
    exact = context.build_context([dense_trace_group], max_items=1, max_chars=wire['rendered_chars'])
    for _ in range(3):
        if exact['max_chars'] != exact['rendered_chars']:
            exact = context.build_context([dense_trace_group], max_items=1, max_chars=exact['rendered_chars'])
    assert exact['presented_count'] == 1 and exact['max_chars'] == exact['rendered_chars']
    below = context.build_context([dense_trace_group], max_items=1, max_chars=exact['max_chars'] - 1)
    # Later whole cards are smaller and remain eligible in the fixed order.
    assert below['presented_count'] <= 1
    assert all(card['receipt_ref'] != 0 for card in below['cards'])
    assert below['rendered_chars'] <= below['max_chars']
    assert below['omissions'][0] == [0, 'memory_char_limit']


def test_admission_identity_binds_policy_and_source_and_is_detached(monkeypatch):
    before = context.admission_identity()
    assert before['policy']['wire_schema'] == context.SCHEMA
    assert before['policy']['card_chars'] == 8192
    assert before['source_sha256'] == hashlib.sha256(Path(context.__file__).read_bytes()).hexdigest()
    changed = context.admission_identity()
    changed['policy']['card_chars'] = 99999
    assert context.admission_identity() == before
    original = Path.read_bytes
    def drift(path):
        value = original(path)
        return value + b'\n# drift' if path.resolve() == Path(context.__file__).resolve() else value
    monkeypatch.setattr(Path, 'read_bytes', drift)
    assert context.admission_identity()['source_sha256'] != before['source_sha256']
