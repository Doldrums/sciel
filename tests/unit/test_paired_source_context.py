"""Exact source transport and whole diffs; only inert text/numeric fixtures."""
import copy
import hashlib
import json

import pytest

from sciel.knowledge_base import paired_comparisons as pair
from sciel.knowledge_base.catalogs import canonical_json
from sciel.knowledge_base.presentation import (comparison_card, reflection_payload,
    expand_reflection_payload, omit_source_context, COMPARISON_CARD_SCHEMA,
    LEGACY_COMPARISON_CARD_SCHEMA)
from test_paired_comparisons import inputs


def sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def parameters(a='x = 1\n', b='x = 2\n'):
    data = inputs()
    for role, source in [('comparator', a), ('challenger', b)]:
        data[role + '_source_sha256'] = sha(source)
        for row in data['pairs']:
            row[role]['source_sha256'] = sha(source)
    return data


def build(a='x = 1\n', b='x = 2\n'):
    return pair.build_comparison(**parameters(a, b), source_pair={'comparator': a, 'challenger': b})


def restore_from_changes(original, view):
    # Independent reconstruction checks coverage without duplicating difflib.
    lines = original.splitlines(keepends=True)
    for change in reversed(view['changes']):
        start, end = change['comparator_lines']
        assert ''.join(lines[start:end]) == change['before']
        lines[start:end] = change['after'].splitlines(keepends=True)
    return ''.join(lines)


def test_optional_source_pair_is_detached_and_preserves_measurements():
    data = parameters()
    source = {'comparator': 'x = 1\n', 'challenger': 'x = 2\n'}
    group = pair.build_comparison(**data, source_pair=source)
    legacy = pair.build_comparison(**data)
    assert {k: v for k, v in group.items() if k != 'source_pair'} == legacy
    assert group['id'] == legacy['id']
    assert pair.validate_comparison(group) == group
    source['challenger'] = 'changed outside'
    assert group['source_pair']['challenger'] == 'x = 2\n'
    assert pair.validate_comparison(legacy) == legacy
    assert pair.build_comparison(**data, source_pair=None) == legacy


@pytest.mark.parametrize('bad', [None, {}, {'comparator': 'x = 1\n'},
    {'comparator': 'x = 1\n', 'challenger': 'wrong'},
    {'comparator': 'x = 1\n', 'challenger': 'x = 2\n', 'operation': 'guessed'},
    {'comparator': 1, 'challenger': 'x = 2\n'}])
def test_stored_null_malformed_or_source_tampering_rejected(bad):
    group = build(); group['source_pair'] = bad
    with pytest.raises(ValueError):
        pair.validate_comparison(group)


def test_source_pair_cannot_replace_receipt_source_identity():
    group = build()
    group['source_pair']['challenger'] = 'x = 3\n'
    group['challenger_source_sha256'] = sha('x = 3\n')
    with pytest.raises(ValueError, match='receipt source'):
        pair.validate_comparison(group)


def test_strict_utf8_limit_counts_bytes_and_rejects_surrogates():
    text = 'é' * (pair.MAX_SOURCE_BYTES // 2)
    assert build(text, 'other')['source_pair']['comparator'] == text
    with pytest.raises(ValueError, match='byte limit'):
        build(text + 'é', 'other')
    bad = parameters()
    with pytest.raises(ValueError, match='strict UTF8'):
        pair.build_comparison(**bad, source_pair={'comparator': '\ud800', 'challenger': 'x = 2\n'})


def test_global_group_json_limit_remains_independent_of_source_byte_limit():
    # Each source fits UTF8, but JSON escaping exceeds the whole-group cap.
    with pytest.raises(ValueError, match='JSON exceeds'):
        build('\x00' * pair.MAX_SOURCE_BYTES, '\x01' * pair.MAX_SOURCE_BYTES)


@pytest.mark.parametrize('before,after', [
    ('one\ntwo\nlast\n', 'one\nNEW\ntwo\nlast\n'),
    ('one\nremoved\nlast\n', 'one\nlast\n'),
    ('α = "é"\r\nunchanged\r\nlast', 'α = "😀"\r\nunchanged\r\nLAST'),
    ('line\n', 'line'), ('line', 'line\n'),
    ('', 'new'), ('old', ''),
    ('first\nkeep\nold\ntail\n', 'FIRST\nkeep\nnew\ntail\nextra'),
    ('a\vb\n', 'a\vc\n'),
])
def test_all_changed_blocks_reconstruct_exact_text_and_ranges(before, after):
    group = build(before, after)
    view = pair.source_pair_view(group)
    assert view['available'] and view['omission_reason'] is None
    assert restore_from_changes(before, view) == after
    lines = after.splitlines(keepends=True)
    for change in view['changes']:
        lo, hi = change['challenger_lines']
        assert ''.join(lines[lo:hi]) == change['after']
    assert view['comparator_source_sha256'] == sha(before)
    assert view['challenger_source_sha256'] == sha(after)
    assert pair.validate_source_pair_view(view, sha(before), sha(after)) == view
    assert 'No semantic, mechanism, runtime, quality or causal claim' in view['interpretation']
    assert len(canonical_json(view)) <= pair.MAX_SOURCE_VIEW_CHARS


def test_large_changed_block_is_wholly_omitted_but_group_retains_exact_source():
    before, after = 'unchanged\n' + 'a' * 6000, 'unchanged\n' + 'b' * 6000
    group = build(before, after); view = pair.source_pair_view(group)
    assert view['available'] is False and view['changes'] == []
    assert view['omission_reason'] == 'changed_text_limit'
    assert group['source_pair'] == {'comparator': before, 'challenger': after}
    assert len(canonical_json(view)) <= pair.MAX_SOURCE_VIEW_CHARS


def test_many_common_lines_do_not_hide_small_edit_or_trigger_work_limit():
    before = 'same\n' * 2000 + 'old\n' + 'suffix\n' * 2000
    after = 'same\n' * 2000 + 'new\n' + 'suffix\n' * 2000
    view = pair.source_pair_view(build(before, after))
    assert view['available'] and len(view['changes']) == 1
    assert view['changes'][0]['comparator_lines'] == [2000, 2001]
    assert restore_from_changes(before, view) == after


def test_pathological_middle_line_work_is_explicitly_wholly_omitted():
    view = pair.source_pair_view(build('a\n' * 1001, 'b\n' * 1001))
    assert not view['available'] and view['changes'] == []
    assert view['omission_reason'] == 'diff_work_limit'


def test_legacy_group_and_card_are_exactly_unchanged_without_guessed_context():
    group = pair.build_comparison(**parameters())
    view = pair.source_pair_view(group)
    assert view['available'] is False and view['omission_reason'] == 'source_pair_absent'
    assert view['changes'] == []
    fields = ('id', 'namespace', 'batch_id', 'branch', 'origin_experiment_id',
              'comparator_source_sha256', 'challenger_source_sha256', 'task_metric',
              'metric', 'limits', 'derived', 'forecast', 'scope')
    expected = {k: copy.deepcopy(group[k]) for k in fields}
    expected.update(schema='sciel-paired-comparison-card/1', evidence_sha256=digest(group),
        adapter_sha256=digest(group['adapter_identity']), receipt_sha256=[
            {'seed': p['seed'], 'comparator': digest(p['comparator']),
             'challenger': digest(p['challenger'])} for p in group['pairs']])
    expected['view_sha256'] = digest(expected)
    assert comparison_card(group) == expected
    assert comparison_card(group)['schema'] == LEGACY_COMPARISON_CARD_SCHEMA
    assert 'source_context' not in expected


def test_source_card_and_compact_reflection_preserve_whole_readable_change():
    group = build('old\nunchanged\n', 'new\nunchanged\n')
    card = comparison_card(group)
    assert card['schema'] == COMPARISON_CARD_SCHEMA == 'sciel-paired-comparison-card/2'
    assert card['source_context'] == pair.source_pair_view(group)
    assert 'source_pair' not in card  # Full sources stay in evidence, not duplicated in prompt.
    payload = reflection_payload([card], [])
    restored = expand_reflection_payload(json.loads(canonical_json(payload)))
    assert restored['observation_catalog'][group['id']] == card
    assert restored['schema'] == 'sciel-reflection-presentation/9-inline'
    assert card['evidence_sha256'] == digest(group)
    assert card['view_sha256'] == digest({k: v for k, v in card.items() if k != 'view_sha256'})


@pytest.mark.parametrize('mutation', [
    lambda c: c['source_context'].update(comparator_source_sha256='0'*64),
    lambda c: c['source_context'].update(interpretation='proven mechanism'),
    lambda c: c['source_context']['changes'][0].update(comparator_lines=[False, 1]),
    lambda c: c['source_context'].update(available=False),
    lambda c: c['source_context'].update(available=False, omission_reason='source_pair_absent', changes=[]),
    lambda c: c.update(unknown='invented'),
    lambda c: c.pop('source_context'),
])
def test_even_rehashed_card_rejects_malformed_or_unbound_source_context(mutation):
    card = comparison_card(build()); mutation(card)
    card['view_sha256'] = digest({k: v for k, v in card.items() if k != 'view_sha256'})
    with pytest.raises(ValueError):
        reflection_payload([card], [])


def test_current_lp_complete_insertions_fit_source_view_without_executing_solver():
    from sciel.hypothesis import circle_lp_initialization as lp
    a, b = lp.source_for('control'), lp.source_for('lp')  # Pure pinned text builder.
    view = pair.source_pair_view(build(a, b))
    assert view['available'] and restore_from_changes(a, view) == b
    assert 'def initialize_radii' in ''.join(x['after'] for x in view['changes'])
    assert len(canonical_json(view)) <= 10000


def test_budget_omission_preserves_every_numeric_fact_and_evidence_binding():
    group = build('old\nkeep\nold2\n', 'new\nkeep\nnew2\n')
    card = comparison_card(group); before = copy.deepcopy(card)
    reduced = omit_source_context(card)
    assert card == before  # Caller and full source evidence remain unchanged.
    assert len(card['source_context']['changes']) == 2
    assert reduced['source_context'] == {**card['source_context'],
        'available': False, 'changes': [], 'omission_reason': 'presentation_budget'}
    assert {k: v for k, v in reduced.items() if k not in ('source_context', 'view_sha256')} == {
        k: v for k, v in card.items() if k not in ('source_context', 'view_sha256')}
    assert reduced['derived'] == group['derived']
    assert reduced['view_sha256'] != card['view_sha256']
    assert reduced['view_sha256'] == digest({k: v for k, v in reduced.items() if k != 'view_sha256'})
    assert omit_source_context(reduced) == reduced
    restored = expand_reflection_payload(reflection_payload([reduced], []))
    assert restored['observation_catalog'][group['id']] == reduced


@pytest.mark.parametrize('mutation', [
    lambda c: c['source_context'].update(omission_reason='unsupported_budget'),
    lambda c: c['source_context'].update(comparator_source_sha256='0'*64),
    lambda c: c['source_context'].update(available=True),
    lambda c: c['source_context'].update(changes=[{'invented': 'partial hunk'}]),
])
def test_rehashed_budget_omission_rejects_invented_context(mutation):
    card = omit_source_context(comparison_card(build())); mutation(card)
    card['view_sha256'] = digest({k: v for k, v in card.items() if k != 'view_sha256'})
    with pytest.raises(ValueError):
        reflection_payload([card], [])
    with pytest.raises(ValueError):
        omit_source_context(card)


def test_budget_omission_does_not_invent_context_for_legacy_or_full_group():
    for value in (comparison_card(pair.build_comparison(**parameters())), build(), None):
        with pytest.raises(ValueError):
            omit_source_context(value)
