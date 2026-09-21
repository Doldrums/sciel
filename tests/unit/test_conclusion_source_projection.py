"""Independent exact-text projection checks; all candidate sources stay inert strings."""
from copy import deepcopy
import difflib
import hashlib
import json

import pytest

from sciel.hypothesis import conclusion_source as source
from sciel.hypothesis.mechanism_review import validate_binding as validate_review_binding


def digest(text):
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def pair_record(parent, candidate):
    return dict(parent_source_sha256=digest(parent), source_sha256=digest(candidate))


def rewrite_pair(lines=400):
    parent = 'def candidate():\n' + ''.join(f'    old_{i:04d} = {i}\n' for i in range(lines))
    candidate = 'def candidate():\n' + ''.join(f'    new_{i:04d} = {i + 1}\n' for i in range(lines))
    return parent, candidate


def test_full_diff_preserves_exact_text_and_has_no_unknown_coverage():
    parent = 'def candidate():\n    """Old documentation."""\n    return "before"\n'
    candidate = 'def candidate():\n    """New documentation."""\n    return "after"\n'
    expected = ''.join(difflib.unified_diff(parent.splitlines(keepends=True),
        candidate.splitlines(keepends=True), fromfile='evaluated_parent', tofile='candidate', n=2))
    projection = source.project(parent, candidate)
    assert projection['mode'] == 'full' and projection['diff'] == expected
    assert projection['full_diff_sha256'] == digest(expected)
    assert projection['full_diff_chars'] == len(expected)
    assert projection['coverage']['complete'] is True
    assert projection['coverage']['omitted_diff_ranges'] == []
    assert projection['coverage']['omitted_changed_chars'] == 0
    assert source.fields(projection) == {'conclusion_source.diff': expected}
    record = pair_record(parent, candidate)
    binding = source.bind(record, parent, candidate, projection)
    assert source.validate_binding(record, json.loads(json.dumps(binding))) == projection


def test_one_large_hunk_samples_changed_text_through_its_late_candidate_lines():
    parent, candidate = rewrite_pair()
    full = source.full_diff(parent, candidate)
    assert sum(line.startswith('@@ ') for line in full.splitlines()) == 1
    assert full.index('+    new_0399') > 6000
    projection = source.project(parent, candidate, window_chars=96, window_count=12)
    windows = projection['windows']
    assert 2 <= len(windows) <= 12
    assert '-    old_0000' in windows[0]['text']
    assert '+    new_0399' in windows[-1]['text']
    assert windows[-1]['start'] > 6000
    assert projection['coverage']['shown_diff_chars'] <= 12 * 96
    assert projection['coverage']['omitted_changed_chars'] > 0
    assert projection['coverage']['complete'] is False
    # Windows cover distinct portions of the entire edit, not 12 hunk starts.
    assert any(len(full) // 3 < row['start'] < len(full) * 2 // 3 for row in windows)
    assert source.validate_binding(pair_record(parent, candidate),
        source.bind(pair_record(parent, candidate), parent, candidate, projection)) == projection


def test_separated_hunks_all_receive_changed_text_without_unchanged_padding_bias():
    old = [f'unchanged_{i:03d} = {i}\n' for i in range(400)]
    new = list(old)
    edits = {2: 'first_edit = 111\n', 200: 'middle_edit = 222\n', 398: 'late_edit = 333\n'}
    for index, text in edits.items():
        new[index] = text
    parent, candidate = ''.join(old), ''.join(new)
    projection = source.project(parent, candidate, window_chars=80, window_count=12)
    assert sum(line.startswith('@@ ') for line in source.full_diff(parent, candidate).splitlines()) == 3
    for text in edits.values():
        assert any('+' + text in window['text'] for window in projection['windows'])
    for window in projection['windows']:
        assert window['text'] == source.full_diff(parent, candidate)[window['start']:window['end']]


def test_unicode_crlf_ranges_use_character_offsets_without_normalizing_source():
    parent = 'def candidate():\r\n' + ''.join(f'    x{i} = "café😀e\u0301旧"\r\n' for i in range(70))
    candidate = parent.replace('旧', '新')
    full = source.full_diff(parent, candidate)
    projection = source.project(parent, candidate, window_chars=31, window_count=9)
    assert len(full.encode()) > len(full)
    assert any('\r\n' in row['text'] for row in projection['windows'])
    assert any('😀' in row['text'] for row in projection['windows'])
    for row in projection['windows']:
        assert row['text'] == full[row['start']:row['end']]
        assert row['sha256'] == digest(row['text'])
        assert row['end'] - row['start'] == len(row['text'])
    record = pair_record(parent, candidate)
    bound = source.bind(record, parent, candidate, projection)
    assert source.validate_binding(record, json.loads(json.dumps(bound))) == projection
    tampered = deepcopy(bound)
    tampered['candidate_source'] = candidate.replace('\r\n', '\n')
    with pytest.raises(ValueError, match='source identities'):
        source.validate_binding(record, tampered)


@pytest.mark.parametrize('mode', ['full', 'windows'])
def test_no_change_has_empty_quote_fields_and_complete_zero_length_coverage(mode):
    text = '# A comment is still exact source.\nvalue = "λ"\n'
    projection = source.project(text, text, window_chars=None if mode == 'full' else 32)
    assert source.fields(projection) == {}
    assert projection['full_diff_chars'] == projection['changed_chars'] == 0
    assert projection['coverage'] == dict(shown_diff_chars=0, shown_changed_chars=0,
        omitted_changed_chars=0, omitted_diff_ranges=[], complete=True)
    assert source.validate_binding(pair_record(text, text), source.bind(
        pair_record(text, text), text, text, projection)) == projection


@pytest.mark.parametrize('mutation', [
    'parent_hash', 'candidate_hash', 'full_hash', 'total_chars', 'total_lines',
    'changed_chars', 'changed_lines', 'coverage_shown', 'coverage_omitted',
    'coverage_ranges', 'coverage_complete', 'window_line_range', 'window_shift',
    'window_rehashed_text', 'selection', 'parent_bytes', 'candidate_bytes',
])
def test_recomputed_binding_rejects_forged_hash_range_coverage_and_rehashed_text(mutation):
    parent, candidate = rewrite_pair(80)
    record = pair_record(parent, candidate)
    projection = source.project(parent, candidate, window_chars=48, window_count=4)
    binding = source.bind(record, parent, candidate, projection)
    bad = deepcopy(binding)
    p = bad['projection']
    if mutation in {'parent_hash', 'candidate_hash', 'full_hash'}:
        key = {'parent_hash': 'parent_source_sha256', 'candidate_hash': 'source_sha256',
            'full_hash': 'full_diff_sha256'}[mutation]
        p[key] = '0' * 64
    elif mutation in {'total_chars', 'total_lines', 'changed_chars', 'changed_lines'}:
        key = {'total_chars': 'full_diff_chars', 'total_lines': 'full_diff_lines'}.get(mutation, mutation)
        p[key] += 1
    elif mutation in {'coverage_shown', 'coverage_omitted'}:
        p['coverage']['shown_changed_chars' if mutation == 'coverage_shown' else 'omitted_changed_chars'] += 1
    elif mutation == 'coverage_ranges':
        p['coverage']['omitted_diff_ranges'][0][0] += 1
    elif mutation == 'coverage_complete':
        p['coverage']['complete'] = True
    elif mutation == 'window_line_range':
        p['windows'][0]['diff_line_start'] += 1
    elif mutation == 'window_shift':
        row = p['windows'][1]
        row['start'] += 1; row['end'] += 1
        row['text'] = source.full_diff(parent, candidate)[row['start']:row['end']]
        row['sha256'] = digest(row['text'])
    elif mutation == 'window_rehashed_text':
        row = p['windows'][1]
        row['text'] = '?' + row['text'][1:]
        row['sha256'] = digest(row['text'])
    elif mutation == 'selection':
        p['selection']['window_chars'] += 1
    elif mutation in {'parent_bytes', 'candidate_bytes'}:
        bad['parent_source' if mutation == 'parent_bytes' else 'candidate_source'] += '# modified\n'
    with pytest.raises(ValueError):
        source.validate_binding(record, bad)
    assert binding['projection'] == projection  # Failed validation is non-mutating.


def test_quote_paths_do_not_splice_disjoint_windows_into_a_single_source_quote():
    parent, candidate = rewrite_pair(100)
    projection = source.project(parent, candidate, window_chars=48, window_count=3)
    field_values = source.fields(projection)
    assert len(field_values) == 3
    paths = list(field_values)
    assert set(paths) == {f'conclusion_source.windows.{i}.text' for i in range(3)}
    assert projection['windows'][0]['end'] < projection['windows'][1]['start']
    joined = field_values[paths[0]] + field_values[paths[1]]
    assert all(joined not in text for text in field_values.values())
    claim = 'The candidate source changes.'
    evidence = dict(hypothesis=dict(text=claim, reasoning=claim), conclusion_source=projection)
    review = [dict(claim_quote=claim, source_quotes=[dict(field=paths[0], text=joined)],
        assessment='directly_supported', direct_effect='Unverified model interpretation.',
        untested_downstream='Quality remains unknown.')]
    with pytest.raises(ValueError, match='source quote'):
        validate_review_binding(review, evidence)
    review[0]['source_quotes'] = [dict(field=path, text=field_values[path]) for path in paths[:2]]
    assert validate_review_binding(review, evidence) == review


def test_binding_and_validated_projection_are_detached_copies():
    parent, candidate = rewrite_pair(40)
    projection = source.project(parent, candidate, window_chars=48, window_count=3)
    record = pair_record(parent, candidate)
    binding = source.bind(record, parent, candidate, projection)
    original = deepcopy(binding)
    projection['windows'][0]['text'] = 'mutated caller'
    validated = source.validate_binding(record, binding)
    validated['coverage']['omitted_diff_ranges'].clear()
    assert binding == original
