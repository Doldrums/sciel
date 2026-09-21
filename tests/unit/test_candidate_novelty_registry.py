"""Whole-history code novelty and retry eligibility; no candidate execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.engine.novelty import CandidateNoveltyRegistry, REPRESENTATION


SOURCE = 'def run(x):\n    return x + 1\n'
FORMATTED = '# a new presentation\ndef run( x ):\n\treturn x+1 # same AST\n'


def resign(state):
    body = {key: value for key, value in state.items() if key != 'sha256'}
    raw = json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    state['sha256'] = hashlib.sha256(raw.encode()).hexdigest()


def successful():
    registry = CandidateNoveltyRegistry()
    registry.register(SOURCE, 'evaluation-1', island=0, evaluation_ok=True)
    registry.bind_program(SOURCE, 'p143', 0)
    return registry


def test_exact_then_ast_equality_use_program_alias_and_inspection_is_pure():
    registry = successful()
    before = registry.state()
    exact = registry.inspect(SOURCE)
    structural = registry.inspect(FORMATTED)
    assert exact['match_kind'] == 'sha256' and exact['accept'] is False
    assert structural['match_kind'] == 'ast' and structural['accept'] is False
    assert exact['source_sha256'] != structural['source_sha256']
    assert exact['ast_sha256'] == structural['ast_sha256']
    assert structural['matched_program_id'] == 'p143'
    assert structural['matched_evaluation_id'] == 'evaluation-1'
    assert structural['neighbors'][0]['ast_diff'] == ''
    assert registry.state() == before


def test_exact_and_structural_matches_skip_all_history_embedding_search(monkeypatch):
    from sciel.engine import novelty
    registry = successful()
    for index in range(20):
        registry.register(f'other = {index}\n', f'other-{index}', evaluation_ok=True)

    def unnecessary(*_, **__):
        pytest.fail('known SHA/AST match must not scan all embeddings')

    monkeypatch.setattr(novelty.heapq, 'nlargest', unnecessary)
    result = registry.inspect(FORMATTED)
    assert result['match_kind'] == 'ast' and result['neighbors_scope'] == 'identity_match'
    monkeypatch.setattr(novelty, '_structural_ast', unnecessary)
    monkeypatch.setattr(novelty, '_descriptor', unnecessary)
    result = registry.inspect(SOURCE)
    assert result['match_kind'] == 'sha256' and result['neighbors'][0]['similarity'] == 1.


def test_historical_p143_p175_fixture_is_structural_duplicate():
    fixtures = Path(__file__).resolve().parents[1] / 'fixtures' / 'candidate_novelty'
    parent = (fixtures / 'p143.py').read_text()
    child = (fixtures / 'p175.py').read_text()
    registry = CandidateNoveltyRegistry()
    registry.register(parent, 'evaluation-143', island=0, evaluation_ok=True)
    registry.bind_program(parent, 'p143', 0)
    result = registry.inspect(child)
    assert result['match_kind'] == 'ast' and result['accept'] is False
    assert result['source_sha256'] != registry.inspect(parent)['source_sha256']
    assert result['matched_program_id'] == 'p143'
    assert result['neighbors'][0]['ast_diff'] == ''


@pytest.mark.parametrize('before,after', [
    ('return 1', 'return 1.0'), ('return 1', 'return True'),
    ('return 1', 'return 2'), ('return "alpha"', 'return "beta"'),
    ('return b"alpha"', 'return b"beta"'),
    ('"""First docstring"""\n    return 1', '"""Other docstring"""\n    return 1'),
])
def test_constants_strings_bytes_and_docstrings_are_not_erased(before, after):
    registry = CandidateNoveltyRegistry()
    previous = 'def run():\n    ' + before + '\n'
    candidate = 'def run():\n    ' + after + '\n'
    registry.register(previous, 'old', evaluation_ok=True)
    result = registry.inspect(candidate)
    assert result['accept'] is True and result['match_kind'] == 'embedding'
    assert result['ast_sha256'] != registry.inspect(previous)['ast_sha256']
    difference = result['neighbors'][0]['ast_diff']
    assert difference.startswith('--- old\n+++ candidate\n')
    assert any(line.startswith('-') and not line.startswith('---') for line in difference.splitlines())
    assert any(line.startswith('+') and not line.startswith('+++') for line in difference.splitlines())


def test_type_ignore_line_numbers_are_locations_not_structural_changes():
    registry = CandidateNoveltyRegistry()
    original = 'value = 1  # type: ignore[attr-defined]\n'
    moved = '# added comment\n\nvalue=1  # type: ignore[attr-defined]\n'
    registry.register(original, 'p1', evaluation_ok=True)
    assert registry.inspect(moved)['match_kind'] == 'ast'
    assert registry.inspect(moved)['accept'] is False


def test_embedding_similarity_never_rejects_even_a_very_close_program():
    registry = CandidateNoveltyRegistry(neighbors=1)
    source = 'def run(x):\n' + ''.join(f'    value{i} = x + {i}\n' for i in range(100)) + '    return value99\n'
    registry.register(source, 'p1', evaluation_ok=True)
    result = registry.inspect(source.replace('value99 = x + 99', 'value99 = x + 100'))
    assert result['neighbors'][0]['similarity'] > .98
    assert result['accept'] is True and result['reason'] == 'code_similarity_only'
    assert result['representation'] == REPRESENTATION


def test_all_islands_and_old_records_remain_searchable_without_active_population():
    registry = successful()
    registry.bind_program(SOURCE, 'p144', island=1)
    for index in range(12):
        registry.register(f'result = {index}\n', f'other-{index}', island=2, evaluation_ok=True)
    result = registry.inspect(FORMATTED)
    assert result['accept'] is False and result['matched_program_id'] == 'p144'
    nearest = result['neighbors'][0]
    assert nearest['aliases'] == ['p143', 'p144'] and nearest['islands'] == [0, 1]
    assert len(registry.state()['records']) == 13


def test_failed_evaluation_allows_explicitly_labeled_remeasurement_then_success_blocks():
    registry = CandidateNoveltyRegistry()
    registry.register(SOURCE, 'first', evaluation_ok=False)
    retry = registry.inspect(SOURCE)
    assert retry['accept'] and retry['remeasurement']
    assert retry['reason'] == 'previous_evaluation_failed'
    assert retry['remeasurement_of'] == ['first']
    assert registry.inspect(FORMATTED)['remeasurement'] is True
    registry.register(SOURCE, 'second')
    pending = registry.inspect(SOURCE)
    assert pending['accept'] is False and pending['remeasurement'] is False
    assert pending['reason'] == 'previous_evaluation_pending_or_unknown'
    assert registry.inspect(SOURCE, remeasurement=True)['accept'] is False
    registry.register(SOURCE, 'second', evaluation_ok=True)
    registry.bind_program(SOURCE, 'p2', 1)
    assert registry.inspect(SOURCE)['accept'] is False
    deliberate = registry.inspect(SOURCE, remeasurement=True)
    assert deliberate['accept'] and deliberate['remeasurement']
    assert deliberate['reason'] == 'explicit_remeasurement'
    assert deliberate['matched_program_id'] == 'p2'


def test_failed_exact_version_cannot_hide_successful_ast_equivalent():
    registry = successful()
    registry.register(FORMATTED, 'failed-variant', evaluation_ok=False)
    result = registry.inspect(FORMATTED)
    assert result['accept'] is False and result['match_kind'] == 'ast'
    assert result['matched_program_id'] == 'p143'


def test_identity_and_terminal_evaluation_cannot_be_rewritten():
    registry = successful()
    before = registry.state()
    with pytest.raises(ValueError, match='different source'):
        registry.register('other = 1', 'evaluation-1', evaluation_ok=True)
    with pytest.raises(ValueError, match='completed evaluation'):
        registry.register(SOURCE, 'evaluation-1', evaluation_ok=False)
    assert registry.state() == before
    registry.register(SOURCE, 'evaluation-2', evaluation_ok=True)
    with pytest.raises(ValueError, match='another evaluation'):
        registry.bind_program(SOURCE, 'p143')
    with pytest.raises(ValueError, match='before its source'):
        registry.bind_program('not_registered = 1', 'p4')


def test_json_checkpoint_round_trip_preserves_full_history_aliases_and_results():
    registry = successful()
    registry.register('value = 3\n', 'failed', island=2, evaluation_ok=False)
    registry.register(SOURCE, 'reevaluation', island=1, evaluation_ok=True)
    registry.bind_program(SOURCE, 'p3', 1)
    registry.register('value = 4\n', 'pending')
    saved = json.loads(json.dumps(registry.state()))
    restored = CandidateNoveltyRegistry()
    restored.load_state(saved)
    assert restored.state() == saved
    for code in (SOURCE, FORMATTED, 'value = 3\n', 'value = 4\n', 'print("new")\n'):
        assert restored.inspect(code) == registry.inspect(code)
    saved['records'][0]['aliases'].append('external mutation')
    assert restored.state() == registry.state()


@pytest.mark.parametrize('mutation', ['schema', 'identity', 'source', 'ast', 'alias', 'status', 'orphan', 'extra'])
def test_corrupted_checkpoint_is_rejected_atomically_even_with_recomputed_outer_hash(mutation):
    registry = successful()
    baseline = registry.state()
    state = deepcopy(baseline)
    if mutation == 'schema':
        state['schema'] = 'future-schema'
    elif mutation == 'identity':
        state['identity']['representation'] = 'score-trained-embedding'
    elif mutation == 'source':
        state['sources'][0]['code'] += '# changed bytes'
    elif mutation == 'ast':
        state['sources'][0]['ast_sha256'] = '0' * 64
    elif mutation == 'alias':
        state['records'][0]['aliases'].append(state['records'][0]['id'])
    elif mutation == 'status':
        state['records'][0]['evaluation_ok'] = 0
    elif mutation == 'orphan':
        state['records'] = []
    else:
        state['records'][0]['score'] = 100
    resign(state)
    with pytest.raises(ValueError):
        registry.load_state(state)
    assert registry.state() == baseline


def test_state_hash_and_neighbor_policy_are_checked():
    state = successful().state()
    state['records'][0]['aliases'].append('changed-without-resigning')
    with pytest.raises(ValueError, match='hash mismatch'):
        CandidateNoveltyRegistry().load_state(state)
    with pytest.raises(ValueError, match='identity mismatch'):
        CandidateNoveltyRegistry(neighbors=1).load_state(successful().state())


def test_deep_ast_and_large_diff_are_bounded_without_discarding_constants():
    registry = CandidateNoveltyRegistry(neighbors=1)
    source = 'value = ' + '+'.join('1' for _ in range(1200)) + '\n'
    registry.register(source, 'deep', evaluation_ok=True)
    result = registry.inspect('# no new AST\n' + source)
    assert result['ast_error'] is None and result['match_kind'] == 'ast'
    large = CandidateNoveltyRegistry(neighbors=1)
    large.register('value = ' + repr('a' * 15000), 'large', evaluation_ok=True)
    difference = large.inspect('value = ' + repr('b' * 15000))['neighbors'][0]
    assert difference['ast_diff_truncated'] is True
    assert len(difference['ast_diff']) <= 12000


def test_source_is_never_executed_and_syntax_validation_remains_separate(tmp_path):
    marker = tmp_path / 'never-created'
    code = f'open({str(marker)!r}, "w").write("bad")\n'
    registry = CandidateNoveltyRegistry()
    registry.register(code, 'inert', evaluation_ok=True)
    assert registry.inspect(code)['accept'] is False
    assert not marker.exists()
    invalid = registry.inspect('def broken(')
    assert invalid['accept'] is True and invalid['ast_sha256'] is None
    assert invalid['ast_error'] == 'SyntaxError'
