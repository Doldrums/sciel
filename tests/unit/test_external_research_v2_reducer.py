"""Pure reconstruction and provenance rejection over the complete saved chain."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.engine import external_research_v2 as history
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def actual():
    from research_history_fixture import build_history
    return build_history()


def validate(capsule, final):
    return history.validate(capsule, config=final['config'], metric=final['metric'])


@pytest.mark.parametrize('dependency', [
    'hypothesis/models.py', 'hypothesis/prompts.py', 'eval/native_geometry.py'])
def test_transitive_policy_drift_rejects_before_history_reconstruction(actual, monkeypatch, dependency):
    capsule, final = actual
    target = ROOT / 'src/sciel' / dependency
    assert dependency in capsule['policy']['sources']
    original_read = Path.read_bytes

    def changed_bytes(path):
        raw = original_read(path)
        return raw + b'\n# synthetic source-policy drift\n' if path == target else raw

    monkeypatch.setattr(Path, 'read_bytes', changed_bytes)
    with pytest.raises(ValueError, match='reducer policy differs'):
        validate(capsule, final)


def change_journal(capsule, index, phase, change, *, action=None):
    rows = list(map(json.loads, capsule['journals'][index]['journal'].splitlines()))
    selected = [r for r in rows if r.get('phase') == phase and (action is None or r.get('action') == action)]
    assert selected
    change(selected[-1])
    capsule['journals'][index]['journal'] = ''.join(json.dumps(r) + '\n' for r in rows)


def test_complete_actual_chain_rebuilds_exact_state_without_controller_or_external_calls(actual, monkeypatch):
    from sciel.engine.research import ResearchController
    def forbidden(*args, **kwargs):
        pytest.fail('foreign reconstruction instantiated a live controller')
    monkeypatch.setattr(ResearchController, '__init__', forbidden)
    capsule, final = actual
    before = canonical_json(capsule)
    result = validate(capsule, final)
    assert result['knowledge'].state() == final['knowledge']
    assert result['conclusions'] == final['conclusions']
    assert result['comparison_reviews'] == {}
    assert result['imported_ids'] == ['r000001a1', 'r000002a1'] and result['floor'] == 2
    assert result['generation']['r000002a1']['record'] == final['knowledge']['experiments'][1]
    assert canonical_json(capsule) == before


def test_boolean_generation_cost_does_not_equal_numeric_ordinary_cost(actual):
    from sciel.knowledge_base.base import KnowledgeBase
    capsule, final = actual
    events = list(map(json.loads, capsule['journals'][0]['journal'].splitlines()))
    result = next(e for e in events if e.get('action') == 'request_result')
    outcome = next(e for e in events if e.get('phase') == 'outcome')
    result['cost'] = True
    outcome['evidence']['generation_cost'] = 1.0
    builder = history.builder_for_profile(history.PROFILE)
    root = capsule['root_parent']
    kb = KnowledgeBase(**{key: final['config'][key] for key in
                          ('max_lessons', 'max_experiments', 'max_fragments')})
    # Exercise the provenance check directly: no unrelated terminal hash can
    # make this regression test pass merely because True and 1.0 compare equal.
    with pytest.raises(ValueError, match='complete original generation response required'):
        history._generation(events, 'r000001a1', metric=final['metric'], root_parent=root,
            known_sources={root['source_sha256']: builder.source_for()},
            known_parent_ids={root['source_sha256']: {root['id']}}, kb=kb)


@pytest.mark.parametrize('damage', [
    'raw_generation', 'prompt', 'attempt_parent', 'attempt_exposure', 'proof',
    'hypothesis', 'wire_generation', 'conclusion_raw', 'conclusion_finish',
    'conclusion_schema', 'conclusion_observed', 'reflection_context', 'reflection_review_context',
    'reflection_response', 'reflection_changed', 'reflection_applied',
])
def test_changed_journal_claims_reject_even_when_capsule_would_be_rehashed(actual, damage):
    original, final = actual
    capsule = copy.deepcopy(original)
    if damage == 'raw_generation':
        change_journal(capsule, 3, 'circle_generation', lambda e: e.update(response=e['response'] + ' '), action='request_result')
    elif damage == 'prompt':
        change_journal(capsule, 3, 'circle_generation', lambda e: e.update(prompt=e['prompt'] + 'invented'), action='request_start')
    elif damage == 'attempt_parent':
        change_journal(capsule, 3, 'attempt', lambda e: e['evidence'].update(parent_id='invented'))
    elif damage == 'attempt_exposure':
        change_journal(capsule, 3, 'attempt', lambda e: e['evidence'].update(injected_ids=['l000001']))
    elif damage == 'proof':
        change_journal(capsule, 3, 'circle_generation', lambda e: e['proof'].update(target_body=e['proof']['target_body'] + '\n'), action='source_binding')
    elif damage == 'hypothesis':
        change_journal(capsule, 3, 'hypothesis', lambda e: e['evidence']['restart_prediction_binding'].update(response_sha256='0' * 64))
    elif damage == 'wire_generation':
        change_journal(capsule, 3, 'circle_generation', lambda e: e['response_format']['json_schema'].update(strict=False), action='request_start')
    elif damage == 'conclusion_raw':
        change_journal(capsule, 4, 'aux_result', lambda e: e.update(response=e['response'] + ' '))
    elif damage == 'conclusion_finish':
        change_journal(capsule, 4, 'aux_result', lambda e: e.update(finish_reason='length'))
    elif damage == 'conclusion_schema':
        # Alter both the envelope and its claimed digest: known schema bytes
        # still reject the unsupported wire, not merely its stale local hash.
        rows = list(map(json.loads, capsule['journals'][4]['journal'].splitlines()))
        start = next(e for e in rows if e.get('phase') == 'aux_start' and e.get('kind') == 'conclusion')
        start['response_format']['json_schema']['schema']['description'] += ' changed'
        start['auxiliary_schema']['response_format_sha256']['conclusion'] = canonical_sha(start['response_format'])
        capsule['journals'][4]['journal'] = ''.join(json.dumps(r) + '\n' for r in rows)
    elif damage == 'conclusion_observed':
        change_journal(capsule, 4, 'conclusion', lambda e: e.update(verdict_matches_observation=False))
    elif damage == 'reflection_context':
        change_journal(capsule, 5, 'reflection_context', lambda e: e.update(experiment_ids=['r000001a1']))
    elif damage == 'reflection_review_context':
        change_journal(capsule, 5, 'reflection_context', lambda e: e.update(review_context_sha256='0' * 64))
    elif damage == 'reflection_response':
        change_journal(capsule, 5, 'aux_result', lambda e: e.update(response_truncated=True))
    elif damage == 'reflection_changed':
        change_journal(capsule, 5, 'reflection', lambda e: e.update(changed=True))
    elif damage == 'reflection_applied':
        change_journal(capsule, 5, 'reflection', lambda e: e.update(applied={'lesson_ids': ['l000001'], 'retired_ids': []}))
    with pytest.raises(ValueError):
        validate(capsule, final)


@pytest.mark.parametrize('damage', ['omit_generation', 'omit_comparison', 'omit_conclusion', 'omit_reflection',
    'duplicate_operation', 'duplicate_journal', 'future_comparison', 'wrong_endpoint', 'boolean_interval',
    'policy', 'terminal', 'root', 'extra_field'])
def test_causal_operation_and_capsule_gates(actual, damage):
    original, final = actual
    capsule = copy.deepcopy(original)
    if damage.startswith('omit_'):
        kind = damage[5:]
        index = next(i for i, op in reversed(list(enumerate(capsule['operations']))) if op['kind'] == kind)
        capsule['operations'].pop(index)
    elif damage == 'duplicate_operation':
        capsule['operations'].append(copy.deepcopy(capsule['operations'][-1]))
    elif damage == 'duplicate_journal':
        capsule['journals'].append(copy.deepcopy(capsule['journals'][0]))
    elif damage == 'future_comparison':
        capsule['operations'].insert(0, capsule['operations'].pop(7))
    elif damage == 'wrong_endpoint':
        capsule['operations'][-1]['event_seq'] = 6
    elif damage == 'boolean_interval':
        capsule['operations'][-1]['interval'] = True
    elif damage == 'policy':
        capsule['policy']['sources']['knowledge_base/base.py'] = '0' * 64
    elif damage == 'terminal':
        capsule['terminal']['knowledge_sha256'] = '0' * 64
    elif damage == 'root':
        capsule['root_parent']['id'] = 'p99'
    elif damage == 'extra_field':
        capsule['allow_replay'] = True
    with pytest.raises(ValueError):
        validate(capsule, final)


@pytest.mark.parametrize('together', [True, False])
def test_evicted_ordinary_review_retires_at_resolution_before_same_step_reflection(tmp_path, together):
    """An old lesson can retain evidence after its ordinary window is evicted.

    Both same-step and separately completed C/R observe the immediate retirement
    boundary. The lesson retains evidence, without reviving a retired review.
    """
    import test_external_history_v2_engine as fixture
    cfg = fixture.research_config()
    cfg['max_experiments'] = 1
    f = fixture.controller(tmp_path / 'source', cfg)
    for index, score in enumerate((.6, .4)):
        outcome = fixture.CascadeOutcome(fixture.EvalResult(True, score, {}, timing=.01), score, 'full', False)
        assert f.worker.run(fixture.snapshot(f), f.model, lambda source: outcome, base_temp=.2).ok
        record = f.c.knowledge.experiments[-1]
        f.operations.append(('ordinary', record['id']))
        if not together:
            f.c.complete_step(reflect=False)
            f.operations.append(('conclusion', record['id']))
        group = fixture.comparison(record, index)
        f.c.import_comparison(group)
        f.groups.append(group)
        f.operations.append(('comparison', group['id']))
        if index == 1:
            prior_review_present = 'r000001a1' in f.c.conclusions
        f.model.reflection = dict(lessons=[dict(rule=fixture.RULES[index], polarity='positive',
            scope='branch', evidence_ids=['r000001a1'])],
            retire_ids=[] if index == 0 else ['l000001'],
            comparison_reviews=[fixture.review_proposal(group)])
        f.c.complete_step()
        if together:
            f.operations.append(('conclusion', record['id']))
        f.operations.append(('reflection', index + 1))
    f.state = f.c.state()
    f.journal = Path(f.log.path).read_text()
    f.log.close()
    capsule = fixture.capsule_for(f)
    if not together:
        capsule['operations'] = [piece for operation in capsule['operations']
            for piece in ([operation, {'kind': 'step_end'}] if operation['kind'] == 'conclusion' else [operation])]
    rebuilt = history.validate(capsule, config=f.c.config, metric=f.c.metric)
    assert rebuilt['knowledge'].state() == f.state['knowledge']
    assert rebuilt['conclusions'] == f.state['conclusions']
    assert set(rebuilt['conclusions']) == {'r000002a1'}
    contexts = [e for e in map(json.loads, f.journal.splitlines()) if e.get('phase') == 'reflection_context']
    assert prior_review_present is False
    # An unavailable ordinary review still occupies a bound context card;
    # presentation count alone must never be treated as interpretation count.
    # Current live policy reserves the old lesson's complete evidence first.
    assert contexts[-1]['allocation_policy'] == 'sciel-reflection-anchor-allocation/1'
    assert contexts[-1]['selection_epoch'] == 1
    assert contexts[-1]['review_selected_experiment_ids'] == ['r000001a1', 'r000002a1']
    wrong = copy.deepcopy(capsule)
    change_journal(wrong, 0, 'reflection_context',
                   lambda row: row.update(review_context_sha256='0' * 64))
    with pytest.raises(ValueError, match='reflection ordinary review context differs'):
        history.validate(wrong, config=f.c.config, metric=f.c.metric)


def test_exact_archived_typed_schema_stays_supported_but_resealed_changes_do_not(actual):
    capsule, _ = actual
    rows = list(map(json.loads, capsule['journals'][4]['journal'].splitlines()))
    start = next(row for row in rows if row.get('phase') == 'aux_start')
    event = next(row for row in rows if row.get('phase') == 'conclusion')
    start['response_format'] = json.loads(
        (ROOT / 'tests/fixtures/auxiliary_schema/legacy_conclusion.json').read_text())
    start['auxiliary_schema']['response_format_sha256']['conclusion'] = canonical_sha(start['response_format'])
    assert canonical_sha(start['response_format']) == '7275dd52cbf86be666d945fad37e29e89092c8cb784b226fdfeb036f9f945eee'
    assert history._auxiliary(rows, event, 'conclusion', event['experiment_id'])[2] is True
    altered = copy.deepcopy(rows)
    changed = next(row for row in altered if row.get('phase') == 'aux_start')
    enum = changed['response_format']['json_schema']['schema']['properties']['mechanism_review']['items']['properties']['source_quotes']['items']['properties']['field']['enum']
    enum.append('invented_unbound_source')
    changed['auxiliary_schema']['response_format_sha256']['conclusion'] = canonical_sha(changed['response_format'])
    with pytest.raises(ValueError, match='unsupported complete historical auxiliary envelope'):
        history._auxiliary(altered, event, 'conclusion', event['experiment_id'])


@pytest.mark.parametrize('damage', ['resealed_source', 'resealed_projection'])
def test_reconstructed_conclusion_authenticates_retained_source_evidence(tmp_path, damage):
    import test_external_history_v2_engine as fixture
    from sciel.hypothesis.conclusion_source import project
    f = fixture.controller(tmp_path/'source', fixture.research_config())
    outcome = fixture.CascadeOutcome(fixture.EvalResult(True, .6, {}, timing=.01), .6, 'full', False)
    assert f.worker.run(fixture.snapshot(f), f.model, lambda source: outcome, base_temp=.2).ok
    f.c.complete_step(reflect=False)
    f.log.close()
    rows = list(map(json.loads, Path(f.log.path).read_text().splitlines()))
    event = next(row for row in rows if row.get('phase') == 'conclusion')
    ident = event['experiment_id']
    conclusions = {}
    history._conclusion(f.c.knowledge, conclusions, rows, event, config=f.c.config)
    assert conclusions[ident] == f.c.conclusions[ident]
    assert conclusions[ident]['source_evidence'] == event['source_evidence']
    altered = copy.deepcopy(event)
    retained = altered['source_evidence']
    if damage == 'resealed_source':
        retained['candidate_source'] += '\n# altered saved source\n'
        retained['projection'] = project(retained['parent_source'], retained['candidate_source'],
                                         **retained['projection']['selection'])
        expected_error = 'differs from evaluated source identities'
    else:
        projection = retained['projection']
        assert projection['mode'] == 'full'
        before = projection['diff']
        projection['diff'] = ('X' if before[0] != 'X' else 'Y') + before[1:]
        projection['full_diff_sha256'] = hashlib.sha256(projection['diff'].encode()).hexdigest()
        expected_error = 'projection differs from exact source pair'
    with pytest.raises(ValueError, match=expected_error):
        history._conclusion(f.c.knowledge, {}, rows, altered, config=f.c.config)
