"""External repeats teach one typed source without rewriting the original trial.

All draws below are fixed numeric geometry fixtures; no candidate/runtime/model
provider is invoked. Controller auxiliary responses are in-memory fakes.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from sciel.engine import research as research_module
from sciel.engine.research import digest
from sciel.eval import packing_geometry as geometry
from sciel.hypothesis.prompts import reflection_prompt
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.catalogs import canonical_json
from sciel.knowledge_base.paired_comparisons import build_comparison
from sciel.knowledge_base.presentation import expand_reflection_payload

from test_research_measurement_state import (
    begin_fixture, circle_metric, controller, observe,
)


ADAPTER = {'schema': 'fixed-numeric-fixture/1', 'no_execution': True}
LIMITS = dict(candidate_wall_seconds=30., certification_wall_seconds=2.,
    candidate_cpu_seconds=30, candidate_mem_bytes=1024,
    certification_cpu_seconds=2, certification_mem_bytes=1024)


def receipt(source, seed, radius):
    capture = geometry.validate_packing({'centers': [
        [.1 + .15 * (i % 5), .1 + .15 * (i // 5)] for i in range(26)],
        'radii': [radius] * 26})
    certificate = geometry.certify_packing(capture)
    assert capture['valid'] and certificate['status'] == 'certified'
    return dict(schema='sciel-repeat-draw/1', status='valid', source_sha256=source,
        seed=seed, adapter_identity=ADAPTER, limits=LIMITS,
        started_at='2026-09-16T10:00:00+00:00',
        completed_at='2026-09-16T10:00:01+00:00', elapsed_seconds=1.,
        seed_applied=dict(python_random=seed, numpy_legacy=seed,
            python_hash_seed=str(seed), before_candidate_import=True,
            determinism_proven=False),
        raw_sum=capture['sum_radii'], certified_sum=certificate['sum_exact'],
        capture=capture, certificate=certificate)


def group(*, record=None, metric=None, batch='b1', branch='p1', radii=(.02, .02, .02)):
    comparator = digest('different confirmed incumbent; never the forecast parent')
    challenger = record['source_sha256'] if record else digest('external challenger')
    pairs = [dict(seed=seed, comparator=receipt(comparator, seed, .01),
                  challenger=receipt(challenger, seed, radius) if radius is not None else None)
             for seed, radius in zip((71, 72, 73), radii)]
    return build_comparison(namespace='fixed-fixture', batch_id=batch, branch=branch,
        origin_experiment_id=record['id'] if record else None,
        comparator_source_sha256=comparator, challenger_source_sha256=challenger,
        adapter_identity=ADAPTER, limits=LIMITS, pairs=pairs,
        provenance={'fixture': 'saved arrays only'},
        task_metric=metric if metric is not None else circle_metric())


def proposal(ident, polarity='positive'):
    return dict(rule='For these exact sources and three seeds the challenger dominated; '
                'no general effect is established.', polarity=polarity,
                scope='branch', evidence_ids=[ident])


def decoded_context(prompt):
    tail = prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1]
    value, _ = json.JSONDecoder().raw_decode(tail)
    return decode_evidence(value)


def test_import_is_one_idempotent_group_without_search_or_exposure_side_effects():
    c, parent, db, events = controller(metric=circle_metric())
    original, _ = observe(c, parent, db, .6)
    before = c.state()
    old_events = copy.deepcopy(events)
    compared = group(record=original)
    assert c.import_comparison(compared) is True
    assert c.import_comparison(copy.deepcopy(compared)) is False
    after = c.state()
    for key in before:
        if key not in ('knowledge',):
            assert after[key] == before[key]
    assert c.summary()['comparison_count'] == 1
    assert c.knowledge.experiments == before['knowledge']['experiments']
    assert c.knowledge.fragments == before['knowledge']['fragments']
    assert events == old_events  # Durable repeat journal owns import evidence.
    assert original['actual_delta'] == .6 - .5
    assert original['forecast_verdict'] == 'INCONCLUSIVE'
    changed = copy.deepcopy(compared)
    changed['provenance']['fixture'] = 'conflicting same ID'
    snapshot = c.state()
    with pytest.raises(ValueError):
        c.import_comparison(changed)
    assert c.state() == snapshot


@pytest.mark.parametrize('corruption', ['unknown_task', 'foreign_metric', 'origin_source', 'origin_branch'])
def test_import_binding_rejection_is_atomic(corruption):
    c, parent, db, _ = controller(metric=circle_metric())
    record, _ = observe(c, parent, db, .6)
    if corruption == 'unknown_task':
        c, _, _, _ = controller()
    elif corruption == 'foreign_metric':
        # A valid descriptor for a different task must not be silently attached.
        from sciel.hypothesis.measurement import metric_context
        item = group(record=record, metric=metric_context())
    elif corruption == 'origin_source':
        item = group(record={**record, 'source_sha256': digest('other child')})
    elif corruption == 'origin_branch':
        item = group(record=record, branch='foreign')
    if corruption == 'unknown_task':
        item = group(record=record)
    before = c.state()
    with pytest.raises(ValueError, match='metric|origin'):
        c.import_comparison(item)
    assert c.state() == before


def test_next_prompt_uses_whole_comparison_preserving_forecast_parent_and_units():
    c, parent, db, _ = controller(metric=circle_metric())
    observed, _ = observe(c, parent, db, .6)
    compared = group(record=observed)
    c.import_comparison(compared)
    context = c.context(db, parent, 0, 2)
    block = context['paired_comparisons']
    assert block['presented_ids'] == [compared['id']]
    assert len(block['cards']) == 1
    _, prompt = begin_fixture(c, context)
    sent = decoded_context(prompt)
    assert sent['paired_comparisons'] == block
    assert sent['parent_score'] == parent.score
    assert sent['parent_source_sha256'] == digest(parent.code)
    assert sent['metric'] == circle_metric()
    encoded = canonical_json(block)
    assert compared['comparator_source_sha256'] in encoded
    assert 'not_scored' in encoded and 'unit_square_side_length' in encoded
    assert 'not three independent teachings' in prompt
    assert c.pending['r000002a1']['injected_ids'] == []


def test_branch_and_injection_ablation_never_leak_group_cards():
    c, parent, db, _ = controller(metric=circle_metric(), inject_lessons=False)
    item = group()
    c.import_comparison(item)
    ctx = c.context(db, parent, 0, 1)
    assert ctx['paired_comparisons']['cards'] == []
    assert ctx['paired_comparisons']['presented_ids'] == []
    assert item['id'] in ctx['paired_comparisons']['omitted_ids']
    assert ctx['paired_comparisons']['omitted_count'] == 1
    _, prompt = begin_fixture(c, ctx)
    assert item['challenger_source_sha256'] not in prompt
    other, p, d, _ = controller(metric=circle_metric())
    other.import_comparison(group(branch='foreign'))
    assert other.context(d, p, 0, 1)['paired_comparisons']['cards'] == []


def test_whole_comparison_consumes_real_card_memory_before_ordinary_admission(monkeypatch):
    c, parent, db, _ = controller(metric=circle_metric())
    record, _ = observe(c, parent, db, .6)
    ident, = c.knowledge.add_lessons([proposal(record['id'])], allowed_evidence_ids=[record['id']])
    c.import_comparison(group(record=record))
    first = c.context(db, parent, 0, 2)
    whole = canonical_json(first['paired_comparisons'])
    text = c.lesson_text(c.knowledge.lessons[0])
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', len(whole) + len(text) - 1)
    second = c.context(db, parent, 0, 2)
    assert second['paired_comparisons'] == first['paired_comparisons']
    assert second['injected_ids'] == []
    assert {'id': ident, 'reason': 'memory_char_limit', 'chars': len(text)} in second['lesson_presentation']['omitted']


@pytest.mark.parametrize('radii,polarity,accepted', [
    ((.02, .02, .02), 'positive', True),
    ((.005, .005, .005), 'negative', True),
    ((.02, .005, .02), 'positive', False),
    ((.01, .01, .01), 'positive', False),
    ((.02, None, .02), 'positive', False),
])
def test_reflection_sees_group_and_only_full_dominance_can_teach(radii, polarity, accepted):
    c, parent, db, events = controller(metric=circle_metric(), max_aux_calls=1, reflection_interval=1)
    record, _ = observe(c, parent, db, .6)
    item = group(record=record, radii=radii)
    c.import_comparison(item)
    prompts = []
    def generate(prompt, model, **kw):
        prompts.append(prompt)
        return SimpleNamespace(text='REFLECTION_JSON: ' + json.dumps({
            'lessons': [proposal(item['id'], polarity)], 'retire_ids': []}),
            cost=0., model='fake', prompt_tokens=1, completion_tokens=1)
    c.llm = SimpleNamespace(generate=generate, models=['fake'], max_retries=1,
                            choose_model=lambda: 'fake')
    c.charge = lambda value: None
    c.complete_step()
    assert len(prompts) == 1
    raw = json.loads(prompts[0].split('EVIDENCE_JSON:\n', 1)[1])
    inline = expand_reflection_payload(decode_evidence(raw))
    assert item['id'] in canonical_json(inline)
    assert 'forecast=not_scored' in prompts[0]
    event = next(e for e in reversed(events) if e['phase'] == 'reflection')
    assert bool(c.knowledge.lessons) is accepted
    if accepted:
        lesson, = c.knowledge.lessons
        assert lesson['evidence_ids'] == lesson['source_support_ids'] == [item['id']]
        assert not lesson['exposure_snapshots']
        assert event['error'] is None
    else:
        assert event['error'] == 'ValueError'
    assert c.attempts == c.outcomes == 1 and c.aux_calls == 1
    assert c.knowledge.experiments[0]['actual_delta'] == record['actual_delta']


def test_resume_roundtrip_imported_groups_without_counter_fabrication():
    c, parent, db, _ = controller(metric=circle_metric())
    record, _ = observe(c, parent, db, .6)
    item = group(record=record)
    c.import_comparison(item)
    state = json.loads(json.dumps(c.state()))
    clone, p2, db2, _ = controller(metric=circle_metric())
    clone.load_state(state)
    assert clone.state() == state
    assert clone.context(db2, p2, 0, 2) == c.context(db, parent, 0, 2)
    assert clone.import_comparison(item) is False
    assert clone.state() == state
    for count in (True, 0, 2, None):
        broken = copy.deepcopy(state)
        if count is None:
            broken.pop('comparison_count')
        else:
            broken['comparison_count'] = count
        with pytest.raises(ValueError, match='comparison count'):
            clone.load_state(broken)
        assert clone.state() == state


def test_no_group_path_keeps_no_new_state_context_keys():
    c, parent, db, _ = controller()
    assert 'paired_comparisons' not in c.context(db, parent, 0, 1)
    assert 'comparison_count' not in c.state()
    before = c.state()
    c.load_state(copy.deepcopy(before))
    assert c.state() == before


@pytest.mark.parametrize('status', ['eval_failed', 'no_child'])
def test_retained_origin_must_be_an_actual_evaluated_candidate(status):
    c, parent, db, _ = controller(metric=circle_metric())
    record, _ = observe(c, parent, db, 0., status=status, evaluation=status == 'eval_failed')
    before = c.state()
    with pytest.raises(ValueError, match='origin'):
        c.import_comparison(group(record=record))
    assert c.state() == before


def test_opposing_repeats_do_not_rescore_original_forecast_or_relabel_parent():
    from sciel.engine.validation import ValidationResult
    c, parent, db, _ = controller(metric=circle_metric())
    ident, _ = begin_fixture(c, c.context(db, parent, 0, 1))
    code = 'def run_search():\n    return .6\n'
    response = SimpleNamespace(text='HYPOTHESIS_JSON: ' + json.dumps(dict(
        hypothesis='A small controller score change.', predicted_low=-.01,
        predicted_high=.01, reasoning='Uncertain local prediction.', confidence=.5))
        + '\n```python\n' + code + '```', cost=0., model='fake',
        prompt_tokens=1, completion_tokens=1)
    c.proposal(ident, response, code, ValidationResult(code, None, [], None, []))
    original = c.resolve(ident, status='evaluated', score=.6, source=code,
        evaluation=dict(ok=True, score=.6, cases={'validity': 1.}, reached='full',
                        confirmed=False, raw_first_score=.6))
    c.complete_step()  # Aux cap zero; clear the original conclusion queue.
    before = copy.deepcopy(c.knowledge.experiments)
    comparison = group(record=original, radii=(.005, .005, .005))
    c.import_comparison(comparison)
    assert comparison['derived']['outcome'] == 'regressed'
    assert comparison['comparator_source_sha256'] != original['parent_source_sha256']
    assert c.knowledge.experiments == before
    assert original['actual_delta'] > 0 and original['forecast_verdict'] == 'REFUTED'
    assert c.conclusions[ident]['observed_forecast_verdict'] == 'REFUTED'
    assert c.conclusions[ident]['observed_comparison']['measurement']['direction'] == 'improved'
    assert c.queue == []


def test_source_group_survives_window_pruning_and_resume_through_lesson_snapshot():
    c, parent, db, _ = controller(metric=circle_metric(), max_experiments=1)
    record, _ = observe(c, parent, db, .6)
    first = group(record=record)
    c.import_comparison(first)
    c.knowledge.add_lessons([proposal(first['id'])], allowed_evidence_ids=[first['id']])
    c.import_comparison(group(batch='b2'))
    assert first['id'] not in {g['id'] for g in c.knowledge.comparisons}
    assert c.knowledge._evidence()[first['id']] == first
    before = c.state()
    clone, _, _, _ = controller(metric=circle_metric(), max_experiments=1)
    clone.load_state(json.loads(json.dumps(before)))
    assert clone.state() == before
    assert clone.import_comparison(first) is False
    assert clone.state() == before


@pytest.mark.parametrize('mismatch', ['metric', 'origin'])
def test_valid_standalone_group_cannot_bypass_controller_restore_binding(mismatch):
    from sciel.hypothesis.measurement import metric_context
    c, parent, db, _ = controller(metric=circle_metric())
    record, _ = observe(c, parent, db, .6)
    item = (group(record=record, metric=metric_context()) if mismatch == 'metric'
            else group(record={**record, 'source_sha256': digest('unrelated evaluated bytes')}))
    # Standalone KB intentionally supports unknown task descriptors; controller
    # restoration must apply the stronger live task/origin boundary atomically.
    c.knowledge.record_comparison(item)
    source = c.state()
    clone, _, _, _ = controller(metric=circle_metric())
    before = clone.state()
    with pytest.raises(ValueError, match='metric|origin'):
        clone.load_state(source)
    assert clone.state() == before


def test_invented_origin_rejected_but_previously_recorded_pruned_origin_is_known():
    c, parent, db, _ = controller(metric=circle_metric(), max_experiments=1)
    original, _ = observe(c, parent, db, .4)  # No positive fragment retains it.
    item = group(record=original)
    stranger = group(record={**original, 'id': 'r999999a1'})
    before = c.state()
    with pytest.raises(ValueError, match='never recorded'):
        c.import_comparison(stranger)
    assert c.state() == before
    observe(c, parent, db, .3)
    assert original['id'] not in c.knowledge._evidence()
    assert original['id'] in c.knowledge._seen
    assert c.import_comparison(item) is True
    clone, _, _, _ = controller(metric=circle_metric(), max_experiments=1)
    clone.load_state(json.loads(json.dumps(c.state())))
    assert clone.state() == c.state()
