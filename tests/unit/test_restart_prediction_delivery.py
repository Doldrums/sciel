"""Authored saved-array observations with explicitly synthetic model forecasts.

Only inert receipts and controller responses are used. No candidate, helper,
optimizer, evaluator or provider executes. Ordinary records bind each forecast
before resolution; presentation never grants causal or experimental credit.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from sciel.engine.prompt.build import build
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Snapshot
from sciel.hypothesis import restart_predictions as predictions
from sciel.hypothesis.prompts import initializer_block, restart_block
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.memory_references import RULES as SHOWN_RULES, SCHEMA as SHOWN_SCHEMA
from sciel.knowledge_base.paired_comparisons import UNKNOWN_RECEIPT, build_comparison
from sciel.knowledge_base.restart_prediction_context import (
    PREFIX, block, build_context, validate_context,
)

import test_real_restart_memory_delivery as existing
from test_real_motion_memory_delivery import location  # Authored saved-array fixture.
from restart_memory_fixtures import LOCATION_SEED


SPEC = {'schema': predictions.SCHEMA, 'expected': {name: True for name in predictions.NAMES}}


def controller_fixture(metric, *, enabled=True, inject=True):
    old, parent, db, sink, events, charges = existing._controller(metric, inject=inject)
    config = copy.deepcopy(old.config)
    config['restart_predictions'] = enabled
    # Direct controller fixture only; no CircleGeneration/Engine dispatch.
    controller = ResearchController(config, sink,
        SimpleNamespace(research=lambda **event: events.append(copy.deepcopy(event))),
        charges.append, metric=metric)
    return controller, parent, db, sink, events, charges


def history_with_synthetic_forecast(controller, parent, db, group):
    records = []
    for index, score in enumerate((.6, .4)):
        ident, _, _ = existing._begin(controller, parent, db)
        source = (existing.PARENT_CODE + '# INERT positive ordinary fixture.\n' if index == 0
                  else group['source_pair']['challenger'])
        text = ('Inert fixture without a hypothesis.' if index == 0 else
            'HYPOTHESIS_JSON: ' + json.dumps(dict(hypothesis=existing.CLAIM,
                predicted_low=0., predicted_high=.2,
                reasoning='RETROSPECTIVE INERT FORECAST FIXTURE; no historical model made this prediction.',
                confidence=.5, restart_predictions=copy.deepcopy(SPEC))))
        controller.proposal(ident, SimpleNamespace(text=text, cost=0., prompt_tokens=0,
            completion_tokens=0), source, SimpleNamespace(code=source))
        if index:
            # Binding is committed before even this synthetic resolve.
            binding = controller.pending[ident]['restart_prediction_binding']
            assert binding['spec_sha256'] == canonical_sha(SPEC)
            assert binding['response_sha256'] == digest(text)
            assert binding['source_sha256'] == binding['proposed_sha256'] == digest(source)
        records.append(controller.resolve(ident, status='evaluated', source=source, score=score,
            evaluation=dict(ok=True, score=score, cases={'sum_radii':score*2.635,
                'validity':1., 'retrospective_fixture_only':1.}, reached='full',
                confirmed=False, raw_first_score=score)))
    controller.complete_step(reflect=False)
    lesson_id, = controller.knowledge.add_lessons([dict(rule=existing.RULE, polarity='positive',
        scope='branch', evidence_ids=[records[0]['id']])], allowed_evidence_ids=[records[0]['id']])
    # KB canonicalization adds explicit legacy child/error nulls. Identity must
    # name the authoritative stored record, not resolve()'s pre-normalized copy.
    return [controller._prediction_records()[record['id']] for record in records], lesson_id


def import_bound_group(controller, original, origin):
    keys = ('batch_id','branch','comparator_source_sha256','challenger_source_sha256',
            'adapter_identity','limits','pairs','source_pair')
    group = build_comparison(**{key:copy.deepcopy(original[key]) for key in keys},
        origin_experiment_id=origin['id'], task_metric=controller.metric,
        namespace='retrospective_model_forecast_fixture:'+canonical_sha(original),
        provenance={**original['provenance'], 'test_context_only':True,
                    'retrospective_synthetic_forecast':True,
                    'original_group_sha256':canonical_sha(original)})
    for key in keys:
        assert canonical_json(group[key]) == canonical_json(original[key])
    assert group['derived'] == original['derived']
    before = (controller.attempts,controller.outcomes,controller.aux_calls,controller.aux_cost)
    assert controller.import_comparison(group)
    assert (controller.attempts,controller.outcomes,controller.aux_calls,controller.aux_cost) == before
    return group


def assert_predicted_context(value, group, record):
    assert value['observables'] == list(predictions.NAMES)
    assert value['omitted'] == []
    card, = value['cards']
    assert card['origin_id'] == record['id'] and card['origin_sha256'] == canonical_sha(record)
    assert card['group_sha256'] == canonical_sha(group)
    assert card['binding'] == record['restart_prediction_binding']
    assert card['expected'] == [True,True,True]
    rows = [dict(zip(value['columns'], row, strict=True)) for row in card['rows']]
    # The authored fixture has one measured seed and two undispatched slots.
    assert [row['seed'] for row in rows] == [LOCATION_SEED, 17, 1657835957]
    assert [row['seed'] for row in rows] == [pair['seed'] for pair in group['pairs']]
    first = rows[0]
    assert first['receipt_sha256'] == canonical_sha(group['pairs'][0]['challenger'])
    assert first['result_sha256'] is not None
    assert first['observed'] == [True,False,False]
    assert first['verdicts'] == ['CONFIRMED','REFUTED','REFUTED']
    assert first['coverage'] == {'retained':16,'accepted':3,'unknown_acceptance':0,'omitted':0}
    assert first['unavailable_reason'] is None
    assert all(row['observed'] == [None]*3 and row['verdicts'] == ['INCONCLUSIVE']*3
        and row['receipt_sha256'] is None and row['result_sha256'] is None
        and row['coverage'] is None and row['unavailable_reason'] == 'receipt_not_dispatched'
        for row in rows[1:])
    return rows


def test_actual_motion_forecasts_reach_reflection_generation_and_identical_restore(location):
    original_bytes = canonical_json(location['group'])
    controller,parent,db,sink,events,charges = controller_fixture(location['metric'])
    records,lesson_id = history_with_synthetic_forecast(controller,parent,db,location['group'])
    record = records[1]
    conclusion, = sink.conclusion_prompts
    assert PREFIX not in conclusion and 'restart_predictions' in conclusion
    assert record['forecast_verdict'] == 'REFUTED'  # Native scalar forecast stays separate.
    group = import_bound_group(controller,location['group'],record)
    knowledge_before = canonical_json(controller.knowledge.state())
    controller.complete_step()
    reflection, = sink.prompts
    assert 'existential over at most the first 16 scheduled calls' in reflection
    assert 'matched center OR radius residual >1e-6' in reflection
    assert 'terminal-minus-optimizer-entered incumbent sum using exact represented binary64' in reflection
    assert 'A true witness wins other unknowns' in reflection
    reflected = existing._parse(reflection,PREFIX)
    assert_predicted_context(reflected,group,record)
    assert canonical_json(controller.knowledge.state()) == knowledge_before
    assert charges == [0.,0.] and controller.aux_calls == 2
    saved = json.loads(canonical_json(controller.state()))
    restored,p2,db2,sink2,events2,charges2 = controller_fixture(location['metric'])
    restored.load_state(saved)
    assert canonical_json(restored.state()) == canonical_json(saved)
    restored.complete_step()
    assert sink2.prompts == sink2.conclusion_prompts == charges2 == []
    ident,prompt,context = existing._begin(controller,parent,db)
    ident2,prompt2,context2 = existing._begin(restored,p2,db2)
    assert ident == ident2 and prompt == prompt2 and canonical_json(context) == canonical_json(context2)
    emitted = existing._parse(prompt,PREFIX)
    assert emitted == context['restart_predictions']
    assert_predicted_context(emitted,group,record)
    shown_chars = len(SHOWN_RULES) if context['paired_comparisons']['schema'] == SHOWN_SCHEMA else 0
    optional_chars = (context['lesson_presentation']['rendered_chars']
        + len(canonical_json(context['paired_comparisons'])) + shown_chars
        + (len(canonical_json(context['comparison_facts'])) if 'comparison_facts' in context else 0)
        + len(restart_block(context.get('restart_observations')))
        + len(block(emitted)) + len(initializer_block(context.get('initializer_evidence')))
        + (len(canonical_json(context['conclusion_feedback'])) if 'conclusion_feedback' in context else 0))
    assert optional_chars <= 32000
    for current in (controller,restored):
        assert canonical_json(current.knowledge.state()) == knowledge_before
        assert len(current.knowledge.experiments) == 2 and len(current.knowledge.comparisons) == 1
        lesson, = current.knowledge.lessons
        assert lesson['id'] == lesson_id and lesson['source_support_ids'] == [records[0]['id']]
        assert lesson['exposure_ids'] == []  # Prepared intent is not model use.
        assert current.knowledge.comparisons[0]['derived'] == location['group']['derived']
    assert canonical_json(location['group']) == original_bytes


@pytest.mark.parametrize('enabled,inject',[(False,True),(True,False)])
def test_disabled_forecast_delivery_preserves_real_receipts_without_exposure(location,enabled,inject):
    controller,parent,db,sink,_,_ = controller_fixture(location['metric'],enabled=enabled,inject=inject)
    records,_ = history_with_synthetic_forecast(controller,parent,db,location['group'])
    group = import_bound_group(controller,location['group'],records[1])
    before = canonical_json(controller.knowledge.state())
    _,prompt,context = existing._begin(controller,parent,db)
    assert PREFIX not in prompt and 'restart_predictions' not in context
    assert canonical_json(controller.knowledge.state()) == before
    assert canonical_json(controller.knowledge.comparisons[0]) == canonical_json(group)
    assert all(not lesson['exposure_ids'] for lesson in controller.knowledge.lessons)
    assert len(sink.conclusion_prompts) == 1 and sink.prompts == []


@pytest.mark.parametrize('change,reason',[
    ('missing_origin','original_hypothesis_unavailable'),
    ('missing_spec','predictions_not_recorded'),
    ('missing_binding','pre_evaluation_binding_unavailable'),
    ('source_mismatch','original_source_binding_mismatch'),
    ('repaired_source','original_source_binding_mismatch'),
    ('spec_mismatch','original_source_binding_mismatch'),
])
def test_missing_or_inconsistent_origin_cannot_borrow_archived_outcomes(location,change,reason):
    controller,parent,db,_,_,_ = controller_fixture(location['metric'])
    records,_ = history_with_synthetic_forecast(controller,parent,db,location['group'])
    group = import_bound_group(controller,location['group'],records[1])
    origins = copy.deepcopy(controller._prediction_records())
    origin = origins[records[1]['id']]
    if change == 'missing_origin':
        origins.pop(origin['id'])
    elif change == 'missing_spec':
        origin['hypothesis'].pop('restart_predictions')
    elif change == 'missing_binding':
        origin.pop('restart_prediction_binding')
    elif change == 'source_mismatch':
        origin['source_sha256'] = '0'*64
    elif change == 'repaired_source':
        origin['source_repaired'] = True
    else:
        origin['hypothesis']['restart_predictions']['expected'][predictions.NAMES[0]] = False
    context = build_context([group],origins)
    card, = context['cards']
    assert card['unavailable_reason'] == reason and 'rows' not in card
    assert validate_context(context,groups=[group],records=origins) == context


def test_forecast_context_rederives_receipts_instead_of_trusting_self_rehashed_results(location):
    controller,parent,db,_,_,_ = controller_fixture(location['metric'])
    records,_ = history_with_synthetic_forecast(controller,parent,db,location['group'])
    group = import_bound_group(controller,location['group'],records[1])
    origins = controller._prediction_records()
    context = build_context([group],origins)
    forged = copy.deepcopy(context)
    observed_index = forged['columns'].index('observed')
    forged['cards'][0]['rows'][0][observed_index] = [True,True,True]
    forged['context_sha256'] = canonical_sha({k:v for k,v in forged.items() if k!='context_sha256'})
    with pytest.raises(ValueError,match='differs from bound evidence'):
        validate_context(forged,groups=[group],records=origins)
    assert build_context([group],origins,max_chars=1) is None
    # A budget below the full card keeps an explicit whole-group omission.
    limited = build_context([group],origins,max_chars=len(canonical_json(context))-1)
    assert limited['cards'] == [] and limited['omitted'] == [group['id']]
    assert len(canonical_json(limited)) <= limited['max_chars']


def test_unknown_dispatch_never_becomes_a_false_forecast_or_valid_quality(location):
    controller,parent,db,_,_,_ = controller_fixture(location['metric'])
    records,_ = history_with_synthetic_forecast(controller,parent,db,location['group'])
    group = import_bound_group(controller,location['group'],records[1])
    pairs = copy.deepcopy(group['pairs'])
    pairs[0]['challenger'] = copy.deepcopy(UNKNOWN_RECEIPT)
    keys = ('namespace','batch_id','branch','origin_experiment_id','comparator_source_sha256',
            'challenger_source_sha256','adapter_identity','limits','source_pair','task_metric')
    unknown = build_comparison(**{key:copy.deepcopy(group[key]) for key in keys},pairs=pairs,
        provenance={'test_context_only':True,'synthetic_unknown_dispatch':True})
    context = build_context([unknown],controller._prediction_records())
    card, = context['cards']
    first = dict(zip(context['columns'],card['rows'][0],strict=True))
    assert first['observed'] == [None]*3 and first['verdicts'] == ['INCONCLUSIVE']*3
    assert first['unavailable_reason'] == 'dispatch_unknown' and first['result_sha256'] is None
    assert unknown['derived']['outcome'] == 'incomplete'
    assert unknown['derived']['all_valid'] is False and unknown['derived']['all_no_loss'] is False
    assert group['pairs'][0]['challenger'] == location['group']['pairs'][0]['challenger']


def test_generation_rejects_self_rehashed_false_results_before_state_mutation(location):
    controller,parent,db,sink,events,charges = controller_fixture(location['metric'])
    records,_ = history_with_synthetic_forecast(controller,parent,db,location['group'])
    import_bound_group(controller,location['group'],records[1])
    context = controller.context(db,parent,0,controller.attempts+1)
    forged = context['restart_predictions']
    observed_index = forged['columns'].index('observed')
    forged['cards'][0]['rows'][0][observed_index] = [True,True,True]
    forged['context_sha256'] = canonical_sha({k:v for k,v in forged.items() if k!='context_sha256'})
    snapshot = Snapshot(0,'diff','fake',controller.attempts,parent.id,parent.code,parent.score,
        research_context=context,kb_lessons=context['lesson_texts'],task_prompt='Inert tamper fixture.')
    before = canonical_json(controller.state())
    event_count = len(events)
    with pytest.raises(ValueError,match='differs from bound evidence'):
        controller.begin(snapshot,0,'diff',build('diff',snapshot))
    assert canonical_json(controller.state()) == before and len(events) == event_count
    assert charges == [0.] and len(sink.conclusion_prompts) == 1 and sink.prompts == []
