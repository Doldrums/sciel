"""Authored saved-array motion through full Research prompts; no numerical replay.

Controllers import immutable inert groups with explicit provenance. Fake auxiliary
replies exercise delivery and restore without providers, candidates or evaluators.
"""
import json
from pathlib import Path

import pytest

from sciel.engine.research import digest
from sciel.hypothesis import restart_observations as observations
from sciel.hypothesis import restart_motion as motion
from sciel.hypothesis import restart_motion_observations as enriched
from sciel.hypothesis.prompts import initializer_block, restart_block
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.memory_references import RULES as SHOWN_RULES, SCHEMA as SHOWN_SCHEMA
from sciel.knowledge_base.restart_context import PREFIX, expand_context

import test_real_restart_memory_delivery as existing
from test_real_restart_memory_delivery import historical
from restart_memory_fixtures import (
    CENTER_DRIFTS, LEGACY_CENTER_DRIFT, LEGACY_RADIUS_DRIFT, location_dataset,
)

MOTION_FIELDS = tuple(motion.COLUMNS[2:])


@pytest.fixture(scope='module')
def location():
    return location_dataset()


def dataset(kind, location, historical):
    if kind == 'location':
        return location
    name = 'pair_vs_single'; group = historical['groups'][name]
    return dict(metric=historical['metric'], group=group, motions=None, first_seed=existing.FIRST_SEED,
        views={role:historical['views'][name,existing.FIRST_SEED,role] for role in ('comparator','challenger')})


def assert_delivered(prompt, group, saved):
    wire = existing._parse(prompt, PREFIX)
    expanded = expand_context(wire)
    assert wire['rendered_chars'] == len(canonical_json(wire)) <= 12000
    assert expanded['selected_count'] == 6
    assert expanded['groups'][0]['group_sha256'] == canonical_sha(group)
    decoded = {}
    for role in ('comparator','challenger'):
        card, = [c for c in expanded['cards'] if c['seed'] == saved['first_seed'] and c['role'] == role]
        view = card['observation_view']; value = view['observation']
        assert view['included'] and len(canonical_json(view)) <= 8192
        assert value['schema'] == enriched.SCHEMA == 'sciel-restart-observation/3'
        base = enriched.base_observation(value)
        assert canonical_json(base) == canonical_json(saved['views'][role]['observation'])
        assert canonical_json(observations.observation_view(base)) == canonical_json(saved['views'][role])
        receipt = group['pairs'][0][role]
        assert card['source_sha256'] == group[role+'_source_sha256']
        assert card['receipt_sha256'] == canonical_sha(receipt)
        assert value['bindings'] == base['bindings'] and value['receipt_sha256'] == canonical_sha(receipt)
        cases = observations.observation_cases(value)
        assert cases == enriched.observation_cases(value) and len(cases) == 16
        if saved['motions'] is not None:
            expected = {case['restart_index']:case for case in motion.motion_cases(saved['motions'][role])}
            for case in cases:
                assert {k:case[k] for k in MOTION_FIELDS} == {k:expected[case['restart_index']][k] for k in MOTION_FIELDS}
        decoded[role] = cases
    coverage = expanded['cards']+expanded['omissions']
    assert {(c['seed'],c['role']) for c in coverage} == {
        (p['seed'],role) for p in group['pairs'] for role in ('comparator','challenger')}
    return wire, decoded


@pytest.mark.parametrize('kind',['location','median'])
def test_real_motion_reaches_reflection_and_identical_next_prompt_after_json_restore(kind,location,historical):
    saved = dataset(kind,location,historical); original = saved['group']
    original_bytes = canonical_json(original)
    controller,parent,db,sink,events,charges = existing._controller(saved['metric'])
    records,lesson_id = existing._history(controller,parent,db)
    assert len(sink.conclusion_prompts) == 1 and PREFIX not in sink.conclusion_prompts[0]
    group = existing._import_group(controller,original)
    kb_before = canonical_json(controller.knowledge.state())
    with pytest.raises(ValueError):
        controller.knowledge.add_lessons([dict(rule='Unjustified motion quality claim',polarity='positive',
            scope='branch',evidence_ids=[group['id']])],allowed_evidence_ids=[group['id']])
    assert canonical_json(controller.knowledge.state()) == kb_before
    controller.complete_step()
    reflection, = sink.prompts
    reflection_wire,cases = assert_delivered(reflection,group,saved)
    assert charges == [0.,0.] and controller.reflected_through == 2
    reflected = next(e for e in events if e['phase']=='restart_observation_presentation' and e['role']=='reflection')
    assert reflected['context'] == reflection_wire and reflected['prompt_sha256'] == digest(reflection)
    assert reflected['quality_changed'] is False and reflected['delivery']=='prompt_prepared_not_model_use'
    if kind == 'location':
        accepted = [c for c in cases['challenger'] if c['helper_accepted'] is True]
        assert [c['restart_index'] for c in accepted] == [16,32,48]
        assert all(c['effective_locations_changed'] is True and c['match_status']=='matched' for c in accepted)
        assert [c['match_center_max'] for c in accepted] == list(CENTER_DRIFTS.values())
        assert all(c['success'] is True and c['status']==0 and c['result_tolerance_valid'] is True for c in accepted)
        assert all(c['phase_sums'][4]==c['phase_sums'][2] and c['phase_improvements'][4]==c['phase_improvements'][2] for c in accepted)
        pair = group['derived']['pairs'][0]
        assert pair['raw_delta'] < 0 and pair['certified_delta']['numerator'] > 0 and not pair['no_loss']
    else:
        case = next(c for c in cases['challenger'] if c['restart_index']==32)
        assert case['effective_locations_changed'] is False and case['match_status']=='matched'
        assert case['match_center_max'] == LEGACY_CENTER_DRIFT
        assert case['match_radius_max'] == LEGACY_RADIUS_DRIFT
        assert all(c.get('success') is None and c.get('status') is None for c in cases['challenger'])
        assert 'success' not in enriched.base_observation(
            expand_context(reflection_wire)['cards'][0]['observation_view']['observation'])['columns']['cases']
    state = json.loads(canonical_json(controller.state()))
    restored,p2,db2,sink2,events2,charges2 = existing._controller(saved['metric'])
    restored.load_state(state)
    assert canonical_json(restored.state()) == canonical_json(state)
    restored.complete_step()
    assert sink2.prompts == sink2.conclusion_prompts == charges2 == []
    ident,prompt,ctx = existing._begin(controller,parent,db)
    ident2,prompt2,ctx2 = existing._begin(restored,p2,db2)
    assert ident==ident2 and prompt==prompt2 and canonical_json(ctx)==canonical_json(ctx2)
    wire,_ = assert_delivered(prompt,group,saved)
    assert ctx['injected_ids']==[lesson_id] and ctx['lesson_texts'][0] in prompt
    shown_rules = len(SHOWN_RULES) if ctx['paired_comparisons']['schema']==SHOWN_SCHEMA else 0
    optional_chars = (ctx['lesson_presentation']['rendered_chars'] + len(canonical_json(ctx['paired_comparisons']))
        + (len(canonical_json(ctx['comparison_facts'])) if 'comparison_facts' in ctx else 0)
        + shown_rules + len(restart_block(wire)) + len(initializer_block(ctx.get('initializer_evidence')))
        + (len(canonical_json(ctx['conclusion_feedback'])) if 'conclusion_feedback' in ctx else 0))
    assert optional_chars <= 32000
    for current,journal in ((controller,events),(restored,events2)):
        assert canonical_json(current.knowledge.comparisons[0])==canonical_json(group)
        lesson, = current.knowledge.lessons
        assert lesson['source_support_ids']==[records[0]['id']] and lesson['exposure_ids']==[]
        assert current.aux_calls==2 and current.aux_cost==0.
        event = [e for e in journal if e['phase']=='restart_observation_presentation' and e['role']=='generation'][-1]
        assert event['context']==wire and event['prompt_sha256']==digest(prompt)
        assert event['delivery']=='prompt_prepared_not_model_use'
    assert canonical_json(original)==original_bytes


def test_ablation_keeps_real_motion_receipts_without_claiming_delivery_or_exposure(location):
    controller,parent,db,sink,_,_ = existing._controller(location['metric'],inject=False)
    existing._history(controller,parent,db)
    group=existing._import_group(controller,location['group'])
    state=json.loads(canonical_json(controller.state()))
    restored,p2,db2,_,_,_ = existing._controller(location['metric'],inject=False)
    restored.load_state(state)
    _,prompt,ctx=existing._begin(restored,p2,db2)
    assert PREFIX not in prompt and ctx['injected_ids']==[] and ctx['lesson_texts']==[]
    assert all(not lesson['exposure_ids'] for lesson in restored.knowledge.lessons)
    assert canonical_json(restored.knowledge.comparisons[0])==canonical_json(group)
    assert sink.prompts==[] and len(sink.conclusion_prompts)==1


def test_tiny_optional_budget_omits_whole_motion_without_mutating_real_evidence(location):
    controller,parent,db,_,_,_ = existing._controller(location['metric'])
    existing._history(controller,parent,db)
    group=existing._import_group(controller,location['group'])
    before=canonical_json(controller.state())
    ctx,ids=controller._restart_context([group],1)
    assert ctx is None and ids==[group['id']]
    assert canonical_json(controller.state())==before


@pytest.mark.parametrize('changed',['policy','restart_motion','restart_motion_observations'])
def test_general_research_motion_policy_or_source_drift_rejects_restore_atomically(location,monkeypatch,changed):
    from sciel.hypothesis import rendering
    controller,parent,db,sink,_,charges = existing._controller(location['metric'])
    existing._history(controller,parent,db)
    saved=json.loads(canonical_json(controller.state()))
    assert saved['aux_calls']==1 and len(saved['conclusions'])==1 and charges==[0.]
    saved_bytes=canonical_json(saved)
    restored,_,_,next_sink,events,next_charges=existing._controller(location['metric'])
    if changed=='policy':
        monkeypatch.setattr(rendering,'MOTION_POLICY',{
            **rendering.MOTION_POLICY,'matching':'Inert fixture: a different motion matching policy.'})
    else:
        target=Path(getattr(rendering,changed).__file__).resolve()
        original_read=Path.read_bytes
        def altered_bytes(path):
            raw=original_read(path)
            return raw+b'\n# Inert identity drift; never written or executed.\n' if path.resolve()==target else raw
        monkeypatch.setattr(Path,'read_bytes',altered_bytes)
    current=rendering.presentation_identity()
    assert current['restart_motion_observation']==saved['presentation']['restart_motion_observation']
    if changed=='policy':
        assert current['restart_motion_policy']!=saved['presentation']['restart_motion_policy']
        assert current['restart_motion_sources']==saved['presentation']['restart_motion_sources']
    else:
        key='hypothesis/'+changed+'.py'
        assert current['restart_motion_policy']==saved['presentation']['restart_motion_policy']
        assert {k for k,v in current['restart_motion_sources'].items()
                if v!=saved['presentation']['restart_motion_sources'][k]}=={key}
    # state() reflects the current identity; baseline after monkeypatch isolates
    # all substantive state/KB/counter mutation from the deliberate identity drift.
    before=canonical_json(restored.state())
    with pytest.raises(ValueError,match='research checkpoint presentation mismatch'):
        restored.load_state(saved)
    assert canonical_json(restored.state())==before and canonical_json(saved)==saved_bytes
    assert next_sink.prompts==next_sink.conclusion_prompts==next_charges==events==[]
    assert len(sink.conclusion_prompts)==1 and controller.aux_calls==1
