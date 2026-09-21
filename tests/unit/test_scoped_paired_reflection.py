"""Study-scoped R intent crosses real memory boundaries with inert saved facts.

All observations and responses are authored primitive fixtures; no candidate,
provider or optimizer runs. Checkpoint tests use the current runtime.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.engine import generated_paired_reflection as reflection
from sciel.engine.external_reflection import ExternalReflectionRejected
from sciel.hypothesis import component_targets
from sciel.hypothesis.comparison_targets import QUALITY_GATE
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.generated_paired_review_context import PREFIX, block
from test_generated_paired_integration import (pilot_artifacts, component, construct,
    import_component, prepare, rows)
from test_generated_paired_review_delivery import policy_for


@pytest.fixture(autouse=True)
def no_execution():
    def forbidden(*_args, **_kwargs):
        pytest.fail('Scoped reflection integration attempted real transport or execution.')
    with pytest.MonkeyPatch.context() as patch:
        names=['sciel.eval.repeat_draw.draw', 'sciel.eval.repeat_draw.adapter_identity',
               'sciel.engine.llm.client.EnsembleLLM.generate']
        names.append('subprocess.Popen')
        for name in names:
            patch.setattr(name, forbidden)
        yield


def synthetic_proposal(component):
    group = next(g for g in component['comparisons'] if g['branch'] == 'on_off')
    control_group = next(g for g in component['comparisons'] if g['branch'] == 'off_control')
    anchor = dict(comparison_id=group['id'], fact_id='quality:strict_verdict')
    additional = dict(comparison_id=control_group['id'], fact_id='quality:strict_reason')
    target = dict(schema=component_targets.SCHEMA, component_sha256=canonical_sha(component),
        receiving_root=copy.deepcopy(component['root_parent']), policy_sha256=canonical_sha(component['policy']),
        motivating_evidence=[anchor, additional], base_source_sha256=group['comparator_source_sha256'],
        comparator_source_sha256=control_group['comparator_source_sha256'],
        change='Synthetic prospective fixture: change OFF activation from every 8th to every 16th restart.',
        quality_gate=QUALITY_GATE, diagnostics=[])
    return dict(comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        claims=[dict(fact_id='quality:strict_verdict', value='INCONCLUSIVE'),
                dict(fact_id='restart:challenger:763606949:result_changed:observed', value='unknown')],
        conjecture='SYNTHETIC SCOPED R: this partial tie does not establish a useful mechanism.',
        limitations='SYNTHETIC LIMIT: the two missing seeds stay unknown; no causal conclusion follows.',
        next_hypothesis=target,
        next_hypothesis_reason='SYNTHETIC INTENT: a new frequency comparison could test a conditional idea; no gain is promised.',
        evidence_refs={key:copy.deepcopy([anchor, additional]) for key in
                       ('conjecture', 'limitations', 'next_hypothesis_reason')})


@pytest.fixture(scope='module')
def saved(component):
    config = dict(external_component_sha256=canonical_sha(component),
        restart_predictions=True, structured_auxiliary=True, comparison_reviews=True,
        comparison_review_version=3, generated_paired_review_version=2, inject_lessons=True,
        max_aux_calls=1, model='anthropic/claude-sonnet-4.5', batch_only_lessons=True,
        comparison_review_text=True,
        comparison_review_text_policy={'schema':'sciel-review-text-policy/1', 'decisions':[]})
    job = reflection.GeneratedPairedReflectionJob(component,
        expected_sha256=canonical_sha(component), config=config)
    plan = job.prepare(run_id='synthetic-scoped-component-R', segment_id='synthetic-scoped-R')
    proposal = synthetic_proposal(component)
    returned = dict(text=json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=[proposal])),
        model=config['model'], cost=.002, prompt_tokens=100, completion_tokens=100,
        finish_reason='stop', refusal=None)
    accepted = job.admit(plan, returned, manifest_sha256='a'*64,
                         journal_prefix=plan['journal_prefix'])
    return NS(component=component,config=config,job=job,plan=plan,proposal=proposal,
              returned=returned,accepted=accepted,sidecar=accepted['extended_capsule'])


def receiver(saved, pilot, path, *, text=True, status='accepted', version=2, sidecar=None):
    sidecar = saved.sidecar if sidecar is None else sidecar
    return construct(pilot, saved.component, path,
        external_component_review_sha256=canonical_sha(sidecar),
        comparison_reviews=True, comparison_review_version=3, generated_paired_review_version=version,
        comparison_review_text=text, comparison_review_text_policy=policy_for(sidecar,status),
        batch_only_lessons=True)


def import_all(f,saved):
    assert import_component(f,saved.component)
    assert f.worker.import_external_component_review(saved.sidecar,
        expected_sha256=canonical_sha(saved.sidecar))


def test_new_job_keeps_full_study_scope_and_exact_motivating_values(saved):
    assert saved.plan['schema']=='sciel-generated-paired-reflection-plan/3'
    assert saved.sidecar['schema']=='sciel-generated-paired-review-sidecar/3'
    assert saved.plan['response_format']['json_schema']['name']=='sciel_generated_paired_reflection_v3'
    for clause in ('at least one raw delta must be > 0', 'at least one exact certified delta must be > 0',
                   'No epsilon, near-zero substitute', 'Total starts requires a validated full-run trace',
                   'never a count inferred from the retained prefix'):
        assert clause in saved.plan['prompt']
    assert saved.plan['new_ordinary_outcomes']==saved.plan['new_search_attempts']==0
    card,=saved.sidecar['review_cards'].values()
    assert card['schema']=='sciel-generated-paired-review-card/2'
    target=card['unverified_interpretation']['next_hypothesis']
    assert target==saved.proposal['next_hypothesis']
    assert target['base_source_sha256']==card['comparator_source_sha256']
    assert target['comparator_source_sha256'] not in {
        card['comparator_source_sha256'],card['challenger_source_sha256']}
    assert target['comparator_source_sha256']==saved.component['root_parent']['source_sha256']
    assert card['unverified_interpretation']['evidence_refs']==saved.proposal['evidence_refs']
    for name,refs in card['evidence_bindings'].items():
        assert [r['value'] for r in refs]==['INCONCLUSIVE','known_paired_loss']
        authored_refs=(target['motivating_evidence'] if name=='target_motivation'
                       else saved.proposal['evidence_refs'][name])
        for authored,resolved in zip(authored_refs,refs):
            assert all(resolved[k]==v for k,v in authored.items())
            assert len(resolved['comparison_sha256'])==64
    assert saved.accepted['new_ordinary_outcomes']==saved.accepted['new_search_attempts']==0
    assert saved.accepted['new_lesson_ids']==saved.accepted['retired_lesson_ids']==[]
    assert saved.accepted['provider_dispatch_verified'] is False
    assert 'generation_allowed' not in saved.accepted
    assert reflection.validate_sidecar(saved.sidecar,component=saved.component,
        expected_sha256=canonical_sha(saved.sidecar))==saved.sidecar


@pytest.mark.parametrize('damage',['raw_reason','card_reason','resolved_fact','target_scope','author_ref'])
def test_sidecar_rehash_cannot_forge_scoped_writer_or_host_resolution(saved,damage):
    forged=copy.deepcopy(saved.sidecar)
    card=next(iter(forged['review_cards'].values()))
    if damage=='raw_reason':
        raw=json.loads(forged['response']['text'])
        raw['comparison_reviews'][0]['next_hypothesis_reason']+=' Forged.'
        forged['response']['text']=json.dumps(raw)
    elif damage=='card_reason':card['unverified_interpretation']['next_hypothesis_reason']+=' Forged.'
    elif damage=='resolved_fact':card['evidence_bindings']['limitations'][0]['value']='CONFIRMED'
    elif damage=='target_scope':card['unverified_interpretation']['next_hypothesis']['component_sha256']='0'*64
    else:card['unverified_interpretation']['evidence_refs']['conjecture'][0]['fact_id']='quality:observed_strict_raw_gain'
    with pytest.raises(ValueError):
        reflection.validate_sidecar(forged,component=saved.component,expected_sha256=canonical_sha(forged))


def test_public_import_and_ON_OFF_delivery_preserve_facts_reserve_and_resume(saved,pilot_artifacts,tmp_path):
    contexts=[]
    for text in (True,False):
        f=receiver(saved,pilot_artifacts,tmp_path/str(text),text=text)
        restored=None
        try:
            import_all(f,saved)
            assert f.c.knowledge.generated_paired_review==saved.sidecar
            assert f.c.knowledge.experiments==f.c.knowledge.comparisons==f.c.knowledge.lessons==[]
            assert all(getattr(f.c,k)==0 for k in ('attempts','outcomes','aux_calls','aux_cost','reflections','reflected_through'))
            context=f.c.context(f.db,f.parent,0,1)
            shown=context['generated_paired_reviews']
            card,=shown['cards']
            stored,=saved.sidecar['review_cards'].values()
            assert card['unverified_interpretation']==(stored['unverified_interpretation'] if text else None)
            assert card.get('evidence_bindings')==(stored['evidence_bindings'] if text else None)
            assert len(block(shown))==shown['rendered_chars']<=shown['reserved_chars']<=8000
            contexts.append(context)
            state=f.c.state()
            restored=receiver(saved,pilot_artifacts,tmp_path/('restored'+str(text)),text=text)
            restored.c.load_state(copy.deepcopy(state))
            restored.worker.load_state(f.worker.state())
            restored.worker.assert_journal(f.log.path,f.log.anchor())
            assert restored.c.state()==state
            assert restored.c.context(restored.db,restored.parent,0,1)['generated_paired_reviews']==shown
            ident,prompt=prepare(f,context)
            delivered,_=json.JSONDecoder().raw_decode(prompt.split(PREFIX,1)[1])
            assert delivered==shown and ident in f.c.pending
            for key in ('conjecture','limitations','next_hypothesis_reason'):
                assert (stored['unverified_interpretation'][key] in prompt)==text
            assert rows(f.log)[-1]['generated_paired_reviews']==shown
            assert f.model.calls==f.charges==restored.model.calls==restored.charges==[]
        finally:
            f.log.close()
            if restored is not None:restored.log.close()
    on,off=contexts
    assert on['generated_paired_reviews']['reserved_chars']==off['generated_paired_reviews']['reserved_chars']
    keys=('generated_paired_history','lesson_texts','paired_comparisons','comparison_facts',
          'restart_predictions','conclusion_feedback','comparison_review_delivery')
    assert canonical_json({k:on.get(k) for k in keys})==canonical_json({k:off.get(k) for k in keys})


def test_pending_refs_are_not_delivered_as_host_selected_facts(saved,pilot_artifacts,tmp_path):
    f=receiver(saved,pilot_artifacts,tmp_path,status='pending')
    try:
        import_all(f,saved)
        ctx=f.c.context(f.db,f.parent,0,1)
        card,=ctx['generated_paired_reviews']['cards']
        assert card['unverified_interpretation'] is None
        assert card.get('evidence_bindings') is None
        assert card['text_eligibility']['status']=='pending'
        before,events=f.c.state(),rows(f.log)
        card['evidence_bindings']=copy.deepcopy(next(iter(saved.sidecar['review_cards'].values()))['evidence_bindings'])
        wire=ctx['generated_paired_reviews']
        wire['context_sha256']=canonical_sha({k:v for k,v in wire.items() if k!='context_sha256'})
        with pytest.raises(ValueError):prepare(f,ctx)
        assert f.c.state()==before and rows(f.log)==events and f.model.calls==[]
    finally:f.log.close()


def test_declared_version_is_frozen_before_any_review_cards_are_imported(saved,pilot_artifacts,tmp_path):
    f=receiver(saved,pilot_artifacts,tmp_path/'original',version=2)
    old=receiver(saved,pilot_artifacts,tmp_path/'old-receiver',version=1)
    try:
        import_component(f,saved.component)
        assert f.c.knowledge.generated_paired_review is None
        state,events=f.c.state(),rows(f.log)
        f.c.config['generated_paired_review_version']=1
        try:
            with pytest.raises(ValueError):
                f.worker.import_external_component_review(saved.sidecar,
                    expected_sha256=canonical_sha(saved.sidecar))
            with pytest.raises(ValueError):f.c.load_state(copy.deepcopy(state))
        finally:f.c.config['generated_paired_review_version']=2
        assert f.c.state()==state and rows(f.log)==events
        before=old.c.state()
        with pytest.raises(ValueError):old.c.load_state(copy.deepcopy(state))
        assert old.c.state()==before
        assert f.model.calls==old.model.calls==f.charges==old.charges==[]
    finally:
        f.log.close()
        old.log.close()


@pytest.mark.parametrize('empty',[False,True])
def test_new_sidecar_cannot_import_under_old_receiver_version_even_when_empty(saved,pilot_artifacts,tmp_path,empty):
    sidecar=saved.sidecar
    if empty:
        returned=dict(saved.returned,text=json.dumps(dict(lessons=[],retire_ids=[],comparison_reviews=[])))
        sidecar=saved.job.admit(saved.plan,returned,manifest_sha256='a'*64,
            journal_prefix=saved.plan['journal_prefix'])['extended_capsule']
    f=receiver(saved,pilot_artifacts,tmp_path,version=1,sidecar=sidecar)
    try:
        import_component(f,saved.component)
        before,events=f.c.state(),rows(f.log)
        with pytest.raises(ValueError):
            f.worker.import_external_component_review(sidecar,expected_sha256=canonical_sha(sidecar))
        assert f.c.state()==before and rows(f.log)==events
        assert f.c.knowledge.generated_paired_review is None and f.model.calls==[]
    finally:f.log.close()


def test_unavailable_legacy_target_is_not_repaired_or_admitted_by_new_contract(saved):
    from test_generated_paired_reflection import response
    returned = response(saved, prospective=True)
    raw = json.loads(returned['text'])
    raw['comparison_reviews'][0]['next_hypothesis']['base_source_sha256'] = '0' * 64
    returned['text'] = json.dumps(raw)
    raw_before = returned['text']
    raw_sha = hashlib.sha256(raw_before.encode()).hexdigest()
    for version in (1, 2):
        config = dict(saved.config, generated_paired_review_version=version)
        job = reflection.GeneratedPairedReflectionJob(saved.component,
            expected_sha256=canonical_sha(saved.component), config=config)
        plan = job.prepare(run_id='legacy-negative-fixture-' + str(version), segment_id='inert-rejection')
        with pytest.raises(ExternalReflectionRejected) as rejected:
            job.admit(plan, returned, manifest_sha256='a' * 64, journal_prefix=plan['journal_prefix'])
        events = [json.loads(x) for x in rejected.value.journal.splitlines()]
        assert events[-1]['comparison_review_ids'] == []
        if version == 1:
            assert events[-1]['error_detail'] == 'comparison target: target references an unavailable source'
        assert next(e for e in events if e.get('phase') == 'aux_result')['response_sha256'] == raw_sha
    assert returned['text'] == raw_before


def test_new_sidecar_full_engine_checkpoint_and_next_inert_generation(saved,pilot_artifacts,tmp_path,monkeypatch):
    # Reuse the complete Engine scenario, including blank-checkpoint rejection,
    # actual checkpoint.load, foreign journal binding and eviction. Only the new
    # opt-in version is added; both models and numerical adapters stay synthetic.
    import test_generated_paired_review_engine as scenario
    original=scenario.engine_config
    def config():
        value=original()
        value['research']['generated_paired_review_version']=2
        return value
    monkeypatch.setattr(scenario,'engine_config',config)
    scenario.test_actual_component_review_survives_engine_checkpoint_eviction_and_next_generation(
        saved.component,pilot_artifacts,saved.sidecar,tmp_path,monkeypatch)
