"""Raw-bound R import and memory delivery over portable measured HOST evidence."""
import copy
import json
from pathlib import Path
import sys
import pytest

from sciel.config import apply_defaults
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.evaluated_review_context import build_context, block, PREFIX
from sciel.engine.external_research_v2 import policy_identity
from test_evaluated_study_integration import (artifact, construct as study_construct, import_study,
    bundle, prepare, rows, settings, InertModel, builder, response, spec, forbid_runtime)
from test_evaluated_target_generation import reflected, approval


@pytest.fixture(scope='module')
def reflection(artifact):
    # Explicitly synthetic new authoring. Never repair the rejected paid response.
    return reflected(bundle(artifact))


def options(r, *, status='accepted', text=True):
    policy = (dict(schema='sciel-review-text-policy/1', decisions=[]) if status == 'pending'
              else approval(r.sidecar['review_card'], status='accepted' if status == 'stale' else status))
    if status == 'stale':
        policy['decisions'][0]['policy_sha256'] = '0'*64
    return dict(external_evaluated_review_sha256=canonical_sha(r.sidecar),
        source_reflection_config_sha256=canonical_sha(r.config), comparison_reviews=True,
        structured_auxiliary=True, comparison_review_version=3, batch_only_lessons=True,
        comparison_review_text=text, comparison_review_text_policy=policy)


def construct(artifact, r, path, **changes):
    opts = options(r)
    opts.update(changes)
    return study_construct(artifact, path, **opts)


def import_review(f, r):
    return f.worker.import_external_evaluated_review(r.sidecar, r.config,
        expected_sha256=canonical_sha(r.sidecar))


def test_public_import_zero_counters_durable_detached_and_idempotent(artifact, reflection, tmp_path):
    f=construct(artifact, reflection, tmp_path)
    try:
        with pytest.raises(ValueError):import_review(f, reflection)
        import_study(f, artifact)
        with pytest.raises(ValueError,match='not been imported'):f.c.context(f.db,f.parent,0,1)
        before=rows(f.log)
        assert import_review(f,reflection) is True
        saved=f.c.knowledge.evaluated_study_review
        assert saved==dict(sidecar=reflection.sidecar,reflection_config=reflection.config)
        saved['reflection_config']['model']='forged'
        assert f.c.knowledge.evaluated_study_review['reflection_config']==reflection.config
        assert all(getattr(f.c,k)==0 for k in ('attempts','outcomes','aux_calls','aux_cost','reflections','reflected_through'))
        assert f.c.knowledge.experiments==f.c.knowledge.comparisons==f.c.knowledge.lessons==[]
        assert f.worker.calls_started==f.worker.evaluations==0 and f.model.calls==f.charges==[]
        event=rows(f.log)[-1];assert event['phase']=='external_evaluated_review_import'
        assert event['new_model_calls']==event['new_lessons']==event['new_exposures']==event['new_cost']==0
        assert event['reflection_config_sha256']==canonical_sha(reflection.config)
        assert len(rows(f.log))==len(before)+1
        events=rows(f.log);assert import_review(f,reflection) is False and rows(f.log)==events
        f.worker.assert_journal(f.log.path,f.log.anchor())
        from sciel.engine.evaluated_review_import import assert_import_journal
        tampered=copy.deepcopy(events);tampered[-1]['new_cost']=False
        with pytest.raises(ValueError,match='journal import binding'):assert_import_journal(f.c,tampered)
    finally:f.log.close()


@pytest.mark.parametrize('status,text',[('accepted',True),('accepted',False),('pending',True),('rejected',True),('stale',True)])
def test_only_whole_accepted_text_enters_G_and_R(artifact, reflection, tmp_path,status,text):
    f=construct(artifact,reflection,tmp_path,**options(reflection,status=status,text=text))
    try:
        import_study(f,artifact);import_review(f,reflection)
        ctx=f.c.context(f.db,f.parent,0,1);view=ctx['evaluated_study_review'];card,=view['cards']
        assert ctx['evaluated_detached_study']['presented_count']==1
        assert card['text_eligibility']['status']==status
        accepted=status=='accepted' and text
        for key in ('unverified_interpretation','evidence_bindings','resolved_target'):
            assert (card[key] is not None)==accepted
        from sciel.engine.reflection_context import prepare_reflection_context
        r=prepare_reflection_context(f.c.knowledge,{},config=f.c.config,interval=1,through=1)
        _,g=prepare(f,ctx)
        assert block(view) in g and block(view) in r['prompt']
        assert r['context_event']['evaluated_study_review']==view
        literal,_=json.JSONDecoder().raw_decode(g.split(PREFIX,1)[1]);assert literal==view
        assert (reflection.proposal['conjecture'] in g)==accepted
        assert (reflection.proposal['conjecture'] in r['prompt'])==accepted
        assert r['records']==r['lessons']==[] and f.c.outcomes==f.c.aux_calls==0
        assert f.model.calls==[]
        assert import_review(f,reflection) is False  # No replay even after local begin.
    finally:f.log.close()


def test_full_reservation_stable_and_whole_card_omissions_keep_facts(artifact,reflection,tmp_path):
    base=reflection.sidecar['review_card'];args=dict(branch=artifact['capsule']['receiving_root']['id'],
        text_policy=approval(base),policy_sha256=canonical_sha(policy_identity()))
    on=build_context(bundle(artifact),base,**args);off=build_context(bundle(artifact),base,text_enabled=False,**args)
    assert on['reserved_chars']==off['reserved_chars'] and off['rendered_chars']<on['rendered_chars']
    for status in ('pending','rejected','stale'):
        policy=options(reflection,status=status)['comparison_review_text_policy']
        view=build_context(bundle(artifact),base,**dict(args,text_policy=policy))
        assert view['reserved_chars']==on['reserved_chars']
    small=build_context(bundle(artifact),base,max_chars=on['reserved_chars']-1,**args)
    assert small['cards']==[] and small['omissions'][0]['reason']=='memory_char_limit'
    nofacts=build_context(bundle(artifact),base,source_facts_presented=False,**args)
    assert nofacts['cards']==[] and nofacts['omissions'][0]['reason']=='source_study_facts_omitted'
    # Corrupt retained cards cannot become admissible merely because hidden/omitted.
    bad=copy.deepcopy(base);bad['writer']['raw_response_sha256']='x'
    with pytest.raises(ValueError):build_context(bundle(artifact),bad,max_chars=0,enabled=False,**args)


@pytest.mark.parametrize('damage',['drop','forged_text','branch','text_flag','registry','review_pin','config_pin'])
def test_snapshot_or_config_mutation_fails_before_any_attempt(artifact,reflection,tmp_path,damage):
    f=construct(artifact,reflection,tmp_path)
    try:
        import_study(f,artifact);import_review(f,reflection);ctx=f.c.context(f.db,f.parent,0,1)
        before=f.c.attempts;events=rows(f.log)
        if damage=='drop':del ctx['evaluated_study_review']
        elif damage=='forged_text':
            v=ctx['evaluated_study_review'];v['cards'][0]['unverified_interpretation']['conjecture']='forged'
            v['context_sha256']=canonical_sha({k:v for k,v in v.items() if k!='context_sha256'})
        elif damage=='branch':ctx['branch']='forged-root'
        elif damage=='text_flag':f.c.config['comparison_review_text']=False
        elif damage=='registry':f.c.config['comparison_review_text_policy']['decisions']=[]
        elif damage=='review_pin':f.c.config['external_evaluated_review_sha256']='0'*64
        else:f.c.config['source_reflection_config_sha256']='0'*64
        with pytest.raises(ValueError):prepare(f,ctx)
        assert f.c.attempts==before and rows(f.log)==events and f.model.calls==[]
    finally:f.log.close()


@pytest.mark.parametrize('damage',['raw','config','drop','identity'])
def test_checkpoint_tamper_is_atomic(artifact,reflection,tmp_path,damage):
    a=construct(artifact,reflection,tmp_path/'a');b=construct(artifact,reflection,tmp_path/'b')
    try:
        import_study(a,artifact);import_review(a,reflection);state=a.c.state()
        if damage=='raw':state['knowledge']['evaluated_study_review']['sidecar']['response']['text']+=' '
        elif damage=='config':state['knowledge']['evaluated_study_review']['reflection_config']['model']='other'
        elif damage=='identity':state['external_evaluated_review_identity']['schema']='other'
        else:
            import_study(b,artifact);import_review(b,reflection)
            del state['knowledge']['evaluated_study_review']
        before=b.c.state()
        with pytest.raises(ValueError):b.c.load_state(state)
        assert b.c.state()==before and b.model.calls==[]
    finally:a.log.close();b.log.close()


@pytest.mark.parametrize('failure',['silent_sink','after_event'])
def test_durable_event_precedes_publish_and_failed_import_cannot_replay(artifact,reflection,tmp_path,monkeypatch,failure):
    f=construct(artifact,reflection,tmp_path)
    try:
        import_study(f,artifact);before=f.c.state();anchor=f.log.anchor
        if failure=='silent_sink':monkeypatch.setattr(f.log,'research',lambda **kw:False)
        else:
            calls=[]
            def broken():
                calls.append(1)
                if len(calls)==1:return anchor()
                raise OSError('synthetic persisted-event/readback failure')
            monkeypatch.setattr(f.log,'anchor',broken)
        with pytest.raises((ValueError,OSError)):import_review(f,reflection)
        assert f.c.state()==before and f.c.knowledge.evaluated_study_review is None
        if failure=='after_event':
            monkeypatch.setattr(f.log,'anchor',anchor)
            with pytest.raises(ValueError,match='refusing replay'):import_review(f,reflection)
        assert f.model.calls==[]
    finally:f.log.close()


@pytest.mark.parametrize('damage',['missing_reflection_pin','missing_review_pin','bool','null'])
def test_config_requires_independent_review_and_original_config_pins(artifact,reflection,damage):
    opts=options(reflection)
    if damage=='missing_reflection_pin':del opts['source_reflection_config_sha256']
    elif damage=='missing_review_pin':del opts['external_evaluated_review_sha256']
    else:opts['source_reflection_config_sha256']=True if damage=='bool' else None
    with pytest.raises(ValueError):apply_defaults(settings(artifact,**opts))


def test_disabled_delivery_retains_review_and_negative_facts_without_text(artifact,reflection,tmp_path):
    f=construct(artifact,reflection,tmp_path,inject_lessons=False)
    try:
        import_study(f,artifact);import_review(f,reflection)
        ctx=f.c.context(f.db,f.parent,0,1)
        assert ctx.get('evaluated_detached_study') is None
        view=ctx['evaluated_study_review'];assert view['cards']==[] and view['omissions'][0]['reason']=='disabled'
        _,prompt=prepare(f,ctx);assert reflection.proposal['conjecture'] not in prompt
        assert f.c.knowledge.evaluated_detached_study['capsule']['comparison']['derived']==artifact['capsule']['comparison']['derived']
        assert f.c.knowledge.evaluated_study_review['sidecar']==reflection.sidecar
    finally:f.log.close()




def test_real_engine_checkpoint_resume_delivers_review_without_reimport_or_paid_R(tmp_path,monkeypatch):
    from sciel.engine import checkpoint
    from sciel.engine.loop import Engine
    from sciel.eval.cascade import CascadeOutcome
    from sciel.eval.facade import EvalResult
    from test_circle_intervention_engine import task_copy
    from test_optimizer_traced_engine import OutcomeDraws
    draws=OutcomeDraws(monkeypatch)
    task=task_copy(tmp_path/'task',initial=builder.source_for())
    from evaluated_study_fixture import build
    from sciel.hypothesis.measurement import metric_context
    artifact=build(task_metric=metric_context(task))
    reflection=reflected(bundle(artifact))
    cfg=settings(artifact,max_experiments=1,**options(reflection))
    bodies=['return best_params.copy()\n','return initial_params.copy()\n']
    def evaluate(_task,path,**kwargs):
        body=builder.body_of(Path(path).read_text());assert body in [builder.DEFAULT_BODY,*bodies]
        score=.5 if body==builder.DEFAULT_BODY else .4
        return CascadeOutcome(EvalResult(True,score,dict(sum_radii=score*artifact['capsule']['metric']['semantics']['denominator'],
            target_ratio=score,validity=1.,combined_score=score),timing=.01),score,'full',False)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',evaluate)
    class Model(InertModel):
        weights=[1.]
        def choose_model(self,rng):return self.models[0]
    folder=tmp_path/'engine'
    first=Engine(cfg,task,str(folder),Model(response(bodies[0],restart_predictions=spec())),python=sys.executable)
    original=first.seed;blank=folder/'before_import.json'
    def seed():
        original();checkpoint.save(first,'before_study_review');blank.write_bytes((folder/'checkpoint.json').read_bytes())
        first.circle_generation.import_external_evaluated_study(artifact['capsule'],artifact['author_dependencies'],
            artifact['author_authority'],expected_sha256=artifact['capsule_sha256'],parent=first.db.get('p1'))
        first.circle_generation.import_external_evaluated_review(reflection.sidecar,reflection.config,
            expected_sha256=canonical_sha(reflection.sidecar))
    first.seed=seed;first._stop_at=1;first.run()
    second=Engine(cfg,task,str(folder),Model(response(bodies[1],restart_predictions=spec())),python=sys.executable)
    checkpoint.load(second,str(folder/'checkpoint.json'));assert second.llm.calls==[]
    second.run()
    assert len(first.llm.calls)==len(second.llm.calls)==1
    assert reflection.proposal['conjecture'] in second.llm.calls[0]['prompt']
    assert PREFIX in second.llm.calls[0]['prompt']
    assert second.research.knowledge.evaluated_study_review==dict(sidecar=reflection.sidecar,reflection_config=reflection.config)
    assert second.research.attempts==second.research.outcomes==second.circle_generation.evaluations==2
    assert second.research.aux_calls==second.research.aux_cost==0
    assert second.research.knowledge.lessons==[]
    assert len(draws.calls)==15  # All are explicit inert fixture draws.
    third=Engine(cfg,task,str(folder),Model(None),python=sys.executable)
    try:
        with pytest.raises(checkpoint.IdentityMismatch,match='uncheckpointed research'):
            checkpoint.load(third,str(blank))
        checkpoint.load(third,str(folder/'checkpoint.json'));assert third.llm.calls==[]
    finally:third.repeat_promotion.close();third.log.close()


def test_missing_review_blocks_complete_step_before_queue_pop_or_aux(artifact,reflection,tmp_path):
    f=construct(artifact,reflection,tmp_path)
    try:
        import_study(f,artifact)
        f.c.queue.append({'sentinel':'must stay queued'})
        with pytest.raises(ValueError,match='not been imported'):f.c.complete_step()
        assert f.c.queue==[{'sentinel':'must stay queued'}] and f.c.aux_calls==0 and f.model.calls==[]
    finally:f.log.close()


def test_reasoned_null_review_is_retained_without_inventing_a_target(artifact,tmp_path):
    r=reflected(bundle(artifact),null_target=True)
    f=construct(artifact,r,tmp_path)
    try:
        import_study(f,artifact);import_review(f,r)
        ctx=f.c.context(f.db,f.parent,0,1)
        card,=ctx['evaluated_study_review']['cards']
        assert card['text_eligibility']['status']=='accepted'
        assert card['unverified_interpretation']['next_hypothesis'] is None
        assert card['resolved_target'] is None
        assert f.c.knowledge.lessons==[] and f.c.attempts==f.c.outcomes==f.c.aux_calls==0
        assert f.model.calls==[]
    finally:f.log.close()
