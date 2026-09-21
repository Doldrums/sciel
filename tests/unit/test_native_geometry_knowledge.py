"""Same-invocation geometry stays full in durable evidence and compact in prompts.

Most cases use inert source and fake returned evaluations. One transport bridge
runs a fixed-array NumPy fixture once, then certifies only its saved geometry.
No optimizer, provider, Docker or runtime probe runs.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController, digest
from sciel.engine.prompt.build import build
from sciel.engine.repeat_support import EVALUATOR_SHA
from sciel.engine.types import Program, Snapshot
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.eval.sandbox import run_stage
from sciel.eval.native_geometry import (SCHEMA, VIEW_SCHEMA, view_geometry, validate_geometry,
                                        certify_artifact)
from sciel.hypothesis.measurement import evaluation_observation, validate_evaluation, metric_context
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis.prompts import conclusion_prompt, reflection_prompt
from sciel.hypothesis.rendering import (decode_evidence, presentation_identity, serialize_context,
    serialize_record, serialize_card, serialize_reflection)
from sciel.knowledge_base.base import _digest, _snapshot
from sciel.knowledge_base.presentation import (observation_card, reflection_payload,
    reflection_inline_payload, expand_reflection_payload)

ROOT = Path(__file__).resolve().parents[2]
PARENT = 'def run_search():\n    return 0.05\n'
CHILD = 'def run_search():\n    return 0.1\n'
HYP = dict(hypothesis='A bounded synthetic change.', predicted_low=.02,
           predicted_high=.1, reasoning='Inert fixture, no execution claim.', confidence=.5)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def geometry(code=CHILD, *, stage='full', confirmation=None, invocation=1):
    numeric = {'centers': [[.25, .25], [.75, .75]] + [[0., 0.] for _ in range(24)],
               'radii': [.125, .125] + [0.] * 24}
    return validate_geometry(dict(schema=SCHEMA, source_sha256=digest(code), evaluator_sha256=EVALUATOR_SHA,
        invocation_id=f'{invocation:032x}', stage=stage, confirmation=confirmation,
        numeric=numeric, numeric_sha256=digest(canonical(numeric)), centers_dtype='float64',
        radii_dtype='float64', native_validity=True, native_sum_radii=.25, native_score=.25/2.635))


def outcome(code=CHILD, *, stage='full', failed_confirmation=False, artifact=True, invocation=1):
    confirmation = (dict(trials=1, selected_trial=0, selection='first_failure_else_lowest_score')
                    if failed_confirmation else None)
    g = geometry(code, stage=stage, confirmation=confirmation, invocation=invocation)
    cases = dict(sum_radii=.25, validity=1., target_ratio=.25/2.635, combined_score=.25/2.635)
    if failed_confirmation:
        cases['n_trials'] = 1.
    result = EvalResult(not failed_confirmation, 0. if failed_confirmation else g['native_score'],
        cases, artifacts={'native_geometry': canonical(g)} if artifact else {})
    return CascadeOutcome(result, .15 if failed_confirmation else result.score,
                          'confirm' if failed_confirmation else stage, confirmed=False)


class FakeModel:
    models = ['fixture']
    max_retries = 1
    def __init__(self):
        self.prompts = []
    def generate(self, prompt, model, temperature=None):
        self.prompts.append(prompt)
        return response()


def response(code=CHILD):
    return LLMResponse('HYPOTHESIS_JSON: '+json.dumps(HYP)+'\n```python\n'+code+'```',
                       'fixture', .001, 10, 20)


def controller(**settings):
    config = apply_defaults({'research': {'enabled': True, 'max_aux_calls': 0,
        'reflection_interval': 0, **settings}})['research']
    llm, events = FakeModel(), []
    c = ResearchController(config, llm, NS(research=lambda **kw: events.append(deepcopy(kw))),
        lambda _: pytest.fail('auxiliary dispatch forbidden'),
        metric=metric_context(load_task(str(ROOT/'tasks/oe/circle_packing'))))
    parent = Program('p1', PARENT, .05, {'sum_radii': .13175, 'validity': 1.,
        'target_ratio': .05, 'combined_score': .05}, 0, 'seed')
    return c, parent, NS(_all={'p1': parent}), events


def snapshot(c, parent, db):
    ctx = c.context(db, parent, 0, c.attempts+1)
    return Snapshot(0, 'full_rewrite', 'fixture', c.attempts, parent.id, parent.code,
                    parent.score, required_entrypoints=['run_search'], research_context=ctx)


def observe(c, parent, db, *, invocation=1, failed=False, artifact=True):
    code = CHILD + f'# observation {invocation}\n'
    snap = snapshot(c, parent, db)
    ident, _ = c.begin(snap, 0, 'full_rewrite', build('full_rewrite', snap))
    assert c.proposal(ident, response(code), code, NS(code=code))
    returned = outcome(code, failed_confirmation=failed, artifact=artifact, invocation=invocation)
    record = c.resolve(ident, status='eval_failed' if failed else 'evaluated',
        score=returned.result.score if not failed else None, source=code,
        evaluation=evaluation_observation(returned))
    c.complete_step()
    return c.knowledge.experiments[-1], returned


def lesson(c, record):
    return c.knowledge.add_lessons([{'rule':'Synthetic positive source observation.',
        'polarity':'positive', 'scope':'branch', 'evidence_ids':[record['id']]}],
        allowed_evidence_ids=[record['id']])[0]


def decoded(text):
    return decode_evidence(json.loads(text))


def assert_view(value, full):
    assert value == view_geometry(full)
    assert value['schema'] == VIEW_SCHEMA and value['arrays_omitted'] is True
    assert 'numeric' not in value and value['certification'] == 'not_performed'
    assert value['artifact_sha256'] == digest(canonical(full))


def test_full_artifact_optional_detached_and_legacy_shape_unchanged():
    returned = outcome();before = deepcopy(returned)
    observed = evaluation_observation(returned)
    full = geometry()
    assert observed['native_geometry'] == full
    assert validate_evaluation(json.loads(json.dumps(observed))) == observed
    observed['native_geometry']['numeric']['radii'][0] = 99.
    assert returned == before
    legacy = evaluation_observation(outcome(artifact=False))
    assert set(legacy) == {'ok','score','cases','reached','confirmed','raw_first_score'}
    assert validate_evaluation(legacy) == legacy
    assert decoded(serialize_record({'evaluation':legacy})) == {'evaluation':legacy}


@pytest.mark.parametrize('damage', ['truncated','repr','view','duplicate','nonfinite'])
def test_present_but_unusable_geometry_never_silently_disappears(damage):
    returned = outcome()
    raw = returned.result.artifacts['native_geometry']
    if damage == 'truncated':raw = raw[:-1]
    elif damage == 'repr':raw = str(geometry())
    elif damage == 'view':raw = canonical(view_geometry(geometry()))
    elif damage == 'duplicate':raw = raw[:-1]+',"stage":"full"}'
    else:raw = raw.replace('0.125', 'NaN', 1)
    returned.result.artifacts['native_geometry'] = raw
    with pytest.raises(ValueError):evaluation_observation(returned)


def test_view_is_never_accepted_as_persisted_evaluation():
    value = evaluation_observation(outcome())
    value['native_geometry'] = view_geometry(value['native_geometry'])
    with pytest.raises(ValueError, match='geometry'):
        validate_evaluation(value)


@pytest.mark.parametrize('damage', ['source','stage','metric','missing_source'])
def test_research_binding_rejects_wrong_origin_before_mutating_pending(damage):
    c,parent,db,_ = controller()
    ident,_ = c.begin(snapshot(c,parent,db),0,'full_rewrite','Inert fixture')
    assert c.proposal(ident,response(),CHILD,NS(code=CHILD))
    value = evaluation_observation(outcome())
    if damage == 'source':value['native_geometry']['source_sha256'] = 'b'*64
    elif damage == 'stage':value['native_geometry']['stage'] = 'smoke'
    elif damage == 'metric':c.pending[ident]['metric'] = metric_context()
    else:c.pending[ident]['source_sha256'] = None
    before = c.state()
    with pytest.raises(ValueError):
        c.resolve(ident,status='evaluated',score=value['score'],source=CHILD,evaluation=value)
    assert c.state() == before


def test_failed_confirmation_keeps_positive_native_geometry_but_no_quality():
    c,parent,db,events = controller()
    record,returned = observe(c,parent,db,failed=True)
    native = record['evaluation']['native_geometry']
    assert record['status']=='eval_failed'
    assert record['score'] is record['actual_delta'] is None
    assert record['forecast_verdict']=='INCONCLUSIVE'
    assert record['evaluation']['score']==0. and native['native_score']>.0
    assert record['evaluation']['raw_first_score']==.15
    assert native['stage']=='full' and native['confirmation']['selected_trial']==0
    assert c.knowledge.lessons==[] and c.outcomes==1
    view = decoded(serialize_record(record))
    assert_view(view['evaluation']['native_geometry'],native)
    assert view['score'] is None and view['evaluation']['score']==0.
    restored,_,_,_=controller();restored.load_state(c.state())
    assert restored.state()==c.state()


def test_real_worker_journal_kb_checkpoint_and_next_generation_projection():
    c,parent,db,events = controller()
    calls=[]
    def inert_evaluator(code):
        calls.append(code)
        return outcome(code)
    result = run_worker('full_rewrite',snapshot(c,parent,db),c.llm,inert_evaluator,
                        dup_cap=1,research=c)
    c.complete_step()
    assert len(c.llm.prompts)==len(calls)==1 and result.ok
    record,=c.knowledge.experiments
    full = record['evaluation']['native_geometry']
    assert full['source_sha256']==digest(calls[0])
    event=next(e for e in events if e['phase']=='outcome')
    assert event['evidence']['evaluation']['native_geometry']==full
    assert record['actual_delta']==result.score-parent.score
    assert record['hypothesis']['predicted_low']==HYP['predicted_low']
    state=deepcopy(c.state());restored,parent2,db2,_=controller();restored.load_state(state)
    assert restored.state()==state
    assert restored.knowledge.fragments[0]['evidence']['evaluation']['native_geometry']==full
    snap=snapshot(restored,parent2,db2)
    _,prompt=restored.begin(snap,0,'full_rewrite',build('full_rewrite',snap))
    wire=json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n',1)[1])[0]
    ctx=decode_evidence(wire)
    assert ctx['fragments']
    for fragment in ctx['fragments']:
        assert_view(fragment['evidence']['evaluation']['native_geometry'],full)
    assert '"numeric"' not in prompt and '"radii":[' not in prompt
    assert c.state()==state  # Another controller's presentation never mutates persisted arrays.


def test_pruned_evidence_source_exposure_reflection_and_restoration_keep_arrays_only_in_storage():
    c,parent,db,_=controller(max_experiments=2)
    first,_=observe(c,parent,db,invocation=1);lesson(c,first)
    second,_=observe(c,parent,db,invocation=2)
    observe(c,parent,db,invocation=3)
    assert first['id'] not in {r['id'] for r in c.knowledge.experiments}
    stored=c.state();full=first['evaluation']['native_geometry']
    retained=c.knowledge.lessons[0]
    assert retained['evidence_snapshots'][0]['evaluation']['native_geometry']==full
    assert retained['exposure_snapshots'][0]['evaluation']['native_geometry']==second['evaluation']['native_geometry']
    card=observation_card(retained)
    assert_view(card['source_observations'][0]['evaluation']['native_geometry'],full)
    for exposure in card['advice_exposures']['representatives']:
        assert exposure['observation']['evaluation']['native_geometry']['arrays_omitted'] is True
    assert decoded(serialize_card(card))==card
    records,lessons=c.knowledge.reflection_context(1)
    assert all('numeric' not in r['evaluation']['native_geometry'] for r in records)
    compact=reflection_payload(records,lessons)
    inline=reflection_inline_payload(records,lessons)
    assert expand_reflection_payload(compact)==inline
    assert decoded(serialize_reflection(records,lessons))==compact
    assert compact['expanded_sha256']==digest(canonical(inline))
    prompts=[conclusion_prompt(first),reflection_prompt(records,lessons)]
    assert all('"numeric"' not in p and '"radii":[' not in p for p in prompts)
    assert c.state()==stored
    restored,_,_,_=controller(max_experiments=2);restored.load_state(stored)
    assert restored.state()==stored
    assert restored.knowledge.lessons[0]['evidence_snapshots'][0]['evaluation']['native_geometry']==full


@pytest.mark.parametrize('damage', ['source','metric','view'])
def test_rehashed_kb_snapshot_cannot_launder_native_binding(damage):
    c,parent,db,_=controller()
    record,_=observe(c,parent,db)
    state=c.state();row=state['knowledge']['experiments'][0]
    if damage=='source':row['evaluation']['native_geometry']['source_sha256']='b'*64
    elif damage=='metric':row['metric']=metric_context()
    else:row['evaluation']['native_geometry']=view_geometry(row['evaluation']['native_geometry'])
    state['knowledge']['seen_experiments'][row['id']]=_digest(_snapshot(row))
    target,_,_,_=controller();before=target.state()
    with pytest.raises(ValueError):target.load_state(state)
    assert target.state()==before


def test_prompt_projection_is_schema_scoped_and_does_not_rewrite_opaque_lookalikes():
    evaluation=evaluation_observation(outcome())
    opaque={'evaluation':deepcopy(evaluation)}
    context={'evaluation':deepcopy(evaluation),'fragments':[{'evidence':{'evaluation':deepcopy(evaluation)},
              'opaque':deepcopy(opaque)}],'opaque':deepcopy(opaque)}
    before=deepcopy(context);wire=decoded(serialize_context(context))
    assert_view(wire['evaluation']['native_geometry'],evaluation['native_geometry'])
    assert_view(wire['fragments'][0]['evidence']['evaluation']['native_geometry'],evaluation['native_geometry'])
    assert wire['opaque']==wire['fragments'][0]['opaque']==opaque
    assert context==before
    # The deterministic comparison is scalar-only, never an alternate array route.
    comparison=observed_comparison({'status':'eval_failed','evaluation':evaluation})
    assert 'native_geometry' not in canonical(comparison)


def test_reflection_catalog_projects_before_hashes_even_without_lessons():
    record={'id':'e1','status':'eval_failed','evaluation':evaluation_observation(outcome())}
    before=deepcopy(record);compact=reflection_payload([record],[])
    projected=compact['observation_catalog']['e1']['evaluation']['native_geometry']
    assert_view(projected,record['evaluation']['native_geometry'])
    assert expand_reflection_payload(compact)==reflection_inline_payload([record],[])
    assert record==before
    compact['observation_catalog']['e1']=record
    with pytest.raises(ValueError,match='presentation view'):
        expand_reflection_payload(compact)


def test_new_identity_blocks_old_runtime_without_inventing_legacy_geometry():
    c,parent,db,_=controller()
    record,_=observe(c,parent,db,artifact=False)
    assert 'native_geometry' not in record['evaluation']
    state=c.state();assert state['presentation']['native_geometry']==VIEW_SCHEMA
    state['presentation'].pop('native_geometry')
    target,_,_,_=controller();before=target.state()
    with pytest.raises(ValueError,match='presentation'):target.load_state(state)
    assert target.state()==before
    assert presentation_identity()['native_geometry']==VIEW_SCHEMA


@pytest.mark.parametrize('force',[False,True])
def test_research_only_checkpoint_cannot_force_old_geometry_presentation(tmp_path,force):
    c,_,_,_=controller()
    journal=tmp_path/'events.jsonl';journal.write_bytes(b'{"event":"run_start"}\n')
    engine=NS(task=None,cfg={'research':c.config},research=c,log=NS(path=str(journal)))
    identity=checkpoint._identity(engine);saved=c.state()
    del saved['presentation']['native_geometry']
    del identity['research']['presentation']['native_geometry']
    state={'identity':identity,'research':saved,'provenance':{
        'schema':checkpoint.PROVENANCE_SCHEMA,'counters':{'db_counter':None},
        'journal_anchor':{'byte_offset':journal.stat().st_size,
                          'prefix_sha256':hashlib.sha256(journal.read_bytes()).hexdigest()}}}
    path=tmp_path/'checkpoint.json';path.write_text(json.dumps(state));before=c.state()
    with pytest.raises(checkpoint.IdentityMismatch,match='presentation' if force else 'research'):
        checkpoint.load(engine,str(path),allow_identity_mismatch=force)
    assert c.state()==before


def test_actual_fixed_array_transport_to_kb_prompt_and_same_artifact_certificate(tmp_path):
    """One tiny native invocation; no solver, followed only by saved-data work."""
    counter=tmp_path/'fixture_invocations.txt'
    code=(
        'from pathlib import Path\n'
        'import numpy as np\n'
        'def run_packing():\n'
        f'    counter = Path({str(counter)!r})\n'
        '    counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else "1")\n'
        '    centers = np.zeros((26, 2), dtype=np.float32)\n'
        '    centers[0] = [.25, .25]\n'
        '    centers[1] = [.75, .75]\n'
        '    radii = np.zeros(26, dtype=np.float32)\n'
        '    radii[:2] = .125\n'
        '    return centers, radii, .25\n'
    )
    candidate=tmp_path/'fixed_arrays.py';candidate.write_text(code)
    task=load_task(str(ROOT/'tasks/oe/circle_packing'))  # Metadata only, never initial_program.
    result=run_stage(task.evaluator_path,'full',str(candidate),timeout=10,python=sys.executable)
    assert result.ok and result.score==.25/2.635 and counter.read_text()=='1'
    returned=CascadeOutcome(result,result.score,'full')
    evaluation=evaluation_observation(returned)
    full=evaluation['native_geometry']
    assert full['source_sha256']==hashlib.sha256(candidate.read_bytes()).hexdigest()==digest(code)
    assert full['evaluator_sha256']==hashlib.sha256(Path(task.evaluator_path).read_bytes()).hexdigest()
    assert full['centers_dtype']==full['radii_dtype']=='float32' and full['stage']=='full'
    assert full['numeric']['centers'][:2]==[[.25,.25],[.75,.75]]
    assert full['numeric']['radii']==[.125,.125]+[0.]*24
    c,parent,db,events=controller()  # Explicit synthetic .05 parent; no parent run is claimed.
    assert c.metric==metric_context(task)
    snap=snapshot(c,parent,db)
    ident,_=c.begin(snap,0,'full_rewrite',build('full_rewrite',snap))
    assert c.proposal(ident,response(code),code,NS(code=code))
    c.resolve(ident,status='evaluated',score=result.score,source=code,evaluation=evaluation)
    c.complete_step()
    record,=c.knowledge.experiments
    assert record['evaluation']['native_geometry']==full
    assert record['score']==result.score and record['actual_delta']==result.score-parent.score
    assert next(e for e in events if e['phase']=='outcome')['evidence']['evaluation']['native_geometry']==full
    saved=json.loads(json.dumps(c.state(),allow_nan=False))
    restored,_,_,_=controller();restored.load_state(saved)
    assert restored.state()==saved
    prompt=conclusion_prompt(restored.knowledge.experiments[0])
    projected=decode_evidence(json.loads(prompt.split('EVIDENCE_JSON:\n',1)[1]))
    assert_view(projected['evaluation']['native_geometry'],full)
    assert '"numeric"' not in prompt
    certificate=certify_artifact(restored.knowledge.experiments[0]['evaluation']['native_geometry'])
    assert certificate['native_artifact_sha256']==digest(canonical(full))
    assert certificate['source_sha256']==full['source_sha256']
    assert certificate['invocation_id']==full['invocation_id']
    assert certificate['certificate']['status']=='certified'
    assert certificate['certificate']['input_numeric_sha256']==full['numeric_sha256']
    assert certificate['certificate']['sum_exact']=={'numerator':1,'denominator':4}
    assert restored.state()==saved and counter.read_text()=='1'
