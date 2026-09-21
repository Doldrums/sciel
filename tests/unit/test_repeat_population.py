"""Pure two-island population transitions with independently certified fake arrays.

No candidate, evaluator, optimizer, subprocess or model is executed.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from types import SimpleNamespace as NS

import pytest

from sciel.engine.repeat_population import RepeatPopulation, _digest
from sciel.engine.repeat_promotion import bootstrap_decision, paired_dominance
from sciel.engine.types import Program
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base.paired_comparisons import UNKNOWN_RECEIPT
from test_paired_comparisons import receipt, ADAPTER, LIMITS

CODE='# inert original source\n'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
METRIC=metric_context(NS(qualified_name='oe/circle_packing',task_hash='a'*64,primary_metric='combined_score',
    research_metric=dict(kind='validity_gated_ratio',score_units='dimensionless',numerator_case='sum_radii',
        numerator_units='unit_square_side_length',denominator=2.635,validity_case='validity',valid_value=1.,
        invalid_score=0.,invalid_numerator_value=0.,invalid_numerator_semantics='penalty_sentinel_not_physical_measurement')))
TASK=dict(task='oe/circle_packing',task_hash='a'*64,canonical_initial_source_sha256=sha(CODE),task_evaluator_sha256='e'*64)
BINDING=dict(task_metric=METRIC,adapter_identity=ADAPTER,limits=LIMITS,task_identity=TASK)


def program(ident, island=0, *, parent=None, score=.4, code=None, validity=1.):
    return Program(ident,code or ('# inert source '+ident+'\n'),score,
        dict(sum_radii=score*2.635,combined_score=score,validity=validity),island,'fake',parent_id=parent)


def packet(pop, child=None, *, old=(.1,.1,.1), new=(.11,.11,.11), failures=None, xs=None):
    """Actual service wire shape with complete real pure geometry receipts."""
    d=pop.state()['projection']; sequence=len(d['batches']); batch=f'b{sequence:06d}'
    bootstrap=child is None; context=None if bootstrap else d['pending']['context']
    source=sha(CODE if bootstrap else child.code)
    comparator=None if bootstrap else context['comparator_source_sha256']
    code=None if bootstrap else d['history'][context['comparator_program_id']]['code']
    rows=[]; values={'bootstrap':[],'challenger':[],'incumbent':[]}
    failures=failures or {};xs=xs or {}
    for index,seed in enumerate(range(10+sequence*3,13+sequence*3)):
        roles=['bootstrap'] if bootstrap else (['challenger','incumbent'] if (sequence+index)%2 else ['incumbent','challenger'])
        for role in roles:
            src=comparator if role=='incumbent' else source
            fail=failures.get((role,index),'valid')
            r=(None if fail is None else deepcopy(UNKNOWN_RECEIPT) if fail=='unknown' else
               receipt(src,seed,old[index] if role in ('incumbent','bootstrap') else new[index],
                   x=xs.get((role,index),.5),status=fail))
            draw=dict(draw_id=f'{batch}-{role}-{index}',role=role,draw_index=index,source_sha256=src,seed=seed)
            rows.append({**draw,'receipt':r});values[role].append(r)
    plan=dict(batch_id=batch,sequence=sequence,kind='bootstrap' if bootstrap else 'comparison',challenger=source,
        incumbent=comparator,program_id=None if bootstrap else child.id,screening=None,
        draws=[{k:v for k,v in row.items() if k!='receipt'} for row in rows],reserved_wall_seconds=(3 if bootstrap else 6)*40.)
    if not bootstrap:
        plan.update(context=deepcopy(context),research_context={k:context[k] for k in ('branch','origin_experiment_id','source_sha256')})
    result=bootstrap_decision(values['bootstrap']) if bootstrap else paired_dominance(values['incumbent'],values['challenger'])
    statuses=[dict(draw_id=r['draw_id'],status=r['receipt']['status'] if r['receipt'] is not None else 'not_dispatched') for r in rows]
    stop=('unknown_dispatch' if any(v['status']=='unknown' for v in statuses) else
          'infrastructure' if any(v['status']=='infrastructure' for v in statuses) else
          'bootstrap_failed' if bootstrap and result['status']!='confirmed' else
          'comparator_failed' if not bootstrap and bootstrap_decision(values['incumbent'])['aggregates'] is None else None)
    decision=dict(decision_id='d'+batch[1:],batch_id=batch,result=result,draw_statuses=statuses,stopped_reason=stop)
    return dict(namespace='fixture:population',batch_id=batch,kind=plan['kind'],plan=plan,decision=decision,receipts=rows,
        source_pair=dict(comparator=code,challenger=CODE if bootstrap else child.code),adapter_identity=deepcopy(ADAPTER),
        limits=deepcopy(LIMITS),task_metric=deepcopy(METRIC),task_identity=deepcopy(TASK),
        decision_anchor=dict(entries=10+sequence*20,sha256=str(sequence%10)*64))


class RNG:
    def __init__(self,value=.99):self.value=value;self.calls=0
    def random(self):self.calls+=1;return self.value


def fresh():
    pop=RepeatPopulation(dict(num_islands=2,pop_cap=2),deepcopy(BINDING))
    seeds=[program('seed0',0,code=CODE),program('seed1',1,code=CODE)]
    initial=packet(pop)
    pop.bootstrap(seeds,initial)
    return pop,{p.id:p for p in seeds},[initial]


def pending(pop, history, ident, island=0, *, score=.4, rng=.99, attribution=None, code=None, validity=1.):
    view=pop.select(island,RNG(rng))
    child=program(ident,island,parent=view['parent_id'],score=score,code=code,validity=validity)
    history[ident]=child
    stage=pop.stage(child,view,attribution)
    return child,view,stage


def finish(pop,history,packets,ident,island=0,**kwargs):
    child,view,stage=pending(pop,history,ident,island,score=kwargs.pop('score',.4),rng=kwargs.pop('rng',.99))
    assert stage['eligible']
    terminal=packet(pop,child,**kwargs);packets.append(terminal)
    return pop.apply(terminal),child,view,terminal


def test_two_islands_two_slots_ordinary_score_does_not_admit_or_select_rejected_source():
    pop,history,packets=fresh(); original=deepcopy({k:asdict(v) for k,v in history.items()})
    rejected,high,_,_=finish(pop,history,packets,'high',score=100.,new=(.09,.09,.09))
    assert rejected['reward']==0 and rejected['status']=='rejected'
    assert pop.active_ids(0)==['seed0'] and pop.active_ids(1)==['seed1']
    assert high.id not in pop.inspirations(0,'seed0')
    admitted,low,view,_=finish(pop,history,packets,'low',score=.01,new=(.2,.2,.2))
    assert admitted['reward']==1 and admitted['status']=='admitted' and view['victim_id'] is None
    assert pop.active_ids(0)==['seed0','low'] and pop.active_ids(1)==['seed1']
    ranks=pop.ranked(0)
    assert [x['program_id'] for x in ranks]==['seed0','low']
    assert ranks[1]['probability']>ranks[0]['probability']>=.1/2
    assert pop.publication_source_sha256==sha(CODE)
    assert {k:asdict(history[k]) for k in original}==original and low.score==.01 and high.score==100.


def test_full_pool_freezes_victim_different_from_parent_and_publication_never_changes():
    pop,history,packets=fresh()
    finish(pop,history,packets,'member',new=(.2,)*3)
    child,view,stage=pending(pop,history,'replacement',rng=.99,attribution={'branch':'seed0','origin_experiment_id':'r3'})
    assert view['parent_id']=='member' and view['comparator_program_id']==view['victim_id']=='seed0'
    assert stage['context']['branch']=='seed0' and stage['context']['origin_experiment_id']=='r3'
    before=pop.active_ids(0);terminal=packet(pop,child,new=(.15,)*3)
    transition=pop.apply(terminal)
    assert before==['seed0','member'] and pop.active_ids(0)==['replacement','member']
    assert transition['status']=='replaced' and transition['parent_id']=='member'
    assert pop.publication_source_sha256==sha(CODE)
    assert pop.inspirations(0,'member')==['replacement']
    before=pop.state();same=pop.apply(deepcopy(terminal))
    assert same['already_applied'] is True and pop.state()==before and len(before['projection']['rewards'])==2


def test_rejected_challenger_refreshes_same_source_evidence_across_islands():
    pop,history,packets=fresh()
    finish(pop,history,packets,'right',1,new=(.15,)*3)
    assert pop.ranked(1)[0]['program_id']=='seed1'
    before=pop.state()['projection']['evidence_heads'][sha(CODE)]['id']
    t,_,_,_=finish(pop,history,packets,'reject',0,old=(.2,)*3,new=(.16,)*3)
    assert t['status']=='rejected' and pop.active_ids(0)==['seed0']
    assert pop.ranked(1)[-1]['program_id']=='seed1'
    current=pop.state()['projection']['evidence_heads'][sha(CODE)]
    assert before!=current['id'] and current['metrics']['raw']['minimum']==.2
    assert pop.ranked(0)[0]['evidence_id']==pop.ranked(1)[-1]['evidence_id']==current['id']


def test_equal_evidence_average_ranks_uniform_and_source_id_victim_tiebreak():
    pop,history,packets=fresh()
    finish(pop,history,packets,'other',1,new=(.2,)*3)
    # An island0 rejection refreshes the shared initial source to exactly .2.
    finish(pop,history,packets,'rejected',old=(.2,)*3,new=(.1,)*3)
    ranks=pop.ranked(1)
    assert [x['rank'] for x in ranks]==[1.5,1.5]
    assert [x['probability'] for x in ranks]==[.5,.5]
    expected=min(('seed1','other'),key=lambda ident:(sha(history[ident].code),ident))
    rng=RNG(0.);view=pop.select(1,rng)
    assert rng.calls==1 and view['comparator_program_id']==view['victim_id']==expected
    assert view['parent_propensity']==.5


@pytest.mark.parametrize('new,old,xs',[
    ((.10000002,)*3,(.10000005,)*3,{('incumbent',i):.1 for i in range(3)}),
    ((.10000005,)*3,(.10000002,)*3,{('challenger',i):.1 for i in range(3)}),
    ((.3,.3,.09),(.1,)*3,{}),
    ((.1,)*3,(.1,)*3,{})])
def test_no_epsilon_means_or_one_metric_can_rescue_a_failed_pair(new,old,xs):
    pop,history,packets=fresh()
    transition,_,_,_=finish(pop,history,packets,'trial',new=new,old=old,xs=xs)
    assert transition['status']=='rejected' and transition['reward']==0 and pop.active_ids(0)==['seed0']
    assert transition['result']['raw_deltas'] is not None


@pytest.mark.parametrize('role,status,stopped',[
    ('challenger','candidate_failure',False),('incumbent','candidate_failure',True),
    ('challenger','infrastructure',True),('incumbent','unknown',True)])
def test_failures_preserve_complete_comparator_or_stop_without_quality_zero(role,status,stopped):
    pop,history,packets=fresh();oldhead=pop.ranked(0)[0]['evidence_id']
    t,_,_,_=finish(pop,history,packets,'failure',old=(.12,)*3,failures={(role,1):status})
    assert pop.active_ids(0)==['seed0'] and bool(pop.stopped_reason)==stopped
    assert t['reward']==(None if stopped else 0) and t['result']['aggregates'] is None
    assert t['result']['raw_deltas'] is None
    newhead=pop.ranked(0)[0]['evidence_id']
    assert (newhead!=oldhead)==(role=='challenger')
    if stopped:
        with pytest.raises(ValueError,match='selection'):pop.select(0,RNG())


def test_unknown_tail_and_not_dispatched_remain_distinct_and_never_subset_mean():
    pop,history,packets=fresh()
    failures={('challenger',0):'unknown',('incumbent',0):None,('incumbent',1):None,
              ('challenger',1):None,('challenger',2):None,('incumbent',2):None}
    t,_,_,p=finish(pop,history,packets,'unknown',failures=failures)
    assert t['reward'] is None and pop.stopped_reason=='comparison_unknown'
    assert p['receipts'][0]['receipt']==UNKNOWN_RECEIPT and p['receipts'][1]['receipt'] is None
    assert pop.ranked(0)[0]['evidence_id']==pop.ranked(1)[0]['evidence_id']


def test_source_duplicate_invalid_candidate_and_engineering_failure_never_repeat_or_reward():
    pop,history,_=fresh()
    original=pop.active_ids(0)
    _,_,stage=pending(pop,history,'clone',code=CODE)
    assert stage==dict(eligible=False,reason='source_duplicate',context=None) and pop.pending is None
    _,_,stage=pending(pop,history,'invalid',validity=0.,score=0.)
    assert stage['reason']=='ordinary_invalid' and pop.active_ids(0)==original
    pop.select(1,RNG());out=pop.finish_attempt('hypothesis_invalid')
    assert out['reward'] is None and pop.summary()['rewards']==[]
    assert len(pop.state()['projection']['history'])==4 and pop.state()['projection']['visits']=={'seed0':2,'seed1':1}


def test_rejected_generated_source_cannot_receive_another_batch_on_another_island():
    pop,history,packets=fresh();_,child,_,_=finish(pop,history,packets,'first',new=(.09,)*3)
    _,_,stage=pending(pop,history,'other_id',1,code=child.code)
    assert stage['reason']=='source_duplicate' and len(pop.state()['projection']['batches'])==2


def test_pending_budget_stop_retains_history_and_snapshot_no_reward_or_admission():
    pop,history,_=fresh();child,view,_=pending(pop,history,'unexamined')
    pop.stop('comparison_budget_unavailable')
    assert pop.pending['program_id']==child.id and pop.state()['projection']['selection']==view
    assert pop.active_ids(0)==['seed0'] and pop.summary()['rewards']==[]
    with pytest.raises(ValueError):pop.select(1,RNG())


@pytest.mark.parametrize('phase',['selected','pending','applied'])
def test_restore_deterministic_membership_probabilities_visits_and_exactly_once_reward(phase):
    pop,history,packets=fresh()
    if phase=='selected':pop.select(1,RNG(.25))
    else:
        child,_,_=pending(pop,history,'new')
        if phase=='applied':
            p=packet(pop,child);packets.append(p);pop.apply(p)
    saved=pop.state();restored=RepeatPopulation(pop.config,BINDING)
    restored.load_state(deepcopy(saved),history)
    assert restored.state()==saved and restored.assert_batches(packets)
    assert restored.ranked(0)==pop.ranked(0)
    if phase=='applied':
        assert restored.apply(packets[-1])['already_applied'] and restored.state()==saved
    else:
        if phase=='pending':
            p=packet(restored,child);restored.apply(p);pop.apply(p)
            assert restored.state()==pop.state()
        else:
            restored.finish_attempt('no_child');pop.finish_attempt('no_child')
            assert restored.select(0,RNG(.31))==pop.select(0,RNG(.31))


def test_loaded_state_and_external_programs_are_bound_not_silently_recomputed():
    pop,history,packets=fresh();finish(pop,history,packets,'member')
    saved=pop.state(); restored=RepeatPopulation(pop.config,BINDING)
    mutated=deepcopy(saved);mutated['projection']['active']['0']=['seed0']
    with pytest.raises(ValueError,match='replay'):restored.load_state(mutated,history)
    changed=deepcopy(history);changed['member'].score+=1
    with pytest.raises(ValueError,match='original Program'):restored.load_state(saved,changed)
    with pytest.raises(ValueError,match='missing'):restored.load_state(saved,{'seed0':history['seed0']})
    restored.load_state(saved,history);history['member'].cases['sum_radii']+=1
    with pytest.raises(ValueError,match='changed'):restored.select(0,RNG())


@pytest.mark.parametrize('target',['result','source','comparator','adapter','limit','seed','array','snapshot','branch','metric'])
def test_terminal_tampering_is_atomic(target):
    pop,history,_=fresh();child,_,_=pending(pop,history,'child');p=packet(pop,child);before=pop.state()
    if target=='result':p['decision']['result']['status']='reject'
    elif target=='source':p['source_pair']['challenger']+='\n'
    elif target=='comparator':p['plan']['incumbent']='f'*64
    elif target=='adapter':p['adapter_identity']['fixture']='swapped'
    elif target=='limit':p['limits']['candidate_wall_seconds']=31.
    elif target=='seed':p['receipts'][0]['receipt']['seed']+=1
    elif target=='array':p['receipts'][0]['receipt']['capture']['radii'][0]+=.01
    elif target=='snapshot':p['plan']['context']['prospective_snapshot']['parent_propensity']=1.
    elif target=='branch':p['plan']['context']['branch']='other'
    else:p['task_metric']['sha256']='f'*64
    # Single-member propensity already1; use a genuinely different value.
    if target=='snapshot':p['plan']['context']['prospective_snapshot']['parent_propensity']=.2
    with pytest.raises(ValueError):pop.apply(p)
    assert pop.state()==before


def test_branch_attribution_validates_real_ancestry_atomically():
    pop,history,packets=fresh();finish(pop,history,packets,'member',new=(.2,)*3)
    view=pop.select(0,RNG());child=program('child',parent='member')
    before=pop.state()
    with pytest.raises(ValueError,match='branch'):pop.stage(child,view,{'branch':'member'})
    assert pop.state()==before
    assert pop.stage(child,view,{'branch':'seed0'})['context']['branch']=='seed0'


def test_durable_batch_binding_refuses_self_consistent_changed_packet_and_allows_valid_tail():
    pop,history,packets=fresh();child,_,_=pending(pop,history,'child');terminal=packet(pop,child)
    assert pop.assert_batches(packets+[terminal])
    pop.apply(terminal);packets.append(terminal)
    assert pop.assert_batches(deepcopy(packets))
    changed=deepcopy(packets);changed[-1]['decision_anchor']['sha256']='a'*64
    with pytest.raises(ValueError,match='durable'):pop.assert_batches(changed)
    with pytest.raises(ValueError,match='missing'):pop.assert_batches(packets[:1])
    with pytest.raises(ValueError,match='decision ID'):pop.apply(changed[-1])


def test_bootstrap_failure_and_mutated_inputs_do_not_create_active_members():
    pop=RepeatPopulation(dict(num_islands=2,pop_cap=2),deepcopy(BINDING))
    initial=packet(pop,failures={('bootstrap',1):'candidate_failure'})
    seeds=[program('a',0,code=CODE),program('b',1,code=CODE)]
    pop.bootstrap(seeds,initial)
    assert pop.initialized and not pop.ready and pop.active_ids(0)==pop.active_ids(1)==[]
    assert pop.publication_source_sha256 is None and pop.stopped_reason=='bootstrap_incomplete'
    initial['receipts'][0]['receipt']['raw_sum']=999
    assert pop.state()['projection']['batches']['b000000']['receipts'][0]['receipt']['raw_sum']==.1


@pytest.mark.parametrize('config',[{'num_islands':True,'pop_cap':2},{'num_islands':2,'pop_cap':1},
                                    {'num_islands':2,'pop_cap':2,'extra':True}])
def test_fixed_profile_rejects_invalid_config(config):
    with pytest.raises(ValueError):RepeatPopulation(config,BINDING)


def test_pending_pre_dispatch_binding_and_stopped_unexamined_exception():
    pop,history,_=fresh();child,_,_=pending(pop,history,'new');p=packet(pop,child)
    assert pop.assert_pending(p['plan'])
    altered=deepcopy(p['plan']);altered['context']['prospective_snapshot']['draw']=.1
    with pytest.raises(ValueError,match='durable'):pop.assert_pending(altered)
    with pytest.raises(ValueError,match='lacks'):pop.assert_pending(None)
    pop.stop('comparison_budget_unavailable');assert pop.assert_pending(None)
    with pytest.raises(ValueError,match='live pending'):pop.apply(p)


def test_source_runtime_identity_separation_and_no_durable_admission_without_bootstrap():
    pop,history,packets=fresh()
    changed=deepcopy(BINDING);changed['limits']['candidate_wall_seconds']=31.
    restored=RepeatPopulation(pop.config,changed)
    with pytest.raises(ValueError,match='identity'):restored.load_state(pop.state(),history)
    blank=RepeatPopulation(pop.config,BINDING)
    assert blank.assert_pending(packets[0]['plan'])
    assert blank.assert_pending(None)
    with pytest.raises(ValueError,match='selection'):blank.select(0,RNG())


def test_real_durable_service_packets_and_recovered_terminal_tail_project_exactly_once(tmp_path):
    from sciel.engine.paired_batches import PairedBatches
    from test_paired_batches import Draws, INITIAL, FIRST, identity as service_identity, config as service_config, ack
    from test_paired_batches import ADAPTER as SERVICE_ADAPTER
    identity=service_identity();draws=Draws()
    service=PairedBatches(tmp_path,service_config(),identity,draws)
    binding=dict(task_metric=identity['task_metric'],adapter_identity=SERVICE_ADAPTER,limits=LIMITS,
        task_identity={k:identity[k] for k in ('task','task_hash','canonical_initial_source_sha256','task_evaluator_sha256')})
    pop=RepeatPopulation(dict(num_islands=2,pop_cap=2),binding)
    originals=[program('a',0,code=INITIAL),program('b',1,code=INITIAL)]
    history={p.id:p for p in originals}
    try:
        service.start('population-fixture');assert service.plan_bootstrap(INITIAL)
        assert pop.assert_pending(service.state()['data']['batches'][-1]['plan'])
        ack(service);service.run_pending();pop.bootstrap(originals,service.terminal_batches()[0])
        view=pop.select(0,RNG(.4));child=program('child',parent='a',code=FIRST);history[child.id]=child
        staged=pop.stage(child,view,{'branch':'a','origin_experiment_id':'r1'})
        assert service.plan_comparison(FIRST,INITIAL,child.id,context=staged['context'])
        checkpoint=pop.state();durable=ack(service)
        assert pop.assert_pending(durable['data']['batches'][-1]['plan'])
        service.run_pending();terminal=service.terminal_batches()[-1];actual_count=len(draws.calls)
        recovered=RepeatPopulation(pop.config,binding);recovered.load_state(checkpoint,history)
        assert recovered.assert_batches(service.terminal_batches())
        result=recovered.apply(terminal)
        assert result['status']=='admitted' and recovered.active_ids(0)==['a','child']
        assert recovered.publication_source_sha256==sha(INITIAL)
        assert recovered.apply(terminal)['already_applied'] and len(recovered.summary()['rewards'])==1
        assert recovered.assert_batches(service.terminal_batches()) and len(draws.calls)==actual_count==9
    finally:service.close()
