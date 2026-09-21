"""Synthetic immutable receipts test evidence binding, never execute source.

The fake task metric is prescribed by this fixture. These tests are not evidence
that a method is useful, and no provider/container is invoked.
"""
from copy import deepcopy

import pytest

from sciel.methods.evidence import EvidenceAssessor, EvidenceRefused
from sciel.methods.models import content_hash
from sciel.methods.registry import Registry

IMAGE = 'sha256:' + 'a' * 64
RUNTIME = dict(schema='synthetic-executor-fixture/1', image_id=IMAGE)
TASK = dict(schema='synthetic-task-fixture/1', metric='quality', direction='max')
LIMITS = dict(wall_seconds=2, cpu_seconds=1, memory_bytes=64*1024*1024,
              pids=8, max_output_bytes=4096, max_stderr_bytes=2048)
CRITERION = dict(metric='quality', direction='max', minimum_effect=0)
SCOPE = 'Only the preassigned synthetic cases; not a generalization experiment.'


def setup(tmp_path):
    registry = Registry(tmp_path/'registry')
    for name in ('author', 'review1', 'review2', 'other'):
        registry.create_lab(name, 'Synthetic evidence binding fixture', 'account-'+name, 'C',
            dict(search_procedure='a'*64, runtime_lock={'image_id':IMAGE},
                 active_methods=[], active_adapters=[], generator={}, capabilities={}, evidence_refs=[]))
    manifest = dict(name='fixture', kind='tool', author_lab='author', entrypoint='method.py:run',
        input_schema={'type':'integer'}, output_schema={'type':'integer'}, protocol='Synthetic fixture',
        runtime_lock={'image_id':IMAGE}, dependencies=[], parents=[], resources=LIMITS,
        scope=SCOPE, examples=[])
    method = registry.register({'method.py':'def run(value):\n    return value\n'}, manifest)
    assessor = EvidenceAssessor(registry)
    return registry, assessor, method


def freeze(assessor, method, **changes):
    args = dict(treatment=[method], control=[], scope=SCOPE, task=TASK,
                runtime=RUNTIME, resources=LIMITS, criterion=CRITERION)
    args.update(changes)
    return assessor.freeze_holdouts(method,
        [dict(reviewer_lab='review1', cases=[dict(id='case-1', input=2)]),
         dict(reviewer_lab='review2', cases=[dict(id='case-2', input=3)])], **args)


def plan(registry, assessor, method, lab='author', protocol=None, omit_measurement=False, **changes):
    data = dict(schema='sciel-method-plan/1', lab_id=lab,
        lab_revision_id=registry.lab(lab)['active_revision'], history_id=lab,
        purpose='heldout' if protocol else 'local', selected=[method], source_plan=None,
        treatment=[method], control=[], inputs=[1], scope=SCOPE, task=TASK,
        measurement_protocols={arm:TASK for arm in ('control','treatment')},
        runtime=RUNTIME, resources=LIMITS, criterion=CRITERION, question='Fixture?',
        input_origin='mere label must never establish independent evidence', proposal_id='fixture')
    if protocol:
        assignment = next(a for a in registry.get_artifact('holdout_protocol', protocol)['assignments']
                          if a['reviewer_lab']==lab)
        data['inputs'] = [c['input'] for c in assignment['cases']]
        data.update(changes)
        if omit_measurement:
            data.pop('measurement_protocols', None)
        data.update(assessor.validate_heldout_plan(protocol, assignment['id'], data))
    else:
        data.update(changes)
        if omit_measurement:
            data.pop('measurement_protocols', None)
    return registry.put_artifact('experiment_plan', data)


def observe(registry, plan_id, *, treatment_metric=1, control_metric=0, failed=False,
            receipt_change=None, corrupt_summary=False):
    p = registry.get_artifact('experiment_plan', plan_id)
    refs, results = [], {}
    for arm in ('control', 'treatment'):
        results[arm] = []
        for case, value in enumerate(p['inputs']):
            output, status = value, 'ok'
            for stage, method in enumerate(p[arm]):
                invocation = content_hash(dict(plan=plan_id, arm=arm, case=case, stage=stage))
                status = 'failed' if failed and arm=='treatment' else 'ok'
                receipt = dict(schema='synthetic-receipt-fixture/1', method_id=method,
                    invocation_id=invocation, plan_id=plan_id, reservation_id='reservation-'+invocation,
                    status=status, dispatch_status='invoked', output=output if status=='ok' else None,
                    input_sha256=content_hash(output), output_sha256=content_hash(output) if status=='ok' else None,
                    source_sha256=content_hash(registry.get_method(method)['files']),
                    runtime_sha256=content_hash(p['runtime']))
                if receipt_change:
                    receipt.update(receipt_change)
                refs.append(registry.put_artifact('call_receipt',dict(schema='sciel-method-call/1',
                    key='invoke:'+invocation, account='account-'+p['lab_id'], category='execution',
                    reservation_id='reservation-'+invocation, result=receipt,
                    status='completed' if status=='ok' else 'failed', actual={'cpu_seconds':.01})))
                if status!='ok':
                    break
            measurement = None
            if status=='ok':
                measurement=dict(valid=arm=='treatment', metrics={'quality':treatment_metric if arm=='treatment' else control_metric})
                registry.put_artifact('call_receipt', dict(schema='sciel-method-call/1',
                    key='measure:'+content_hash(dict(plan=plan_id,arm=arm,case=case)),
                    account='account-'+p['lab_id'], category='measurement',status='completed',
                    result=dict(status='ok',value=measurement)))
            results[arm].append(dict(input_sha256=content_hash(value), status=status,
                output_sha256=content_hash(output) if status=='ok' else None, measurement=measurement))
    return registry.put_artifact('experiment',dict(schema='sciel-method-experiment/1',plan_id=plan_id,
        results=results,invocation_refs=refs,invoked_methods=p['treatment'],
        # Assessor must recompute rather than trust this deliberately false summary.
        rule_passed=True,mean_effect=999 if corrupt_summary else treatment_metric-control_metric))


def cohort(tmp_path, *, metric=1, failed=False, receipt_change=None, direction='max'):
    registry, assessor, method = setup(tmp_path)
    criterion={**CRITERION,'direction':direction}
    protocol = freeze(assessor,method,criterion=criterion)
    original = observe(registry,plan(registry,assessor,method,criterion=criterion),
                       treatment_metric=1 if direction=='max' else -1)
    heldout = [observe(registry, plan(registry,assessor,method,lab,protocol,criterion=criterion),
                       treatment_metric=metric, failed=failed, receipt_change=receipt_change,
                       corrupt_summary=True)
               for lab in ('review1','review2')]
    return registry,assessor,method,protocol,original,heldout


def test_complete_preassigned_cohort_recomputes_benefit_and_is_idempotent(tmp_path):
    r,a,m,p,o,h = cohort(tmp_path)
    ident=a.assess_benefit(m,o,h,protocol_id=p)
    evidence=r.get_artifact('method_evidence',ident)
    assert evidence['level']=='benefit_replicated'
    assert [v['mean_effect'] for v in evidence['heldout_comparisons']]==[1,1]
    assert len(evidence['supporting'])==3
    assert evidence['scope']==SCOPE
    assert a.assess_benefit(m,o,h,protocol_id=p)==ident
    assert 'no statistical confidence' in evidence['uncertainty']


@pytest.mark.parametrize('metric,failed,expected',[(0,False,'contradicting'),(-1,False,'contradicting'),(1,True,'inconclusive')])
def test_all_assigned_negative_ties_and_failures_retained(tmp_path,metric,failed,expected):
    r,a,m,p,o,h=cohort(tmp_path,metric=metric,failed=failed)
    evidence=r.get_artifact('method_evidence',a.assess_benefit(m,o,h,protocol_id=p))
    assert evidence['level']=='benefit_not_replicated'
    assert evidence[expected]==h
    assert len(evidence['heldout_comparisons'])==2


def test_minimization_uses_task_direction_not_larger_is_better(tmp_path):
    r,a,m,p,o,h=cohort(tmp_path,metric=-2,direction='min')
    evidence=r.get_artifact('method_evidence',a.assess_benefit(m,o,h,protocol_id=p))
    assert evidence['level']=='benefit_replicated'
    assert evidence['heldout_comparisons'][0]['mean_effect']==2


@pytest.mark.parametrize('which', ['subset','same','author'])
def test_no_cherry_picking_cache_replay_or_author_check(tmp_path,which):
    r,a,m,p,o,h=cohort(tmp_path)
    bad={'subset':h[:1],'same':[h[0],h[0]],'author':[o,h[0]]}[which]
    with pytest.raises(EvidenceRefused):
        a.assess_benefit(m,o,bad,protocol_id=p)


@pytest.mark.parametrize('change', [
    {'dispatch_status':'not_dispatched'}, {'source_sha256':'b'*64},
    {'runtime_sha256':'b'*64}, {'input_sha256':'b'*64}, {'output_sha256':'b'*64},
    {'invocation_id':'b'*64}, {'reservation_id':'different'}, {'status':'unknown'}])
def test_actual_invocation_and_exact_source_runtime_dataflow_are_required(tmp_path,change):
    r,a,m,p,o,h=cohort(tmp_path,receipt_change=change)
    with pytest.raises(EvidenceRefused):
        a.assess_benefit(m,o,h,protocol_id=p)


def test_known_local_development_allowed_but_not_preselected_original(tmp_path):
    r,a,m=setup(tmp_path)
    old=observe(r,plan(r,a,m,inputs=[20]))
    p=freeze(a,m)
    with pytest.raises(EvidenceRefused,match='preregistration'):
        a.assess_benefit(m,old,['a'*64,'b'*64],protocol_id=p)
    # A genuinely subsequent author comparison can use the frozen protocol.
    o=observe(r,plan(r,a,m))
    h=[observe(r,plan(r,a,m,lab,p)) for lab in ('review1','review2')]
    assert r.get_artifact('method_evidence',a.assess_benefit(m,o,h,protocol_id=p))['level']=='benefit_replicated'


def test_prior_input_use_or_late_input_leak_refused(tmp_path):
    r,a,m=setup(tmp_path)
    plan(r,a,m,inputs=[2])
    with pytest.raises(EvidenceRefused,match='previously observed'):
        freeze(a,m)
    r,a,m,p,o,h=cohort(tmp_path/'late')
    plan(r,a,m,inputs=[3])
    with pytest.raises(EvidenceRefused,match='leaked'):
        a.assess_benefit(m,o,h,protocol_id=p)


@pytest.mark.parametrize('field,value', [('inputs',[99]),('scope','another scope'),
    ('criterion',dict(metric='quality',direction='max',minimum_effect=2)),
    ('resources',{**LIMITS,'cpu_seconds':2}), ('history_id','author'),('purpose','replay')])
def test_assigned_plan_cannot_change_cases_criterion_resources_or_identity(tmp_path,field,value):
    r,a,m=setup(tmp_path)
    p=freeze(a,m)
    with pytest.raises(EvidenceRefused):
        plan(r,a,m,'review1',p,**{field:value})


def test_reviewer_cannot_have_prior_author_answer_or_other_history(tmp_path):
    r,a,m=setup(tmp_path)
    r.put_artifact('proposal_context',dict(lab_id='review1',answer='prior evidence'))
    with pytest.raises(EvidenceRefused,match='history'):
        freeze(a,m)
    r,a,m,p,o,h=cohort(tmp_path/'later')
    r.put_artifact('proposal_context',dict(lab_id='review2',answer='after assignment'))
    with pytest.raises(EvidenceRefused,match='history'):
        a.assess_benefit(m,o,h,protocol_id=p)


def test_method_only_in_dependency_composition_is_not_credited(tmp_path):
    r,a,m=setup(tmp_path)
    bundle=r.get_method(m)
    manifest={**bundle['manifest'],'name':'wrapper','dependencies':[m]}
    wrapper=r.register({'method.py':'def run(value):\n    return value + 0\n'},manifest)
    with pytest.raises(EvidenceRefused,match='top-level'):
        freeze(a,m,treatment=[wrapper])
    with pytest.raises(EvidenceRefused,match='control includes'):
        freeze(a,m,control=[wrapper])


def test_label_alone_cannot_create_heldout_evidence(tmp_path):
    r,a,m=setup(tmp_path)
    p=freeze(a,m)
    o=observe(r,plan(r,a,m))
    h=[observe(r,plan(r,a,m,lab,inputs=[index+2],purpose='heldout'))
       for index,lab in enumerate(('review1','review2'))]
    with pytest.raises(EvidenceRefused,match='assignment'):
        a.assess_benefit(m,o,h,protocol_id=p)


def test_cannot_pick_later_better_author_comparison(tmp_path):
    r,a,m=setup(tmp_path)
    p=freeze(a,m)
    observe(r,plan(r,a,m,inputs=[1]),treatment_metric=0)
    later=observe(r,plan(r,a,m,inputs=[4]))
    h=[observe(r,plan(r,a,m,lab,p)) for lab in ('review1','review2')]
    with pytest.raises(EvidenceRefused,match='first matching author'):
        a.assess_benefit(m,later,h,protocol_id=p)


def test_author_comparison_cannot_be_selected_after_heldout_runs(tmp_path):
    r,a,m=setup(tmp_path)
    p=freeze(a,m)
    h=[observe(r,plan(r,a,m,lab,p)) for lab in ('review1','review2')]
    original=observe(r,plan(r,a,m))
    with pytest.raises(EvidenceRefused,match='before held-out'):
        a.assess_benefit(m,original,h,protocol_id=p)


def test_reviewer_history_is_refused_before_plan_dispatch(tmp_path):
    r,a,m=setup(tmp_path)
    p=freeze(a,m)
    r.put_artifact('proposal_context',dict(lab_id='review1',answer='contamination'))
    with pytest.raises(EvidenceRefused,match='history'):
        plan(r,a,m,'review1',p)


def test_measurement_cannot_be_replaced_by_unbound_experiment_numbers(tmp_path):
    r,a,m,p,o,h=cohort(tmp_path)
    body=r.get_artifact('experiment',h[0])
    body['results']['treatment'][0]['measurement']['metrics']['quality']=100
    altered=r.put_artifact('experiment',body)
    with pytest.raises(EvidenceRefused,match='trusted task measurement'):
        a.assess_benefit(m,o,[altered,h[1]],protocol_id=p)


def test_different_case_labels_do_not_make_same_input_independent(tmp_path):
    r,a,m=setup(tmp_path)
    with pytest.raises(EvidenceRefused,match='repeats a cohort'):
        a.freeze_holdouts(m,
            [dict(reviewer_lab='review1',cases=[dict(id='case-a',input=2)]),
             dict(reviewer_lab='review2',cases=[dict(id='case-b',input=2)])],
            treatment=[m],control=[],scope=SCOPE,task=TASK,runtime=RUNTIME,
            resources=LIMITS,criterion=CRITERION)


def test_imported_protocol_cannot_claim_zero_reviewer_replication(tmp_path):
    r,a,m=setup(tmp_path)
    p=freeze(a,m)
    body=r.get_artifact('holdout_protocol',p)
    body['assignments']=[]
    fake=r.put_artifact('holdout_protocol',body)
    o=observe(r,plan(r,a,m))
    with pytest.raises(EvidenceRefused,match='cohort size'):
        a.assess_benefit(m,o,[],protocol_id=fake)


@pytest.mark.parametrize('arm', ['control','treatment','missing'])
def test_cascade_or_other_task_measurement_cannot_replace_frozen_comparison(tmp_path,arm):
    r,a,m=setup(tmp_path)
    p=freeze(a,m)
    protocols={name:TASK for name in ('control','treatment')}
    if arm=='missing':
        protocols=None
    else:
        protocols[arm]={'schema':'another-measurement-protocol/1'}
    with pytest.raises(EvidenceRefused,match='measurement protocol'):
        plan(r,a,m,'review1',p,measurement_protocols=protocols,omit_measurement=arm=='missing')
    original=observe(r,plan(r,a,m,measurement_protocols=protocols,omit_measurement=arm=='missing'))
    h=[observe(r,plan(r,a,m,lab,p)) for lab in ('review1','review2')]
    with pytest.raises(EvidenceRefused,match='measurement protocol'):
        a.assess_benefit(m,original,h,protocol_id=p)
