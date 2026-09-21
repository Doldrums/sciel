"""Validate physical receipts before replaying a scientific outcome."""
import base64
from copy import deepcopy
import math
from pathlib import Path
import re
from types import SimpleNamespace

from ..methods.models import canonical_bytes, content_hash
from .candidate_evidence import SCHEMA, FAILURE_SCHEMA, BOUND, _read, sha
from .facade import EvalResult, EvaluationInfrastructureError
from .trusted_scoring.adapter import aggregate_completed, _failure
from .trusted_scoring.transport import MAX_BYTES


def validate_failure(value):
    fields={'schema','reason','execution','execution_sha256'}
    if type(value) is not dict or set(value)!=fields or value['schema']!=FAILURE_SCHEMA:
        raise ValueError('invalid candidate evaluation failure diagnostic')
    if value['reason'] not in {'execution_unknown','execution_refused','scorer_failure','receipt_invalid','attachment_failure','operation_context_missing'}:
        raise ValueError('invalid candidate evaluation failure reason')
    if value['execution_sha256']!=content_hash(value['execution']):
        raise ValueError('candidate evaluation failure evidence differs')
    validate_execution(value['execution'])
    return deepcopy(value)

class EvaluationBackendError(EvaluationInfrastructureError):
    def __init__(self,diagnostic):
        self.diagnostic=validate_failure(diagnostic)
        super().__init__('candidate-only evaluator infrastructure failure: '+diagnostic['reason'])

def validate_execution(value):
    fields={'schema','request_sha256','evaluation_id','identity','status','receipts','stages','resources','stage_scope'}
    if type(value) is not dict or set(value)!=fields or value['schema']!=SCHEMA:
        raise ValueError('invalid candidate evaluation execution record')
    if value['status'] not in ('completed','infrastructure_failed','unknown'):
        raise ValueError('invalid candidate evaluation execution status')
    if type(value['evaluation_id']) is not str or not value['evaluation_id']:
        raise ValueError('missing evaluation identity')
    if type(value['request_sha256']) is not str or re.fullmatch('[a-f0-9]{64}',value['request_sha256']) is None:
        raise ValueError('invalid evaluation request identity')
    if type(value['identity']) is not dict or value['identity'].get('schema')!='sciel-candidate-backend/1':
        raise ValueError('missing execution runtime identity')
    if type(value['stages']) is not list or len(value['stages'])>3:
        raise ValueError('invalid stage summary count')
    for ref in value['stages']:
        if type(ref) is not dict or set(ref)!={'name','sha256'} or type(ref['name']) is not str or re.fullmatch(r'(smoke|full|confirm)\.stage\.json',ref['name']) is None or type(ref['sha256']) is not str or re.fullmatch('[a-f0-9]{64}',ref['sha256']) is None:
            raise ValueError('invalid stage summary reference')
    refs=value['receipts']
    if type(refs) is not list or len(refs)>5:
        raise ValueError('invalid stage receipt count')
    names=set()
    for ref in refs:
        if type(ref) is not dict or set(ref)!={'name','sha256'} or type(ref['name']) is not str or re.fullmatch(r'(smoke|full|confirm)-[0-2]\.receipt\.json',ref['name']) is None or ref['name'] in names:
            raise ValueError('invalid stage receipt reference')
        names.add(ref['name'])
        if type(ref['sha256']) is not str or re.fullmatch('[a-f0-9]{64}',ref['sha256']) is None:
            raise ValueError('invalid stage receipt digest')
    resources=value['resources']
    if type(resources) is not dict or set(resources)!={'cpu_seconds','known_cpu_seconds','cpu_complete','wall_seconds','scope'} or type(resources['cpu_complete']) is not bool:
        raise ValueError('invalid evaluation resource record')
    for key in ('known_cpu_seconds','wall_seconds'):
        if type(resources[key]) not in (int,float) or not math.isfinite(resources[key]) or resources[key]<0:
            raise ValueError('invalid evaluation resource measurement')
    if resources['cpu_seconds'] is not None and (type(resources['cpu_seconds']) not in (int,float) or not math.isfinite(resources['cpu_seconds']) or resources['cpu_seconds']<0):
        raise ValueError('invalid complete CPU measurement')
    if resources['cpu_seconds'] != (resources['known_cpu_seconds'] if resources['cpu_complete'] else None):
        raise ValueError('unknown evaluation CPU cannot be zero-filled')
    if value['status']=='completed' and not resources['cpu_complete']:
        raise ValueError('completed evaluation lacks resource accounting')
    canonical_bytes(value)
    return value

def verify_execution(value,directory,request,request_hash,*,outcome=None):
    """Authenticate referenced physical evidence before any cached scientific result."""
    validate_execution(value)
    if (value['request_sha256']!=request_hash or value['evaluation_id']!=request.get('evaluation_id')
            or value['identity']!=request.get('evaluation_backend')):
        raise ValueError('evaluation execution differs from WAL request')
    actual_names={p.name for p in Path(directory).glob('*.receipt.json') if re.fullmatch(r'(smoke|full|confirm)-[0-2]\.receipt\.json',p.name)}
    if actual_names!={r['name'] for r in value['receipts']}:
        raise ValueError('evaluation physical receipt inventory differs')
    records=[]; observed_cpu=[]; complete_cpu=True
    for ref in value['receipts']:
        record=_read(Path(directory)/ref['name'])
        if record['sha256']!=ref['sha256'] or record['request_sha256']!=request_hash:
            raise ValueError('physical evaluation receipt differs')
        intent=_read(Path(directory)/ref['name'].replace('.receipt.','.intent.'))
        if record['intent_sha256']!=intent['sha256'] or intent['request_sha256']!=request_hash:
            raise ValueError('physical evaluation receipt lost its dispatch intent')
        expected=sha((request_hash+':'+record['stage']+':'+str(record['trial'])).encode())
        if intent['container_name']!='sciel-eval-'+expected or intent['invocation_id']!=expected[:32]:
            raise ValueError('physical evaluation ownership identity differs')
        records.append(record)
        method=record.get('method_receipt')
        if method is not None:
            if method.get('receipt_sha256')!=content_hash({k:v for k,v in method.items() if k!='receipt_sha256'}):
                raise ValueError('physical method receipt seal differs')
            if any(method.get(k)!=intent['method_request'].get(k) for k in BOUND):
                raise ValueError('physical method request differs')
            if method.get('container',{}).get('name')!=intent['container_name']:
                raise ValueError('physical container ownership differs')
        if value['status']=='completed':
            status,_=_validate_candidate_receipt(method,intent['method_request'],intent['container_name'],
                value['identity']['executor'],value['identity']['policy']['image_id'])
            if status not in ('ok','failed'):
                raise ValueError('completed evaluation contains unresolved candidate transport')
        scorer=record.get('scorer_receipt')
        scorer_ref=record.get('scorer_intent')
        if scorer_ref is not None:
            if scorer_ref.get('name')!=ref['name'].replace('.receipt.','.scorer-intent.'):
                raise ValueError('scorer dispatch reference differs')
            scorer_intent=_read(Path(directory)/scorer_ref['name'])
            if scorer_intent['sha256']!=scorer_ref.get('sha256') or scorer_intent['request_sha256']!=request_hash:
                raise ValueError('scorer dispatch intent differs')
            scorer_name='sciel-eval-'+sha((request_hash+':'+record['stage']+':'+str(record['trial'])+':scorer').encode())
            if scorer_intent['container_name']!=scorer_name:
                raise ValueError('scorer container ownership differs')
            if scorer is not None:
                if (scorer.get('receipt_sha256')!=content_hash({k:v for k,v in scorer.items() if k!='receipt_sha256'})
                        or any(scorer.get(k)!=scorer_intent['method_request'].get(k) for k in BOUND)
                        or scorer.get('container',{}).get('name')!=scorer_name):
                    raise ValueError('scorer physical receipt binding differs')
                if value['status']=='completed':
                    from .scorer_container import _bound_receipt
                    _bound_receipt(SimpleNamespace(identity=value['identity']['executor']),
                        scorer_intent['method_request'],scorer,scorer_name)
                    if (scorer.get('status')!='ok' or scorer.get('reason') is not None
                            or scorer.get('dispatch_status')!='invoked'
                            or scorer.get('resource_measurement',{}).get('cpu')!='trusted_cgroup_observation'
                            or scorer.get('output_sha256')!=content_hash(scorer.get('output'))
                            or scorer.get('container',{}).get('removed') is not True):
                        raise ValueError('completed scorer physical contract differs')
                    expected_input={'kind':value['identity']['kind'],
                        'payload_base64':base64.b64encode(method['output'].encode('utf-8')).decode('ascii'),
                        'binding':dict(source_sha256=request['source_sha256'],
                            evaluator_sha256=value['identity']['task']['evaluator_sha256'],
                            invocation_id=method['invocation_id'],
                            stage='full' if record['stage']=='confirm' else record['stage'],
                            elapsed_seconds=float(method['resources']['method_wall_seconds']))}
                    actual_request=scorer_intent['method_request']
                    if (actual_request['input']!=expected_input
                            or actual_request['input_sha256']!=content_hash(expected_input)
                            or actual_request['request_sha256']!=content_hash({k:v for k,v in actual_request.items() if k!='request_sha256'})):
                        raise ValueError('scorer input differs from exact candidate output and host binding')
        elif scorer is not None:
            raise ValueError('scorer receipt without intent')
        for physical in (method,scorer):
            if physical is None: continue
            cpu=(physical.get('resources') or {}).get('cpu_seconds')
            if cpu is not None:
                if type(cpu) not in (int,float) or not math.isfinite(cpu) or cpu<0:
                    raise ValueError('invalid physical CPU measurement')
                observed_cpu.append(float(cpu))
            if cpu is None or physical.get('status')=='unknown' or physical.get('container',{}).get('removed') is not True:
                complete_cpu=False
        if method is None and record.get('candidate_not_dispatched') is not True: complete_cpu=False
        if scorer_ref is not None and scorer is None and record.get('scorer_not_dispatched') is not True: complete_cpu=False
        for key, physical in (('candidate_not_dispatched', method), ('scorer_not_dispatched', scorer)):
            if type(record.get(key)) is not bool or record[key] and physical is not None:
                raise ValueError('invalid pretransport refusal evidence')
    resources=value['resources']
    known_cpu=math.fsum(observed_cpu)
    if resources['known_cpu_seconds']!=known_cpu or resources['cpu_complete'] and not complete_cpu:
        raise ValueError('aggregate evaluation CPU differs from physical evidence')
    # Each finished stage summary is another immutable attachment within this
    # same evaluation operation, never a separate journal or retry authority.
    actual_stages={p.name for p in Path(directory).glob('*.stage.json')}
    if actual_stages!={r['name'] for r in value['stages']}:
        raise ValueError('evaluation stage summary inventory differs')
    summaries=[]; expected_order=['smoke','full','confirm']
    for index,ref in enumerate(value['stages']):
        summary=_read(Path(directory)/ref['name'])
        if (summary['sha256']!=ref['sha256'] or summary['request_sha256']!=request_hash
                or summary['stage']!=expected_order[index] or ref['name']!=summary['stage']+'.stage.json'):
            raise ValueError('evaluation stage identity/order differs')
        summaries.append(summary)
        stage=summary['stage']; rows=[r for r in records if r['stage']==stage]
        if [r['trial'] for r in rows]!=list(range(len(rows))):
            raise ValueError('physical trial ordering differs')
        spec=value['identity']['effective_task']
        count=spec['confirm_n'] if stage=='confirm' else 1
        if summary['timeout']!=spec[stage+'_timeout'] or not 0<len(rows)<=count:
            raise ValueError('physical stage count or deadline differs')
        for row in rows:
            if row['failure'] is not None or row['result'] is None:
                raise ValueError('finished stage contains infrastructure failure')
            physical=row['method_receipt']; scored=row['scorer_receipt']
            if physical.get('status')=='failed':
                expected=_failure(value['identity']['kind'],'Candidate execution failed: '+str(physical.get('reason'))[:120])
            elif scored is not None and scored.get('status')=='ok' and scored['container'].get('removed') is True:
                wrapper=scored['output']
                if type(wrapper) is not dict or set(wrapper)!={'schema','result','runtime'} or wrapper['schema']!='sciel-trusted-scorer-output/1':
                    raise ValueError('invalid trusted scorer runtime envelope')
                runtime=wrapper['runtime']
                if type(runtime) is not dict or set(runtime)!={'python','numpy'} or any(type(v) is not str or not 0<len(v)<=128 or not v.isascii() or not v.isprintable() for v in runtime.values()):
                    raise ValueError('invalid trusted scorer runtime observation')
                expected=wrapper['result']
            else:
                try: physical['output'].encode('utf-8')
                except UnicodeError: expected=_failure(value['identity']['kind'],'Candidate output is not valid UTF-8')
                else:
                    expected=(_failure(value['identity']['kind'],'Candidate output exceeds bounded JSON transport')
                        if not 0 < len(physical['output'].encode('utf-8')) <= MAX_BYTES else
                        _failure(value['identity']['kind'],'Evaluation stage resource budget exhausted'))
            if row['result']!=expected:
                raise ValueError('candidate result differs from owned scorer receipt')
        cpu=math.fsum(float(physical['resources']['cpu_seconds']) for row in rows
            for physical in (row['method_receipt'],row['scorer_receipt']) if physical is not None)
        if type(summary['elapsed_seconds']) not in (int,float) or not math.isfinite(summary['elapsed_seconds']) or summary['elapsed_seconds']<0 or summary['cpu_seconds']!=cpu:
            raise ValueError('stage resource measurements differ')
        exhausted=(summary['elapsed_seconds']>=summary['timeout'] or cpu>=value['identity']['policy']['cpu_seconds'])
        if summary['resource_exhausted'] is not exhausted:
            raise ValueError('stage deadline/CPU decision differs')
        if exhausted:
            expected=_failure(value['identity']['kind'],'Evaluation stage resource budget exhausted')
            if summary['complete'] is not False: raise ValueError('exhausted stage claims completion')
        else:
            if len(rows)!=count or summary['complete'] is not True:
                raise ValueError('completed stage lacks all physical trials')
            expected=(aggregate_completed(value['identity']['kind'],[r['result'] for r in rows],count)
                      if stage=='confirm' else rows[0]['result'])
        if summary['result']!=expected:
            raise ValueError('stage selected result differs from physical results')
        if index and not summaries[index-1]['result']['ok']:
            raise ValueError('cascade advanced after failed stage')
        if stage=='confirm' and not (request['confirm_records'] and spec['stochastic']):
            raise ValueError('unexpected confirmation stage')
    if resources['wall_seconds'] < math.fsum(s['elapsed_seconds'] for s in summaries):
        raise ValueError('aggregate evaluation wall time is shorter than completed stages')
    if outcome is not None:
        if value['status']!='completed' or not resources['cpu_complete'] or not summaries:
            raise ValueError('returned scientific outcome lacks completed physical execution')
        consumed=sum(len([r for r in records if r['stage']==s['stage']]) for s in summaries)
        if consumed!=len(records): raise ValueError('unaccounted physical trials after final stage')
        last=summaries[-1]; result=EvalResult.from_json(last['result'])
        confirmed=bool(last['stage']=='confirm' and last['complete'] and result.ok and spec['confirm_n']>=2)
        result.confirmed_sampled=confirmed;result.timing=last['elapsed_seconds']
        raw=summaries[1]['result']['score'] if len(summaries)>1 else summaries[0]['result']['score']
        if (outcome['result']!=result.to_json() or outcome['reached']!=last['stage']
                or type(outcome['confirmed']) is not bool or outcome['confirmed']!=confirmed
                or type(outcome['raw_first_score']) not in (int,float) or outcome['raw_first_score']!=raw):
            raise ValueError('WAL scientific result differs from physical stage selection')
        if result.ok and (last['stage']=='smoke' or last['stage']=='full' and request['confirm_records'] and spec['stochastic']):
            raise ValueError('successful cascade omitted a required next stage')
    elif value['status']=='completed':
        raise ValueError('error receipt claims completed evaluation')
    return value

def _validate_candidate_receipt(receipt,expected,container,runtime,image):
    if (type(receipt) is not dict or receipt.get('schema')!='sciel-method-execution/1'
            or receipt.get('receipt_sha256')!=content_hash({k:v for k,v in receipt.items() if k!='receipt_sha256'})
            or any(receipt.get(k)!=expected[k] for k in BOUND)
            or receipt.get('limits')!=expected['limits'] or receipt.get('runtime')!=runtime
            or receipt.get('dependency_ids')!=[] or receipt.get('container',{}).get('name')!=container
            or receipt.get('container',{}).get('image_id')!=image):
        raise ValueError('unbound method execution receipt')
    resources=receipt.get('resources',{})
    cpu=resources.get('cpu_seconds')
    if cpu is not None and (type(cpu) not in (int,float) or not math.isfinite(cpu) or cpu<0):
        raise ValueError('invalid CPU observation')
    if receipt['status']=='unknown' or receipt['container'].get('removed') is not True:
        return 'unknown',cpu
    if receipt.get('dispatch_status')=='not_dispatched':
        if cpu!=0 or receipt['status']!='failed': raise ValueError('invalid nondispatch receipt')
        return 'not_dispatched',0.
    if receipt.get('dispatch_status')!='invoked' or receipt.get('resource_measurement',{}).get('cpu')!='trusted_cgroup_observation' or cpu is None:
        raise ValueError('missing trusted CPU observation')
    if receipt['status']=='failed': return 'failed',cpu
    if receipt['status']!='ok' or receipt.get('reason') is not None or type(receipt.get('output')) is not str or receipt.get('output_sha256')!=content_hash(receipt['output']):
        raise ValueError('invalid successful method output')
    return 'ok',cpu
