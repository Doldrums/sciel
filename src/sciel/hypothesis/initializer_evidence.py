"""Pure compact evidence from finite Circle helper witnesses; no execution.

The full receipt belongs in the journal. This bounded object preserves effective
binary64 outputs and their identities for deterministic historical projection.
Self-consistent hashes are integrity bindings, not external authentication.
"""
from __future__ import annotations

from .initializer_evidence_contracts import (
    COMPARISON_SCHEMA,
    FIXTURE_SHA256,
    LEGACY_COMPARISON_SCHEMA,
    LEGACY_VIEW_SCHEMA,
    MAX_EVIDENCE_CHARS,
    MAX_RECEIPT_CHARS,
    ORIGIN_VIEW_SCHEMA,
    ORIGIN_VIEW_SCOPE,
    SCHEMA,
    SCOPE,
    VIEW_SCHEMA,
    VIEW_SCOPE,
    _IDENTITY,
    _OBS,
    _ROLES,
    _SOURCE,
)

import base64
import copy
from fractions import Fraction
from functools import lru_cache
import itertools
import json
import math
import re
import struct

from . import circle_initializer_witness as core



def _require(ok, reason):
    if not ok:
        raise ValueError('initializer evidence: '+reason)


def _keys(value, fields, reason):
    _require(type(value) is dict and set(value)==set(fields),reason)


def _same(a,b):
    return core._canonical(a)==core._canonical(b)


def _digest(value):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'invalid digest')


def _code(value):
    _require(value is None or (type(value) is str and 0<len(value)<=128),'invalid status reason')


def _fixtures(digest=FIXTURE_SHA256):
    # Versioned data recipes only. Never prepare_witness(), identity(), or launch.
    return core.fixture_bundle_by_sha256(digest)


def _identity(value):
    _keys(value,_IDENTITY,'identity fields')
    for key,item in value.items():
        if key.endswith('_sha256'): _digest(item)
        elif key=='builder_files':
            _keys(item,{'circle_interventions.py','circle_proposals.py','circle_restart_body.py','circle_restart_scales.py'},'builder identity fields')
            for digest in item.values(): _digest(digest)
        else:
            _require(type(item) is str and 0<len(item)<=2048,'identity text')
    _require(value['builder_contract']=='circle_restart_body/1','body contract')


def _limits(value):
    _keys(value,{'max_sources','max_cases','cpu_seconds_per_source','wall_seconds_per_source',
        'max_total_child_wall_seconds','memory_bytes','memory_limit_kind','max_stdout_bytes_per_source',
        'max_stderr_bytes_per_source','max_array_bytes','threads'},'limits fields')
    for key,expected in {'max_sources':3,'max_cases':16,'cpu_seconds_per_source':2,
        'max_stdout_bytes_per_source':1024**2,'max_stderr_bytes_per_source':1024**2,
        'max_array_bytes':32768,'threads':1}.items():
        _require(type(value[key]) is int and value[key]==expected,'witness limit differs')
    wall=value['wall_seconds_per_source']
    _require(type(wall) is float and math.isfinite(wall) and 0<wall<=5.,'wall limit')
    _require(type(value['max_total_child_wall_seconds']) is float and value['max_total_child_wall_seconds']==3*wall,'total wall limit')
    _require((value['memory_limit_kind']=='unavailable' and value['memory_bytes'] is None) or
        (value['memory_limit_kind']=='address_space' and type(value['memory_bytes']) is int and value['memory_bytes']==2*1024**3),'memory policy')


def _roles(value, labels):
    _require(type(value) is dict and set(value) in ({'candidate','parent'},set(_ROLES)),'role fields')
    _require(all(type(x) is str for x in value.values()) and len(set(value.values()))==len(value)
             and set(value.values())==set(labels),'role/source membership')


def _compact(array):
    numbers=core._decode_array(array)
    _require(numbers.shape==(78,) and numbers.dtype.kind=='f' and numbers.dtype.itemsize==8,'effective shape/dtype')
    values=[float(x) for x in numbers]
    _require(all(math.isfinite(x) for x in values),'nonfinite effective output')
    raw=b''.join(struct.pack('<d',x) for x in values)
    return {'encoding':'binary64-le/78','data_base64':base64.b64encode(raw).decode(),'bytes_sha256':core._sha(raw)}


def _vector(value):
    _keys(value,{'encoding','data_base64','bytes_sha256'},'output fields')
    _require(value['encoding']=='binary64-le/78' and type(value['data_base64']) is str and len(value['data_base64'])==832,'output encoding')
    _digest(value['bytes_sha256'])
    try: raw=base64.b64decode(value['data_base64'],validate=True)
    except (ValueError,TypeError) as exc: raise ValueError('initializer evidence: output base64') from exc
    _require(len(raw)==624 and core._sha(raw)==value['bytes_sha256'],'output digest/length')
    numbers=list(struct.unpack('<78d',raw))
    _require(all(math.isfinite(x) for x in numbers),'nonfinite output')
    return numbers


def _compare_sources(left,right,case_ids):
    rows=[]
    for ident,a,b in zip(case_ids,left['observations'],right['observations'],strict=True):
        equal=first=None
        if a['status']==b['status']=='observed':
            aa=_vector(a['effective_output']);bb=_vector(b['effective_output'])
            differences=[i for i,(x,y) in enumerate(zip(aa,bb)) if struct.pack('<d',x)!=struct.pack('<d',y)]
            equal=not differences;first=differences[0] if differences else None
        rows.append({'case_id':ident,'equal_bits':equal,'first_differing_coordinate':first})
    status=('different_on_witness' if any(row['equal_bits'] is False for row in rows) else
            'same_on_witnesses' if all(row['equal_bits'] is True for row in rows) and left['complete'] and right['complete'] else 'incomplete')
    return {'status':status,'cases':rows}


def _counts(values):
    return {'decreased':sum(v<0 for v in values),'equal':sum(v==0 for v in values),'increased':sum(v>0 for v in values)}


def _ratio(value):
    return {'numerator':value.numerator,'denominator':value.denominator}


def _candidate_facts(observation,case):
    if observation['status']!='observed':
        return {'case_id':case['id'],'status':observation['status'],'facts':None}
    before=[float(x) for x in core._decode_array(case['inputs']['best_params'])]
    after=_vector(observation['effective_output'])
    ordinary=[float(x) for x in core._decode_array(case['inputs']['ordinary_params'])]
    changed=[i for i,(x,y) in enumerate(zip(before,after)) if struct.pack('<d',x)!=struct.pack('<d',y)]
    circles=sorted({i//2 for i in changed if i<52})
    a=[Fraction.from_float(x) for x in before];b=[Fraction.from_float(x) for x in after]
    distances=[];radii=[];old_overlap=new_overlap=undefined=0;two=None
    for i in range(26):
        for j in range(i+1,26):
            old=(a[2*i]-a[2*j])**2+(a[2*i+1]-a[2*j+1])**2
            new=(b[2*i]-b[2*j])**2+(b[2*i+1]-b[2*j+1])**2
            delta=new-old;rdelta=b[52+i]+b[52+j]-a[52+i]-a[52+j]
            distances.append(delta);radii.append(rdelta)
            oo=old<(a[52+i]+a[52+j])**2
            no=new<(b[52+i]+b[52+j])**2 if min(b[52+i],b[52+j])>=0 else None
            old_overlap+=int(oo);new_overlap+=int(no is True);undefined+=int(no is None)
            if circles==[i,j]:
                di=[b[2*i+k]-a[2*i+k] for k in range(2)]
                dj=[b[2*j+k]-a[2*j+k] for k in range(2)]
                two={'ids':[i,j],'squared_distance_delta':_ratio(delta),'radius_sum_delta':_ratio(rdelta),
                     'before_overlap':oo,'after_overlap':no,'common_translation_exact':di==dj,
                     'translation':[_ratio(v) for v in di] if di==dj else None}
    facts={'changed_center_coordinates':sum(i<52 for i in changed),'changed_circles':len(circles),
           'changed_radii':sum(i>=52 for i in changed),
           'changed_from_ordinary':sum(struct.pack('<d',x)!=struct.pack('<d',y) for x,y in zip(ordinary,after)),
           'pair_squared_distance':_counts(distances),'pair_radius_sum':_counts(radii),
           'overlaps':{'before':old_overlap,'after':new_overlap,'after_undefined':undefined},
           'two_moved_circles':two}
    return {'case_id':case['id'],'status':'observed','facts':facts}


@lru_cache(maxsize=256)
def _candidate_derivation(fixture_sha256, payload):
    # Bounded, content-addressed pure computation, never a validation verdict.
    # The canonical payload is in the key, not merely its claimed digest.
    fixtures=_fixtures(fixture_sha256)
    _require(fixture_sha256==fixtures['sha256'],'cached fixture identity')
    observations=json.loads(payload)
    rows=[_candidate_facts(obs,case) for obs,case in zip(observations,fixtures['cases'],strict=True)]
    return core._canonical(rows)


def _derive(value,fixtures):
    roles=value['roles'];sources=value['sources'];ids=[c['id'] for c in fixtures['cases']]
    comparisons=[]
    for left,right in itertools.combinations([r for r in _ROLES if r in roles],2):
        comparisons.append({'left':left,'right':right,**_compare_sources(sources[roles[left]],sources[roles[right]],ids)})
    candidate=sources[roles['candidate']]
    return {'comparisons':comparisons,
        'candidate_cases':json.loads(_candidate_derivation(fixtures['sha256'],core._canonical([
            {k:obs[k] for k in ('status','effective_output')} for obs in candidate['observations']])))}


def _checked_source(source,fixtures):
    _keys(source,_SOURCE,'source fields')
    for key in ('source_sha256','body_sha256','helper_sha256'): _digest(source[key])
    _require(type(source['complete']) is bool and source['status'] in ('observed','incomplete'),'source status')
    _code(source['error_code'])
    _require((source['error_code'] is None)==source['complete'],'source terminal reason')
    observations=source['observations'];cases=fixtures['cases']
    _require(type(observations) is list and len(observations)==len(cases),'case count')
    stopped=False
    for obs,case in zip(observations,cases,strict=True):
        _keys(obs,_OBS,'observation fields')
        _require(obs['case_id']==case['id'],'case order/identity')
        status=obs['status'];_require(status in ('observed','failed','unknown','not_dispatched'),'case status')
        _code(obs['error_code'])
        if stopped: _require(status=='not_dispatched','observations after missing suffix')
        if status in ('unknown','not_dispatched'): stopped=True
        counts=obs['handoff_counts']
        if status=='observed':
            _require(obs['error_code'] is None,'successful error')
            numbers=_vector(obs['effective_output'])
            fallback=obs['effective_origin']=='saved_ordinary'
            _require(obs['effective_origin'] in ('return','saved_ordinary') and
                obs['raw_return_kind']==('none' if fallback else 'value'),'fallback metadata')
            expected=dict(proposal_attempts=1,proposal_accepted=int(not fallback),proposal_none=int(fallback),proposal_errors=0)
            _require(_same(counts,expected),'successful handoff counts')
            if fallback: _require(_same(obs['effective_output'],_compact(case['inputs']['ordinary_params'])),'saved fallback differs')
        else:
            _require(obs['effective_output'] is None and obs['effective_origin'] is None,'missing output must be null')
            if status=='failed':
                _require(obs['raw_return_kind'] in (None,'value','none') and obs['error_code'] is not None,'failed metadata')
                _require(_same(counts,dict(proposal_attempts=1,proposal_accepted=0,proposal_none=0,proposal_errors=1)),'failed handoff counts')
            else: _require(counts is None and obs['raw_return_kind'] is None and obs['error_code'] is not None,'unknown handoff cannot be invented')
    _require(not source['complete'] or not stopped,'complete source with missing cases')
    expected='observed' if source['complete'] and all(o['status']=='observed' for o in observations) else 'incomplete'
    _require(source['status']==expected,'source completion mismatch')


def _plan(value,fixtures):
    return {'schema':value['witness_schema'],'identity':value['identity'],'fixtures':fixtures,
        'sources':[{'label':label,**{k:source[k] for k in ('source_sha256','body_sha256','helper_sha256')}}
                   for label,source in sorted(value['sources'].items())],
        'limits':value['limits'],'scope':core.SCOPE}


def validate_evidence(value, *, source_sha256=None, parent_source_sha256=None):
    """Validate a retained historical value without probing its former runtime."""
    try:
        _keys(value,{'schema','witness_schema','receipt_sha256','plan_sha256','identity','fixture_schema',
            'fixture_sha256','limits','roles','sources','derived','scope','evidence_sha256'},'evidence fields')
        _require(len(core._canonical(value))<=MAX_EVIDENCE_CHARS,'evidence capacity')
        _require(value['schema']==SCHEMA and value['witness_schema']==core.SCHEMA and value['scope']==SCOPE,'evidence schema/scope')
        for key in ('receipt_sha256','plan_sha256','fixture_sha256','evidence_sha256'): _digest(value[key])
        _require(core._hash({k:v for k,v in value.items() if k!='evidence_sha256'})==value['evidence_sha256'],'evidence digest')
        fixtures=_fixtures(value['fixture_sha256'])
        _require(value['fixture_schema']==fixtures['schema'] and value['fixture_sha256']==fixtures['sha256'],'fixture identity')
        _identity(value['identity']);_limits(value['limits'])
        _require(type(value['sources']) is dict and 2<=len(value['sources'])<=3,'source count')
        for label in value['sources']:
            _require(type(label) is str and 0<len(label)<=64 and label.isascii() and all(c.isalnum() or c in '_-' for c in label),'source label')
        _roles(value['roles'],value['sources'])
        for source in value['sources'].values(): _checked_source(source,fixtures)
        _require(core._hash(_plan(value,fixtures))==value['plan_sha256'],'retained plan binding differs')
        for role,expected in (('candidate',source_sha256),('parent',parent_source_sha256)):
            if expected is not None:
                _digest(expected)
                _require(value['sources'][value['roles'][role]]['source_sha256']==expected,'ordinary '+role+' source mismatch')
        _require(_same(value['derived'],_derive(value,fixtures)),'derived counts/fractions/comparisons differ')
        return copy.deepcopy(value)
    except (KeyError,TypeError,OverflowError,RecursionError) as exc:
        raise ValueError('initializer evidence: malformed value') from exc


def _receipt_sources(receipt):
    _keys(receipt,{'schema','plan','sources','comparisons','status','scope','candidate_rejection_authorized',
        'semantic_equivalence_proven','new_model_calls','new_solver_calls','new_native_evaluations','receipt_sha256'},'receipt fields')
    _require(len(core._canonical(receipt))<=MAX_RECEIPT_CHARS,'receipt capacity')
    _require(receipt['schema']==core.SCHEMA and receipt['scope']==core.SCOPE and
        receipt['semantic_equivalence_proven'] is False and receipt['candidate_rejection_authorized'] is False,'receipt scope')
    _require(all(type(receipt[k]) is int and receipt[k]==0 for k in ('new_model_calls','new_solver_calls','new_native_evaluations')),'receipt work counts')
    _digest(receipt['receipt_sha256'])
    _require(core._hash({k:v for k,v in receipt.items() if k!='receipt_sha256'})==receipt['receipt_sha256'],'receipt digest')
    plan=receipt['plan']
    _keys(plan,{'schema','identity','fixtures','sources','limits','scope','plan_sha256'},'plan fields')
    _require(plan['schema']==core.SCHEMA and plan['scope']==core.SCOPE and _same(plan['fixtures'],_fixtures(plan['fixtures']['sha256'])),'plan contract/fixtures')
    _identity(plan['identity']);_limits(plan['limits']);_digest(plan['plan_sha256'])
    _require(core._hash({k:v for k,v in plan.items() if k!='plan_sha256'})==plan['plan_sha256'],'plan digest')
    _require(type(plan['sources']) is list and type(receipt['sources']) is list and 2<=len(plan['sources'])==len(receipt['sources'])<=3,'receipt source count')
    result={}
    for pin,source in zip(plan['sources'],receipt['sources'],strict=True):
        _keys(pin,{'label','source_sha256','body_sha256','helper_sha256'},'source pin fields')
        label=pin['label'];_require(type(label) is str and label not in result,'duplicate source label')
        for key in ('source_sha256','body_sha256','helper_sha256'): _digest(pin[key])
        process=source['process']
        _keys(process,{'returncode','failure','elapsed_seconds'},'process fields')
        _require(process['returncode'] is None or type(process['returncode']) is int,'returncode type')
        _code(process['failure'])
        _require(type(process['elapsed_seconds']) in (int,float) and math.isfinite(process['elapsed_seconds']) and process['elapsed_seconds']>=0,'process time')
        streams={}
        for name in ('stdout','stderr'):
            encoded=source['raw_'+name+'_base64']
            _require(type(encoded) is str and len(encoded)<=((core.MAX_STREAM_BYTES+2)//3)*4,'stream capacity')
            streams[name]=base64.b64decode(encoded,validate=True)
            _require(len(streams[name])<=core.MAX_STREAM_BYTES,'stream bytes')
        # Pure parsing/normalization/Fraction rederivation from the captured stream.
        checked={**pin,**core._consume({**process,**streams},plan,label)}
        _require(_same(checked,source),'receipt observations/raw streams/facts differ')
        compact=[]
        for obs in checked['observations']:
            compact.append({'case_id':obs['case_id'],'status':obs['status'],
                'effective_output':_compact(obs['effective_output']) if obs['status']=='observed' else None,
                'effective_origin':obs.get('effective_origin'),
                'raw_return_kind':obs['raw_return']['kind'] if obs.get('raw_return') else None,
                'handoff_counts':obs.get('handoff_counts'),'error_code':obs['error_code']})
        result[label]={**{k:pin[k] for k in ('source_sha256','body_sha256','helper_sha256')},
            **{k:checked[k] for k in ('complete','status','error_code')},'observations':compact}
    _require(list(result)==sorted(result),'source order')
    ids=[c['id'] for c in plan['fixtures']['cases']]
    comparisons=[{'left':a,'right':b,**_compare_sources(result[a],result[b],ids)} for a,b in itertools.combinations(result,2)]
    _require(_same(comparisons,receipt['comparisons']),'receipt comparison mismatch')
    _require(receipt['status']==('observed' if all(x['status']=='observed' for x in result.values()) else 'incomplete'),'receipt completion mismatch')
    return result


def build_evidence(receipt, *, roles):
    """Validate a full receipt once, then retain bounded outputs and provenance."""
    try:
        sources=_receipt_sources(receipt);_roles(roles,sources);plan=receipt['plan']
        value={'schema':SCHEMA,'witness_schema':core.SCHEMA,'receipt_sha256':receipt['receipt_sha256'],
            'plan_sha256':plan['plan_sha256'],'identity':copy.deepcopy(plan['identity']),
            'fixture_schema':plan['fixtures']['schema'],'fixture_sha256':plan['fixtures']['sha256'],
            'limits':copy.deepcopy(plan['limits']),'roles':copy.deepcopy(roles),'sources':sources,'scope':SCOPE}
        value['derived']=_derive(value,plan['fixtures']);value['evidence_sha256']=core._hash(value)
        return validate_evidence(value)
    except (KeyError,TypeError,OverflowError,RecursionError) as exc:
        raise ValueError('initializer evidence: malformed receipt') from exc



from .initializer_evidence_views import (
    _COLUMNS,
    _HANDOFF_KEYS,
    _ORIGIN_FIELDS,
    _ORIGIN_VIEW_COLUMNS,
    _VIEW_COLUMNS,
    _bindings,
    _bounded_int,
    _case_view,
    _comparison_status,
    _exact_ratio,
    _expanded_case_views,
    _fallback_view_facts,
    _origin_case_view,
    _pack_case_views,
    _seal,
    _validate_bindings,
    _validate_case_view,
    _validate_origin,
    compare_evidence,
    evidence_case_views,
    evidence_view,
    validate_evidence_comparison,
    validate_evidence_view,
)
