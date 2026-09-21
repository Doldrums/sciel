"""Bounded presentation and validation of retained initializer evidence."""
from __future__ import annotations

import copy
import itertools
import math
from functools import lru_cache
from . import circle_initializer_witness as core
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

def _bindings(value):
    return {'evidence_sha256':value['evidence_sha256'],'receipt_sha256':value['receipt_sha256'],
        'role_sources':{role:value['sources'][label]['source_sha256'] for role,label in value['roles'].items()}}


_COLUMNS={'changes':['center_coordinates','circles','radii','from_saved_ordinary'],
          'pair_signs':['decreased','equal','increased'],
          'overlaps':['before','after','after_undefined']}
_ORIGIN_VIEW_COLUMNS = {**_COLUMNS,'handoff':['attempts','accepted','none','errors']}
_VIEW_COLUMNS = {**_ORIGIN_VIEW_COLUMNS,'case':['status','facts','handoff_ref']}
_HANDOFF_KEYS = ('proposal_attempts','proposal_accepted','proposal_none','proposal_errors')
_ORIGIN_FIELDS = {'origin','return_kind','handoff','error'}


def _case_view(row):
    result={'case_id':row['case_id'],'status':row['status'],'facts':None}
    if row['facts'] is not None:
        f=row['facts']
        result['facts']={'changes':[f[k] for k in ('changed_center_coordinates','changed_circles','changed_radii','changed_from_ordinary')],
            'distance':[f['pair_squared_distance'][k] for k in _COLUMNS['pair_signs']],
            'radius_sum':[f['pair_radius_sum'][k] for k in _COLUMNS['pair_signs']],
            'overlaps':[f['overlaps'][k] for k in _COLUMNS['overlaps']],
            'two_moved_circles':f['two_moved_circles']}
    return result


def _origin_case_view(row,observation):
    # Copy recorded provenance; equal output values do not imply equal origins.
    from . import initializer_evidence as _public
    _public._require(row['case_id']==observation['case_id'] and row['status']==observation['status'],
             'candidate observation/view join')
    counts=observation['handoff_counts']
    return {**_case_view(row),'origin':observation['effective_origin'],
        'return_kind':observation['raw_return_kind'],
        'handoff':[counts[k] for k in _HANDOFF_KEYS] if counts is not None else None,
        'error':observation['error_code']}


def _pack_case_views(rows):
    """Intern only identical recorded metadata, in deterministic first-use order."""
    handoffs=[];indices={};cases=[]
    for row in rows:
        handoff={key:copy.deepcopy(row[key]) for key in sorted(_ORIGIN_FIELDS)}
        key=core._canonical(handoff)
        if key not in indices:
            indices[key]=len(handoffs);handoffs.append(handoff)
        cases.append([row['status'],copy.deepcopy(row['facts']),indices[key]])
    return cases,handoffs


def _expanded_case_views(value):
    from . import initializer_evidence as _public
    if value['schema']!=VIEW_SCHEMA:
        return copy.deepcopy(value['candidate_cases'])
    handoffs=value['handoffs']
    _public._require(type(handoffs) is list and 1<=len(handoffs)<=16,'view handoff catalog size')
    keys=[]
    for entry in handoffs:
        _public._keys(entry,_ORIGIN_FIELDS,'view handoff catalog fields')
        keys.append(core._canonical(entry))
    _public._require(len(set(keys))==len(keys),'duplicate view handoff catalog entries')
    rows=[];used=[]
    for ident,row in zip(value['case_ids'],value['candidate_cases'],strict=True):
        _public._require(type(row) is list and len(row)==3,'view case columns')
        status,facts,ref=row
        _public._require(type(ref) is int and 0<=ref<len(handoffs),'view handoff reference')
        if ref not in used:used.append(ref)
        rows.append({'case_id':ident,'status':status,'facts':copy.deepcopy(facts),
                     **copy.deepcopy(handoffs[ref])})
    _public._require(used==list(range(len(handoffs))),'view handoff catalog first-use order or unused entry')
    return rows


@lru_cache(maxsize=32)
def _fallback_view_facts(fixture_sha256,case_id):
    from . import initializer_evidence as _public
    fixtures=_public._fixtures(fixture_sha256)
    case=next(case for case in fixtures['cases'] if case['id']==case_id)
    row=_public._candidate_facts({'status':'observed','effective_output':_public._compact(case['inputs']['ordinary_params'])},case)
    # Cache immutable bytes, not a caller-mutable facts dictionary.
    return core._canonical(_case_view(row)['facts'])


def _validate_origin(row,fixture_sha256):
    from . import initializer_evidence as _public
    status=row['status'];origin=row['origin'];kind=row['return_kind'];counts=row['handoff'];error=row['error']
    _public._code(error)
    if status=='observed':
        _public._require(error is None,'observed view error')
        _public._require(origin in ('return','saved_ordinary'),'observed view origin')
        fallback=origin=='saved_ordinary'
        _public._require(kind==('none' if fallback else 'value'),'view return/origin mismatch')
        _public._require(_public._same(counts,[1,int(not fallback),int(fallback),0]),'view successful handoff counts')
        if fallback:
            _public._require(core._canonical(row['facts'])==_fallback_view_facts(fixture_sha256,row['case_id']),
                     'saved ordinary view facts differ')
    elif status=='failed':
        _public._require(origin is None and kind in (None,'value','none') and error is not None,
                 'failed view origin/return/error')
        _public._require(_public._same(counts,[1,0,0,1]),'failed view handoff counts')
    else:
        _public._require(origin is None and kind is None and counts is None and error is not None,
                 'unknown view handoff cannot be invented')


def _seal(value,key):
    return {**value,key:core._hash(value)}


def evidence_view(value, max_chars=8192):
    """Return a whole card or explicit omission metadata, never a prefix of facts."""
    from . import initializer_evidence as _public
    _public._require(type(max_chars) is int and max_chars>=0,'view budget')
    value=_public.validate_evidence(value)
    cases,handoffs=_pack_case_views([_origin_case_view(row,obs) for row,obs in zip(
        value['derived']['candidate_cases'],value['sources'][value['roles']['candidate']]['observations'],strict=True)])
    card={'schema':VIEW_SCHEMA,'status':'presented',**_bindings(value),
        'fixture_sha256':value['fixture_sha256'],'identity_sha256':core._hash(value['identity']),
        'case_ids':[x['case_id'] for x in value['derived']['candidate_cases']],
        'comparisons':[{'left':c['left'],'right':c['right'],'status':c['status'],
            'equal_bits':[r['equal_bits'] for r in c['cases']]} for c in value['derived']['comparisons']],
        'columns':copy.deepcopy(_VIEW_COLUMNS),
        'candidate_cases':cases,'handoffs':handoffs,'scope':VIEW_SCOPE}
    card=_seal(card,'view_sha256')
    size=len(core._canonical(card))
    if size<=max_chars: return card
    omitted={'schema':VIEW_SCHEMA,'status':'omitted',**_bindings(value),'reason':'memory_char_limit',
             'card_chars':size}
    # Metadata is an omission report, not permission to overrun a prompt budget.
    omitted['fits_budget']=len(core._canonical(_seal({**omitted,'fits_budget':False},'view_sha256')))<=max_chars
    return _seal(omitted,'view_sha256')


def compare_evidence(left,right):
    """Compare candidates across compatible retained receipts without new runs."""
    from . import initializer_evidence as _public
    left=_public.validate_evidence(left);right=_public.validate_evidence(right)
    result={'schema':COMPARISON_SCHEMA,'left':_bindings(left),'right':_bindings(right),'scope':SCOPE}
    compatible=(left['fixture_schema']==right['fixture_schema'] and left['fixture_sha256']==right['fixture_sha256']
                and _public._same(left['identity'],right['identity']))
    if not compatible:
        return _seal({**result,'status':'incompatible','reason':'fixture_or_checker_runtime_identity_differs','fixture_sha256':None,'cases':[]},'comparison_sha256')
    a=left['sources'][left['roles']['candidate']];b=right['sources'][right['roles']['candidate']]
    ids=[case['id'] for case in _public._fixtures(left['fixture_sha256'])['cases']]
    return _seal({**result,**_public._compare_sources(a,b,ids),'reason':None,'fixture_sha256':left['fixture_sha256']},'comparison_sha256')


def _validate_bindings(value):
    from . import initializer_evidence as _public
    _public._keys(value,{'evidence_sha256','receipt_sha256','role_sources'},'view source binding fields')
    _public._digest(value['evidence_sha256']);_public._digest(value['receipt_sha256'])
    roles=value['role_sources']
    _public._require(type(roles) is dict and set(roles) in ({'candidate','parent'},set(_ROLES)),'view roles')
    for digest in roles.values(): _public._digest(digest)


def _bounded_int(value,maximum):
    from . import initializer_evidence as _public
    _public._require(type(value) is int and 0<=value<=maximum,'view count')


def _exact_ratio(value):
    from . import initializer_evidence as _public
    _public._keys(value,{'numerator','denominator'},'fraction fields')
    n,d=value['numerator'],value['denominator']
    _public._require(type(n) is int and type(d) is int and d>0 and max(abs(n).bit_length(),d.bit_length())<=4096,'fraction values')
    _public._require(math.gcd(n,d)==1,'fraction not reduced')


def _validate_case_view(row,ident,*,legacy=True,fixture_sha256=None):
    from . import initializer_evidence as _public
    fields={'case_id','status','facts'}
    _public._keys(row,fields if legacy else fields|_ORIGIN_FIELDS,'view case fields')
    _public._require(row['case_id']==ident and row['status'] in ('observed','failed','unknown','not_dispatched'),'view case identity/status')
    if not legacy: _validate_origin(row,fixture_sha256)
    facts=row['facts']
    if row['status']!='observed':
        _public._require(facts is None,'missing view facts must be null');return
    _public._keys(facts,{'changes','distance','radius_sum','overlaps','two_moved_circles'},'view fact fields')
    _public._require(type(facts['changes']) is list and len(facts['changes'])==4,'change counts')
    for v,cap in zip(facts['changes'],(52,26,26,78),strict=True): _bounded_int(v,cap)
    coords,circles,_,_=facts['changes']
    _public._require((coords+1)//2<=circles<=coords,'changed circle/coordinate mismatch')
    for key in ('distance','radius_sum'):
        _public._require(type(facts[key]) is list and len(facts[key])==3,'pair counts')
        for v in facts[key]: _bounded_int(v,325)
        _public._require(sum(facts[key])==325,'pair total')
    _public._require(type(facts['overlaps']) is list and len(facts['overlaps'])==3,'overlap counts')
    for v in facts['overlaps']: _bounded_int(v,325)
    _public._require(sum(facts['overlaps'][1:])<=325,'undefined overlap counts')
    two=facts['two_moved_circles']
    if circles!=2:
        _public._require(two is None,'unexpected two-circle relation');return
    _public._keys(two,{'ids','squared_distance_delta','radius_sum_delta','before_overlap','after_overlap',
        'common_translation_exact','translation'},'two-circle fields')
    _public._require(type(two['ids']) is list and len(two['ids'])==2,'two-circle IDs')
    for i in two['ids']: _bounded_int(i,25)
    _public._require(two['ids'][0]<two['ids'][1],'ordered two-circle IDs')
    for key in ('squared_distance_delta','radius_sum_delta'): _exact_ratio(two[key])
    _public._require(type(two['before_overlap']) is bool and (two['after_overlap'] is None or type(two['after_overlap']) is bool),'overlap types')
    _public._require(type(two['common_translation_exact']) is bool,'translation type')
    if two['common_translation_exact']:
        _public._require(type(two['translation']) is list and len(two['translation'])==2,'translation vector')
        for value in two['translation']: _exact_ratio(value)
        _public._require(two['squared_distance_delta']['numerator']==0,'common translation cannot change exact distance')
    else: _public._require(two['translation'] is None,'non-common translation metadata')


def _comparison_status(status,values):
    from . import initializer_evidence as _public
    _public._require(status in ('same_on_witnesses','different_on_witness','incomplete'),'comparison status')
    _public._require(all(v is None or type(v) is bool for v in values),'comparison value types')
    if status=='same_on_witnesses': _public._require(all(v is True for v in values),'equality includes missing/different')
    elif status=='different_on_witness': _public._require(any(v is False for v in values),'difference without witness')
    else: _public._require(not any(v is False for v in values),'known difference cannot be hidden')


def validate_evidence_view(value):
    """Validate transport structure/digest, not authenticate an unavailable receipt."""
    from . import initializer_evidence as _public
    try:
        _public._require(type(value) is dict and value.get('schema') in (VIEW_SCHEMA,ORIGIN_VIEW_SCHEMA,LEGACY_VIEW_SCHEMA),'view schema')
        legacy=value['schema']==LEGACY_VIEW_SCHEMA
        compact=value['schema']==VIEW_SCHEMA
        _public._digest(value.get('view_sha256'))
        _public._require(core._hash({k:v for k,v in value.items() if k!='view_sha256'})==value['view_sha256'],'view digest')
        binding={k:value[k] for k in ('evidence_sha256','receipt_sha256','role_sources')};_validate_bindings(binding)
        common={'schema','status','evidence_sha256','receipt_sha256','role_sources','view_sha256'}
        if value['status']=='omitted':
            _public._keys(value,common|{'reason','card_chars','fits_budget'},'omitted view fields')
            _public._require(value['reason']=='memory_char_limit' and type(value['card_chars']) is int and value['card_chars']>0
                and type(value['fits_budget']) is bool,'omission metadata')
        else:
            fields=common|{'fixture_sha256','identity_sha256','case_ids','comparisons','columns','candidate_cases','scope'}
            _public._keys(value,fields|{'handoffs'} if compact else fields,'presented view fields')
            scope=SCOPE if legacy else VIEW_SCOPE if compact else ORIGIN_VIEW_SCOPE
            columns=_COLUMNS if legacy else _VIEW_COLUMNS if compact else _ORIGIN_VIEW_COLUMNS
            _public._require(value['status']=='presented' and value['scope']==scope and
                _public._same(value['columns'],columns),'view status/scope/columns')
            fixtures=_public._fixtures(value['fixture_sha256']);_public._digest(value['identity_sha256'])
            ids=[c['id'] for c in fixtures['cases']]
            _public._require(_public._same(value['case_ids'],ids),'view cases')
            _public._require(type(value['candidate_cases']) is list and len(value['candidate_cases'])==16,'view case count')
            for row,ident in zip(_expanded_case_views(value),ids,strict=True):
                _validate_case_view(row,ident,legacy=legacy,fixture_sha256=value['fixture_sha256'])
            expected=list(itertools.combinations([r for r in _ROLES if r in value['role_sources']],2))
            _public._require(type(value['comparisons']) is list and len(value['comparisons'])==len(expected),'view comparison count')
            for row,(left,right) in zip(value['comparisons'],expected,strict=True):
                _public._keys(row,{'left','right','status','equal_bits'},'view comparison fields')
                _public._require((row['left'],row['right'])==(left,right) and type(row['equal_bits']) is list and len(row['equal_bits'])==16,'view comparison membership')
                _comparison_status(row['status'],row['equal_bits'])
        return copy.deepcopy(value)
    except (KeyError,TypeError,OverflowError,RecursionError) as exc:
        raise ValueError('initializer evidence: malformed view') from exc


def evidence_case_views(value):
    """Validate then return detached expanded rows; legacy /1 has no origins.

    An omitted whole card has no case rows. This resolves transport references,
    not an unavailable receipt or an inferred helper result.
    """
    value=validate_evidence_view(value)
    return [] if value['status']=='omitted' else _expanded_case_views(value)


def validate_evidence_comparison(value):
    """Validate a cross-receipt comparison's exact bounded transport schema."""
    from . import initializer_evidence as _public
    try:
        _public._require(type(value) is dict,'comparison fields')
        legacy=value.get('schema')==LEGACY_COMPARISON_SCHEMA
        fields={'schema','left','right','scope','status','reason','cases','comparison_sha256'}
        _public._keys(value,fields if legacy else fields|{'fixture_sha256'},'comparison fields')
        _public._require(value['schema'] in (COMPARISON_SCHEMA,LEGACY_COMPARISON_SCHEMA) and value['scope']==SCOPE,'comparison schema/scope')
        _public._digest(value['comparison_sha256'])
        _public._require(core._hash({k:v for k,v in value.items() if k!='comparison_sha256'})==value['comparison_sha256'],'comparison digest')
        _validate_bindings(value['left']);_validate_bindings(value['right'])
        if value['status']=='incompatible':
            _public._require(value['cases']==[] and value['reason']=='fixture_or_checker_runtime_identity_differs','incompatible comparison')
            _public._require(legacy or value['fixture_sha256'] is None,'incompatible fixture binding')
        else:
            _public._require(value['reason'] is None and type(value['cases']) is list and len(value['cases'])==16,'comparison cases')
            fixtures=_public._fixtures(FIXTURE_SHA256 if legacy else value['fixture_sha256'])
            for row,case in zip(value['cases'],fixtures['cases'],strict=True):
                _public._keys(row,{'case_id','equal_bits','first_differing_coordinate'},'comparison case fields')
                _public._require(row['case_id']==case['id'],'comparison case order')
                if row['equal_bits'] is False: _bounded_int(row['first_differing_coordinate'],77)
                else: _public._require(row['first_differing_coordinate'] is None,'unexpected difference index')
            _comparison_status(value['status'],[row['equal_bits'] for row in value['cases']])
        return copy.deepcopy(value)
    except (KeyError,TypeError,OverflowError,RecursionError) as exc:
        raise ValueError('initializer evidence: malformed comparison') from exc
