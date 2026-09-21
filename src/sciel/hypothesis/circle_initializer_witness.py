"""Finite, source-bound Circle initializer observations; never an equivalence proof.

Only the exact accepted helper and the pinned handoff functions execute, in a
bounded child. No solver module, evaluator, RNG or model is invoked. Equality
here must not drive rejection, deduplication, quality or lesson admission.
"""
from __future__ import annotations

import ast
import base64
import builtins
import copy
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
from sciel.hypothesis import circle_restart_body as builder
from sciel.hypothesis.initializer_fixtures import (
    FIXTURE_PROFILES,
    _initial,
    _legacy_fixture_bundle,
    _registered_fixture,
    _unequal_fixture_bundle,
    fixture_bundle_by_sha256,
    make_fixture_bundle,
)

SCHEMA = 'sciel-circle-initializer-witness/1'
FIXTURE_SCHEMA = 'sciel-circle-initializer-fixtures/1'
MAX_SOURCES = 3
MAX_CASES = 16
CPU_SECONDS = 2
WALL_SECONDS = 5.0
MEMORY_BYTES = 2 * 1024**3
MAX_STREAM_BYTES = 1024**2  # Independently for stdout and stderr, checked while reading.
MAX_ARRAY_BYTES = 32768
MAX_REQUEST_BYTES = 512 * 1024
ALIASES = ('best_params', 'initial_params', 'uniform_noise', 'ordinary_params')
SCOPE = ('Finite initializer observations only, not universal equivalence, mechanism truth, '
         'packing validity, native evaluation, timing/quality benefit or a security sandbox. '
         'No rejection, deduplication, lesson admission or exposure credit follows from equality.')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    return _sha(_canonical(value).encode())


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))
    _canonical(value)
    return value


_LOADED_FILES = {str(Path(module.__file__).resolve()): _sha(Path(module.__file__).read_bytes())
                 for module in (builder, builder.proposals, builder.proposals.scales,
                                builder.proposals.scales.references)}
def _checker_hash():
    files = ('circle_initializer_witness.py', 'initializer_fixtures.py')
    return _hash({name: _sha(Path(__file__).with_name(name).read_bytes()) for name in files})


_LOADED_CHECKER = _checker_hash()
_LOADED_RUNTIME = _sha(builder.proposals._RUNTIME.encode())


def _function_text(source, name):
    functions = [x for x in ast.parse(source).body if isinstance(x, ast.FunctionDef) and x.name == name]
    _require(len(functions) == 1, 'function_anchor_not_unique')
    node = functions[0]
    _require(not node.decorator_list and node.col_offset == 0, 'unexpected_function_surface')
    # Keep original source bytes, including aliases/comments; never unparse AST.
    # AST line numbers count LF/CRLF, not Unicode separators inside literals.
    lines = source.split('\n')
    return '\n'.join(lines[node.lineno-1:node.end_lineno]) + ('\n' if node.end_lineno < len(lines) else '')


def identity():
    """Pin the loaded checker/builders, exact handoff, scaffold and local runtime."""
    _require(_checker_hash() == _LOADED_CHECKER, 'loaded_checker_drift')
    for path, digest in _LOADED_FILES.items():
        _require(_sha(Path(path).read_bytes()) == digest, 'loaded_builder_drift')
    _require(_sha(builder.proposals._RUNTIME.encode()) == _LOADED_RUNTIME, 'handoff_runtime_drift')
    control = builder.source_for()
    import numpy._core._multiarray_umath as numpy_core
    return {'checker_sha256': _LOADED_CHECKER,
        'builder_files': {Path(path).name: digest for path, digest in _LOADED_FILES.items()},
        'builder_contract': builder.CONTRACT,
        'control_source_sha256': _sha(control.encode()),
        'common_scaffold_sha256': builder.PINNED_SCAFFOLD_SHA256,
        'proposal_runtime_sha256': _LOADED_RUNTIME,
        'handoff_sha256': _sha(_function_text(builder.proposals._RUNTIME, '_proposal_invoke').encode()),
        'initial_parameters_sha256': _sha(_function_text(control, 'initial_parameters').encode()),
        'python': sys.version, 'python_executable_sha256': _sha(Path(sys.executable).read_bytes()),
        'platform': platform.platform(), 'numpy_version': np.__version__,
        'numpy_init_sha256': _sha(Path(np.__file__).read_bytes()),
        'numpy_core_sha256': _sha(Path(numpy_core.__file__).read_bytes())}


def _array(value):
    """Bounded, lossless array bytes. Numeric convenience values never contain NaN."""
    array = np.asarray(value)
    result = {'dtype': array.dtype.str, 'shape': list(array.shape), 'order': 'C',
              'data_base64': None, 'bytes_sha256': None, 'numeric': None, 'complete': False,
              'unavailable_reason': None}
    reason = ('object_dtype' if array.dtype.hasobject else 'structured_dtype' if array.dtype.fields is not None
              else 'subarray_dtype' if array.dtype.subdtype is not None else 'zero_itemsize' if array.dtype.itemsize==0
              else 'array_byte_limit' if array.nbytes>MAX_ARRAY_BYTES else None)
    if reason is not None:
        result['unavailable_reason'] = reason
        return result
    raw = array.tobytes(order='C')
    result.update(data_base64=base64.b64encode(raw).decode(), bytes_sha256=_sha(raw), complete=True)
    if array.dtype.kind in 'fiu' and array.size <= 4096:
        with np.errstate(all='ignore'):
            numbers = array.astype(np.float64)
        if np.isfinite(numbers).all():
            result['numeric'] = numbers.tolist()
    return result


def _decode_array(value):
    _require(type(value) is dict and set(value) == {'dtype','shape','order','data_base64',
        'bytes_sha256','numeric','complete','unavailable_reason'}, 'array_fields')
    _require(value['complete'] is True and value['order'] == 'C', 'array_unavailable')
    shape = value['shape']
    _require(type(shape) is list and len(shape) <= 8 and all(type(x) is int and x >= 0 for x in shape), 'array_shape')
    dtype = np.dtype(value['dtype'])
    _require(not dtype.hasobject and dtype.fields is None and dtype.subdtype is None and dtype.itemsize>0, 'array_dtype')
    _require(math.prod(shape) * dtype.itemsize <= MAX_ARRAY_BYTES, 'array_byte_limit')
    raw = base64.b64decode(value['data_base64'], validate=True)
    _require(len(raw) == math.prod(shape)*dtype.itemsize and _sha(raw) == value['bytes_sha256'], 'array_bytes')
    array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    _require(_canonical(_array(array)) == _canonical(value), 'array_representation')
    return array


def _limits(timeout):
    return {'max_sources':MAX_SOURCES,'max_cases':MAX_CASES,'cpu_seconds_per_source':CPU_SECONDS,
        'wall_seconds_per_source':float(timeout),'max_total_child_wall_seconds':MAX_SOURCES*float(timeout),
        'memory_bytes':None if sys.platform == 'darwin' else MEMORY_BYTES,
        'memory_limit_kind':'unavailable' if sys.platform == 'darwin' else 'address_space',
        'max_stdout_bytes_per_source':MAX_STREAM_BYTES,'max_stderr_bytes_per_source':MAX_STREAM_BYTES,
        'max_array_bytes':MAX_ARRAY_BYTES,'threads':1}


def prepare_witness(sources, fixture_bundle=None, *, expected_source_sha256, timeout_seconds=WALL_SECONDS):
    """Pure preparation: exact source validation and immutable plan, no helper run."""
    _require(type(timeout_seconds) in (int,float) and math.isfinite(timeout_seconds) and 0<timeout_seconds<=WALL_SECONDS, 'wall_limit')
    _require(type(sources) is dict and 1<=len(sources)<=MAX_SOURCES, 'source_count')
    _require(type(expected_source_sha256) is dict and set(sources)==set(expected_source_sha256), 'source_pin_keys')
    fixed = make_fixture_bundle() if fixture_bundle is None else _registered_fixture(fixture_bundle)
    bound = []
    for label,source in sorted(sources.items()):
        _require(type(label) is str and 0<len(label)<=64 and label.isascii() and all(c.isalnum() or c in '_-' for c in label), 'source_label')
        _require(type(source) is str and len(source.encode())<=131072, 'source_size')
        digest = _sha(source.encode())
        _require(expected_source_sha256[label]==digest, 'source_pin_mismatch')
        body = builder.body_of(source)  # Includes exact source roundtrip and strict codec.
        helper = _function_text(source, 'propose_restart')
        bound.append({'label':label,'source_sha256':digest,'body_sha256':_sha(body.encode()),
                      'helper_sha256':_sha(helper.encode())})
    plan = {'schema':SCHEMA,'identity':identity(),'fixtures':fixed,'sources':bound,
            'limits':_limits(timeout_seconds),'scope':SCOPE}
    return {**plan,'plan_sha256':_hash(plan)}


def _emit(value):
    print(_canonical(value), flush=True)


def _child(request):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,(CPU_SECONDS,CPU_SECONDS))
    if sys.platform != 'darwin':
        resource.setrlimit(resource.RLIMIT_AS,(MEMORY_BYTES,MEMORY_BYTES))
    plan = prepare_witness(request['sources'],request['plan']['fixtures'],
        expected_source_sha256=request['expected_source_sha256'],
        timeout_seconds=request['plan']['limits']['wall_seconds_per_source'])
    _require(_canonical(plan)==_canonical(request['plan']), 'child_plan_mismatch')
    label = request['label']; source = request['sources'][label]
    helpers = {'np':np,'math':math,'__builtins__':{name:getattr(builtins,name) for name in builder.proposals._BUILTINS}}
    exec(compile(_function_text(source,'propose_restart'),'<witness-exact-helper>','exec'),helpers)
    helper = helpers['propose_restart']
    # Execute unchanged trusted handoff functions; the assembled solver never runs.
    runtime = {'np':np,'time':time}
    for name in ('_proposal_stats','_proposal_invoke'):
        exec(compile(_function_text(builder.proposals._RUNTIME,name),'<witness-pinned-handoff>','exec'),runtime)
    _emit({'kind':'header','plan_sha256':plan['plan_sha256'],'label':label,'identity':identity()})
    for case in plan['fixtures']['cases']:
        _emit({'kind':'intent','case_id':case['id']})
        arrays = [_decode_array(case['inputs'][k]) for k in ALIASES]
        observation = {'kind':'observation','case_id':case['id'],'status':'failed',
            'raw_return':None,'post_inputs':None,'effective_output':None,'error_code':None,
            'input_storage_detached':None}
        def capture(*args):
            observation['input_storage_detached'] = all(not np.shares_memory(a,b)
                for a,b in itertools.combinations(args[:4],2))
            try:
                value = helper(*args)
                if value is None:
                    observation['raw_return'] = {'kind':'none','array':None,
                        'python_type':'NoneType','shares_input_storage':None}
                else:
                    try:
                        observed = _array(value)
                    except (ValueError,TypeError,OverflowError):
                        observed = None
                    observation['raw_return'] = {'kind':'value','array':observed,
                        'python_type':type(value).__name__[:80],
                        'shares_input_storage':([bool(np.shares_memory(value,a)) for a in args[:4]]
                            if isinstance(value,np.ndarray) else None)}
                return value
            finally:
                observation['post_inputs'] = {k:_array(a) for k,a in zip(ALIASES,args[:4])}
        runtime['propose_restart'] = capture
        stats = runtime['_proposal_stats']()
        try:
            output = runtime['_proposal_invoke'](*arrays,case['restart_index'],stats)
            observation['effective_output'] = _array(output)
            observation['status'] = 'observed'
        except BaseException as exc:
            # Exception type only: no arbitrary model string/repr is evaluated.
            observation['error_code'] = type(exc).__name__[:80]
        observation['handoff_counts'] = {k:stats[k] for k in ('proposal_attempts','proposal_accepted','proposal_none','proposal_errors')}
        _emit(observation)
    _require(identity()==plan['identity'],'child_identity_drift')
    _emit({'kind':'terminal','completed':True})


def _scrubbed_env():
    return {'PYTHONPATH':os.pathsep.join((str(Path(builder.__file__).resolve().parents[2]),str(Path(np.__file__).resolve().parents[1]))),
        'PYTHONHASHSEED':'0','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','LC_ALL':'C',
        **{k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')}}


def _launch(request, python, timeout):
    raw = _canonical(request).encode()
    _require(len(raw)<=MAX_REQUEST_BYTES,'request_size')
    streams = {'stdout':bytearray(),'stderr':bytearray()}; failure=None
    with tempfile.TemporaryDirectory(prefix='sciel-initializer-witness-') as directory:
        # A regular stdin file cannot block the parent before its wall watchdog.
        path=Path(directory)/'request.json'; path.write_bytes(raw)
        with path.open('rb') as incoming:
            process=subprocess.Popen([str(python),'-B','-S',str(Path(__file__).resolve()),'--worker'],
                stdin=incoming,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=directory,
                env=_scrubbed_env(),start_new_session=True,close_fds=True)
            started=time.monotonic()
            try:
                with selectors.DefaultSelector() as selector:
                    for name in streams: selector.register(getattr(process,name),selectors.EVENT_READ,name)
                    while selector.get_map():
                        remaining=timeout-(time.monotonic()-started)
                        if remaining<=0: failure='timeout'; break
                        for key,_ in selector.select(min(remaining,.1)):
                            chunk=os.read(key.fd,65536)
                            if not chunk: selector.unregister(key.fileobj); continue
                            data=streams[key.data]; room=MAX_STREAM_BYTES-len(data)
                            data.extend(chunk[:room])
                            if len(chunk)>room: failure='output_limit'; break
                        if failure: break
                if not failure:
                    try: process.wait(timeout=max(.001,timeout-(time.monotonic()-started)))
                    except subprocess.TimeoutExpired: failure='timeout'
            finally:
                if process.poll() is None or failure:
                    try: os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                process.wait()
                process.stdout.close(); process.stderr.close()
    return {'returncode':process.returncode,'failure':failure,
        'elapsed_seconds':time.monotonic()-started,
        **{name:bytes(data) for name,data in streams.items()}}


def _fraction(value):
    return {'numerator':value.numerator,'denominator':value.denominator}


def _facts(before, after):
    changed=[i for i in range(78) if struct.pack('<d',float(before[i]))!=struct.pack('<d',float(after[i]))]
    a=[Fraction.from_float(float(x)) for x in before]; b=[Fraction.from_float(float(x)) for x in after]
    distances=[]
    for i in range(26):
        for j in range(i+1,26):
            old=(a[2*i]-a[2*j])**2+(a[2*i+1]-a[2*j+1])**2
            new=(b[2*i]-b[2*j])**2+(b[2*i+1]-b[2*j+1])**2
            distances.append({'i':i,'j':j,'squared_distance_delta':_fraction(new-old),
                'radius_sum_delta':_fraction(b[52+i]+b[52+j]-a[52+i]-a[52+j]),
                'before_overlap':(old<(a[52+i]+a[52+j])**2 if min(a[52+i],a[52+j])>=0 else None),
                'after_overlap':(new<(b[52+i]+b[52+j])**2 if min(b[52+i],b[52+j])>=0 else None)})
    return {'changed_center_coordinates':[i for i in changed if i<52],
        'changed_radius_indices':[i-52 for i in changed if i>=52],
        'pair_relations':distances,
        'scope':'Exact rational properties of represented binary64 inputs/outputs, not ideal real arithmetic or solver effects. Overlap is null for a pair with negative radius; no packing validity is inferred.'}


def _checked_observation(event, case):
    _require(type(event) is dict and set(event)=={'kind','case_id','status','raw_return',
        'post_inputs','effective_output','error_code','input_storage_detached','handoff_counts'},'observation_fields')
    _require(event['status'] in ('observed','failed') and event['case_id']==case['id'],'observation_status')
    _require(event['input_storage_detached'] is True,'argument_storage_not_detached')
    _require(set(event['post_inputs'])==set(ALIASES),'post_inputs')
    for item in event['post_inputs'].values(): _decode_array(item)
    result=copy.deepcopy(event)
    result['input_mutated']={k:event['post_inputs'][k]['bytes_sha256']!=case['inputs'][k]['bytes_sha256'] for k in ALIASES}
    counts=event['handoff_counts']
    _require(all(type(v) is int and v>=0 for v in counts.values()),'handoff_counts')
    raw=event['raw_return']
    if raw is not None:
        _require(type(raw) is dict and set(raw)=={'kind','array','python_type','shares_input_storage'},'raw_fields')
        _require(type(raw['python_type']) is str and len(raw['python_type'])<=80,'raw_type')
        shared=raw['shares_input_storage']
        _require(shared is None or (type(shared) is list and len(shared)==4 and all(type(x) is bool for x in shared)),'raw_alias_metadata')
        if raw['array'] is not None and raw['array'].get('complete') is True:
            _decode_array(raw['array'])
    if event['status']=='failed':
        _require(event['effective_output'] is None and counts==dict(proposal_attempts=1,proposal_accepted=0,proposal_none=0,proposal_errors=1),'failed_handoff')
        result['facts']=None
        return result
    output=_decode_array(event['effective_output'])
    _require(output.dtype==np.dtype('float64') and output.shape==(78,) and np.isfinite(output).all(),'effective_output')
    if raw['kind']=='none':
        _require(raw['array'] is None,'none_array')
        normalized=_decode_array(case['inputs']['ordinary_params']);origin='saved_ordinary'
        expected_counts=dict(proposal_attempts=1,proposal_accepted=0,proposal_none=1,proposal_errors=0)
    else:
        _require(raw['kind']=='value','raw_return_kind')
        array=_decode_array(raw['array'])
        _require(array.shape==(78,) and array.dtype.kind in 'fiu','raw_output_contract')
        normalized=np.asarray(array,dtype=float).copy();origin='return'
        expected_counts=dict(proposal_attempts=1,proposal_accepted=1,proposal_none=0,proposal_errors=0)
    _require(counts==expected_counts and normalized.tobytes()==output.tobytes(),'handoff_normalization')
    result['effective_origin']=origin
    result['facts']={'relative_to_best':_facts(_decode_array(case['inputs']['best_params']),output),
        'changed_from_saved_ordinary':[i for i,(a,b) in enumerate(zip(_decode_array(case['inputs']['ordinary_params']),output)) if struct.pack('<d',float(a))!=struct.pack('<d',float(b))]}
    return result


def _consume(process, plan, label):
    """Check each returned observation; preserve completed prefix after a timeout."""
    cases=plan['fixtures']['cases']; observations=[]; pending=None; terminal=False; error=None
    try:
        lines=process['stdout'].splitlines()
        header=_read_json(lines[0])
        _require(header=={'kind':'header','plan_sha256':plan['plan_sha256'],'label':label,'identity':plan['identity']},'child_header')
        for line in lines[1:]:
            event=_read_json(line)
            _require(not terminal,'events_after_terminal')
            if event['kind']=='intent':
                _require(pending is None and len(observations)<len(cases) and event=={'kind':'intent','case_id':cases[len(observations)]['id']},'intent_order')
                pending=event['case_id']
            elif event['kind']=='observation':
                _require(pending is not None,'observation_without_intent')
                observations.append(_checked_observation(event,cases[len(observations)]));pending=None
            else:
                _require(event=={'kind':'terminal','completed':True} and pending is None and len(observations)==len(cases),'terminal_event')
                terminal=True
    except (ValueError,KeyError,TypeError,IndexError,OverflowError,RecursionError):
        error='invalid_child_receipt'
    for case in cases[len(observations):]:
        observations.append({'case_id':case['id'],'status':'unknown' if pending==case['id'] else 'not_dispatched',
            'raw_return':None,'post_inputs':None,'effective_output':None,'facts':None,'error_code':error or process['failure'] or 'child_stopped'})
    complete=terminal and error is None and process['failure'] is None and process['returncode']==0
    return {'label':label,'complete':complete,'status':'observed' if complete and all(x['status']=='observed' for x in observations) else 'incomplete',
        'error_code':error or process['failure'] or (None if complete else 'child_stopped'),
        'observations':observations,'process':{k:v for k,v in process.items() if k not in ('stdout','stderr')},
        'raw_stdout_base64':base64.b64encode(process['stdout']).decode(),
        'raw_stderr_base64':base64.b64encode(process['stderr']).decode()}


def witness_sources(sources, fixture_bundle=None, *, expected_source_sha256,
                    expected_plan_sha256=None, python=None, timeout_seconds=WALL_SECONDS):
    """Execute only small accepted helpers; finite equality is evidence, never a gate."""
    plan=prepare_witness(sources,fixture_bundle,expected_source_sha256=expected_source_sha256,
        timeout_seconds=timeout_seconds)
    if expected_plan_sha256 is not None:
        _require(expected_plan_sha256==plan['plan_sha256'],'plan_pin_mismatch')
    results=[]
    for source in plan['sources']:
        label=source['label']
        request={'sources':sources,'expected_source_sha256':expected_source_sha256,'plan':plan,'label':label}
        try: process=_launch(request,python or sys.executable,float(timeout_seconds))
        except OSError:
            process={'returncode':None,'failure':'process_start_failed','elapsed_seconds':0.,'stdout':b'','stderr':b''}
        results.append({**source,**_consume(process,plan,label)})
    _require(identity()==plan['identity'],'parent_identity_drift')
    comparisons=[]
    for left,right in itertools.combinations(results,2):
        rows=[]
        for a,b in zip(left['observations'],right['observations']):
            equal=None; first=None
            if a['status']==b['status']=='observed':
                aa=_decode_array(a['effective_output']);bb=_decode_array(b['effective_output'])
                equal=aa.tobytes()==bb.tobytes()
                if not equal:
                    first=next(i for i in range(78) if aa[i:i+1].tobytes()!=bb[i:i+1].tobytes())
            rows.append({'case_id':a['case_id'],'equal_bits':equal,'first_differing_coordinate':first})
        status=('different_on_witness' if any(x['equal_bits'] is False for x in rows) else
                'same_on_witnesses' if all(x['equal_bits'] is True for x in rows) and left['complete'] and right['complete'] else 'incomplete')
        comparisons.append({'left':left['label'],'right':right['label'],'status':status,'cases':rows})
    receipt={'schema':SCHEMA,'plan':plan,'sources':results,'comparisons':comparisons,
        'status':'observed' if all(x['status']=='observed' for x in results) else 'incomplete',
        'scope':SCOPE,'candidate_rejection_authorized':False,'semantic_equivalence_proven':False,
        'new_model_calls':0,'new_solver_calls':0,'new_native_evaluations':0}
    return {**receipt,'receipt_sha256':_hash(receipt)}


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('Only --worker is supported; use the source-pinned public API.')
    request_bytes=sys.stdin.buffer.read(MAX_REQUEST_BYTES+1)
    _require(len(request_bytes)<=MAX_REQUEST_BYTES,'request_size')
    _child(_read_json(request_bytes))
