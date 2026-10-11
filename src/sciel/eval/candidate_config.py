"""Pure opt-in backend policy validation, before provider construction."""
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import re

from .candidate_registry import PINS


def validate_policy(value):
    if value is None:
        return None  # historical trusted-local mode, never an isolation claim
    if type(value) is not dict or set(value) != {'backend','image_id','cpu_seconds','memory_bytes','pids'}:
        raise ValueError('evaluation requires exactly backend, image_id, cpu_seconds, memory_bytes and pids')
    if value['backend'] != 'candidate_only':
        raise ValueError('unsupported evaluation backend; no native fallback')
    if type(value['image_id']) is not str or re.fullmatch(r'sha256:[0-9a-f]{64}',value['image_id']) is None:
        raise ValueError('candidate_only requires an immutable local image ID')
    cpu=value['cpu_seconds']
    if type(cpu) not in (int,float) or not math.isfinite(cpu) or not 0 < cpu <= 3600:
        raise ValueError('candidate_only requires positive finite stage CPU seconds <=3600')
    if type(value['memory_bytes']) is not int or not 64*1024*1024 <= value['memory_bytes'] <= 64*1024**3:
        raise ValueError('candidate_only requires explicit finite memory_bytes between64MiB and64GiB')
    if type(value['pids']) is not int or not 2 <= value['pids'] <= 256:
        raise ValueError('candidate_only requires pids in2..256')
    return deepcopy(value)


def validate_task(cfg,task):
    policy=validate_policy(cfg.get('evaluation'))
    if policy is None: return None
    parallel=cfg.get('parallel_search') or {}
    if parallel.get('enabled') is not True:
        raise ValueError('candidate_only requires parallel_search.enabled=true; workers=1 is supported')
    if (cfg.get('candidate_novelty') or {}).get('enabled') is not True:
        raise ValueError('candidate_only requires durable candidate novelty')
    # These native helper/repeat layouts are outside the ordinary wave driver.
    for key in ('repeat_population','repeat_promotion','circle_generation','circle_intervention'):
        if (cfg.get(key) or {}).get('enabled'):
            raise ValueError('candidate_only does not support '+key)
    if (cfg.get('container') or {}).get('selection')=='centres':
        raise ValueError('candidate_only requires the supported ordinary wave layout')
    from .registry import load_task
    current=load_task(task.root)
    if vars(task)!=vars(current):
        raise ValueError('candidate_only effective TaskSpec differs from the exact saved task')
    evaluator=hashlib.sha256(Path(task.evaluator_path).read_bytes()).hexdigest()
    matches=[kind for kind,pin in PINS.items() if pin['task_hash']==task.task_hash
             and pin['evaluator_sha256']==evaluator and pin['task']==task.qualified_name]
    if len(matches)!=1: raise ValueError('candidate_only has no exact registered task/evaluator adapter')
    for name in ('smoke','full','confirm'):
        limit=getattr(task,name+'_timeout')
        if type(limit) not in (int,float) or not math.isfinite(limit) or not 0 < limit <=7200:
            raise ValueError('candidate_only stage deadline must be finite and <=7200 seconds')
    if task.confirm_n not in (1,3): raise ValueError('registered task confirmation contract differs')
    return matches[0]
