"""Durable candidate-evaluation attachments and runtime identity."""
import hashlib
import json
import os
from pathlib import Path
import stat

from ..methods.models import canonical_bytes, content_hash

SCHEMA = 'sciel-candidate-evaluation/1'
FAILURE_SCHEMA = 'sciel-candidate-evaluation-failure/1'
MAX_ATTACHMENT_BYTES = 4 * 1024 * 1024
BOUND = ('method_id', 'invocation_id', 'plan_id', 'reservation_id',
         'source_sha256', 'runtime_sha256', 'input_sha256', 'request_sha256')
ROOT = Path(__file__).resolve().parent


def sha(raw): return hashlib.sha256(raw).hexdigest()

def _sealed(value): return {**value,'sha256':content_hash(value)}

def _read(path):
    if path.parent.is_symlink(): raise ValueError('evaluation attachment directory is a symlink')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size>MAX_ATTACHMENT_BYTES:
            raise ValueError('invalid bounded evaluation attachment')
        raw=stream.read(MAX_ATTACHMENT_BYTES+1)
    if len(raw)>MAX_ATTACHMENT_BYTES: raise ValueError('evaluation attachment too large')
    value=json.loads(raw)
    if type(value) is not dict or value.get('sha256')!=content_hash({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('evaluation attachment seal differs')
    return value

def _write(path,value):
    from ..engine.parallel_calls import durable_json
    if path.is_symlink() or path.exists():
        raise ValueError('evaluation attachment already exists; no physical redispatch')
    value=_sealed(value)
    if len(canonical_bytes(value))>MAX_ATTACHMENT_BYTES: raise ValueError('evaluation attachment too large')
    durable_json(path,value,ensure_ascii=True)
    return {'name':path.name,'sha256':value['sha256']}


def runtime_sources():
    """Bind every trusted Python module, including extracted orchestration helpers."""
    package = ROOT.parent
    return {
        str(path.relative_to(package)): sha(path.read_bytes())
        for path in sorted(package.rglob('*.py'))
    }
