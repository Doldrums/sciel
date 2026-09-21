"""Trusted immutable bundle entrypoint: primitive data only, no candidate file."""
import base64
import os
import sys


def run(inputs):
    if sys.platform != 'linux' or os.geteuid() != 10001 or os.getpid() == 1:
        raise RuntimeError('trusted scorer requires the unprivileged Linux container child')
    if type(inputs) is not dict or set(inputs) != {'kind', 'payload_base64', 'binding'}:
        raise ValueError('invalid scorer input fields')
    if type(inputs['payload_base64']) is not str or len(inputs['payload_base64']) > 4 * ((128 * 1024 + 2) // 3):
        raise ValueError('scorer payload encoding bound')
    payload = base64.b64decode(inputs['payload_base64'], validate=True)
    if not 0 < len(payload) <= 128 * 1024:
        raise ValueError('scorer payload byte bound')
    # The method loader namespaces only this immutable trusted_scoring sibling.
    # Neither candidate.py, evaluator.py nor any task assets are present.
    from trusted_scoring.adapter import HostBinding, score_payload
    import platform
    import numpy as np
    return {'schema': 'sciel-trusted-scorer-output/1',
            'result': score_payload(inputs['kind'], payload, HostBinding(**inputs['binding'])),
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__}}
