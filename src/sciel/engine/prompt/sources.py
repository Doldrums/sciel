"""Exact source digests for prompt inputs."""
from __future__ import annotations

import hashlib


def source_digest(code):
    if type(code) is not str:
        raise ValueError('source must be text')
    return hashlib.sha256(code.encode('utf-8', errors='surrogatepass')).hexdigest()
