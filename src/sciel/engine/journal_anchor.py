"""Refuse to replay paid model work recorded after the last checkpoint."""
from __future__ import annotations

import hashlib
import json


def assert_no_uncheckpointed_research(log_path, anchor):
    """Do not replay paid work after a crash beyond the last checkpoint."""
    if not anchor:
        raise ValueError('research resume requires a bound journal anchor')
    with open(log_path, 'rb') as source:
        offset = anchor['byte_offset']
        if type(offset) is not int or offset < 1:
            raise ValueError('invalid research journal offset')
        prefix = source.read(offset)
        if len(prefix) != offset or hashlib.sha256(prefix).hexdigest() != anchor['prefix_sha256']:
            raise ValueError('research journal does not match checkpoint prefix')
        tail = source.read()
    for line in tail.splitlines():
        try:
            item = json.loads(line)
        except (ValueError, UnicodeError):
            raise ValueError('journal has an incomplete post-checkpoint tail') from None
        if item.get('event') == 'research':
            raise ValueError('uncheckpointed research activity; refusing to replay model calls')
