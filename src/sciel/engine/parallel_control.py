"""Physical dispatch proof and cooperative cancellation signals."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from ..campaign.budget import CampaignStopped, RequestNotDispatched


def dispatch_evidence(directory, request_hash):
    """Return the digest of regular, bounded dispatch proof shared by both replays."""
    path = Path(directory) / 'dispatched.json'
    try:
        if path.parent.is_symlink():
            raise ValueError('Parallel dispatch directory is a symlink')
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
                raise ValueError('Parallel dispatch evidence is not a bounded regular file')
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise ValueError('Parallel dispatch evidence is too large')
        marker = json.loads(raw)
        if (type(marker) is not dict or marker.get('request_sha256') != request_hash
                or type(marker.get('pid')) is not int or marker['pid'] <= 0
                or type(marker.get('timestamp')) is not str):
            raise ValueError('Parallel dispatch marker differs from its intent')
        return hashlib.sha256(raw).hexdigest()
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError('Parallel call lacks valid physical dispatch evidence marker') from exc

class UnknownParallelCall(CampaignStopped):
    """A submitted call without a durable terminal receipt must be reconciled."""

class ParallelPause(RequestNotDispatched):
    """This leaf operation was paused before dispatch and may run on resume."""

class ParallelAborted(RequestNotDispatched):
    """A sibling failed; this leaf operation was never dispatched."""

def _paused_error(error):
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, ParallelPause):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False
