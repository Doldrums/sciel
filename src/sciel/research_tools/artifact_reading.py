"""Bounded reads of immutable JSON artifacts through regular-file descriptors."""
import errno
import os
import stat

from .cells import _decode


def _signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def read_json(path, limit):
    """Reject special files and authenticate a stable descriptor before decoding."""
    if path.is_symlink():
        raise ValueError('research artifact path is a symlink')
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError('research artifact path is a symlink') from error
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError('research artifact is oversized or not a regular file')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(descriptor)
        if len(raw) != before.st_size or _signature(before) != _signature(after):
            raise ValueError('research artifact changed during read')
    finally:
        os.close(descriptor)
    return _decode(raw)
