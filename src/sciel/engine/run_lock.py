"""One process owns a run directory until its engine closes.

The lock file is permanent: unlinking it would let different processes lock
different inodes. The OS releases the advisory lock on process exit.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path


class RunDirectoryBusy(RuntimeError):
    pass


class RunDirectoryLock:
    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._file = open(Path(directory) / '.writer.lock', 'a+b')
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            self._file.close()
            raise RunDirectoryBusy(f'Run directory already has a writer: {directory}') from exc
        self._pid = os.getpid()

    def close(self):
        if not self._file.closed:
            # A forked child must not unlock its parent's shared open-file lock.
            if os.getpid() == self._pid:
                fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()

    def __del__(self):
        if hasattr(self, '_pid'):
            self.close()
