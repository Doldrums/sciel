"""Container PID1 watchdog. Never import candidate code in this root process.

Exiting PID1 ends the entire private PID namespace, including setsid descendants.
The unprivileged child cannot signal this supervisor. Its deadline survives death
of the Docker CLI or the host evaluator. No objective or secrets enter this image.
"""

import os
import json
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import time


MAX_SOURCE_BYTES = 262144


def _exact(fd, length, deadline):
    data = bytearray()
    while len(data) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise TimeoutError("bootstrap deadline")
        part = os.read(fd, length - len(data))
        if not part:
            raise EOFError("bootstrap closed")
        data.extend(part)
    return bytes(data)


def main():
    if os.getpid() != 1 or os.geteuid() != 0:
        raise RuntimeError("supervisor requires private PID namespace as root PID1")
    timeout = float(sys.argv[1])
    if not 0 < timeout <= 30:
        raise ValueError("invalid watchdog timeout")
    deadline = time.monotonic() + timeout
    os.write(1, json.dumps({"status": "ready", "protocol": "sciel-hidden-oracle/1"}).encode() + b"\n")
    size = struct.unpack("!I", _exact(0, 4, deadline))[0]
    if not 0 < size <= MAX_SOURCE_BYTES:
        raise ValueError("invalid candidate size")
    source = _exact(0, size, deadline)
    path = Path("/input/candidate.py")
    path.write_bytes(source)
    path.chmod(0o444)
    os.mkdir("/tmp/work", mode=0o777)
    os.chmod("/tmp/work", 0o777)
    child = subprocess.Popen(
        [sys.executable, "-B", "/opt/oracle/candidate_runner.py"],
        stdin=0, stdout=1, stderr=2, cwd="/tmp/work",
        user=65534, group=65534, extra_groups=[], start_new_session=True,
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
             "HOME": "/tmp/work", "TMPDIR": "/tmp/work",
             "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1",
             "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    try:
        return 0 if child.wait(timeout=max(0.001, deadline - time.monotonic())) == 0 else 1
    except subprocess.TimeoutExpired:
        return 124
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # Do not wait for detached descendants or inherited pipes. PID1 exits.


if __name__ == "__main__":
    try:
        code = main()
    except BaseException as exc:
        os.write(2, f"supervisor: {type(exc).__name__}: {exc}\n".encode())
        code = 125
    os._exit(code if 0 <= code <= 255 else 125)
