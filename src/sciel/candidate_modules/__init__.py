"""Load source files under fresh names that a spawned Python worker can import.

The name contains its original absolute path, source digest and a per-load nonce.
Importing this package in a fresh interpreter installs the same narrow finder;
no parent-only registry, source copy, bytecode cache or process-mode override is
needed. This provides importability, not isolation or a sandbox.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import types
import uuid


_PREFIX = __name__ + "."
_IDENTITY = re.compile(r"_p((?:[0-9a-f]{2})+)_s([0-9a-f]{64})_n([0-9a-f]{32})", re.ASCII)


_NAMED = re.compile(re.escape(_PREFIX) + r"_p((?:[0-9a-f]{2})+)_s[0-9a-f]{64}_n[0-9a-f]{32}", re.ASCII)


def readable(text):
    """Text with each candidate module name replaced by the last two parts of its file path, as an error reads it."""
    def name(match):
        try:
            parts = Path(os.fsdecode(bytes.fromhex(match.group(1)))).parts
        except ValueError:
            return match.group(0)
        return "/".join(parts[-2:])
    return _NAMED.sub(name, text) if isinstance(text, str) else text


def _identity(fullname):
    """Decode only this namespace's canonical, single-level module identities."""
    if not fullname.startswith(_PREFIX):
        return None
    match = _IDENTITY.fullmatch(fullname[len(_PREFIX):])
    if match is None:
        raise ModuleNotFoundError("Invalid SCiEL candidate module identity", name=fullname)
    encoded, expected_sha, _nonce = match.groups()
    path = os.fsdecode(bytes.fromhex(encoded))
    if ("\0" in path or not os.path.isabs(path) or os.path.normpath(path) != path
            or os.fsencode(path).hex() != encoded):
        raise ModuleNotFoundError("Invalid SCiEL candidate source path", name=fullname)
    return path, expected_sha


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, path, expected_sha):
        self.path = path
        self.expected_sha = expected_sha

    def create_module(self, spec):
        return None

    def _source(self):
        source = Path(self.path).read_bytes()
        if hashlib.sha256(source).hexdigest() != self.expected_sha:
            raise ImportError("SCiEL candidate source changed after loading: " + self.path)
        return source

    def get_filename(self, fullname):
        return self.path

    def get_source(self, fullname):
        return importlib.util.decode_source(self._source())

    def get_code(self, fullname):
        # Compiling the bytes directly honors Python encoding cookies and skips
        # both reading and writing timestamp-based .pyc files.
        return compile(self._source(), self.path, "exec", dont_inherit=True)

    def exec_module(self, module):
        # Normal importlib has already registered the module before execution,
        # as required by dataclasses, recursive imports and ordinary pickling.
        module.__file__ = self.path
        module.__cached__ = None
        exec(self.get_code(module.__name__), module.__dict__)


class _CandidateFinder(importlib.abc.MetaPathFinder):
    _sciel_candidate_namespace = __name__

    def find_spec(self, fullname, path=None, target=None):
        identity = _identity(fullname)
        if identity is None:
            return None
        original, expected_sha = identity
        return importlib.util.spec_from_file_location(
            fullname, original, loader=_SourceLoader(original, expected_sha))


def _install():
    # Import/reload must not install duplicate hooks. The hook handles no names
    # outside this package and cannot redirect ordinary application imports.
    if not any(getattr(finder, "_sciel_candidate_namespace", None) == __name__
               for finder in sys.meta_path):
        sys.meta_path.insert(0, _CandidateFinder())


# The evaluation runner names the program it scores; time spent importing it and in its
# module-level functions is that program's own, the rest of the stage is scoring overhead.
_NOTHING = object()
_TIMED = {"path": None, "seconds": 0.0, "called": False, "depth": 0, "calls": 0,
          "kept": _NOTHING, "kept_used": False}


class StageBudgetExceeded(BaseException):
    """Raised inside the timed program when its stage budget runs out. A BaseException, so the `except Exception`
    of a search loop does not swallow it; it is re-raised every half second until the program's entry function
    unwinds."""


def time_program(program_path):
    _TIMED.update(path=str(Path(program_path).resolve(strict=True)), seconds=0.0, called=False, depth=0, calls=0,
                  kept=_NOTHING, kept_used=False)


def program_seconds():
    """Seconds spent in the timed program, or None when none of its functions ran in this process."""
    return _TIMED["seconds"] if _TIMED["called"] else None


def keep(result):
    """Record a copy of result as the timed program's answer so far (sciel.task.keep)."""
    _TIMED["kept"] = copy.deepcopy(result)


def kept_used():
    """Whether an entry function of the timed program returned its kept result at the end of its budget."""
    return _TIMED["kept_used"]


def _expire(signum, frame):
    raise StageBudgetExceeded


def _arm():
    """Interrupt the program when its stage budget runs out (SCIEL_INTERRUPT_SECONDS, which the sandbox sets at most
    SCIEL_STAGE_SECONDS): (previous SIGALRM handler,) or None when no budget is set or no timer can be armed here.
    The import of the program counts against its first call."""
    try:
        budget = float(os.environ["SCIEL_INTERRUPT_SECONDS"])
    except (KeyError, ValueError):
        return None
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        return None
    if not _TIMED["calls"]:
        budget -= _TIMED["seconds"]
    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, max(0.05, budget), 0.5)
    return (previous,)


def _disarm(armed):
    if armed is not None:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, armed[0] if armed[0] is not None else signal.SIG_DFL)


def _timed(fn):
    def call(*args, **kwargs):
        outer = not _TIMED["depth"]
        if outer:
            _TIMED["kept"] = _NOTHING
        _TIMED["depth"] += 1
        _TIMED["called"] = True
        started = time.monotonic()
        armed = _arm() if outer else None
        try:
            try:
                return fn(*args, **kwargs)
            finally:
                _disarm(armed)
        except StageBudgetExceeded:
            # Only the entry function the evaluator called answers for the program: it returns the last kept result,
            # or fails as an ordinary exception the evaluator reports.
            if not outer:
                raise
            if _TIMED["kept"] is _NOTHING:
                raise TimeoutError(
                    "the program ran past its stage budget (SCIEL_STAGE_SECONDS) and was interrupted without a "
                    "kept result; call sciel.task.keep(best) whenever the best improves, and the last kept result "
                    "is scored when the budget runs out") from None
            _TIMED["kept_used"] = True
            return _TIMED["kept"]
        finally:
            _TIMED["depth"] -= 1
            if not _TIMED["depth"]:
                _TIMED["calls"] += 1
                _TIMED["seconds"] += time.monotonic() - started
    for key in ("__module__", "__name__", "__qualname__", "__doc__", "__wrapped__"):
        setattr(call, key, fn if key == "__wrapped__" else getattr(fn, key))
    return call


def load_candidate(program_path, module_name="program"):
    """Execute exact source in a fresh module importable by spawn/forkserver.

    ``module_name`` is retained as a caller compatibility label; the actual
    module name is collision-free and must not be replaced by that alias.
    Source must remain at its original path, unchanged, while workers use it.
    Existing ambient import paths are preserved. This does not add the source
    directory to ``sys.path``, isolate conflicting sibling-module names or add
    package-relative import support to a standalone candidate file.
    """
    if not isinstance(module_name, str) or not module_name:
        raise ValueError("candidate module label must be a nonempty string")
    original = str(Path(program_path).resolve(strict=True))
    source = Path(original).read_bytes()
    identity = ("_p" + os.fsencode(original).hex()
                + "_s" + hashlib.sha256(source).hexdigest()
                + "_n" + uuid.uuid4().hex)
    if original != _TIMED["path"]:
        # import_module also removes a partly initialized module if execution raises.
        return importlib.import_module(_PREFIX + identity)
    started = time.monotonic()
    try:
        module = importlib.import_module(_PREFIX + identity)
    finally:
        _TIMED["seconds"] += time.monotonic() - started
    for name, value in list(vars(module).items()):
        if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
            setattr(module, name, _timed(value))
    return module


_install()
