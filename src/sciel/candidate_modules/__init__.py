"""Load source files under fresh names that a spawned Python worker can import.

The name contains its original absolute path, source digest and a per-load nonce.
Importing this package in a fresh interpreter installs the same narrow finder;
no parent-only registry, source copy, bytecode cache or process-mode override is
needed. This provides importability, not isolation or a sandbox.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.util
import os
from pathlib import Path
import re
import sys
import uuid


_PREFIX = __name__ + "."
_IDENTITY = re.compile(r"_p((?:[0-9a-f]{2})+)_s([0-9a-f]{64})_n([0-9a-f]{32})", re.ASCII)


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
    # import_module also removes a partly initialized module if execution raises.
    return importlib.import_module(_PREFIX + identity)


_install()
