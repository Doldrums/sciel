"""Refactored source closures must retain standalone and historical contracts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sciel.engine import source_identity
from sciel.hypothesis import circle_initializer_witness as witness
from sciel.hypothesis import restart_trace


def test_recursive_policy_closure_covers_extracted_and_nested_helpers(tmp_path, monkeypatch):
    root = tmp_path / 'sciel'
    names = (
        'engine/source_identity.py', 'engine/llm/new_transport.py',
        'hypothesis/codec/validation.py', 'knowledge_base/views/records.py',
        'config.py',
    )
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# ' + name + '\n')
    (root / 'hypothesis' / 'ignored.txt').write_text('not Python')
    monkeypatch.setattr(source_identity, '__file__', str(root / 'engine/source_identity.py'))
    before = source_identity.package_source_hashes(files=('config.py',))
    assert list(before) == sorted(names)
    changed = root / 'knowledge_base/views/records.py'
    changed.write_text('# new projection semantics\n')
    after = source_identity.package_source_hashes(files=('config.py',))
    assert after[changed.relative_to(root).as_posix()] != before[changed.relative_to(root).as_posix()]
    assert all(after[name] == before[name] for name in names if name != 'knowledge_base/views/records.py')


def test_historical_collector_is_still_exact_standalone_source():
    source = restart_trace.collector_source()
    assert hashlib.sha256(source.encode()).hexdigest() == (
        '4239a6990d7fc99ff6eaa41fd9a05762fdd4f1bbea5e5c683e3379d51717864c'
    )
    compile(source, '<historical-restart-codec>', 'exec')
    assert 'from sciel' not in source and 'from .' not in source


def test_witness_loaded_identity_rejects_fixture_source_drift(monkeypatch):
    # A moved recipe still participates in the loaded checker identity. No file
    # is edited and no initializer or child process is invoked.
    original = Path.read_bytes
    target = Path(witness.__file__).with_name('initializer_fixtures.py')

    def changed(path):
        raw = original(path)
        return raw + b'\n# changed recipe\n' if path == target else raw

    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(ValueError, match='loaded_checker_drift'):
        witness.identity()


@pytest.mark.parametrize('module', [
    'restart_trace_recorder', 'restart_trace_validation', 'initializer_fixtures',
    'initializer_evidence_views', 'conclusion_fragments', 'prompt_rules',
])
def test_extracted_helpers_import_without_public_facade_import_order(module):
    source_root = Path(witness.__file__).resolve().parents[2]
    script = ('import sys; sys.path.insert(0, ' + repr(str(source_root)) + '); '
              'import importlib; importlib.import_module(' + repr('sciel.hypothesis.' + module) + ')')
    result = subprocess.run([sys.executable, '-I', '-c', script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
