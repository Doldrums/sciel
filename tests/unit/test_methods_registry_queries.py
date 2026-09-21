"""Selective reads keep full integrity checks and detached result ownership."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

import sciel.methods.registry as registry_module
from sciel.methods.registry import Registry, RegistryCorruption


def test_kind_filter_returns_ordered_detached_records_without_copying_other_events(tmp_path, monkeypatch):
    registry = Registry(tmp_path)
    registry.append('unrelated', {'large': 'x' * 200_000})
    registry.append('selected', {'nested': {'value': 1}})
    registry.append('selected', {'nested': {'value': 2}})
    registry.events()  # Populate the verified cache before measuring the read path.
    copied = []

    def record_copy(value):
        copied.append(value)
        return deepcopy(value)

    monkeypatch.setattr(registry_module, 'deepcopy', record_copy)
    selected = registry.events('selected')
    assert [row['data']['nested']['value'] for row in selected] == [1, 2]
    assert len(copied) == 1
    assert [row['kind'] for row in copied[0]] == ['selected', 'selected']
    selected[0]['data']['nested']['value'] = 99
    assert registry.events('selected')[0]['data']['nested']['value'] == 1
    assert registry.events('missing') == []


def test_filtered_read_checks_artifacts_of_unselected_events(tmp_path):
    registry = Registry(tmp_path)
    artifact = registry.put_artifact('unselected', {'value': 1})
    registry.append('selected', {})
    registry.events('selected')
    (tmp_path / 'artifacts' / (artifact + '.json')).write_text('{}')
    with pytest.raises(RegistryCorruption, match='content hash mismatch'):
        registry.events('selected')


def test_filtered_read_observes_other_writers_without_mutating_the_full_journal(tmp_path):
    registry = Registry(tmp_path)
    registry.append('selected', {'value': 1})
    assert len(registry.events('selected')) == 1
    other = Registry(tmp_path)
    other.append('unrelated', {})
    other.append('selected', {'value': 2})
    before = (tmp_path / 'journal.jsonl').read_bytes()
    assert [row['data']['value'] for row in registry.events('selected')] == [1, 2]
    assert len(registry.events()) == 3
    assert (tmp_path / 'journal.jsonl').read_bytes() == before


@pytest.mark.parametrize('operation', ['read', 'immutable_write'])
def test_special_file_is_refused_before_a_blocking_open(tmp_path, operation):
    fifo = tmp_path / 'artifact.json'
    os.mkfifo(fifo)
    code = '''
from pathlib import Path
import sys
from sciel.methods.registry_records import RegistryCorruption, _atomic, _read
path = Path(sys.argv[1])
try:
    if sys.argv[2] == 'read':
        _read(path)
    else:
        _atomic(path, b'{}', immutable=True)
except RegistryCorruption:
    pass
else:
    raise AssertionError('special file was accepted')
'''
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-c', code, str(fifo), operation],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH': str(root / 'src')},
        capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.glob('.write-*'))
