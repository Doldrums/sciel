"""Artifact reads reject blocking special files and descriptor-open races."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sciel.methods.models import content_hash
from sciel.research_tools import tool_artifacts


@pytest.mark.parametrize('operation', ['read', 'reuse'])
def test_fifo_artifact_is_rejected_without_blocking(tmp_path, operation):
    artifact = {'schema': tool_artifacts.ARTIFACT_SCHEMA, 'kind': 'proposal', 'payload': {'x': 1}}
    ident = content_hash(artifact)
    os.mkfifo(tmp_path / (ident + '.json'))
    probe = '''
from pathlib import Path
from types import SimpleNamespace
import sys
from sciel.research_tools import tool_artifacts
owner = SimpleNamespace(path=Path(sys.argv[1]))
try:
    if sys.argv[3] == 'read':
        tool_artifacts._read_artifact(owner, sys.argv[2], 'proposal')
    else:
        tool_artifacts._artifact(owner, 'proposal', {'x': 1})
except ValueError as error:
    assert 'regular file' in str(error), str(error)
else:
    raise AssertionError('FIFO was accepted as an artifact')
'''
    completed = subprocess.run(
        [sys.executable, '-c', probe, str(tmp_path), ident, operation],
        capture_output=True, text=True, timeout=2,
    )
    assert completed.returncode == 0, completed.stderr


def test_existing_artifact_roundtrip_and_corruption(tmp_path):
    owner = SimpleNamespace(path=tmp_path)
    ident = tool_artifacts._artifact(owner, 'proposal', {'x': 1})
    assert tool_artifacts._artifact(owner, 'proposal', {'x': 1}) == ident
    assert tool_artifacts._read_artifact(owner, ident, 'proposal') == {'x': 1}
    (tmp_path / (ident + '.json')).write_text('{"changed":true}')
    with pytest.raises(ValueError, match='modified'):
        tool_artifacts._artifact(owner, 'proposal', {'x': 1})
    with pytest.raises(ValueError, match='identity differs'):
        tool_artifacts._read_artifact(owner, ident, 'proposal')


def test_symlink_substituted_during_open_is_not_followed(tmp_path, monkeypatch):
    from sciel.research_tools import artifact_reading

    target = tmp_path / 'artifact.json'
    target.write_text('{}')
    other = tmp_path / 'other.json'
    other.write_text('{"unrelated":true}')
    original_open = os.open

    def substitute(path, flags):
        target.unlink()
        target.symlink_to(other)
        return original_open(path, flags)

    monkeypatch.setattr(artifact_reading.os, 'open', substitute)
    with pytest.raises(ValueError, match='symlink'):
        artifact_reading.read_json(target, 4096)


def test_oversized_and_nonobject_artifacts_are_rejected(tmp_path):
    from sciel.research_tools.artifact_reading import read_json

    target = tmp_path / 'large.json'
    target.write_text('{"value":"too large"}')
    with pytest.raises(ValueError, match='oversized'):
        read_json(target, 3)
    owner = SimpleNamespace(path=tmp_path)
    ident = content_hash([])
    (tmp_path / (ident + '.json')).write_text('[]')
    with pytest.raises(ValueError, match='identity differs'):
        tool_artifacts._read_artifact(owner, ident, 'proposal')
