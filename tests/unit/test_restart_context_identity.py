"""Restart admission binds its extracted wire codec, not only its public facade."""
import hashlib
from pathlib import Path

import pytest

from sciel.knowledge_base import restart_context


@pytest.mark.parametrize('name', ['restart_encoding.py', 'restart_codec.py'])
def test_extracted_codec_drift_changes_admission_identity(monkeypatch, name):
    before = restart_context.admission_identity()
    target = Path(restart_context.__file__).with_name(name).resolve()
    original = Path.read_bytes
    assert before['implementation'][name] == hashlib.sha256(original(target)).hexdigest()

    def drift(path):
        data = original(path)
        return data + b'\n# fixture source drift' if path.resolve() == target else data

    monkeypatch.setattr(Path, 'read_bytes', drift)
    after = restart_context.admission_identity()
    assert after['source_sha256'] == before['source_sha256']
    assert after['policy'] == before['policy']
    assert after['implementation'][name] != before['implementation'][name]
