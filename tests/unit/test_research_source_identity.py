"""Moved Engine policy code remains part of durable research identity."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.research_tools.controller import ResearchTools


def controller(tmp_path):
    library = SimpleNamespace(path=tmp_path / 'library', identity={'executor': 'inert fixture'})
    return ResearchTools({'enabled': True, 'image_id': 'sha256:' + '1' * 64}, tmp_path,
                         task_identity={'task_hash': 'a' * 64}, emit=lambda **_: None,
                         charge=lambda _: None, library=library)


@pytest.mark.parametrize('module', ['evaluation.py', 'research_conclusions.py', 'cooperative_io.py'])
def test_moved_engine_policy_invalidates_research_checkpoint(tmp_path, monkeypatch, module):
    previous = controller(tmp_path / 'before')
    saved = previous.state()
    original = Path.read_bytes

    def changed(path):
        raw = original(path)
        return raw + b'\n# altered policy\n' if path.name == module and path.parent.name == 'engine' else raw

    # Only the identity reader sees changed bytes. No code, source file or
    # execution leaf is modified or run by this fixture.
    monkeypatch.setattr(Path, 'read_bytes', changed)
    current = controller(tmp_path / 'after')
    assert current.identity != previous.identity
    assert module in current.identity['integration']
    with pytest.raises(ValueError, match='checkpoint identity'):
        current.load_state(saved)
