"""Descriptor fingerprints follow loaded bytes even if their paths are replaced."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def artifacts(tmp_path):
    torch = pytest.importorskip('torch')
    from sciel.components.policy.code_embed import BehaviorAxis, Embedder
    embed = tmp_path / 'embed.pt'
    torch.save({'in_dim': 8, 'state': Embedder(8).state_dict()}, embed)
    embed_hash = hashlib.sha256(embed.read_bytes()).hexdigest()
    axis = tmp_path / 'axis.npz'
    BehaviorAxis(np.eye(64)[0], np.zeros(64), embed_sha=embed_hash[:12],
                 version='fixture').save(str(axis))
    return embed, embed_hash, axis, hashlib.sha256(axis.read_bytes()).hexdigest()


def replace_after_read(monkeypatch, target):
    original = Path.read_bytes

    def read_bytes(path):
        raw = original(path)
        if path == target:
            path.write_bytes(b'replaced after the original artifact was read')
        return raw

    monkeypatch.setattr(Path, 'read_bytes', read_bytes)


@pytest.mark.parametrize('replaced', ['embedder', 'axis'])
def test_fitted_axis_checks_and_stamps_loaded_artifact_bytes(tmp_path, monkeypatch, replaced):
    embed, embed_hash, axis, axis_hash = artifacts(tmp_path)
    from sciel.components.policy.code_embed import LearnedDescriptor
    replace_after_read(monkeypatch, embed if replaced == 'embedder' else axis)
    descriptor = LearnedDescriptor.load(str(embed), axis_path=str(axis))
    assert descriptor.embedder.artifact_identity['sha256'] == embed_hash
    assert descriptor.axis.artifact_identity['sha256'] == axis_hash
    assert descriptor.axis.embed_sha == embed_hash[:12]
    assert descriptor.axis_sha == axis_hash[:12]
    assert descriptor.kind == 'pca2_fixture@' + axis_hash[:8]


@pytest.mark.parametrize('kind', ['grid', 'vq'])
def test_engine_stamps_loaded_descriptor_hash_not_replaced_path(tmp_path, monkeypatch, kind):
    embed, embed_hash, _, _ = artifacts(tmp_path)
    from sciel.engine.loop import Engine
    seed = tmp_path / 'seed.py'
    seed.write_text('def run(): return 1\n')  # parsed for interface only, never executed
    task = SimpleNamespace(qualified_name='test/fixture', initial_program_path=str(seed), evaluator_path=None)
    replace_after_read(monkeypatch, embed)

    def no_calls(*_, **__):
        pytest.fail('constructing descriptor identity must not dispatch a model call')

    llm = SimpleNamespace(models=['inert'], generate=no_calls)
    engine = Engine({'embed_grid': kind == 'grid', 'embed_grid_path': str(embed),
                     'embed_axis_path': None, 'embed_niche': kind == 'vq',
                     'embed_niche_path': str(embed), 'research': {'enabled': False},
                     'policy': {'enabled': False}}, task, str(tmp_path / 'run'), llm)
    try:
        assert engine.descriptor_info['kind'] == ('legacy_random_raw' if kind == 'grid' else 'vq_niche')
        assert engine.descriptor_info['embed_sha'] == embed_hash[:12]
    finally:
        engine.close()
