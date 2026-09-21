"""Frozen policy identity describes the artifact loaded, not just its path."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.components.policy.controller import Controller
from sciel.components.policy.mode_policy_mlp import ReturnConditionedModeMLP
from sciel.engine import checkpoint


def write_mode(path, preferred):
    model = ReturnConditionedModeMLP(hidden=2)
    model.W1 = np.zeros((model.in_dim, 2)); model.b1 = np.zeros(2)
    model.W2 = np.zeros((2, 3)); model.b2 = np.zeros(3)
    model.b2[preferred] = 10
    model.save(str(path))


def engine(policy=None, three_head=None):
    return SimpleNamespace(task=SimpleNamespace(qualified_name='test/fixture', task_hash='same'),
        cfg={'policy': policy or {}}, descriptor_info={},
        controller=Controller(policy or {}, 1), three_head=three_head)


def test_replaced_mode_artifact_refuses_resume_and_original_loaded_identity_is_stable(tmp_path):
    path = tmp_path / 'mode.npz'
    write_mode(path, 0)
    cfg = {'enabled': True, 'mode_source': 'mlp', 'mode_policy_path': str(path)}
    first = engine(cfg)
    saved = {'identity': checkpoint._identity(first)}
    write_mode(path, 1)
    # Checkpointing the existing engine still binds the original loaded weights.
    assert checkpoint._identity(first) == saved['identity']
    restored = engine(cfg)
    assert any('frozen_policies' in p for p in checkpoint.check_identity(restored, saved))
    assert first.controller.mode_mlp.dist(np.zeros(first.controller.mode_mlp.in_dim - 1), 1) != restored.controller.mode_mlp.dist(np.zeros(restored.controller.mode_mlp.in_dim - 1), 1)
    assert checkpoint.check_identity(first, saved) == []
    # An old checkpoint without the fingerprint cannot prove compatibility.
    old = deepcopy(saved); old['identity'].pop('frozen_policies')
    assert any('frozen_policies' in p for p in checkpoint.check_identity(restored, old))


@pytest.mark.parametrize('changed', ['model', 'embedder', 'beta'])
def test_three_head_identity_binds_both_loaded_artifacts_and_tilt(tmp_path, changed):
    torch = pytest.importorskip('torch')
    from sciel.components.policy.code_embed import Embedder
    from sciel.components.policy.three_head import ThreeHeadModel, ThreeHeadTilt
    model_path, embed_path = tmp_path / 'model.pt', tmp_path / 'embed.pt'
    model = ThreeHeadModel(64, n_niche=3, d=8, heads=2, blocks=1)
    model.save(str(model_path))
    embedder = Embedder(8)
    torch.save({'in_dim': 8, 'state': embedder.state_dict()}, embed_path)
    first = engine(three_head=ThreeHeadTilt.load(str(model_path), str(embed_path)))
    saved = {'identity': checkpoint._identity(first)}
    assert checkpoint.check_identity(first, saved) == []
    beta = .3
    if changed == 'model':
        with torch.no_grad():
            next(model.net.parameters()).add_(1)
        model.save(str(model_path))
    elif changed == 'embedder':
        with torch.no_grad():
            next(embedder.parameters()).add_(1)
        torch.save({'in_dim': 8, 'state': embedder.state_dict()}, embed_path)
    else:
        beta = .7
    assert checkpoint._identity(first) == saved['identity']
    restored = engine(three_head=ThreeHeadTilt.load(str(model_path), str(embed_path), beta))
    assert any('frozen_policies' in p for p in checkpoint.check_identity(restored, saved))
