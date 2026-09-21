"""Feedback reaches the next request and survives eviction/resume without new calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from sciel.components.selection.novelty_chain import Selected
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult


ROOT = Path(__file__).resolve().parents[2]


class Provider(MockLLM):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def generate(self, prompt, model, temperature=None, **kwargs):
        self.prompts.append(prompt)
        code = f'def run_search():\n    return {10 + len(self.prompts)}.0\n'
        return LLMResponse('```python\n' + code + '```', model, .03, 20, 20)


def construct(path, provider):
    cfg = dict(task='test/toy_quadratic', budget=None, max_iterations=2,
        schedule_horizon=10, num_islands=1, inflight_cap=1, dup_cap=1,
        random_seed=23, schedule_policy='round_robin', milestone_fracs=[],
        embed_grid=False, embed_niche=False, portfolio_seed=False,
        dynamic_islands=False, idea_island=False, crossover=False,
        op_fixed='full_rewrite', policy={'enabled': False}, llm={'mock': True},
        pop_cap=1, archive_size=1)
    return build_engine(cfg, str(ROOT / 'tasks'), str(path),
                        python=sys.executable, llm=provider)


def evaluator(monkeypatch):
    calls = []

    def evaluate(task, path, **kwargs):
        calls.append(Path(path).read_text())
        return CascadeOutcome(EvalResult(ok=True, score=.1, cases={}, timing=.01),
                              .1, 'full', False)

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    return calls


def test_next_generation_follows_child_source_after_eviction_and_resume(tmp_path, monkeypatch):
    provider = Provider()
    calls = evaluator(monkeypatch)
    engine = construct(tmp_path / 'first', provider)
    try:
        engine.seed()
        engine._step(0)
        child = engine.db.get('p2')
        assert child.id not in engine.db.islands[0].programs
        feedback = engine.observed_changes.block(child.code)
        assert 'score_tie_unpaired' in feedback
        assert 'same_geometry' not in feedback
        assert 'behaviorally inert' not in feedback
        assert engine.B_evals == 1 and len(calls) == 2 and len(provider.prompts) == 1
        assert engine.n_records == 0
        saved = checkpoint.save(engine, 'feedback_fixture')
        saved_state = json.loads(Path(saved).read_text())['observed_changes']
    finally:
        engine.close()

    restored_provider = Provider()
    restored_provider.prompts.append('already generated in earlier segment')
    restored = construct(tmp_path / 'restored', restored_provider)
    try:
        checkpoint.load(restored, saved)
        assert restored.observed_changes.state() == saved_state
        assert restored.observed_changes.block(child.code) == feedback
        # Force an archived child to exercise source-bound feedback, independent
        # of the unchanged selection policy's choice among active members.
        monkeypatch.setattr(restored.selection, 'select', lambda *a, **k:
            Selected(parent_id=child.id, inspiration_ids=[], propensity=1., debug={}))
        restored._step(0)
        assert feedback in restored_provider.prompts[-1]
        assert len(calls) == 3 and restored.cost_total == pytest.approx(.06)
        assert restored.B_evals == 2 and restored.n_records == 0
    finally:
        restored.close()


@pytest.mark.parametrize('damage', ['missing', 'invalid', 'identity'])
def test_feedback_state_cannot_be_silently_dropped_or_forced(tmp_path, monkeypatch, damage):
    evaluator(monkeypatch)
    engine = construct(tmp_path / 'first', Provider())
    try:
        engine.seed()
        engine._step(0)
        saved = Path(checkpoint.save(engine, 'feedback_fixture'))
        state = json.loads(saved.read_text())
    finally:
        engine.close()
    if damage == 'missing':
        del state['observed_changes']
    elif damage == 'invalid':
        state['observed_changes'] = {'schema': 'broken'}
    else:
        state['identity']['observed_changes'] = {'schema': 'different'}
    saved.write_text(json.dumps(state))
    restored = construct(tmp_path / 'restored', Provider())
    before = deepcopy(restored.observed_changes.state())
    with pytest.raises(checkpoint.IdentityMismatch, match='observed change'):
        checkpoint.load(restored, str(saved), allow_identity_mismatch=True)
    assert restored.observed_changes.state() == before


def test_migrated_parent_uses_original_measurement_without_another_evaluation(tmp_path, monkeypatch):
    calls = evaluator(monkeypatch)
    engine = construct(tmp_path / 'migration', Provider())
    try:
        engine.seed()
        seed = engine.db.get('p1')
        island = engine.db.add_island()
        assert engine.db.migrate_unique_elites(0, [island]) == 1
        migrant = engine.db.get('p2')
        child = engine.db.add('def run_search():\n    return 91.0\n', .1, {}, island,
                              'gen', parent_id=migrant.id)
        receipt = CascadeOutcome(EvalResult(ok=True, score=.1, cases={}, timing=.01), .1, 'full', False)
        record = engine._observe_program(child, receipt, parent=migrant)
        assert record['status'] == 'score_tie_unpaired'
        assert record['parent_program_id'] == migrant.id
        assert record['parent_source_sha256'] == hashlib.sha256(seed.code.encode()).hexdigest()
        assert record['parent_observation']['score'] == seed.score
        assert len(calls) == 1
    finally:
        engine.close()
