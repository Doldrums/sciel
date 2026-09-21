"""New profile lifecycle with inert draws/models, never numerical execution."""
import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.engine import checkpoint
from sciel.engine.repeat_support import create
from sciel.engine.repeat_promotion import RepeatPromotion
from sciel.config import apply_defaults
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis.restart_optimizer_trace import RestartTrace
from sciel.hypothesis.restart_observations import observation_cases
from sciel.knowledge_base.catalogs import canonical_sha, canonical_json
from test_circle_intervention_engine import task_copy
from test_repeat_engine_integration import FakeDraws
import test_traced_circle_engine as flow


class OutcomeDraws(FakeDraws):
    identity = copy.deepcopy(flow.Draws.identity)

    def draw(self, source, *, restart_trace=False, **kwargs):
        assert restart_trace is True
        builder.body_of(source)
        receipt = super().draw(source, **kwargs)
        self.calls[-1]['restart_trace'] = True
        q = receipt['raw_sum']
        binding = dict(source_sha256=receipt['source_sha256'], seed=receipt['seed'],
            runtime_sha256=canonical_sha(self.identity['runtime']),
            scaffold_sha256=builder.identity()['common_scaffold_sha256'],
            invocation_id=hashlib.sha256((source+str(receipt['seed'])).encode()).hexdigest()[:32])
        trace = RestartTrace(binding)
        value = np.array([coordinate for center in receipt['capture']['centers'] for coordinate in center]
                         + receipt['capture']['radii'])
        args = {name: value.copy() for name in ('best', 'initial', 'noise', 'ordinary')}
        trace.begin_start(0, incumbent_sum=q, improvements=0)
        trace.begin_helper(args)
        trace.helper_end(raw_kind='none', post_inputs=args, effective=value,
            origin='saved_ordinary_fallback', incumbent_sum=q, improvements=0)
        trace.optimizer_enter(incumbent_sum=q, improvements=0)
        trace.optimizer_exit(incumbent_sum=q, improvements=0)
        # A failed optimizer can return feasible geometry without admitting a gain.
        trace.optimizer_outcome(False, SimpleNamespace(status=9))
        trace.optimizer_result(value)
        trace.end_start('completed', incumbent_sum=q, improvements=0)
        receipt['restart_trace'] = dict(status='captured', reason=None, sidecar_sha256='c'*64,
            trace=trace.finish('restart_limit', incumbent_sum=q, improvements=0))
        return receipt


@pytest.fixture
def profile(monkeypatch):
    # Reuse the existing strict lifecycle fixture with the explicitly new family.
    monkeypatch.setattr(flow, 'traced', builder)
    monkeypatch.setattr(flow, 'Draws', OutcomeDraws)


def outcomes(prompt):
    context = flow.parse(prompt)
    return [case for card in context['cards'] if card['observation_view'] is not None
            for case in observation_cases(card['observation_view']['observation'])]


@pytest.mark.parametrize('route', ['repeat_promotion', 'repeat_population'])
@pytest.mark.parametrize('pair_mode', [False, True])
def test_optimizer_profile_reaches_reflection_and_generation_after_resume(
        tmp_path, monkeypatch, profile, route, pair_mode):
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=builder.source_for())
    folder = tmp_path/'run'
    first, model = flow.construct(task, folder, draws, route, pair_mode, monkeypatch)
    first._stop_at = 1
    first.run()
    reflected = next(call['prompt'] for call in model.calls if call['kind'] == 'reflection')
    before = canonical_json(first.research.knowledge.comparisons)
    cases = outcomes(reflected)
    assert cases and all(c['success'] is False and c['status'] == 9 and
                         c['result_tolerance_valid'] is True and c['phase_improvements'] == [0]*5 for c in cases)
    state = json.loads((folder/'checkpoint.json').read_text())
    assert state['identity']['circle_generation']['identity']['execution_profile'] == builder.EXECUTION_PROFILE
    resumed, following = flow.construct(task, folder, draws, route, pair_mode, monkeypatch, start=1)
    checkpoint.load(resumed, str(folder/'checkpoint.json'))
    assert canonical_json(resumed.research.knowledge.comparisons) == before
    resumed.run()
    generated = next(call['prompt'] for call in following.calls if call['kind'] == 'mutation')
    delivered = outcomes(generated)
    assert delivered and all(c['success'] is False and c['status'] == 9 and c['result_tolerance_valid'] is True for c in delivered)
    assert len(draws.calls) == 15 and len(resumed.fixture_evaluated) == 1
    assert all(builder.body_of(call['source']) in [builder.DEFAULT_BODY, *flow.BODIES] for call in draws.calls)
    if pair_mode:
        group = first.research.knowledge.experiments[0]['proposal_group']
        assert group['execution_profile'] == builder.EXECUTION_PROFILE
        assert group['entries'][1]['candidate_source_sha256'] == hashlib.sha256(builder.source_for().encode()).hexdigest()


@pytest.mark.parametrize('damage', ['profile', 'scaffold', 'collector'])
def test_profile_two_identity_rejects_cross_profile_or_corrupt_capture(tmp_path, monkeypatch, profile, damage):
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=builder.source_for())
    config = apply_defaults(flow.config())['repeat_promotion']
    valid = create(config, task, tmp_path/'original', sys.executable, initial_source_policy=builder.OPERATION)
    damaged = copy.deepcopy(valid.identity)
    if damage == 'profile':
        damaged['execution_profile'] = 'paired_restart_trace/1'
    else:
        damaged['restart_trace'][damage+'_sha256'] = '0'*64
    try:
        with pytest.raises(ValueError, match='traced execution policy'):
            RepeatPromotion(tmp_path/'damaged', config, damaged, draws.draw)
        assert not draws.calls
    finally:
        valid.close()


@pytest.mark.parametrize('damage', ['profile', 'capture_policy', 'proof_family'])
def test_new_profile_checkpoint_rejection_is_staged(tmp_path, monkeypatch, profile, damage):
    flow.test_traced_checkpoint_cannot_force_relabel_sources_or_capture(tmp_path, monkeypatch, damage)


@pytest.mark.parametrize('change', ['disabled', 'no_repeat', 'initializer_witness'])
def test_new_profile_rejects_unsupported_config(tmp_path, monkeypatch, profile, change):
    flow.test_unsupported_profile_combinations_rejected_before_side_effects(change)
