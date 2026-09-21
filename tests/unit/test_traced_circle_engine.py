"""Complete profile lifecycle with fake model/evaluator/draws; no numerical search."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.loop import Engine
from sciel.engine.llm.client import LLMResponse
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_traced_body as traced
from sciel.hypothesis import circle_restart_body as legacy
from sciel.hypothesis import circle_proposal_pair as pair
from sciel.hypothesis.restart_trace import RestartTrace
from sciel.knowledge_base.catalogs import canonical_sha, canonical_json
from sciel.knowledge_base.restart_context import PREFIX, expand_context
from test_circle_intervention_engine import task_copy
from test_circle_generation_engine import response
from test_repeat_engine_integration import FakeDraws, config as repeat_config, journal
from test_repeat_reflection_lifecycle import ReflectingModel

BODIES = ['# fixture_step = 1\nreturn best_params + uniform_noise * 0.25\n',
          '# fixture_step = 2\nreturn best_params + uniform_noise * 0.5\n']


class Draws(FakeDraws):
    identity = {'schema': 'fake-repeat-adapter/2', 'runtime': {'kind': 'inert-fixture'}}
    identity['identity_sha256'] = canonical_sha(identity)

    def draw(self, source, *, restart_trace=False, **kwargs):
        assert restart_trace is True
        traced.body_of(source)  # Parse only; never invoke the source.
        receipt = super().draw(source, **kwargs)
        self.calls[-1]['restart_trace'] = restart_trace
        q = receipt['raw_sum']
        binding = dict(source_sha256=receipt['source_sha256'], seed=receipt['seed'],
            runtime_sha256=canonical_sha(self.identity['runtime']),
            scaffold_sha256=traced.identity()['common_scaffold_sha256'],
            invocation_id=hashlib.sha256((source+str(receipt['seed'])).encode()).hexdigest()[:32])
        trace = RestartTrace(binding)
        value = np.zeros(78); value[52:] = q / 26
        inputs = {k: value.copy() for k in ('best', 'initial', 'noise', 'ordinary')}
        trace.begin_start(0, incumbent_sum=q, improvements=0)
        trace.begin_helper(inputs)
        trace.helper_end(raw_kind='none', post_inputs=inputs, effective=value,
            origin='saved_ordinary_fallback', incumbent_sum=q, improvements=0)
        trace.optimizer_enter(incumbent_sum=q, improvements=0)
        trace.optimizer_exit(incumbent_sum=q, improvements=0)
        trace.optimizer_result(value)
        trace.end_start('completed', incumbent_sum=q, improvements=0)
        receipt['restart_trace'] = dict(status='captured', reason=None,
            sidecar_sha256='c'*64,
            trace=trace.finish('restart_limit', incumbent_sum=q, improvements=0))
        return receipt


class Model(ReflectingModel):
    def __init__(self, draws, pair_mode, start=0):
        super().__init__(draws, start)
        self.pair_mode = pair_mode

    def generate(self, prompt, model, temperature=None, *, response_format=None):
        if response_format is None:
            return super().generate(prompt, model, temperature)
        assert response_format == (pair.response_format() if self.pair_mode else legacy.response_format())
        index = self.mutation_count
        assert index < 2, 'unexpected generation replay'
        self.mutation_count += 1
        selected = json.loads(response(BODIES[index]))
        text = (json.dumps({'candidates': [selected, json.loads(response('return None\n'))]})
                if self.pair_mode else json.dumps(selected))
        self.calls.append(dict(kind='mutation', prompt=prompt, text=text))
        return LLMResponse(text, model, .001, 10, 10, finish_reason='stop')


def config(route='repeat_promotion', pair_mode=False):
    cfg = repeat_config(budget=2, max_iterations=2, op_fixed='circle_generate',
        circle_generation={'enabled': True, 'execution_profile': traced.EXECUTION_PROFILE})
    if pair_mode:
        cfg['circle_generation']['proposal_policy'] = 'first_eligible_of_two'
    if route == 'repeat_population':
        cfg[route] = cfg.pop('repeat_promotion')
        cfg.update(container={'kind': 'pool', 'selection': 'shared'}, archive_size=0, pop_cap=2)
    cfg['research'].update(batch_only_lessons=True, reflection_interval=1, max_aux_calls=4)
    return cfg


def construct(task, folder, draws, route, pair_mode, monkeypatch, *, start=0):
    model = Model(draws, pair_mode, start)
    engine = Engine(config(route, pair_mode), task, str(folder), model, python=sys.executable)
    engine.fixture_evaluated = []
    def evaluate(_task, path, **kwargs):
        source = Path(path).read_text()
        body = traced.body_of(source)
        index = 0 if body == traced.DEFAULT_BODY else BODIES.index(body) + 1
        raw = [.5, 1., 1.1][index]
        score = raw / 2.635
        engine.fixture_evaluated.append(source)
        cases = dict(sum_radii=raw, target_ratio=score, validity=1., combined_score=score)
        return CascadeOutcome(EvalResult(True, score, cases, timing=.01), score, 'full', False)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    return engine, model


def parse(prompt):
    return expand_context(json.JSONDecoder().raw_decode(prompt.split(PREFIX, 1)[1].lstrip())[0])


@pytest.mark.parametrize('route', ['repeat_promotion', 'repeat_population'])
@pytest.mark.parametrize('pair_mode', [False, True])
def test_same_source_profile_repeat_reflection_next_generation_and_resume(tmp_path, monkeypatch, route, pair_mode):
    draws = Draws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=traced.source_for())
    folder = tmp_path/'run'
    first, model = construct(task, folder, draws, route, pair_mode, monkeypatch)
    first._stop_at = 1
    first.run()
    assert first.fixture_evaluated == [traced.source_for(), traced.source_for(BODIES[0])]
    assert len(draws.calls) == 9
    assert all(c['restart_trace'] for c in draws.calls)
    assert set(c['source'] for c in draws.calls) == set(first.fixture_evaluated)
    assert [c['kind'] for c in model.calls] == ['mutation', 'conclusion', 'reflection']
    assert PREFIX not in model.calls[0]['prompt'] and PREFIX not in model.calls[1]['prompt']
    assert parse(model.calls[2]['prompt'])['presented_count'] > 0
    group, = first.research.knowledge.comparisons
    group_before = canonical_json(group)
    observation, = first.research.knowledge.experiments
    assert observation['controlled_change'] == traced.apply(traced.source_for(), BODIES[0])['proof']
    if pair_mode:
        proposals = observation['proposal_group']
        assert proposals['execution_profile'] == traced.EXECUTION_PROFILE
        assert proposals['entries'][1]['candidate_source_sha256'] == hashlib.sha256(traced.source_for().encode()).hexdigest()
        assert proposals['selected_index'] == 0
    saved = json.loads((folder/'checkpoint.json').read_text())
    assert saved['identity']['circle_generation']['identity']['execution_profile'] == traced.EXECUTION_PROFILE
    second, following = construct(task, folder, draws, route, pair_mode, monkeypatch, start=1)
    checkpoint.load(second, str(folder/'checkpoint.json'))
    second.run()
    assert second.fixture_evaluated == [traced.source_for(BODIES[1])]
    assert len(draws.calls) == 15 and second.research.aux_calls == 4
    generated = next(c for c in following.calls if c['kind'] == 'mutation')
    context = parse(generated['prompt'])
    assert context['presented_count'] > 0
    assert context['selected_count'] == context['presented_count'] + context['omitted_count']
    assert canonical_json(second.research.knowledge.comparisons[0]) == group_before
    events = journal(folder)
    attempt = next(e for e in events if e.get('phase') == 'attempt' and e.get('experiment_id') == 'r000002a1')
    # Delivery must work with the real, larger source-bound proof as well as the
    # old tiny generic fixture. Do not quietly weaken this to whole omission.
    assert attempt['lesson_presentation']['presented']
    lesson = next(l for l in second.research.knowledge.lessons if l['source_support_ids'] == [group['id']])
    assert lesson['exposure_ids'] == ['r000002a1']
    assert second.circle_generation.evaluations == 2
    assert all(traced.body_of(c['source']) in [traced.DEFAULT_BODY, *BODIES] for c in draws.calls)


@pytest.mark.parametrize('value', [None, False, 1, 'unknown'])
def test_explicit_invalid_profile_rejected(value):
    cfg = config(); cfg['circle_generation']['execution_profile'] = value
    with pytest.raises(ValueError, match='execution_profile'):
        apply_defaults(cfg)


@pytest.mark.parametrize('change', ['disabled', 'no_repeat', 'initializer_witness'])
def test_unsupported_profile_combinations_rejected_before_side_effects(change):
    cfg = config()
    if change == 'disabled':
        cfg['circle_generation']['enabled'] = False
    elif change == 'no_repeat':
        cfg.pop('repeat_promotion')
    else:
        cfg['research']['initializer_witness'] = True
    with pytest.raises(ValueError):
        apply_defaults(cfg)


@pytest.mark.parametrize('damage', ['profile', 'scaffold', 'collector'])
def test_repeat_identity_policy_rejects_corruption_without_dispatch(tmp_path, monkeypatch, damage):
    from sciel.engine.repeat_support import create
    from sciel.engine.repeat_promotion import RepeatPromotion
    draws = Draws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=traced.source_for())
    repeat_cfg = apply_defaults(config())['repeat_promotion']
    valid = create(repeat_cfg, task, tmp_path/'original', sys.executable, initial_source_policy=traced.OPERATION)
    damaged = copy.deepcopy(valid.identity)
    if damage == 'profile':
        damaged['execution_profile'] = 'unknown'
    else:
        damaged['restart_trace'][damage+'_sha256'] = '0'*64
    with pytest.raises(ValueError, match='traced execution policy'):
        RepeatPromotion(tmp_path/'damaged', repeat_cfg, damaged, draws.draw)
    assert draws.calls == []


@pytest.mark.parametrize('damage', ['profile', 'capture_policy', 'proof_family'])
def test_traced_checkpoint_cannot_force_relabel_sources_or_capture(tmp_path, monkeypatch, damage):
    draws = Draws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=traced.source_for())
    folder = tmp_path/'run'
    first, _ = construct(task, folder, draws, 'repeat_promotion', False, monkeypatch)
    first._stop_at = 1; first.run()
    state = json.loads((folder/'checkpoint.json').read_text())
    if damage == 'profile':
        del state['identity']['circle_generation']['identity']['execution_profile']
    elif damage == 'capture_policy':
        state['identity']['repeat_promotion']['identity']['restart_trace']['enabled'] = False
    else:
        state['circle_generation']['seen_proofs'][0] = legacy.apply(legacy.source_for(), BODIES[0])['proof']
    tampered = folder/'tampered.json'; tampered.write_text(canonical_json(state))
    resumed, model = construct(task, folder, draws, 'repeat_promotion', False, monkeypatch, start=1)
    before = canonical_json(resumed.research.state())
    try:
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(resumed, str(tampered), allow_identity_mismatch=True)
        assert not model.calls and len(draws.calls) == 9
        assert canonical_json(resumed.research.state()) == before
    finally:
        resumed.repeat_promotion.close(); resumed.log.close()
