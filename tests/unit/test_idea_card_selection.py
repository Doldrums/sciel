"""Card-distance selection mechanics; fake generation/evaluation, no source execution."""
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import re
import sys

import pytest

from sciel.components.policy.base import floor_clip
from sciel.components.selection.idea_cards import EPSILON, prepare_card_tilt
from sciel.components.selection.novelty_chain import IslandView, NoveltyChainKernel
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.idea import card_text
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task


COMMON = {'family': 'grid', 'strategy': 'regular square lattice', 'mechanism': 'fixed spacing'}
RARE = {'family': 'annealing', 'strategy': 'stochastic energy walk', 'mechanism': 'temperature decay'}


def configured(**updates):
    cfg = {'task': 'test/toy_quadratic', 'budget': 3, 'max_iterations': 6,
           'num_islands': 1, 'random_seed': 31, 'dup_cap': 1, 'op_fixed': 'idea',
           'policy': {'enabled': False}, 'portfolio_seed': False, 'dynamic_islands': False,
           'idea_island': True, 'embed_grid': False, 'embed_niche': False, 'three_head': False,
           'crossover': False, 'milestone_fracs': [], 'confirm_records': False,
           'llm': {'mock': True}, 'idea_two_stage': {'enabled': True},
           'idea_card_selection': {'enabled': True},
           'research': {'enabled': True, 'reflection_interval': 0, 'max_aux_calls': 0}}
    cfg.update(updates)
    return cfg


def test_card_target_mixture_has_exact_propensity_floor_and_one_visit_draw():
    view = IslandView(['a', 'b', 'c'], {'a': .5, 'b': .5, 'c': .5}, {})
    kernel = NoveltyChainKernel(floor=.01)
    kernel.visits = {'a': 1, 'b': 0, 'c': 0}
    before = deepcopy(kernel.state())
    tilt, details = prepare_card_tilt(view, {'a': COMMON, 'b': COMMON, 'c': RARE,
                                            'ignored_foreign': RARE}, kernel.visits,
                                     beta=.5, neighbors=1)
    assert set(details['card_metadata']) == set(view.ids)
    novelty = details['card_space']['novelty']
    assert novelty['c'] > novelty['a'] == novelty['b'] == 0
    weights = {pid: (EPSILON + novelty[pid]) / (1 + kernel.visits.get(pid, 0)) for pid in view.ids}
    target = {pid: weight / sum(weights.values()) for pid, weight in weights.items()}
    base_weights = {pid: 1 / (1 + kernel.visits.get(pid, 0)) for pid in view.ids}
    base = {pid: value / sum(base_weights.values()) for pid, value in base_weights.items()}
    expected = floor_clip({pid: .5 * base[pid] + .5 * target[pid] for pid in view.ids}, .01)
    assert details['target_distribution'] == target
    assert target['a'] < target['b'] < target['c']

    class OneDraw:
        calls = 0

        def choices(self, ids, *, weights, k):
            self.calls += 1
            assert list(ids) == view.ids and k == 1
            assert weights == [expected[pid] for pid in ids]
            return ['c']

    draw, logged = OneDraw(), []

    def before_visit(pid, distribution, debug):
        assert kernel.state() == before
        logged.append((pid, deepcopy(distribution)))
        distribution.clear()  # Diagnostic callbacks receive a policy-safe copy.

    selected = kernel.select(view, tilt, draw, before_visit=before_visit)
    assert draw.calls == 1 and selected.parent_id == 'c'
    assert selected.propensity == expected['c']
    assert logged == [('c', expected)]
    assert kernel.visits == {'a': 1, 'b': 0, 'c': 1}
    assert min(expected.values()) >= .01 and math.isclose(sum(expected.values()), 1)


@pytest.mark.parametrize('cards', [{'a': '', 'b': 'legacy prose'},
                                 {'a': COMMON, 'b': None},
                                 {'a': COMMON, 'b': COMMON}])
def test_unavailable_card_signal_is_exact_old_kernel_fallback(cards):
    view = IslandView(['a', 'b'], {'a': .1, 'b': .9}, {})
    left, right = NoveltyChainKernel(), NoveltyChainKernel()
    left.visits = right.visits = {'a': 2}
    right.visits = dict(left.visits)
    lrng, rrng = random.Random(17), random.Random(17)
    tilt, detail = prepare_card_tilt(view, cards, right.visits, beta=.5, neighbors=3)
    assert tilt is None and not detail['card_space']['available']
    assert detail['target_distribution'] is None
    selected = right.select(view, tilt, rrng, before_visit=lambda *args: None)
    assert selected == left.select(view, None, lrng)
    assert left.state() == right.state() and lrng.getstate() == rrng.getstate()


def test_missing_cards_receive_neutral_novelty_not_fake_unique_identity():
    view = IslandView(['a', 'b', 'missing', 'legacy'], dict.fromkeys(['a', 'b', 'missing', 'legacy'], .4), {})
    tilt, detail = prepare_card_tilt(view, {'a': COMMON, 'b': RARE, 'legacy': 'an old plan'}, {},
                                     beta=.5, neighbors=3)
    assert tilt is not None
    novelty = detail['card_space']['novelty']
    assert novelty['missing'] == novelty['legacy'] == (novelty['a'] + novelty['b']) / 2
    assert detail['card_metadata']['missing']['kind'] == 'missing'
    assert detail['card_metadata']['legacy']['kind'] == 'legacy'


@pytest.mark.parametrize('updates', [
    {'idea_two_stage': {'enabled': False}}, {'idea_island': False}, {'op_fixed': 'diff'},
    {'selection_policy': 'stock_70_20_10'}, {'policy': {'enabled': True}},
    {'policy': {'enabled': False, 'mode_bandit': True}},
    {'policy': {'enabled': False, 'mode_source': 'mlp'}},
    {'preset_selection': True}, {'three_head': True},
    {'container': {'kind': 'cvt', 'selection': 'centres'}}])
def test_incompatible_card_selection_config_fails_before_work(updates):
    with pytest.raises(ValueError, match='idea_card_selection'):
        apply_defaults(configured(**updates))


@pytest.mark.parametrize('settings', [None, {'unknown': 1}, {'enabled': 1}, {'beta': 0},
    {'beta': True}, {'beta': float('nan')}, {'beta': float('inf')}, {'beta': 1.1},
    {'neighbors': False}, {'neighbors': 0}, {'neighbors': 1001}, {'neighbors': 1.5}])
def test_card_selection_settings_are_strict(settings):
    with pytest.raises(ValueError, match='idea_card_selection'):
        apply_defaults(configured(idea_card_selection=settings))


class FakeModel:
    models = ['fake']
    weights = [1.0]
    max_retries = 1

    def __init__(self):
        self.calls = []

    def choose_model(self, rng):
        return 'fake'

    def generate(self, prompt, model, temperature=None):
        self.calls.append(prompt)
        if 'IDEA STAGE 1:' in prompt:
            n = int(re.search(r'ITERATION: (\d+)', prompt)[1])
            text = json.dumps({'family': 'generated', 'strategy': f'fixture step {n}',
                               'mechanism': f'new fixed mechanism {n}'})
        else:
            n = int(re.search(r'fixture step (\d+)', prompt)[1])
            hypothesis = {'hypothesis': 'Fixture score increases', 'predicted_low': -1,
                          'predicted_high': 1, 'reasoning': 'Numeric fixture only', 'confidence': .5}
            text = ('HYPOTHESIS_JSON: ' + json.dumps(hypothesis) + '\n```python\n'
                    f'def run_search():\n    return {n / 10 + .5!r}, 0.0\n```\n')
        return LLMResponse(text, model, .01, 10, 20)


def build_engine(path, **updates):
    task = load_task(str(Path(__file__).resolve().parents[2] / 'tasks/test/toy_quadratic'))
    instance = Engine(configured(**updates), task, str(path), FakeModel(), python=sys.executable)
    # Already measured fixture population; the real candidate/evaluator is never loaded.
    instance.seed_score = .4
    for island in range(len(instance.db.islands)):
        for n, card in enumerate((COMMON, COMMON, RARE)):
            instance.db.add(f'def run_search(): return ({island+n}, 0)\n', .4, {}, island,
                            origin='seed', card=card_text(card))
    instance.evaluated = []

    def fake_evaluate(code):
        instance.evaluated.append(code)
        value = float(re.search(r'return ([0-9.]+), 0.0', code)[1])
        return CascadeOutcome(EvalResult(True, value, {}, timing=.01), value, 'full')

    instance._evaluate = fake_evaluate
    return instance


def read_events(path):
    return [json.loads(line) for line in (path / 'sciel_events.jsonl').read_text().splitlines()]


def test_only_idea_island_tilt_is_logged_before_visit_and_generation(tmp_path):
    engine = build_engine(tmp_path, num_islands=2)
    original = engine.log.decision
    seen = []

    def capture(**fields):
        if fields.get('policy') == 'card_diversity':
            assert engine.llm.calls == []
            assert fields['visits_before'] == {pid: engine.selection.visits.get(pid, 0)
                                                for pid in fields['ids']}
            assert set(fields['ids']) == set(engine.db.islands[1].programs)
            assert fields['propensity'] == fields['distribution'][fields['parent_id']]
            assert 'child_id' not in fields
            seen.append(deepcopy(fields))
        original(**fields)

    engine.log.decision = capture
    try:
        engine._step(1)
        assert len(seen) == 1 and seen[0]['action'] == 'tilt'
        assert len(engine.llm.calls) == 2 and len(engine.evaluated) == 1
        engine._step(0)
        assert len(seen) == 1  # Two-stage dispatch still works on a non-card-selection island.
        assert len(engine.llm.calls) == 4 and len(engine.evaluated) == 2
        assert engine.cost_total == pytest.approx(.04)
    finally:
        engine.log.close()
    rows = read_events(tmp_path)
    selection = next(e for e in rows if e.get('policy') == 'card_diversity')
    request = next(e for e in rows if e.get('phase') == 'idea' and e.get('action') == 'request_start')
    assert rows.index(selection) < rows.index(request)
    iteration = next(e for e in rows if e['event'] == 'iteration')
    assert iteration['mode'] == iteration['effective_mode'] == 'card_diversity'
    assert iteration['parent_propensity'] == selection['propensity']


def test_disabled_option_never_reads_cards_and_preserves_rng_selection(tmp_path, monkeypatch):
    # build_engine supplies defaults, so remove the key before dispatch explicitly.
    absent = build_engine(tmp_path / 'absent', idea_card_selection={'enabled': False})
    absent.cfg.pop('idea_card_selection')
    disabled = build_engine(tmp_path / 'disabled', idea_card_selection={'enabled': False})

    def forbidden(*args, **kwargs):
        pytest.fail('disabled card selection accessed card representation')

    monkeypatch.setattr('sciel.components.selection.idea_cards.card_novelty', forbidden)
    try:
        absent._step(0)
        disabled._step(0)
        for field in ('db', 'rng', 'selection', 'research', 'idea'):
            assert getattr(absent, field).state() == getattr(disabled, field).state()
        assert absent.llm.calls == disabled.llm.calls
        assert absent.evaluated == disabled.evaluated and absent.cost_total == disabled.cost_total
    finally:
        absent.log.close()
        disabled.log.close()
    assert not [e for e in read_events(tmp_path / 'disabled') if e.get('policy') == 'card_diversity']


def test_checkpoint_resume_reconstructs_card_probabilities_and_all_search_state(tmp_path):
    full = build_engine(tmp_path / 'full')
    expected = full.run()
    split = build_engine(tmp_path / 'split')
    split._stop_at = 1
    assert split.run()['reason'] == 'paused'
    resumed = build_engine(tmp_path / 'split')
    checkpoint.load(resumed, str(tmp_path / 'split/checkpoint.json'))
    assert resumed.run() == expected
    for name in ('db', 'rng', 'selection', 'research', 'idea', 'schedule'):
        assert getattr(resumed, name).state() == getattr(full, name).state()
    keys = ('ids', 'visits_before', 'card_metadata', 'input_sha256', 'card_space',
            'target_distribution', 'distribution', 'parent_id', 'propensity')
    extract = lambda path: [{k: e[k] for k in keys} for e in read_events(path)
                            if e.get('policy') == 'card_diversity']
    assert extract(tmp_path / 'full') == extract(tmp_path / 'split')
    assert len(full.llm.calls) == 6 and full.cost_total == pytest.approx(.06)


@pytest.mark.parametrize('updates', [
    {'idea_card_selection': {'enabled': False}},
    {'idea_card_selection': {'enabled': True, 'beta': .7}},
    {'idea_card_selection': {'enabled': True, 'neighbors': 1}},
    {'selection_floor': .02}, {'tau_floor': .03}, {'num_islands': 2}])
def test_enabled_card_selection_resume_identity_cannot_silently_change(tmp_path, updates):
    original = build_engine(tmp_path / 'run')
    state = {'identity': checkpoint._identity(original)}
    changed = build_engine(tmp_path / 'changed', **updates)
    try:
        assert any('idea_card_selection' in issue for issue in checkpoint.check_identity(changed, state))
        assert changed.llm.calls == [] and changed.evaluated == []
    finally:
        original.log.close()
        changed.log.close()
