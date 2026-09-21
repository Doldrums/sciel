"""Optional auxiliary lifetime cap; all responses are in-memory test stubs."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.research import ResearchController


class InertModel:
    models = ['fixture']
    max_retries = 1

    def __init__(self):
        self.calls = []

    def generate(self, prompt, model, temperature=None):
        self.calls.append((prompt, model, temperature))
        return SimpleNamespace(text='{}', model=model, cost=.02,
                               prompt_tokens=3, completion_tokens=1)


def controller(limit):
    # Keep this test focused on the controller; config validation is tested
    # separately. History capacities remain finite in the unlimited-call case.
    cfg = apply_defaults({'research': dict(enabled=True, reflection_interval=0,
        max_lessons=2, max_experiments=4, max_fragments=2, retrieve_k=2)})['research']
    cfg['max_aux_calls'] = limit
    model, events, charges = InertModel(), [], []
    log = SimpleNamespace(research=lambda **row: events.append(copy.deepcopy(row)))
    return ResearchController(cfg, model, log, charges.append), model, events, charges


@pytest.mark.parametrize('limit', [0, 2, 32])
def test_zero_and_finite_caps_preserve_dispatch_and_charge_limits(limit):
    research, model, events, charges = controller(limit)
    results = [research._aux('conclusion', 'inert prompt', f'r{i}') for i in range(limit + 2)]
    assert results == ['{}'] * limit + [None, None]
    assert len(model.calls) == research.aux_calls == limit
    assert charges == [.02] * limit
    assert research.aux_cost == pytest.approx(.02 * limit)
    assert [row['call_index'] for row in events if row['phase'] == 'aux_start'] == list(range(1, limit + 1))
    assert all(row['reason'] == 'aux_call_limit' for row in events[-2:])
    restored, _, _, _ = controller(limit)
    restored.load_state(research.state())
    assert restored.aux_calls == limit
    tampered = research.state()
    tampered['aux_calls'] = limit + 1
    with pytest.raises(ValueError, match='inconsistent research checkpoint accounting'):
        restored.load_state(tampered)


def test_none_dispatches_beyond_old_default_and_roundtrips_accounting():
    research, model, events, charges = controller(None)
    for i in range(40):
        assert research._aux('conclusion', 'inert prompt', f'r{i}') == '{}'
    assert len(model.calls) == research.aux_calls == 40
    assert len(charges) == 40 and research.aux_cost == pytest.approx(.8)
    assert not any(row['phase'] == 'aux_skipped' for row in events)
    state = research.state()
    assert state['config']['max_aux_calls'] is None
    assert state['config']['max_lessons'] == 2
    assert state['config']['max_experiments'] == 4
    assert state['config']['max_fragments'] == 2
    restored, resumed_model, resumed_events, resumed_charges = controller(None)
    restored.load_state(state)
    assert restored.state() == state
    assert restored._aux('reflection', 'next inert prompt', 0) == '{}'
    assert restored.aux_calls == 41
    assert len(resumed_model.calls) == 1 and resumed_charges == [.02]
    assert resumed_events[0]['call_index'] == 41


@pytest.mark.parametrize('damage', [
    {'aux_calls': -1}, {'aux_calls': True}, {'aux_cost': -.01},
    {'aux_cost': float('nan')}, {'outcomes': 1}, {'reflected_through': 1},
])
def test_none_does_not_relax_checkpoint_accounting_validation(damage):
    research, _, _, _ = controller(None)
    assert research._aux('conclusion', 'inert prompt', 'r1') == '{}'
    valid = research.state()
    tampered = copy.deepcopy(valid)
    tampered.update(damage)
    with pytest.raises(ValueError, match='research checkpoint (counter|accounting)'):
        research.load_state(tampered)
    assert research.state() == valid
