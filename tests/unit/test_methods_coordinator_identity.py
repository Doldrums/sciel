"""Direct Coordinator initialization with real locks and inert external leaves."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest

from sciel.methods.budget import ResourceLedger, RESOURCES
from sciel.methods.coordinator import Coordinator
from sciel.methods.registry import Registry


def inputs(tmp_path):
    return Registry(tmp_path/'registry'), ResourceLedger(tmp_path/'resources.jsonl', dict.fromkeys(RESOURCES, None))


def construct(registry, ledger, max_steps):
    return Coordinator(registry, ledger, SimpleNamespace(identity={'kind':'inert executor'}),
        SimpleNamespace(identity={'kind':'inert proposer'}),
        SimpleNamespace(identity={'kind':'inert task'}), resources={}, max_steps=max_steps)


@pytest.mark.parametrize('same_config', [False, True])
def test_concurrent_initialization_has_one_frozen_identity(tmp_path, monkeypatch, same_config):
    registry_a, ledger_a = inputs(tmp_path)
    registry_b, ledger_b = inputs(tmp_path)
    first_read, release_first, second_read, second_started = [threading.Event() for _ in range(4)]
    original = Coordinator._events

    def events(self, kind):
        result = original(self, kind)
        if kind == 'coordinator_identity':
            if threading.current_thread().name.endswith('_0'):
                first_read.set()
                assert release_first.wait(2)
            else:
                second_read.set()
        return result

    monkeypatch.setattr(Coordinator, '_events', events)
    def second():
        second_started.set()
        return construct(registry_b, ledger_b, 1 if same_config else 2)

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='identity') as pool:
        first = pool.submit(construct, registry_a, ledger_a, 1)
        assert first_read.wait(2)
        other = pool.submit(second)
        assert second_started.wait(2)
        # Old code enters the second identity read before the first append;
        # fixed code serializes this entire read/check/append transaction.
        entered_before_release = second_read.wait(.1)
        release_first.set()
        first.result(timeout=2)
        if same_config:
            other.result(timeout=2)
        else:
            with pytest.raises(ValueError, match='identity differs'):
                other.result(timeout=2)
    assert entered_before_release is False
    identities = [e['data']['identity'] for e in registry_a.events() if e['kind']=='coordinator_identity']
    assert len(identities) == 1 and identities[0]['max_steps'] == 1
    assert not list((registry_a.root/'dispatch').glob('*/intent.json'))


def test_existing_conflicting_identity_history_is_refused_without_rewrite(tmp_path):
    registry, ledger = inputs(tmp_path)
    first = construct(registry, ledger, 1)
    changed = deepcopy(first.identity)
    changed['max_steps'] = 2
    registry.append('coordinator_identity', {'identity':changed, 'started_at':0})
    before = registry.events()
    with pytest.raises(ValueError, match='identity differs'):
        construct(registry, ledger, 1)
    assert registry.events() == before
