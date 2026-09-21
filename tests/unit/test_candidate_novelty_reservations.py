"""Parallel source claims are atomic; no providers or candidate execution."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier

import pytest

from sciel.engine.novelty import CandidateNoveltyRegistry


SOURCE = 'def solve():\n    return 1\n'


def test_simultaneous_structural_candidates_have_one_reservation():
    registry = CandidateNoveltyRegistry()
    barrier = Barrier(8)

    def claim(index):
        code = f'# proposal {index}\n' + SOURCE
        barrier.wait(timeout=3)
        return registry.reserve(code, f'e{index}', index)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(8)))
    assert sum(result['accept'] for result in results) == 1
    rejected = [result for result in results if not result['accept']]
    assert all(result['match_kind'] == 'ast' for result in rejected)
    assert all(result['reason'] == 'previous_evaluation_pending_or_unknown' for result in rejected)
    assert len(registry.state()['records']) == len(registry.state()['sources']) == 1


def test_pending_claim_survives_state_and_detached_checkpoint_copy():
    registry = CandidateNoveltyRegistry()
    reserved = registry.reserve(SOURCE, 'e1', 2)
    assert reserved['accept'] and reserved['reservation_id'] == 'e1'
    cloned = deepcopy(registry)
    restored = CandidateNoveltyRegistry()
    restored.load_state(registry.state())
    for copy in (cloned, restored):
        assert copy.state() == registry.state()
        assert copy.reserve('# formatting\n' + SOURCE, 'e2')['accept'] is False
        copy.register(SOURCE, 'e1', evaluation_ok=False)
        assert copy.reserve(SOURCE, 'retry', remeasurement=False)['remeasurement'] is True
    assert registry.inspect(SOURCE)['reason'] == 'previous_evaluation_pending_or_unknown'


def test_failed_evaluation_retry_and_successful_explicit_remeasurement():
    registry = CandidateNoveltyRegistry()
    registry.reserve(SOURCE, 'failed')
    registry.register(SOURCE, 'failed', evaluation_ok=False)
    retried = registry.reserve(SOURCE, 'retry')
    assert retried['accept'] and retried['remeasurement']
    registry.register(SOURCE, 'retry', evaluation_ok=True)
    before = registry.state()
    assert registry.reserve(SOURCE, 'duplicate')['accept'] is False
    assert registry.state() == before
    assert registry.reserve(SOURCE, 'deliberate', remeasurement=True)['accept'] is True
    assert registry.reserve(SOURCE, 'also-deliberate', remeasurement=True)['accept'] is False


def test_explicit_binding_uses_job_evaluation_not_latest_attempt():
    registry = CandidateNoveltyRegistry()
    registry.register(SOURCE, 'e1', evaluation_ok=True)
    registry.reserve(SOURCE, 'e2', remeasurement=True)
    registry.bind_program(SOURCE, 'p1', 3, evaluation_id='e1')
    records = {row['id']: row for row in registry.state()['records']}
    assert records['e1']['aliases'] == ['p1']
    assert records['e1']['islands'] == [3]
    assert records['e2']['aliases'] == []
    registry.bind_program(SOURCE, 'p2')
    assert registry.state()['records'][-1]['aliases'] == ['p2']


def test_invalid_binding_or_reused_reservation_id_does_not_mutate_registry():
    registry = CandidateNoveltyRegistry()
    registry.register(SOURCE, 'e1', evaluation_ok=True)
    registry.bind_program(SOURCE, 'p1')
    registry.register('other = 2\n', 'e2', evaluation_ok=True)
    before = registry.state()
    for evaluation_id in ('unknown', 'p1', 'e2'):
        with pytest.raises(ValueError):
            registry.bind_program(SOURCE, 'p3', evaluation_id=evaluation_id)
    with pytest.raises(ValueError, match='already registered'):
        registry.reserve(SOURCE, 'e1', remeasurement=True)
    assert registry.state() == before
