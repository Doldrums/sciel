"""Real local locks with fake work only; no provider, evaluator or engine."""
from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import queue
import threading
import time

import pytest

from sciel.engine.capacity import (
    HostCapacity, CapacityCancelled, CapacityConfigurationError, CapacityTimeout,
)


def _holder(root, ready, release):
    manager = HostCapacity(root, llm_slots=2, evaluator_slots=2, poll_interval=.01)
    with manager.acquire("evaluator", timeout=5) as lease:
        ready.put(lease.receipt())
        deadline = time.monotonic() + 5
        # Shared Event/Condition locks are not crash-robust. This test kills a
        # holder deliberately, so use a lock-free stop flag for fake work.
        while not release.value and time.monotonic() < deadline:
            time.sleep(.01)


def test_context_exception_releases_and_metadata_is_detached(tmp_path):
    manager = HostCapacity(tmp_path, evaluator_slots=1)
    with pytest.raises(RuntimeError, match="fake work"):
        with manager.acquire("evaluator") as lease:
            assert lease.wait_seconds >= 0
            receipt = lease.receipt()
            assert receipt["pools"] == [dict(pool="evaluator", limit=1)]
            json.dumps(receipt, allow_nan=False)
            receipt["pools"][0]["limit"] = 99
            assert lease.receipt()["pools"][0]["limit"] == 1
            raise RuntimeError("fake work")
    assert lease.released
    lease.release()
    with manager.acquire("evaluator", timeout=0):
        pass


def test_threads_share_slots_across_independent_manager_instances(tmp_path):
    lock = threading.Lock()
    active = maximum = 0
    def work(_):
        nonlocal active, maximum
        manager = HostCapacity(tmp_path, evaluator_slots=2, poll_interval=.005)
        with manager.acquire("evaluator", timeout=2):
            with lock:
                active += 1
                maximum = max(active, maximum)
            time.sleep(.025)
            with lock:
                active -= 1
    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(work, range(12)))
    assert maximum == 2 and active == 0


def test_independent_processes_share_two_slots_and_dead_owner_releases(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, release = context.Queue(), context.Value("b", 0, lock=False)
    children = [context.Process(target=_holder, args=(str(tmp_path), ready, release)) for _ in range(3)]
    try:
        for child in children:
            child.start()
        receipts = [ready.get(timeout=5), ready.get(timeout=5)]
        with pytest.raises(queue.Empty):
            ready.get(timeout=.15)
        owner = next(child for child in children if child.pid == receipts[0]["pid"])
        owner.kill()
        owner.join(3)
        third = ready.get(timeout=3)
        assert third["pid"] not in {receipt["pid"] for receipt in receipts}
        assert third["wait_seconds"] > .05
    finally:
        release.value = 1
        for child in children:
            child.join(3)
            if child.is_alive():
                child.kill()
                child.join(3)
        ready.close()
    with HostCapacity(tmp_path, evaluator_slots=2).acquire("evaluator", timeout=0):
        pass


def test_provider_wait_releases_partial_global_slot(tmp_path):
    manager = HostCapacity(tmp_path, llm_slots=2, provider_limits={"a": 1, "b": 1}, poll_interval=.005)
    stop = threading.Event()
    with manager.acquire("llm", provider="a"):
        with ThreadPoolExecutor(max_workers=1) as executor:
            waiting = executor.submit(manager.acquire, "llm", provider="a", cancel=stop)
            time.sleep(.025)
            # The queued a-call must not strand the other global permit.
            with manager.acquire("llm", provider="b", timeout=1):
                pass
            stop.set()
            with pytest.raises(CapacityCancelled) as error:
                waiting.result(timeout=1)
            assert error.value.wait_seconds >= .025
    with manager.acquire("llm", provider="a", timeout=0):
        pass


def test_cancelled_and_timed_out_waits_have_no_orphan_permits(tmp_path):
    manager = HostCapacity(tmp_path, evaluator_slots=1, poll_interval=.005)
    event = threading.Event()
    event.set()
    with pytest.raises(CapacityCancelled):
        manager.acquire("evaluator", cancel=event)
    with manager.acquire("evaluator"):
        with pytest.raises(CapacityTimeout) as error:
            manager.acquire("evaluator", timeout=.025)
        assert error.value.wait_seconds >= .025
    with manager.acquire("evaluator", timeout=0):
        pass


def test_conflicting_limits_or_omitted_provider_policy_fail_closed(tmp_path):
    first = HostCapacity(tmp_path, llm_slots=2, evaluator_slots=1, provider_limits={"a": 1})
    second = HostCapacity(tmp_path, llm_slots=2, evaluator_slots=2)
    with first.acquire("evaluator"):
        with pytest.raises(CapacityConfigurationError):
            second.acquire("evaluator", timeout=0)
    with first.acquire("llm", provider="a"):
        with pytest.raises(CapacityConfigurationError):
            second.acquire("llm", provider="a", timeout=0)
    # Failed provider acquisition must release the already obtained host slot.
    with first.acquire("llm", provider="a", timeout=0):
        with first.acquire("llm", provider="b", timeout=0):
            pass
    with pytest.raises(ValueError, match="provider"):
        first.acquire("llm")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork lifecycle")
def test_fork_child_cannot_release_or_keep_parents_permit(tmp_path):
    manager = HostCapacity(tmp_path, evaluator_slots=1, poll_interval=.005)
    lease = manager.acquire("evaluator")
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.close(read_fd)
            lease.release()  # inherited object must not unlock its parent
            try:
                manager.acquire("evaluator", timeout=.03)
            except CapacityTimeout:
                os.write(write_fd, b"blocked")
            else:
                os.write(write_fd, b"incorrect")
            time.sleep(.2)  # remain alive while the parent releases
        finally:
            os._exit(0)
    try:
        os.close(write_fd)
        assert os.read(read_fd, 9) == b"blocked"
        lease.release()
        # A forked child's inherited descriptors cannot prolong the parent's
        # permit. The child is deliberately still alive at this point.
        with manager.acquire("evaluator", timeout=.1):
            pass
    finally:
        lease.release()
        os.close(read_fd)
        os.waitpid(child, 0)


def test_symlink_policy_and_corrupt_policy_do_not_grant_capacity(tmp_path):
    manager = HostCapacity(tmp_path, evaluator_slots=1)
    with manager.acquire("evaluator"):
        pass
    policy = next(tmp_path.glob("*.json"))
    policy.write_text("incomplete")
    with pytest.raises(CapacityConfigurationError):
        manager.acquire("evaluator", timeout=0)
    policy.unlink()
    target = tmp_path / "unrelated"
    target.write_text("do not touch")
    policy.symlink_to(target)
    with pytest.raises(CapacityConfigurationError):
        manager.acquire("evaluator", timeout=0)
    assert target.read_text() == "do not touch"


@pytest.mark.parametrize("slots", [0, -1, True, 1.5, 4097])
def test_invalid_limits_rejected_before_creating_root(tmp_path, slots):
    root = tmp_path / "not-created"
    with pytest.raises(ValueError):
        HostCapacity(root, evaluator_slots=slots)
    assert not root.exists()
