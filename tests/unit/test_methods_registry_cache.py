"""Cached projections must retain content validation and detached public reads."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import pytest

import sciel.methods.registry as module
from sciel.methods.registry import Registry, RegistryCorruption
from test_methods_registry import initial, register


def test_warm_projection_checks_paths_without_rehashing_historical_bodies(tmp_path, monkeypatch):
    registry = Registry(tmp_path)
    artifact = registry.put_artifact('large_prompt', {'text': 'x' * 100_000})
    registry.events()
    original = module._read
    reads = []
    def read(path):
        reads.append(Path(path))
        return original(path)
    monkeypatch.setattr(module, '_read', read)
    assert len(registry.events()) == 1
    assert registry.get_artifact('large_prompt', artifact)['text'].startswith('x')
    assert reads == []
    registry.append('progress', {'index': 1})
    registry.events()
    assert all(path.name == 'journal.jsonl' for path in reads)


def test_same_size_restored_mtime_tamper_invalidates_warm_object(tmp_path):
    registry = Registry(tmp_path)
    artifact = registry.put_artifact('result', {'text': 'original'})
    registry.events()
    path = tmp_path / 'artifacts' / (artifact + '.json')
    before = path.stat()
    path.write_bytes(path.read_bytes().replace(b'original', b'modified'))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    with pytest.raises(RegistryCorruption, match='content hash mismatch'):
        registry.events()


def test_same_size_restored_mtime_journal_tamper_invalidates_warm_state(tmp_path):
    registry = Registry(tmp_path)
    registry.append('notice', {'text': 'original'})
    registry.events()
    path = tmp_path / 'journal.jsonl'
    before = path.stat()
    path.write_bytes(path.read_bytes().replace(b'original', b'modified'))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RegistryCorruption, match='hash chain mismatch'):
        registry.events()


@pytest.mark.parametrize('change', ['remove', 'symlink'])
def test_missing_or_linked_historical_artifact_is_not_hidden_by_cache(tmp_path, change):
    registry = Registry(tmp_path / 'registry')
    artifact = registry.put_artifact('result', {'value': 1})
    registry.events()
    path = registry.root / 'artifacts' / (artifact + '.json')
    saved = tmp_path / 'saved.json'
    saved.write_bytes(path.read_bytes())
    path.unlink()
    if change == 'symlink':
        path.symlink_to(saved)
    with pytest.raises(RegistryCorruption):
        registry.events()


def test_another_registry_writer_invalidates_cached_journal_without_rollback(tmp_path):
    registry = Registry(tmp_path)
    registry.append('first', {})
    registry.events()
    other = Registry(tmp_path)
    other.append('second', {})
    assert [row['kind'] for row in registry.events()] == ['first', 'second']
    registry.append('third', {})
    assert [row['kind'] for row in other.events()] == ['first', 'second', 'third']


def test_cached_public_views_are_detached_and_failed_append_cannot_poison_state(tmp_path, monkeypatch):
    registry = Registry(tmp_path)
    method = register(registry)
    lab = registry.create_lab('lab', 'question', 'budget', 'B', initial(active_methods=[method]))
    evidence = registry.put_artifact('result', {'nested': {'value': 1}})
    registry.events()
    registry.get_method(method)['manifest']['name'] = 'mutated'
    registry.get_artifact('result', evidence)['nested']['value'] = 99
    registry.revision(lab['active_revision'])['active_methods'].clear()
    registry.events()[0]['data']['id'] = 'broken'
    assert registry.get_method(method)['manifest']['name'] != 'mutated'
    assert registry.get_artifact('result', evidence)['nested']['value'] == 1
    assert registry.revision(lab['active_revision'])['active_methods'] == [method]
    monkeypatch.setattr(registry, '_append', lambda *a, **kw: (_ for _ in ()).throw(OSError('fixture disk failure')))
    with pytest.raises(OSError):
        registry.status_method(method, 'released', [evidence])
    assert registry.method_status(method)['status'] == 'draft'


def test_body_lru_budget_does_not_force_old_large_artifact_reloads_on_append(tmp_path, monkeypatch):
    monkeypatch.setattr(module, '_OBJECT_CACHE_BYTES', 1024)
    registry = Registry(tmp_path)
    for index in range(4):
        registry.put_artifact('large', {'index': index, 'text': 'x' * 4096})
    registry.events()
    assert registry._object_cache_bytes <= 1024
    original = module._read
    reads = []
    def read(path):
        reads.append(Path(path))
        return original(path)
    monkeypatch.setattr(module, '_read', read)
    registry.append('progress', {})
    registry.events()
    assert not any(path.parent.name == 'artifacts' for path in reads)
    # Cold resume still validates every historical body from its bytes.
    resumed = Registry(tmp_path)
    assert sum(path.parent.name == 'artifacts' for path in reads) == 4
    assert len(resumed.events()) == 5


def test_context_after_fifty_synthetic_steps_stays_within_host_cpu_bound(tmp_path):
    """No generated code/provider calls: only realistic immutable history shape."""
    from test_methods_coordinator import setup, lab, method
    coordinator, registry, _, executor, proposer = setup(tmp_path)
    lab(coordinator, 'author')
    refs = []
    for index in range(50):
        value = method(name='version-' + str(index),
                       code='def run(values):\n    return sorted(values)\n# ' + 'x' * 24000 + '\n')
        method_id = registry.register(value['files'], value['manifest'])
        registry.put_artifact('proposal_context', {'index': index, 'saved_prompt': 'x' * 160000})
        receipt = registry.put_artifact('proposal_receipt', {'index': index, 'error': None,
                                           'diagnostic': {'code': 'synthetic_fixture'}})
        refs.append(receipt)
        revision = registry.revise_lab('author', registry.lab('author')['active_revision'],
                                       {'evidence_refs': deepcopy(refs)}, 'synthetic step ' + str(index))
        registry.append('episode_step', {'lab_id': 'author', 'index': index, 'proposal_id': receipt,
                                         'method_id': method_id, 'revision_id': revision, 'status': 'continued'})
    revision_id, revision = coordinator._revision('author')
    started = time.process_time()
    context = coordinator._context('author', revision_id, revision)
    elapsed = time.process_time() - started
    assert elapsed < 10, 'Context retrieval exceeds its actual finite host-operation reservation'
    assert len(context['available_methods']) <= 16 and len(context['evidence']) == 8
    assert not executor.calls and proposer.calls == 0
    assert registry._object_cache_bytes <= module._OBJECT_CACHE_BYTES
