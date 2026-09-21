"""Offline crash injection for the scientific wave commit boundary."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.engine import parallel_store as store
from sciel.engine.run_lock import RunDirectoryBusy, RunDirectoryLock
from sciel.run_log import RunLog


def _read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def engine(tmp_path, monkeypatch):
    log = RunLog(str(tmp_path / 'sciel_events.jsonl'))
    log.run_start(config={'parallel_search': {'enabled': True}})
    value = SimpleNamespace(run_dir=str(tmp_path), log=log, counter=0)

    def save(engine, milestone, *, path=None):
        target = Path(path or Path(engine.run_dir) / 'checkpoint.json')
        state = {'counter': engine.counter, 'milestone': milestone,
                 'provenance': {'run_id': engine.log.run_id, 'journal_anchor': engine.log.anchor()}}
        store._write_json(target, state)
        engine._saved_checkpoint = {'checkpoint_sha256': store._digest(target),
                                    'provenance': deepcopy(state['provenance'])}
        return str(target)

    monkeypatch.setattr(store.checkpoint, 'save', save)
    save(value, 'initial')
    yield value
    value.log.close()


def test_stage_is_unpublished_and_keeps_same_log_identity(engine):
    main = Path(engine.log.path)
    original = main.read_bytes()
    research_reference = engine.log
    original_segment = engine.log.segment_id
    transaction = store.begin(engine, 0)
    assert engine.log is research_reference
    assert engine.log.segment_id == original_segment
    engine.counter = 3
    engine.log.iteration(child_id='p2')
    assert main.read_bytes() == original
    assert _read(Path(engine.run_dir) / 'checkpoint.json')['counter'] == 0
    assert engine.log.event_seq == 2
    transaction.commit()
    assert engine.log is research_reference
    assert engine.log.path == str(main)
    assert _read(Path(engine.run_dir) / 'checkpoint.json')['counter'] == 3
    assert [json.loads(line)['event_seq'] for line in main.read_text().splitlines()] == [1, 2]
    assert store.recover(engine.run_dir)['phase'] == 'committed'
    assert transaction.commit() == str(Path(engine.run_dir) / 'checkpoint.json')
    engine.log.iteration(child_id='p3')
    assert engine.log.anchor()['event_seq'] == 3


def test_abort_replays_same_directory_preserving_external_receipts(engine):
    transaction = store.begin(engine, '0')
    receipt = transaction.directory / 'call-receipt.json'
    receipt.write_text('{"immutable":true}')
    original_path = transaction._original['path']
    engine.counter = 99
    engine.log.iteration(child_id='unpublished')
    transaction.restore_log()
    transaction.restore_log()
    assert engine.log.path == original_path
    assert engine.log.event_seq == 1
    assert store.recover(engine.run_dir)['phase'] == 'active'
    # Actual replay loads checkpoint, then starts a new segment. The same wave
    # keeps its immutable call receipts but receives a fresh staged journal.
    engine.counter = _read(Path(engine.run_dir) / 'checkpoint.json')['counter']
    engine.log.run_start(restored_checkpoint=engine._saved_checkpoint)
    replay = store.begin(engine, 0)
    assert replay.directory == transaction.directory
    assert receipt.read_text() == '{"immutable":true}'
    assert b'unpublished' not in Path(engine.log.path).read_bytes()
    engine.counter = 1
    engine.log.iteration(child_id='replayed')
    replay.commit()
    assert _read(Path(engine.run_dir) / 'checkpoint.json')['counter'] == 1


@pytest.mark.parametrize('partial', [0, 7, 'all'])
def test_recover_prepared_partial_append_idempotently(engine, monkeypatch, partial):
    transaction = store.begin(engine, 0)
    engine.counter = 2
    engine.log.iteration(child_id='p2', detail='a long staged scientific record')
    real_finish = store._finish_prepared
    monkeypatch.setattr(store, '_finish_prepared', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('crash before append')))
    with pytest.raises(OSError):
        transaction.commit()
    transaction.restore_log()
    monkeypatch.setattr(store, '_finish_prepared', real_finish)
    manifest = _read(transaction.directory / 'transaction.json')
    staged = (transaction.directory / manifest['journal']).read_bytes()
    original = (Path(engine.run_dir) / 'sciel_events.jsonl').read_bytes()
    amount = len(staged) if partial == 'all' else partial
    with open(Path(engine.run_dir) / 'sciel_events.jsonl', 'ab') as stream:
        stream.write(staged[:amount])
    assert store.recover(engine.run_dir)['phase'] == 'committed'
    assert (Path(engine.run_dir) / 'sciel_events.jsonl').read_bytes() == original + staged
    assert _read(Path(engine.run_dir) / 'checkpoint.json')['counter'] == 2
    before = (Path(engine.run_dir) / 'checkpoint.json').read_bytes()
    assert store.recover(engine.run_dir)['phase'] == 'committed'
    assert (Path(engine.run_dir) / 'checkpoint.json').read_bytes() == before


def test_recover_after_checkpoint_replace_before_commit_marker(engine, monkeypatch):
    transaction = store.begin(engine, 0)
    engine.counter = 4
    engine.log.iteration(child_id='p4')
    write = store._write_json

    def fail_marker(path, value):
        if value.get('phase') == 'committed':
            raise OSError('crash after checkpoint replacement')
        write(path, value)

    monkeypatch.setattr(store, '_write_json', fail_marker)
    with pytest.raises(OSError):
        transaction.commit()
    transaction.restore_log()
    monkeypatch.setattr(store, '_write_json', write)
    assert _read(Path(engine.run_dir) / 'checkpoint.json')['counter'] == 4
    assert store.recover(engine.run_dir)['phase'] == 'committed'


def test_active_baseline_install_recovers_crash_before_copy(engine, monkeypatch):
    engine.counter = 8
    original = store._baseline
    monkeypatch.setattr(store, '_baseline', lambda *args: (_ for _ in ()).throw(OSError('crash before baseline install')))
    with pytest.raises(OSError):
        store.begin(engine, 0)
    monkeypatch.setattr(store, '_baseline', original)
    assert _read(Path(engine.run_dir) / 'checkpoint.json')['counter'] == 0
    assert store.recover(engine.run_dir)['phase'] == 'active'
    assert _read(Path(engine.run_dir) / 'checkpoint.json')['counter'] == 8


def test_fail_closed_if_scientific_suffix_or_checkpoint_changed(engine):
    transaction = store.begin(engine, 0)
    transaction.restore_log()
    engine.log.iteration(child_id='illegal-outside-wave')
    with pytest.raises(store.WaveTransactionError, match='scientific journal suffix'):
        store.recover(engine.run_dir)


def test_commit_refuses_accidental_authoritative_checkpoint_save(engine):
    transaction = store.begin(engine, 0)
    engine.counter = 3
    store.checkpoint.save(engine, 'accidental-root-save')
    with pytest.raises(store.WaveTransactionError, match='checkpoint changed'):
        transaction.commit()
    transaction.restore_log()


@pytest.mark.parametrize('tamper', ['candidate', 'journal', 'main'])
def test_prepared_tampering_rejected_without_publishing(engine, monkeypatch, tamper):
    transaction = store.begin(engine, 0)
    engine.counter = 7
    engine.log.iteration(child_id='p7')
    finish = store._finish_prepared
    monkeypatch.setattr(store, '_finish_prepared', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        transaction.commit()
    transaction.restore_log()
    monkeypatch.setattr(store, '_finish_prepared', finish)
    manifest = _read(transaction.directory / 'transaction.json')
    target = (Path(engine.run_dir) / 'sciel_events.jsonl' if tamper == 'main'
              else transaction.directory / manifest[tamper])
    with open(target, 'ab') as stream:
        stream.write(b'garbage')
    before = (Path(engine.run_dir) / 'checkpoint.json').read_bytes()
    with pytest.raises(store.WaveTransactionError):
        store.recover(engine.run_dir)
    assert (Path(engine.run_dir) / 'checkpoint.json').read_bytes() == before


def test_wave_conflicts_and_recovery_respects_writer_lock(engine):
    transaction = store.begin(engine, 0)
    with pytest.raises(store.WaveTransactionError, match='nested'):
        store.begin(engine, 1)
    transaction.restore_log()
    with pytest.raises(store.WaveTransactionError, match='another wave'):
        store.begin(engine, 1)
    lock = RunDirectoryLock(engine.run_dir)
    try:
        with pytest.raises(RunDirectoryBusy):
            store.recover(engine.run_dir)
    finally:
        lock.close()
    with pytest.raises(ValueError):
        store.begin(engine, '../escape')


def test_saved_llm_context_and_monotonic_origin_survive_swap(engine):
    engine.log.research(phase='attempt', experiment_id='r1', prompt_sha256='123', evidence={'iteration': 1})
    before = store._metadata(engine.log)
    transaction = store.begin(engine, 0)
    assert store._metadata(engine.log) == before
    engine.log.research(phase='aux_start', kind='conclusion', reference='r1')
    expected = store._metadata(engine.log)
    transaction.commit()
    assert store._metadata(engine.log) == expected


def test_baseline_and_wal_not_discarded_on_rebase_failure(engine, monkeypatch):
    transaction = store.begin(engine, 0)
    (transaction.directory / 'receipt').write_text('paid response')
    original_baseline = transaction.manifest['baseline']
    transaction.restore_log()
    engine.log.run_start(restored_checkpoint=engine._saved_checkpoint)
    write = store._atomic_bytes

    def fail_stage(target, data):
        if str(target).endswith('.journal.jsonl'):
            raise OSError('crash creating new delta')
        write(target, data)

    monkeypatch.setattr(store, '_atomic_bytes', fail_stage)
    with pytest.raises(OSError):
        store.begin(engine, 0)
    assert _read(transaction.directory / 'transaction.json')['baseline'] == original_baseline
    assert store.recover(engine.run_dir)['phase'] == 'active'
    assert (transaction.directory / 'receipt').read_text() == 'paid response'


def test_cleanup_only_after_commit_marker_and_resumes_after_crash(engine, monkeypatch):
    transaction = store.begin(engine, 0)
    calls = transaction.directory / 'calls'
    calls.mkdir()
    (calls / 'returned.json').write_text('immutable paid receipt')
    (transaction.directory / 'jobs.json').write_text('immutable dispatch')
    (transaction.directory / 'result-G-1.json').write_text('candidate receipt')
    engine.counter = 9
    engine.log.iteration(child_id='p9')
    finish = store._finish_prepared
    monkeypatch.setattr(store, '_finish_prepared', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('crash prepared')))
    with pytest.raises(OSError):
        transaction.commit()
    transaction.restore_log()
    manifest = _read(transaction.directory / 'transaction.json')
    copies = [transaction.directory / manifest[key] for key in ('baseline', 'journal', 'candidate')]
    assert all(path.exists() for path in copies)
    monkeypatch.setattr(store, '_finish_prepared', finish)
    cleanup = store._cleanup_committed
    monkeypatch.setattr(store, '_cleanup_committed', lambda *args: (_ for _ in ()).throw(OSError('crash cleanup')))
    with pytest.raises(OSError, match='cleanup'):
        store.recover(engine.run_dir)
    assert _read(transaction.directory / 'transaction.json')['phase'] == 'committed'
    assert _read(Path(engine.run_dir) / 'checkpoint.json')['counter'] == 9
    assert all(path.exists() for path in copies)
    monkeypatch.setattr(store, '_cleanup_committed', cleanup)
    assert store.recover(engine.run_dir)['phase'] == 'committed'
    assert not any(path.exists() for path in copies)
    assert (calls / 'returned.json').read_text() == 'immutable paid receipt'
    assert (transaction.directory / 'jobs.json').read_text() == 'immutable dispatch'
    assert (transaction.directory / 'result-G-1.json').read_text() == 'candidate receipt'


def test_normal_commit_removes_full_copies(engine):
    transaction = store.begin(engine, 0)
    engine.log.iteration(child_id='p2')
    transaction.commit()
    assert list(transaction.directory.iterdir()) == [transaction.directory / 'transaction.json']
    # A subsequent wave starts from the authoritative files, not removed copies.
    following = store.begin(engine, 1)
    following.restore_log()


def test_normal_waves_never_reread_or_copy_historical_journal(engine, monkeypatch):
    import builtins
    main = Path(engine.log.path).resolve()
    for i in range(25):
        engine.log.decision(policy='history_fixture', index=i, payload='x' * 4000)
    original_fh = engine.log._fh
    expected_sequence = engine.log.event_seq
    open_file = builtins.open
    original_constructor = RunLog.__init__

    def guarded_open(file, mode='r', *args, **kwargs):
        if not isinstance(file, int) and Path(file).resolve() == main and ('r' in mode or '+' in mode):
            raise AssertionError('normal wave reread historical journal')
        return open_file(file, mode, *args, **kwargs)

    def guarded_constructor(self, path):
        assert Path(path).resolve() != main, 'normal commit reconstructed the full RunLog'
        original_constructor(self, path)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'open', guarded_open)
        patch.setattr(RunLog, '__init__', guarded_constructor)
        for wave in range(3):
            transaction = store.begin(engine, wave)
            assert Path(engine.log.path).stat().st_size == 0
            assert engine.log._prefix_bytes == main.stat().st_size
            engine.log.iteration(child_id=f'p{wave + 1}')
            assert Path(engine.log.path).stat().st_size < 1000
            transaction.commit()
            expected_sequence += 1
            assert engine.log._fh is original_fh
            assert engine.log._prefix_bytes == 0
            assert engine.log.anchor()['event_seq'] == expected_sequence
    contents = main.read_bytes()
    assert engine.log.anchor()['prefix_sha256'] == hashlib.sha256(contents).hexdigest()
    assert engine.log.anchor()['byte_offset'] == len(contents)


def test_delta_recovery_streams_baseline_once_and_authenticates_full_anchor(engine, monkeypatch):
    transaction = store.begin(engine, 0)
    engine.log.iteration(child_id='p2')
    finish = store._finish_prepared
    monkeypatch.setattr(store, '_finish_prepared', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        transaction.commit()
    transaction.restore_log()
    monkeypatch.setattr(store, '_finish_prepared', finish)
    manifest = _read(transaction.directory / 'transaction.json')
    delta = (transaction.directory / manifest['journal']).read_bytes()
    assert len(delta) == manifest['delta_bytes']
    assert hashlib.sha256(delta).hexdigest() == manifest['delta_sha256']
    assert manifest['journal_bytes'] == manifest['base_journal_bytes'] + len(delta)
    calls = []
    read_base = store._base_digest

    def counted(path, boundary):
        calls.append(boundary)
        return read_base(path, boundary)

    monkeypatch.setattr(store, '_base_digest', counted)
    store.recover(engine.run_dir)
    assert calls == [manifest['base_journal_bytes']]
    checkpoint = _read(Path(engine.run_dir) / 'checkpoint.json')
    full = (Path(engine.run_dir) / 'sciel_events.jsonl').read_bytes()
    assert checkpoint['provenance']['journal_anchor']['prefix_sha256'] == hashlib.sha256(full).hexdigest()


def test_delta_recovery_rejects_same_length_prefix_tampering(engine, monkeypatch):
    transaction = store.begin(engine, 0)
    engine.log.iteration(child_id='p2')
    finish = store._finish_prepared
    monkeypatch.setattr(store, '_finish_prepared', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        transaction.commit()
    transaction.restore_log()
    monkeypatch.setattr(store, '_finish_prepared', finish)
    main = Path(engine.run_dir) / 'sciel_events.jsonl'
    contents = main.read_bytes()
    assert b'run_start' in contents
    main.write_bytes(contents.replace(b'run_start', b'run_stOrt', 1))
    before = (Path(engine.run_dir) / 'checkpoint.json').read_bytes()
    with pytest.raises(store.WaveTransactionError, match='baseline journal hash mismatch'):
        store.recover(engine.run_dir)
    assert (Path(engine.run_dir) / 'checkpoint.json').read_bytes() == before


def test_delta_normal_commit_detects_external_baseline_edit_before_append(engine):
    transaction = store.begin(engine, 0)
    engine.log.iteration(child_id='unpublished')
    main = Path(engine.run_dir) / 'sciel_events.jsonl'
    contents = main.read_bytes()
    main.write_bytes(contents.replace(b'run_start', b'run_stOrt', 1))
    with pytest.raises(store.WaveTransactionError, match='authoritative journal changed'):
        transaction.commit()
    transaction.restore_log()
    assert b'unpublished' not in main.read_bytes()


@pytest.mark.parametrize('fail_after_first', [False, True])
def test_short_delta_writes_complete_or_leave_prepared_recoverable(engine, monkeypatch, fail_after_first):
    import builtins
    transaction = store.begin(engine, 0)
    engine.counter = 8
    engine.log.iteration(child_id='p8', payload='x' * 128)
    main = Path(engine.run_dir) / 'sciel_events.jsonl'
    open_file = builtins.open

    class ShortWriter:
        def __init__(self, stream):
            self.stream, self.writes = stream, 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def fileno(self):
            return self.stream.fileno()
        def write(self, data):
            self.writes += 1
            if fail_after_first and self.writes > 1:
                return 0
            return self.stream.write(data[:7])

    def short_open(path, mode='r', *args, **kwargs):
        stream = open_file(path, mode, *args, **kwargs)
        return ShortWriter(stream) if Path(path) == main and mode == 'ab' else stream

    with monkeypatch.context() as patch:
        patch.setattr(builtins, 'open', short_open)
        if fail_after_first:
            with pytest.raises(OSError, match='short parallel journal append'):
                transaction.commit()
        else:
            transaction.commit()
    if fail_after_first:
        transaction.restore_log()
        assert _read(main.parent / 'checkpoint.json')['counter'] == 0
        assert _read(transaction.directory / 'transaction.json')['phase'] == 'prepared'
        store.recover(engine.run_dir)
    saved = _read(main.parent / 'checkpoint.json')
    assert saved['counter'] == 8
    assert saved['provenance']['journal_anchor']['prefix_sha256'] == hashlib.sha256(main.read_bytes()).hexdigest()
