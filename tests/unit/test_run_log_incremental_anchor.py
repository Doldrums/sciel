"""Journal anchors bind exact bytes without rereading every historical prefix."""
import hashlib
import os
from pathlib import Path

import pytest

from sciel.run_log import RunLog


def assert_anchor(log, path):
    raw = path.read_bytes()
    anchor = log.anchor()
    assert anchor['prefix_sha256'] == hashlib.sha256(raw).hexdigest()
    assert anchor['byte_offset'] == len(raw)
    assert anchor['event_seq'] == log.event_seq


def test_appended_and_restored_anchors_bind_exact_prefix(tmp_path):
    path = tmp_path / 'events.jsonl'
    log = RunLog(str(path))
    try:
        log.run_start()
        for i in range(4):
            log.research(phase='hypothesis', experiment_id=str(i), prompt='世界\r\n' * 500)
            assert_anchor(log, path)
        restored = {'provenance': {'run_id': log.run_id, 'journal_anchor': log.anchor()}}
    finally:
        log.close()
    log = RunLog(str(path))
    try:
        log.run_start(restored_checkpoint=restored)
        log.decision(action='continued')
        assert_anchor(log, path)
    finally:
        log.close()


def test_checkpoints_do_not_read_the_historical_prefix(tmp_path, monkeypatch):
    path = tmp_path / 'events.jsonl'
    log = RunLog(str(path))
    try:
        log.run_start()
        log.research(phase='hypothesis', prompt='x' * 2_000_000)
        # The file is already open. Repeated anchors and new writes must not
        # open/read the historical prefix again, regardless of its size.
        def unexpected(*args, **kwargs):
            pytest.fail('checkpoint reread historical journal')
        monkeypatch.setattr('builtins.open', unexpected)
        for i in range(4):
            log.decision(action='fixture', value=i)
            assert log.anchor()['event_seq'] == 3 + i
    finally:
        log.close()


def test_restore_refuses_to_append_to_an_incomplete_record(tmp_path):
    path = tmp_path / 'events.jsonl'
    path.write_bytes(b'{"event":"run_end"}')
    log = RunLog(str(path))
    try:
        with pytest.raises(OSError, match='cannot start'):
            log.run_start()
    finally:
        log.close()
    assert path.read_bytes() == b'{"event":"run_end"}'


@pytest.mark.parametrize('mutation', ['append', 'same_size', 'replace', 'truncate'])
@pytest.mark.parametrize('before_emit', [False, True])
def test_external_journal_changes_are_rejected(tmp_path, mutation, before_emit):
    path = tmp_path / 'events.jsonl'
    log = RunLog(str(path))
    try:
        log.run_start()
        raw = path.read_bytes()
        if mutation == 'append':
            with path.open('ab') as stream:
                stream.write(b'{}\n')
        elif mutation == 'same_size':
            # Change a prefix byte without changing the last event/size; restoring
            # mtime does not hide the changed ctime.
            old = path.stat()
            path.write_bytes(raw.replace(b'run_start', b'run_other'))
            os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
        elif mutation == 'replace':
            alternate = Path(str(path) + '.replacement')
            alternate.write_bytes(raw)
            alternate.replace(path)
        else:
            path.write_bytes(raw[:-1])
        if before_emit:
            assert log.emit('decision', action='must_not_append') is False
        with pytest.raises(OSError, match='journal'):
            log.anchor()
    finally:
        log.close()
