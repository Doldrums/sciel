"""Known broken reads are typed; no credentials, networking or generated code."""
import hashlib
from http.client import IncompleteRead
import json
import socket
import ssl
from urllib.error import URLError

import pytest

from sciel.engine.llm.client import LLMError, llm_error_metadata
from sciel.engine.llm.codex import CodexResponsesLLM
from sciel.engine.parallel_recovery import _classify, _protocol_failure
from test_codex_responses import MODEL, Opener, Stream, config, events


def interruption(kind):
    if kind == 'incomplete_read':
        # Complete-looking text in partial HTTP bytes is not a terminal receipt.
        return IncompleteRead(b'private data: {"type":"response.completed"}', 5)
    return ssl.SSLEOFError(8, 'private TLS detail fixture-secret')


class BrokenRead(Stream):
    def __init__(self, rows, error):
        super().__init__(rows, chunk=65536)
        self.failure = error

    def read1(self, size):
        try:
            return next(self.chunks)
        except StopIteration:
            raise self.failure


@pytest.mark.parametrize('kind', ['incomplete_read', 'tls_eof'])
@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('rows', [[], events('private complete-looking text')[:-1]])
def test_only_opened_stream_read_interruptions_get_bound_codes(kind, wrapped, rows):
    error = interruption(kind)
    if wrapped:
        error = URLError(error)

    class ReadOpener(Opener):
        def open(self, request, timeout):
            super().open(request, timeout)
            return BrokenRead(rows, error)

    opener = ReadOpener()
    llm = CodexResponsesLLM(config(), 'fixture-secret', opener=opener)
    with pytest.raises(LLMError) as caught:
        llm.generate('private prompt', MODEL)
    metadata = llm_error_metadata(caught.value)
    diagnostic = metadata['codex_protocol']
    assert diagnostic['code'] == 'transport_' + kind
    assert diagnostic['body_sha256'] == hashlib.sha256(opener.calls[0][0].data).hexdigest()
    assert diagnostic['model'] == MODEL and diagnostic['terminal_status'] == 'none'
    assert diagnostic['events_seen'] == len(rows)
    assert _protocol_failure(metadata) == diagnostic
    assert _classify(caught.value)['kind'] == 'provider'
    assert len(opener.calls) == 1
    assert 'private' not in json.dumps(metadata) and 'fixture-secret' not in json.dumps(metadata)
    assert len(json.dumps(metadata)) < 1024
    # An observed terminal cannot be erased by relabelling its later failure.
    metadata['codex_protocol']['terminal_status'] = 'completed'
    assert _protocol_failure(metadata) is None


@pytest.mark.parametrize('kind', ['incomplete_read', 'tls_eof'])
@pytest.mark.parametrize('location', ['open', 'close'])
@pytest.mark.parametrize('wrapped', [False, True])
def test_handshake_and_completed_response_cleanup_do_not_gain_read_diagnostics(kind, location, wrapped):
    error = interruption(kind)
    if wrapped:
        error = URLError(error)

    class CloseFailure(Stream):
        def __exit__(self, *args):
            raise error

    class BoundaryOpener(Opener):
        def open(self, request, timeout):
            super().open(request, timeout)
            if location == 'open':
                raise error
            return CloseFailure(events())

    opener = BoundaryOpener()
    llm = CodexResponsesLLM(config(), 'fixture-secret', opener=opener)
    with pytest.raises(LLMError) as caught:
        llm.generate('p', MODEL)
    metadata = llm_error_metadata(caught.value)
    assert metadata == dict(stage='request', reason='request_failed', cause_type='OtherError')
    assert _classify(caught.value)['kind'] == 'fatal' and len(opener.calls) == 1


@pytest.mark.parametrize('error', [
    ssl.SSLCertVerificationError(1, 'private certificate'),
    ssl.SSLError(1, 'private generic TLS'),
    socket.gaierror(-2, 'private DNS'),
    ValueError('private local bug'),
])
def test_unclassified_read_errors_remain_fatal(error):
    class ErrorOpener(Opener):
        def open(self, request, timeout):
            super().open(request, timeout)
            return BrokenRead([], URLError(error))

    opener = ErrorOpener()
    llm = CodexResponsesLLM(config(), 'fixture-secret', opener=opener)
    with pytest.raises(LLMError) as caught:
        llm.generate('p', MODEL)
    metadata = llm_error_metadata(caught.value)
    assert 'codex_protocol' not in metadata
    assert _classify(caught.value)['kind'] == 'fatal'
    assert 'private' not in json.dumps(metadata) and len(opener.calls) == 1
