"""v0.2 engine + real signed loopback protocol; numeric/evaluation leaves are synthetic."""
from contextlib import nullcontext
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sciel.adaptation.types import InferenceSpec, canonical_bytes, content_hash
from sciel.adaptation.deployment import request_headers
from sciel.campaign.budget import RequestNotDispatched
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.factory import build_llm, build_engine
from sciel.engine.llm.client import response_cost_provenance, response_accounting
from sciel.engine.llm.trace import TracedLLM
from sciel.engine.parallel import WaveLLM
from sciel.engine.parallel_calls import CallLedger
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult

ROOT = Path(__file__).resolve().parents[2]


def local_config():
    return dict(task='test/toy_quadratic', num_islands=1, budget=1, max_iterations=1,
        random_seed=7, embed_grid=False, portfolio_seed=False, idea_island=False,
        dynamic_islands=False, policy={'enabled':False}, milestone_fracs=[],
        op_fixed='full_rewrite', llm={'backend':'local_openai','models':['base-pinned'],
            'api_base':'http://127.0.0.1:8000/v1','generation_seed':123,'max_tokens':128,
            'local_inference': {'base_model':'fixture/model','base_revision':'a'*40,
                'tokenizer_revision':'b'*40,'chat_template_sha256':'c'*64,
                'base_alias':'base-pinned','learner_by_island':{'0':'learner-0'},
                'adapters':{},'node_ledger_id':'fixture-node'}})


class ProtocolServer:
    def __init__(self):
        self.key = Ed25519PrivateKey.generate()
        self.public_key = self.key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
        self.instance, self.worker = '1' * 32, '2' * 32
        self.bootstrap = '3' * 64
        self.receipts, self.calls = {}, []
        self.status_calls = 0
        self.bad_echo = False
        self.status_override = None
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def reply(self, status, value, headers=()):
                data = canonical_bytes(value)
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                for key, value in headers:
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)
            def do_GET(self):
                assert self.path == '/v1/sciel/deployment/status'
                owner.status_calls += 1
                self.reply(200, owner.status_override or owner.status())
            def do_POST(self):
                assert self.path == '/v1/chat/completions'
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                binding = {name: self.headers.get(name) for name in ('x-sciel-server-instance', 'x-sciel-deployment')}
                owner.calls.append(dict(body=body, headers=binding))
                receipt = owner.receipts.get(binding['x-sciel-deployment'])
                if (binding['x-sciel-server-instance'] != owner.instance or receipt is None
                        or receipt['body']['inference_spec']['serving_alias'] != body['model']):
                    return self.reply(409, {'error': {'message':'wrong deployment', 'type':'invalid_request_error'}})
                value = len(owner.calls)
                content = f'```python\nVALUE = {value}\ndef run_search():\n    return VALUE\n```'
                result = dict(id='chat-fixture', object='chat.completion', created=0, model=body['model'],
                    choices=[dict(index=0, finish_reason='stop', message=dict(role='assistant', content=content))],
                    usage=dict(prompt_tokens=64, completion_tokens=16, total_tokens=80))
                if owner.bad_echo:
                    binding['x-sciel-deployment'] = '0' * 64
                self.reply(200, result, binding.items())
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.api_base = f'http://127.0.0.1:{self.http.server_port}/v1'
        self.thread = Thread(target=self.http.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start()
        self.config = local_config()
        self.config['llm']['api_base'] = self.api_base
        self.config['llm']['local_inference']['adapters'] = {}
        self.config['llm']['managed_deployment'] = self.pins()
        source = self.config['llm']['local_inference']
        self.base = InferenceSpec(owner='base', role='base', serving_alias=source['base_alias'],
            **{key:source[key] for key in ('base_model','base_revision','tokenizer_revision','chat_template_sha256')})
        self.base_receipt = self.publish(self.base)

    def close(self):
        self.http.shutdown()
        self.thread.join(timeout=2)
        self.http.server_close()

    def pins(self):
        return dict(public_key=self.public_key, server_instance_id=self.instance, bootstrap_sha256=self.bootstrap)

    def sign(self, body):
        return dict(body=deepcopy(body), sha256=content_hash(body), public_key=self.public_key,
                    signature=self.key.sign(canonical_bytes(body)).hex())

    def publish(self, spec, *, files=None, manifest_hash=None):
        adapted = spec.adapter_id is not None
        body = dict(schema='sciel-managed-deployment/1', server_instance_id=self.instance,
            worker_boot_id=self.worker, bootstrap_sha256=self.bootstrap, api_base=self.api_base,
            inference_spec=spec.to_dict(), immutable_alias=True, inplace_updates=False,
            previous_versions_retained=True, retention=dict(kind='bounded_pinned_cpu_gpu',
                max_public_versions=4, private_lora_ids=[1000]), kind='adapter' if adapted else 'base',
            worker_ack=None)
        if adapted:
            body['files'] = files
            body['registry_manifest_sha256'] = manifest_hash
            body['worker_ack'] = dict(alias=spec.serving_alias, path=f'/owned/{self.instance}/{spec.adapter_sha256}',
                lora_int_id=len(self.receipts)+1, manifest_sha256=manifest_hash, adapter_sha256=spec.adapter_sha256)
        receipt = self.sign(body)
        self.receipts = {**self.receipts, receipt['sha256']: receipt}
        return deepcopy(receipt)


    def status(self):
        return dict(schema='sciel-managed-deployment-status/1', server_instance_id=self.instance,
                    public_key=self.public_key, receipts=deepcopy(self.receipts))



@pytest.fixture
def server():
    value = ProtocolServer()
    try:
        yield value
    finally:
        value.close()


def test_actual_signed_wire_trace_and_new_upstream_accounting(server, tmp_path):
    provider = build_llm(server.config)
    trace = TracedLLM(provider, tmp_path, lambda _: {'role':'G', 'run_id':'fixture'})
    spec = provider.resolve_inference(island=0, role='G').to_dict()
    result = trace.generate('prompt', 'base-pinned', inference_spec=spec)
    assert result.resolved_model == result.served_model == 'base-pinned'
    assert result.response_error is None
    assert response_accounting(result) == {'known_usd':0.0, 'unpriced_calls':0}
    assert response_cost_provenance(result)['scope'] == 'external_api_only'
    wire = server.calls[0]
    assert wire['body']['seed'] == 123
    assert wire['body']['chat_template_kwargs'] == {'enable_thinking':False}
    assert wire['headers'] == request_headers(provider.bind_deployment(spec))
    rows = [json.loads(s) for s in (tmp_path/'artifacts/llm_calls.jsonl').read_text().splitlines()]
    assert rows[0]['inference_spec'] == spec == rows[1]['inference_spec']
    assert rows[0]['inference_deployment'] == rows[1]['inference_deployment']
    assert rows[0]['transport_parameters']['deployment_headers'] == wire['headers']
    assert rows[1]['resolved_model'] == 'base-pinned'
    assert rows[1]['cost_provenance']['node_ledger_id'] == 'fixture-node'
    assert len(server.calls) == 1


def test_bad_live_signature_refuses_before_generation(server):
    server.status_override = server.status()
    receipt = next(iter(server.status_override['receipts'].values()))
    receipt['signature'] = '0'*128
    with pytest.raises(ValueError, match='signature'):
        build_llm(server.config)
    assert not server.calls


def test_bad_echo_is_paid_returned_error_not_retry(server, tmp_path):
    provider = build_llm(server.config)
    server.bad_echo = True
    trace = TracedLLM(provider, tmp_path, lambda _: {'role':'C'})
    response = trace.generate('aux', 'base-pinned')
    assert response.response_error['reason'] == 'deployment_identity_mismatch'
    assert response.cost == 0 and len(server.calls) == 1
    row = json.loads((tmp_path/'artifacts/llm_calls.jsonl').read_text().splitlines()[-1])
    assert row['generation_returned'] and row['response_error'] == response.response_error


def test_real_upstream_engine_one_generation_and_checkpoint(server, tmp_path):
    cfg = deepcopy(server.config)
    engine = build_engine(cfg, str(ROOT/'tasks'), str(tmp_path))
    engine._evaluate = lambda code: CascadeOutcome(EvalResult(True, .5), .5, 'full')
    try:
        engine.run()
        assert engine.iteration_no == 1 and engine.B_evals == 1
        saved = tmp_path/'checkpoint.json'
        state = json.loads(saved.read_text())
        assert state['identity']['engine_contract'] == 'espalier-3.64'
        assert state['identity']['local_inference']['generation_seed'] == 123
        assert state['identity']['managed_deployment']['pins']['public_key'] == server.public_key
        assert len(server.calls) == 1
        rows = [json.loads(s) for s in (tmp_path/'artifacts/llm_calls.jsonl').read_text().splitlines()]
        request = next(row for row in rows if row['event']=='request')
        assert request['inference_spec']['role'] == 'G'
        assert request['inference_spec']['owner'] == 'learner-0'
    finally:
        engine.close()
    # Exact native resume, no additional generation.
    restored = build_engine(cfg, str(ROOT/'tasks'), str(tmp_path))
    try:
        checkpoint.load(restored, str(saved))
        assert restored.iteration_no == 1 and len(server.calls) == 1
    finally:
        restored.close()


def test_wal_same_request_replay_keeps_signed_metadata_without_dispatch(server, tmp_path):
    provider = build_llm(server.config)
    ledger = CallLedger(tmp_path/'wave', SimpleNamespace(acquire=lambda *a,**kw:nullcontext()), 'local')
    class IO:
        current = 'G:1'
        def call(self, kind, request, operation):
            directory, digest, cached = ledger.prepare('G:1:0', kind, request)
            return ledger.result(kind, cached or ledger.perform(directory, digest, kind, operation))
    engine = SimpleNamespace(run_dir=tmp_path, _parallel_wave=0,
        _llm_trace_context=lambda _: {'role':'G','run_id':'fixture'})
    wave = WaveLLM(engine, provider, IO())
    spec = provider.resolve_inference(island=0, role='G')
    first = wave.generate('prompt','base-pinned',inference_spec=spec)
    assert wave.generate('prompt','base-pinned',inference_spec=spec) == first
    assert len(server.calls) == 1
    assert first.inference_deployment and first.deployment_response and first.call_id
    assert first.resolved_model == 'base-pinned'
    cfg = deepcopy(server.config); cfg['llm']['generation_seed'] = 124
    wave.original = build_llm(cfg)
    with pytest.raises(ValueError, match='durable intent'):
        wave.generate('prompt','base-pinned',inference_spec=spec)
    assert len(server.calls) == 1


@pytest.mark.parametrize('change', ['seed','signer','old-contract','adaptation','adaptation-config','adaptation-identity','missing-local'])
def test_force_cannot_drop_or_rebind_saved_local_authority(server, tmp_path, change):
    engine = build_engine(server.config, str(ROOT/'tasks'), str(tmp_path))
    try:
        saved = Path(checkpoint.save(engine, 'before'))
        state = json.loads(saved.read_text())
        if change == 'seed': state['identity']['local_inference']['generation_seed'] = 99
        if change == 'signer': state['identity']['managed_deployment']['pins']['public_key'] = '0'*64
        if change == 'old-contract': state['identity']['engine_contract'] = 'espalier-3.44'
        if change == 'adaptation': state['online_execution'] = {'intent':{'request':'paid'}}
        if change == 'adaptation-config': state['config']['online_adaptation'] = {}
        if change == 'adaptation-identity': state['identity']['continuous_routing'] = {}
        if change == 'missing-local': del state['identity']['local_inference']
        saved.write_text(json.dumps(state))
        with pytest.raises(checkpoint.IdentityMismatch, match='cannot force|not enabled'):
            checkpoint.load(engine, str(saved), allow_identity_mismatch=True)
        assert not server.calls
    finally:
        engine.close()


@pytest.mark.parametrize('change', ['online','oracle','routing','adapter','mock','provider','seed'])
def test_unported_modes_fail_before_provider_or_run_mutation(tmp_path, monkeypatch, change):
    cfg = local_config()
    if change == 'online': cfg['online_adaptation'] = {}
    if change == 'oracle': cfg['oracle_evaluation'] = {}
    if change == 'routing': cfg['llm']['continuous_routing'] = {}
    if change == 'adapter': cfg['llm']['local_inference']['adapters'] = {'learner-0':{}}
    if change == 'mock': cfg['llm']['mock'] = True
    if change == 'provider': cfg['llm']['provider'] = 'codex_responses'
    if change == 'seed': cfg['llm']['generation_seed'] = True
    sdk = Mock(); monkeypatch.setattr('openai.OpenAI', sdk)
    for operation in (apply_defaults, build_llm):
        with pytest.raises(ValueError): operation(cfg)
    with pytest.raises(ValueError):
        build_engine(cfg, str(ROOT/'tasks'), str(tmp_path/'run'))
    assert not (tmp_path/'run').exists()
    sdk.assert_not_called()
