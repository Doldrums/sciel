"""Durable IO protocol tests with fake callables, never providers/containers."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
import time

import pytest

from sciel.campaign.budget import RequestNotDispatched
from sciel.engine.llm.client import LLMResponse
from sciel.methods.budget import BudgetExceeded, BudgetUnknown, ResourceLedger
from sciel.methods.proposals import BudgetedProposer, ScriptedProposer, parse_proposal, prompt_for, response_record
from sciel.methods.registry import Registry
from sciel.methods.transport import DispatchStore


BOUNDS = dict(dollars=1, wall_seconds=10, cpu_seconds=10, gpu_seconds=0, storage_bytes=10000, steps=1)
ACTUAL = dict(dollars=.2, wall_seconds=.1, cpu_seconds=.01, gpu_seconds=0, storage_bytes=100, steps=1)


def store(tmp_path):
    registry = Registry(tmp_path / 'registry')
    ledger = ResourceLedger(tmp_path / 'budget.jsonl',
        dict(dollars=20, wall_seconds=100, cpu_seconds=100, gpu_seconds=0, storage_bytes=100000, steps=20))
    ledger.create_account('lab', {})
    return DispatchStore(tmp_path / 'calls', registry, ledger)


def call(dispatch, function, *, key='first', account='lab', request=None):
    return dispatch.run(key, account, 'proposal', request or {'prompt_sha256': 'a' * 64}, BOUNDS, function)


def test_known_receipt_replays_without_second_call_or_charge(tmp_path):
    dispatch = store(tmp_path)
    calls = []
    def provider(reservation):
        calls.append(reservation)
        return {'status': 'completed', 'text': 'real response', 'prompt_tokens': 4, 'completion_tokens': 6}, ACTUAL
    first = call(dispatch, provider)
    second = call(DispatchStore(dispatch.root, dispatch.registry, dispatch.ledger), provider)
    assert len(calls) == 1
    assert first['replayed'] is False and second['replayed'] is True
    assert first['call_receipt_id'] == second['call_receipt_id']
    assert second['prompt_tokens'] == 4
    assert dispatch.ledger.snapshot()['spent']['dollars'] == '0.2'
    with pytest.raises(ValueError, match='another request'):
        call(dispatch, provider, request={'prompt_sha256': 'b' * 64})
    assert len(calls) == 1


def test_concurrent_hosts_dispatch_one_logical_call_once(tmp_path):
    dispatch = store(tmp_path)
    calls = []
    def provider(reservation):
        calls.append(reservation)
        time.sleep(.02)
        return {'status': 'completed'}, ACTUAL
    def invoke(_):
        return call(DispatchStore(dispatch.root, dispatch.registry, dispatch.ledger), provider)
    with ThreadPoolExecutor(max_workers=3) as pool:
        receipts = list(pool.map(invoke, range(3)))
    assert len(calls) == 1
    assert sum(not receipt['replayed'] for receipt in receipts) == 1


def test_same_key_in_different_accounts_is_not_the_same_budget_request(tmp_path):
    dispatch = store(tmp_path)
    dispatch.ledger.create_account('recipient', {})
    a = call(dispatch, lambda _: ({'status': 'completed'}, ACTUAL))
    b = call(dispatch, lambda _: ({'status': 'completed'}, ACTUAL), account='recipient')
    assert a['reservation_id'] != b['reservation_id']
    assert dispatch.ledger.snapshot()['spent']['dollars'] == '0.4'


def test_unknown_is_durable_private_fail_closed_and_explicitly_resolvable(tmp_path):
    dispatch = store(tmp_path)
    calls = []
    def provider(_):
        calls.append(1)
        raise ConnectionError('Authorization: secret-provider-key')
    with pytest.raises(BudgetUnknown) as caught:
        call(dispatch, provider)
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    with pytest.raises(BudgetUnknown):
        call(dispatch, provider)
    assert len(calls) == 1
    assert dispatch.ledger.snapshot()['reserved']['dollars'] == '1'
    assert dispatch.ledger.snapshot()['unknown_reservations']
    for path in tmp_path.rglob('*.json'):
        assert 'secret-provider-key' not in path.read_text()
    original = next(dispatch.root.glob('*/receipt.json')).read_bytes()
    evidence = dispatch.registry.put_artifact('provider_audit', {'billing_reference': 'fixture', 'charge': '.2'})
    result = dispatch.resolve_unknown('first', 'lab', result={'status': 'failed', 'reason': 'audited abandonment'},
        actual={**ACTUAL, 'wall_seconds': 1, 'cpu_seconds': 1}, reason='external receipt reconciled',
        evidence_refs=[evidence])
    assert result['status'] == 'failed'
    assert next(dispatch.root.glob('*/receipt.json')).read_bytes() == original
    resumed = call(dispatch, provider)
    assert resumed['replayed'] is True and resumed['status'] == 'failed'
    assert len(calls) == 1
    assert dispatch.ledger.snapshot()['spent']['dollars'] == '0.2'
    assert not dispatch.ledger.snapshot()['unknown_reservations']


def test_missing_usage_cannot_be_completed_and_keeps_known_charge_and_response(tmp_path):
    dispatch = store(tmp_path)
    with pytest.raises(BudgetUnknown):
        call(dispatch, lambda _: ({'status': 'completed', 'text': 'paid response'}, {'dollars': .3}))
    receipt = json.loads(next(dispatch.root.glob('*/receipt.json')).read_text())
    assert receipt['status'] == 'unknown'
    assert receipt['result']['text'] == 'paid response'
    assert dispatch.ledger.snapshot()['spent']['dollars'] == '0.3'
    assert dispatch.ledger.snapshot()['reserved']['dollars'] == '0.7'


def test_proven_predispatch_rejection_is_failed_known_zero_provider_charge(tmp_path):
    dispatch = store(tmp_path)
    def rejected(_):
        raise RequestNotDispatched('private ledger diagnostic')
    value = call(dispatch, rejected)
    assert value['status'] == 'failed' and value['dispatched'] is False
    assert dispatch.ledger.snapshot()['spent']['dollars'] == '0'
    assert not dispatch.ledger.snapshot()['unknown_reservations']


def test_crash_after_terminal_receipt_before_settlement_replays_known_result(tmp_path, monkeypatch):
    dispatch = store(tmp_path)
    calls = []
    settle = dispatch.ledger.settle
    def fail(*args, **kwargs):
        raise OSError('fixture crash')
    monkeypatch.setattr(dispatch.ledger, 'settle', fail)
    with pytest.raises(OSError):
        call(dispatch, lambda _: (calls.append(1) or {'status': 'completed'}, ACTUAL))
    monkeypatch.setattr(dispatch.ledger, 'settle', settle)
    value = call(dispatch, lambda _: pytest.fail('cannot redispatch known receipt'))
    assert value['replayed'] is True and len(calls) == 1
    assert dispatch.ledger.snapshot()['spent']['dollars'] == '0.2'


def test_crash_without_durable_terminal_receipt_never_reexecutes(tmp_path, monkeypatch):
    dispatch = store(tmp_path)
    write = dispatch._write
    def fail_terminal(path, value):
        if path.name == 'receipt.json':
            raise OSError('fixture disk error')
        return write(path, value)
    monkeypatch.setattr(dispatch, '_write', fail_terminal)
    with pytest.raises(BudgetUnknown):
        call(dispatch, lambda _: ({'status': 'completed', 'text': 'unpersisted'}, ACTUAL))
    monkeypatch.setattr(dispatch, '_write', write)
    with pytest.raises(BudgetUnknown, match='terminal receipt'):
        call(dispatch, lambda _: pytest.fail('unknown dispatch must never repeat'))
    assert dispatch.ledger.snapshot()['spent']['dollars'] == '0.2'


def test_tampered_receipt_and_missing_audit_evidence_refused(tmp_path):
    dispatch = store(tmp_path)
    call(dispatch, lambda _: ({'status': 'completed'}, ACTUAL))
    path = next(dispatch.root.glob('*/receipt.json'))
    value = json.loads(path.read_text())
    value['actual']['dollars'] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='integrity'):
        call(dispatch, lambda _: pytest.fail('tamper must not reexecute'))
    with pytest.raises(ValueError, match='registered artifacts'):
        dispatch.resolve_unknown('first', 'lab', result={'status': 'failed'}, actual=ACTUAL,
                                 reason='unsupported claim', evidence_refs=['f' * 64])


def test_known_overrun_is_booked_before_halt_and_does_not_retry(tmp_path):
    dispatch = store(tmp_path)
    with pytest.raises(BudgetExceeded):
        call(dispatch, lambda _: ({'status': 'completed'}, {**ACTUAL, 'dollars': 2}))
    assert dispatch.ledger.snapshot()['spent']['dollars'] == '2'
    with pytest.raises(BudgetExceeded):
        call(dispatch, lambda _: pytest.fail('overrun is already recorded'))


def test_scripted_proposer_real_response_ctor_and_receipt_usage():
    proposer = ScriptedProposer([{'kind': 'stop', 'reason': 'fixture'}])
    response = proposer.generate('prompt')
    assert response.cost == 0 and response.prompt_tokens == 0 and response.completion_tokens == 0
    assert response_record(response)['status'] == 'completed'
    assert parse_proposal(response.text)['kind'] == 'stop'
    with pytest.raises(ValueError, match='exhausted'):
        proposer.generate('again')
    assert response_record(LLMResponse('paid?', 'model', None, None, None))['status'] == 'unknown'


def test_scripted_resume_uses_explicit_per_lab_proposal_ordinal():
    fixture = {'author': ['first', 'repair'], 'recipient': ['adopt', 'stop']}
    proposer = ScriptedProposer(fixture)
    assert proposer.generate(prompt_for({'lab_id': 'author', 'proposal_index': 0})).text == 'first'
    resumed = ScriptedProposer(fixture)
    assert resumed.generate(prompt_for({'lab_id': 'author', 'proposal_index': 1})).text == 'repair'
    assert resumed.generate(prompt_for({'lab_id': 'recipient', 'proposal_index': 0})).text == 'adopt'


class Client:
    def __init__(self, options=None):
        self.options = options or {}
    def with_options(self, **kwargs):
        return Client(kwargs)


class PricedLLM:
    models = ['pinned-model']
    max_tokens = 100
    def __init__(self):
        self.prices = {'pinned-model': {'prompt': Decimal('.001'), 'completion': Decimal('.002'), 'request': Decimal(0)}}
        self.client = Client()
    def generate(self, prompt, model, temperature, *, response_format):
        assert self.client.options == {'timeout': 3., 'max_retries': 0}
        assert response_format is None
        return LLMResponse('{}', model, .02, 10, 2)


def test_budgeted_proposer_pins_real_sdk_options_decimal_prices_and_identity():
    original = PricedLLM()
    proposer = BudgetedProposer(original, 'pinned-model', timeout=3)
    assert proposer.identity['transport_retries'] == 0
    assert proposer.identity['pricing']['prompt'] == '0.001'
    assert original.client.options == {}
    original.prices['pinned-model']['prompt'] = Decimal('999')
    original.max_tokens = 900000
    bounds = proposer.bounds('prompt')
    assert Decimal(bounds['dollars']) < 2
    assert proposer.llm.max_tokens == 100
    assert response_record(proposer.generate('prompt'))['completion_tokens'] == 2
    json.dumps(proposer.identity)


@pytest.mark.parametrize('text', ['{"kind":[]}', '{"kind":"stop","reason":"x","reason":"y"}',
                                  '{"kind":"stop","reason":NaN}', '{"kind":"solution","code":3,"experiment":{}}'])
def test_malformed_proposals_raise_validation_errors(text):
    with pytest.raises(ValueError):
        parse_proposal(text)


def test_audited_success_keeps_inner_execution_ok_and_reuses_without_dispatch(tmp_path):
    dispatch=store(tmp_path)
    def lost(_):
        raise ConnectionError('fixture lost reply')
    with pytest.raises(BudgetUnknown):
        call(dispatch,lost)
    original=next(dispatch.root.glob('*/receipt.json')).read_bytes()
    audit=dispatch.registry.put_artifact('execution_audit',{'fixture':'externally recovered output and resource receipt'})
    actual={**ACTUAL,'cpu_seconds':1,'wall_seconds':1}
    recovered={'status':'ok','output':[1,2],'method_id':'a'*64,'plan_id':'b'*64}
    result=dispatch.resolve_unknown('first','lab',result=recovered,actual=actual,status='completed',
        reason='external receipt recovered',evidence_refs=[audit])
    assert result['status']=='ok' and result['output']==[1,2]
    artifact=dispatch.registry.get_artifact('call_receipt',result['call_receipt_id'])
    assert artifact['status']=='completed' and artifact['result']==recovered
    replay=call(dispatch,lambda _:pytest.fail('audited terminal success cannot be re-dispatched'))
    assert replay['status']=='ok' and replay['call_receipt_id']==result['call_receipt_id']
    assert next(dispatch.root.glob('*/receipt.json')).read_bytes()==original
    assert dispatch.ledger.snapshot()['spent']['dollars']=='0.2'
    assert not dispatch.ledger.snapshot()['unknown_reservations']


def test_proposal_instruction_documents_actual_pinned_dependency_invocation():
    from sciel.methods.proposals import PROCEDURE
    text=PROCEDURE['instructions']
    assert 'manifest.dependencies' in text and 'm_<hash>.invoke(JSONvalue)' in text
    assert 'share the invocation budget' in text
