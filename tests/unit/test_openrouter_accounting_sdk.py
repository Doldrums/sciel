"""Exercise actual SDK JSON parsing without any network or model generation."""
import json

import httpx
from openai import OpenAI

from sciel.campaign.budget import BudgetedLLM, CostLedger


def test_sdk_preserves_provider_charge_and_serializes_price_caps(tmp_path):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            'id':'fixture-generation', 'object':'chat.completion', 'created':1, 'model':'m',
            'choices':[{'index':0, 'finish_reason':'stop',
                        'message':{'role':'assistant', 'content':'fixture code'}}],
            'usage':{'prompt_tokens':5, 'completion_tokens':8, 'total_tokens':13, 'cost':.0001}})
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        client = OpenAI(api_key='fixture-not-a-real-key', base_url='https://openrouter.ai/api/v1',
                        max_retries=0, http_client=transport)
        cfg = {'models':['m'], 'weights':[1.0], 'max_tokens':100,
               'api_base':'https://openrouter.ai/api/v1'}
        prices = {'models':[{'id':'m', 'pricing':{'prompt':'.000001','completion':'.000002'}}]}
        ledger = CostLedger(tmp_path / 'cost.json', 1)
        model = BudgetedLLM(cfg, prices, ledger, 'fixture', 'fixture-not-a-real-key', client=client)
        result = model.generate('prompt', 'm')
    assert result.text == 'fixture code'
    assert result.cost == .0001
    assert ledger.snapshot()['known_cost_usd'] == '0.0001'
    assert len(requests) == 1
    assert requests[0]['provider']['max_price'] == {'prompt':1.0, 'completion':2.0, 'request':0.0}
    assert requests[0]['reasoning']['effort'] == 'none'
