"""Selected model metadata must permit the fixed effort=none before any I/O."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import BudgetedLLM, validate_llm_config

CFG={'models':['m'],'weights':[1],'max_tokens':100,'api_base':'https://openrouter.ai/api/v1'}
BASE={'id':'m','pricing':{'prompt':'.0000003','completion':'.0000025'}}


def test_mandatory_reasoning_rejected_before_client_or_reservation(monkeypatch):
    pricing={'models':[dict(BASE,reasoning={'mandatory':True,'supported_efforts':['low','high']})]}
    saved=deepcopy(pricing)
    cfg=dict(CFG,max_tokens=4096)
    constructor=Mock(side_effect=AssertionError('must not create SDK client'))
    monkeypatch.setattr('openai.OpenAI',constructor)
    ledger=Mock()
    with pytest.raises(ValueError,match='incompatible with fixed reasoning effort none'):
        BudgetedLLM(cfg,pricing,ledger,'fixture','unused')
    assert not constructor.mock_calls and not ledger.mock_calls and pricing==saved


@pytest.mark.parametrize('reasoning',[{}, {'mandatory':False}, {'mandatory':False,'supported_efforts':None},
    {'mandatory':False,'supported_efforts':['none','low']}])
def test_optional_selected_model_permits_original_policy(reasoning):
    row=dict(BASE,reasoning=reasoning)
    assert validate_llm_config(CFG,{'models':[row]})['m']['prompt']>0


def test_legacy_absence_and_unselected_mandatory_model_preserved():
    other=dict(BASE,id='unselected',reasoning={'mandatory':True})
    assert 'm' in validate_llm_config(CFG,{'models':[BASE,other]})


@pytest.mark.parametrize('reasoning',[None,False,[],{'mandatory':1},{'mandatory':'false'},
    {'supported_efforts':'none'},{'supported_efforts':[]},{'supported_efforts':[False]},
    {'supported_efforts':['']}, {'supported_efforts':['low','high']},
    {'mandatory':True,'supported_efforts':None}])
def test_malformed_or_incompatible_explicit_selected_metadata_stops(reasoning):
    sdk,ledger=Mock(),Mock()
    with pytest.raises(ValueError,match='reasoning'):
        BudgetedLLM(CFG,{'models':[dict(BASE,reasoning=reasoning)]},ledger,'fixture','unused',sdk)
    assert not sdk.mock_calls and not ledger.mock_calls
