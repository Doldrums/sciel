"""Prospective paired forecasts never masquerade as ordinary native intervals."""
import copy
from dataclasses import FrozenInstanceError
import json

import pytest

from sciel.hypothesis import component_generation_hypothesis as wire
from sciel.hypothesis import restart_predictions
from sciel.hypothesis.parsing import parse_hypothesis_value


TARGET = 'a' * 64


def payload(body='return None\n'):
    return dict(target_sha256=TARGET, body=body, abstention_reason=None, hypothesis=dict(
        text='  Prospective fixture: change one restart decision.  ', reasoning='Unmeasured conditional test.',
        confidence=.5, paired_forecast=dict(schema=wire.FORECAST_SCHEMA,
            target_sha256=TARGET, expected_gate_pass=False),
        restart_predictions=dict(schema=restart_predictions.SCHEMA,
            expected=dict(zip(restart_predictions.NAMES, (True, False, False))))))


def test_detached_immutable_DTO_preserves_prose_and_excludes_native_fields():
    original=payload(); decoded=wire.decode_response(json.dumps(original),target_sha256=TARGET)
    assert decoded['status']=='proposed' and decoded['error'] is None
    dto=decoded['hypothesis']; projected=dto.to_dict()
    assert projected==original['hypothesis'] and dto.text.startswith('  ')
    with pytest.raises(FrozenInstanceError):dto.text='mutated'
    projected['restart_predictions']['expected']['result_changed']=True
    assert dto.to_dict()==original['hypothesis']
    assert parse_hypothesis_value(original['hypothesis']) is None
    assert all(k not in projected for k in ('predicted_low','predicted_high','parent_score'))


@pytest.mark.parametrize('mutation', ['native_low','native_high','parent_score','missing_restart',
    'missing_paired','target','forecast_bool','confidence_bool','confidence_nan','extra',
    'empty_text','long_text','long_reasoning','restart_bool','restart_extra','body_null','hyp_null','both_modes'])
def test_exact_new_wire_rejects_invented_or_ambiguous_forecasts(mutation):
    value=payload();h=value['hypothesis']
    if mutation in ('native_low','native_high','parent_score'):h[mutation]=0
    elif mutation=='missing_restart':h.pop('restart_predictions')
    elif mutation=='missing_paired':h.pop('paired_forecast')
    elif mutation=='target':h['paired_forecast']['target_sha256']='b'*64
    elif mutation=='forecast_bool':h['paired_forecast']['expected_gate_pass']=1
    elif mutation=='confidence_bool':h['confidence']=True
    elif mutation=='confidence_nan':h['confidence']=float('nan')
    elif mutation=='extra':value['extra']='forbidden'
    elif mutation=='empty_text':h['text']=' '
    elif mutation=='long_text':h['text']='x'*2049
    elif mutation=='long_reasoning':h['reasoning']='x'*4097
    elif mutation=='restart_bool':h['restart_predictions']['expected']['result_changed']=0
    elif mutation=='restart_extra':h['restart_predictions']['expected']['native_gain']=True
    elif mutation=='body_null':value['body']=None
    elif mutation=='hyp_null':value['hypothesis']=None
    else:value['abstention_reason']='cannot'
    result=wire.decode_response(json.dumps(value),target_sha256=TARGET)
    assert result['status']=='invalid' and result['body'] is result['hypothesis'] is None
    assert set(result['error'])=={'code','path'}


@pytest.mark.parametrize('raw',[None,{},'[]','null','NaN','```json\n{}\n```','{} prose',
    '{"target_sha256":"'+TARGET+'","target_sha256":"'+TARGET+'"}',
    ' '*32769,'"\ud800"','['*2000])
def test_decoder_never_repairs_duplicate_fences_nonJSON_or_unbounded_input(raw):
    result=wire.decode_response(raw,target_sha256=TARGET)
    assert result['status']=='invalid' and result['error'] is not None


def test_null_is_explicit_bounded_abstention_and_target_binding_still_required():
    value=dict(target_sha256=TARGET,body=None,hypothesis=None,abstention_reason=' No implementation justified. ')
    result=wire.decode_response(json.dumps(value),target_sha256=TARGET)
    assert result['status']=='abstained' and result['abstention_reason']==value['abstention_reason']
    for reason in ('', ' ', 'x'*1025, None):
        value['abstention_reason']=reason
        assert wire.decode_response(json.dumps(value),target_sha256=TARGET)['status']=='invalid'
    value['abstention_reason']='No';value['target_sha256']='b'*64
    assert wire.decode_response(json.dumps(value),target_sha256=TARGET)['error']['code']=='target_binding'


def test_wire_is_detached_strict_closed_and_both_forecasts_bind_target():
    a=wire.response_format(TARGET);b=copy.deepcopy(a)
    schema=a['json_schema']['schema'];h=schema['properties']['hypothesis']
    assert schema['additionalProperties'] is False and set(schema['required'])==set(schema['properties'])
    assert h['additionalProperties'] is False and h['type']==['object','null']
    assert h['properties']['paired_forecast']['properties']['target_sha256']['enum']==[TARGET]
    a['json_schema']['name']='changed'
    assert wire.response_format(TARGET)==b
