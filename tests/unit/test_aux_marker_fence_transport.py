"""Explicit future auxiliary transport; no model, candidate or runtime execution."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.hypothesis import parsing
from sciel.hypothesis.mechanism_review import validate_binding


@pytest.mark.parametrize('parser,marker,payload',[
    (parsing.parse_conclusion,'CONCLUSION_JSON',dict(verdict='REFUTED',explanation='Observed.',lessons=[])),
    (parsing.parse_reflection,'REFLECTION_JSON',dict(lessons=[],retire_ids=[])),
])
@pytest.mark.parametrize('newline,outer,gap',[('\n','','\n'),('\r\n',' \t\n','\r\n\r\n'),('\n','\t',' ')])
def test_exact_marker_and_sole_fence_preserve_payload(parser,marker,payload,newline,outer,gap):
    raw=json.dumps(payload,indent=2).replace('\n',newline)
    response=outer+marker+':'+gap+'```json'+newline+raw+newline+'```'+outer
    assert parser(response)==parser(marker+':'+raw)
    assert parsing.AUX_TRANSPORT=='sciel-aux-json-transport/3'


@pytest.mark.parametrize('parser,marker,payload',[
    (parsing.parse_conclusion,'CONCLUSION_JSON',dict(verdict='REFUTED',explanation='Observed.',lessons=[])),
    (parsing.parse_reflection,'REFLECTION_JSON',dict(lessons=[],retire_ids=[])),
])
@pytest.mark.parametrize('template',[
    'Prose\n{marker}:\n```json\n{raw}\n```',
    '{marker}:\n```json\n{raw}\n```\nProse',
    'OTHER_JSON:\n```json\n{raw}\n```',
    '{marker}:\n{marker}:\n```json\n{raw}\n```',
    '{marker}:\n```json\n{marker}: {raw}\n```',
    '{marker}:\n```json\n{raw}\n```\n{marker}: {raw}',
    '{marker}:\n```json\n{raw}\n```\n```json\n{raw}\n```',
    '{marker}:\n```json\n{raw}\n{raw}\n```',
    '{marker}:\n```JSON\n{raw}\n```',
    '{marker}:\n```python\n{raw}\n```',
    '{marker}:\n```json \n{raw}\n```',
    '{marker}:\n```json\n{raw}\n````',
    '{marker}:\n```json\n[{raw}]\n```',
    '{marker}:\n~~~json\n{raw}\n~~~',
])
def test_marked_fence_does_not_repair_ambiguous_envelopes(parser,marker,payload,template):
    assert parser(template.format(marker=marker,raw=json.dumps(payload))) is None


@pytest.mark.parametrize('parser,marker,raw',[
    (parsing.parse_conclusion,'CONCLUSION_JSON','{"verdict":"REFUTED","explanation":"Observed.","lessons":[],"lessons":[]}'),
    (parsing.parse_conclusion,'CONCLUSION_JSON','{"verdict":"REFUTED","explanation":"Observed.","lessons":[],"extra":1}'),
    (parsing.parse_reflection,'REFLECTION_JSON','{"lessons":NaN,"retire_ids":[]}'),
    (parsing.parse_reflection,'REFLECTION_JSON','{"lessons":[1e9999],"retire_ids":[]}'),
    (parsing.parse_reflection,'REFLECTION_JSON','{"lessons":[],"retire_ids":[],"extra":0}'),
    (parsing.parse_reflection,'REFLECTION_JSON','{"lessons":'+('['*2000)+'0'+(']'*2000)+',"retire_ids":[]}'),
    (parsing.parse_reflection,'REFLECTION_JSON','{"extra":"'+('x'*parsing.MAX_BLOCK_CHARS)+'"}'),
])
def test_marked_fence_retains_content_limits(parser,marker,raw):
    assert parser(marker+':\n```json\n'+raw+'\n```') is None


def test_response_limit_is_checked_before_unwrapping_and_generation_is_unchanged():
    raw='REFLECTION_JSON:\n```json\n{"lessons":[],"retire_ids":[]}\n```'
    assert parsing.parse_reflection(' '*parsing.MAX_RESPONSE_CHARS+raw) is None
    hypothesis=dict(hypothesis='Claim',reasoning='Reason',predicted_low=0.,predicted_high=1.,confidence=.5)
    assert parsing.parse_hypothesis('HYPOTHESIS_JSON:\n```json\n'+json.dumps(hypothesis)+'\n```') is None


def test_exact_historically_rejected_reply_is_only_future_static_compatibility():
    path=Path(__file__).resolve().parents[1]/'fixtures/aux_marker_fence_conclusion_r2.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='898abb8461d639640dcab9b58e915084eddd2e8b7ae87643d46503aff868f833'
    fixture=json.loads(path.read_text());raw=fixture['response']
    assert hashlib.sha256(raw.encode()).hexdigest()==fixture['response_sha256']=='29d559e12c5d894fde853753fbd4cff156aab02d75e7aa535308c69f4c924f5f'
    assert fixture['historical_conclusion_status']=='invalid'
    result=parsing.decode_conclusion(raw,require_mechanism=True)
    assert result['error'] is None and result['conclusion'].verdict=='REFUTED'
    assert result['conclusion'].lessons[0]['evidence_ids']==['r000001a1']
    validate_binding(result['conclusion'].mechanism_review,fixture)
    changed=copy.deepcopy(fixture);changed['controlled_change']['target_body']='different inert source'
    with pytest.raises(ValueError):validate_binding(result['conclusion'].mechanism_review,changed)
    assert raw==fixture['response']  # No persisted repair, admission or historical reclassification.


def test_previous_aux_transport_checkpoint_is_rejected():
    from sciel.config import apply_defaults
    from sciel.engine.research import ResearchController
    config=apply_defaults({'research':{'enabled':True}})['research']
    controller=ResearchController(config,SimpleNamespace(models=['fixture'],max_retries=1),
        SimpleNamespace(research=lambda **kw:pytest.fail('unexpected event')),lambda _:pytest.fail('charge'))
    state=controller.state();assert state['presentation']['auxiliary_transport']==parsing.AUX_TRANSPORT
    controller.load_state(state)
    state['presentation']['auxiliary_transport']='sciel-aux-json-transport/2'
    with pytest.raises(ValueError,match='presentation mismatch'):controller.load_state(state)


def test_exact_r2_reflection_with_post_fence_rationale_remains_invalid():
    path=Path(__file__).resolve().parents[1]/'fixtures/aux_marker_fence_reflection_r2.txt'
    raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='6bc0310d1cc30475b91b6e7c4dd9d718482624bfb61c441c7216362cb40c4065'
    assert parsing.parse_reflection(raw.decode()) is None
