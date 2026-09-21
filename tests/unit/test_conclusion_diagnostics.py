"""Precise rejection metadata without changing conclusion acceptance or evidence."""
from copy import deepcopy
import json

import pytest

from sciel.hypothesis.mechanism_review import (MAX_REVIEW_CHARS, MechanismReviewError,
                                              validate_binding, validate_review)
from sciel.hypothesis.parsing import decode_conclusion, parse_conclusion


def entry(**changes):
    value={'claim_quote':'The update might improve quality.',
           'source_quotes':[{'field':'controlled_change.target_body','text':'return value\n'}],
           'assessment':'unknown','direct_effect':'The code returns a value.',
           'untested_downstream':'Later quality is not established.'}
    value.update(changes);return value


def conclusion(entries=None):
    return {'verdict':'REFUTED','explanation':'Measured loss; mechanism unverified.','lessons':[],
            'mechanism_review':[entry()] if entries is None else entries}


def dump(value):return json.dumps(value,ensure_ascii=False)


def test_sole_json_fence_is_accepted_with_identical_literal_payload():
    raw=conclusion();text='```json\n'+dump(raw)+'\n```'
    result=decode_conclusion(text,require_mechanism=True)
    assert result['error'] is None and result['conclusion'].to_dict()==raw
    assert parse_conclusion(text,require_mechanism=True)==result['conclusion']


def test_r2_sized_quote_reports_first_precise_limit_not_fence_or_aggregate():
    quote='x'*1986
    raw=conclusion([entry(source_quotes=[{'field':'controlled_change.target_body','text':quote}])]+[entry()]*3)
    text='```json\n'+dump(raw)+'\n```';before=deepcopy(raw)
    result=decode_conclusion(text,require_mechanism=True)
    assert result=={'conclusion':None,'error':{'code':'text_too_long',
        'path':'mechanism_review[0].source_quotes[0].text','actual':1986,'maximum':512}}
    assert parse_conclusion(text,require_mechanism=True) is None and raw==before
    assert quote not in dump(result) and '```' not in dump(result)


def test_complete_review_budget_reports_actual_canonical_escaped_length():
    raw=conclusion([entry(direct_effect='界'*400)]*4)
    canonical=json.dumps(raw['mechanism_review'],ensure_ascii=True,sort_keys=True,separators=(',',':'))
    assert len(canonical)>MAX_REVIEW_CHARS and all(len(e['direct_effect'])<=512 for e in raw['mechanism_review'])
    error=decode_conclusion(dump(raw),require_mechanism=True)['error']
    assert error=={'code':'review_too_long','path':'mechanism_review','actual':len(canonical),'maximum':4096}
    assert '界' not in dump(error)


@pytest.mark.parametrize('field',['claim_quote','direct_effect','untested_downstream'])
def test_each_entry_text_limit_has_an_exact_path(field):
    raw=conclusion([entry(**{field:'x'*513})])
    assert decode_conclusion(dump(raw))['error']=={'code':'text_too_long','path':f'mechanism_review[0].{field}','actual':513,'maximum':512}


def test_wrong_declared_field_never_retags_a_real_quote_from_another_field():
    body='if active:\n    return value\n';compiled='def helper():\n    if active:\n        return value\n'
    data={'hypothesis':{'text':'The update might improve quality.'},
          'controlled_change':{'target_body':body,'after':{'text':compiled}}}
    copied='if active:\n        return value\n'
    assert copied in compiled and copied not in body
    raw=conclusion([entry(source_quotes=[{'field':'controlled_change.target_body','text':copied}])])
    decoded=decode_conclusion(dump(raw),require_mechanism=True)
    assert decoded['error'] is None  # Structurally valid; evidence binding owns this rejection.
    before=deepcopy((data,raw))
    with pytest.raises(MechanismReviewError) as caught:validate_binding(decoded['conclusion'].mechanism_review,data)
    assert caught.value.diagnostic=={'code':'source_quote_mismatch','path':'mechanism_review[0].source_quotes[0].text'}
    assert (data,raw)==before and copied not in dump(caught.value.diagnostic)
    changed=caught.value.diagnostic;changed['path']='caller mutation'
    assert caught.value.diagnostic['path']=='mechanism_review[0].source_quotes[0].text'


def test_claim_and_missing_field_binding_have_stable_non_echoing_diagnostics():
    data={'hypothesis':{'text':'Another original claim.'}}
    with pytest.raises(MechanismReviewError) as caught:validate_binding([entry()],data)
    assert caught.value.diagnostic=={'code':'claim_quote_mismatch','path':'mechanism_review[0].claim_quote'}
    data['hypothesis']['text']='The update might improve quality.'
    with pytest.raises(MechanismReviewError) as caught:validate_binding([entry()],data)
    assert caught.value.diagnostic=={'code':'source_field_unavailable','path':'mechanism_review[0].source_quotes[0].field'}
    assert 'Another original claim' not in dump(caught.value.diagnostic)


def test_untrusted_field_names_are_never_echoed_and_old_valueerror_contract_remains():
    untrusted='UNTRUSTED\n'*1000
    raw=conclusion([entry(source_quotes=[{'field':untrusted,'text':'x'}])])
    result=decode_conclusion(dump(raw))
    assert result['error']=={'code':'invalid_source_field','path':'mechanism_review[0].source_quotes[0].field'}
    assert len(dump(result['error']))<160 and untrusted not in dump(result)
    with pytest.raises(ValueError,match='invalid mechanism source quote'):validate_review(raw['mechanism_review'])


def test_legacy_api_shapes_and_transport_rejections_are_preserved():
    old=conclusion();del old['mechanism_review']
    assert parse_conclusion(dump(old)).to_dict()==old
    assert decode_conclusion(dump(old),require_mechanism=True)=={'conclusion':None,'error':{'code':'missing_mechanism_review','path':'mechanism_review'}}
    for text in ('not json','```python\n'+dump(conclusion())+'\n```',dump(conclusion())+' trailing'):
        assert decode_conclusion(text)['error']=={'code':'invalid_conclusion_transport','path':'response'}
        assert parse_conclusion(text) is None
    extra={**conclusion(),'rejected_raw_text':'do not echo'}
    assert decode_conclusion(dump(extra))['error']=={'code':'invalid_conclusion_fields','path':'conclusion'}
