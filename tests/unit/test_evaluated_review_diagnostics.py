"""Safe local shape diagnostics; no card admission or model/numeric execution."""
import copy
import json

import pytest

from sciel.hypothesis import evaluated_study_review as review
from sciel.knowledge_base.catalogs import canonical_json

SECRET = 'UNTRUSTED_MODEL_TEXT_NEVER_ECHO'


def proposal(count=1):
    # These names only exercise the shape decoder, not authenticated fact admission.
    return dict(claims=[dict(fact_id='synthetic:'+str(i), value='unknown') for i in range(count)],
        conjecture='Synthetic unverified prose.', limitations='No measured claim.',
        next_hypothesis_reason='Synthetic prospective reasoning.',
        evidence_refs={field:['synthetic:0'] for field in review.PROSE_FIELDS},
        next_hypothesis=dict(base_source_sha256='a'*64, comparator_source_sha256='b'*64,
            change='Synthetic source edit, not executed.', diagnostics=[], motivating_evidence=['synthetic:0']))


def test_local_boundary_24_accepts_detached_and_25_rejects_with_exact_counts():
    value=proposal(24);saved=copy.deepcopy(value)
    decoded=review.decode_response(canonical_json(value))
    assert decoded==dict(review=value,error=None)
    decoded['review']['claims'][0]['value']='changed'
    assert value==saved
    value=proposal(25)
    assert review.decode_response(canonical_json(value))==dict(review=None,error=dict(
        code='invalid_claim_count',path='claims',details=dict(count=25,minimum=1,maximum=24)))
    with pytest.raises(ValueError,match='one to 24 claims required'):
        review.validate_proposal(value)


@pytest.mark.parametrize('damage,code,path',[
    ('root_type','invalid_object_type','response'),
    ('root_fields','invalid_object_fields','response'),
    ('prose_type','invalid_text_type','conjecture'),
    ('blank_reason','blank_text','next_hypothesis_reason'),
    ('long_conjecture','text_bound','conjecture'),
    ('long_limitations','text_bound','limitations'),
    ('long_reason','text_bound','next_hypothesis_reason'),
    ('refs_type','invalid_object_type','evidence_refs'),
    ('refs_fields','invalid_object_fields','evidence_refs'),
    ('refs_not_array','invalid_reference_count','evidence_refs.conjecture'),
    ('refs_empty','invalid_reference_count','evidence_refs.conjecture'),
    ('refs_many','invalid_reference_count','evidence_refs.conjecture'),
    ('refs_duplicate','duplicate_reference','evidence_refs.conjecture'),
    ('refs_invalid','invalid_reference','evidence_refs.conjecture'),
    ('claims_type','invalid_claims_type','claims'),
    ('claims_empty','invalid_claim_count','claims'),
    ('claim_type','invalid_object_type','claims[0]'),
    ('claim_fields','invalid_object_fields','claims[0]'),
    ('claim_id','invalid_reference','claims[0].fact_id'),
    ('claim_duplicate','duplicate_claim','claims[1].fact_id'),
    ('claim_value_type','invalid_text_type','claims[0].value'),
    ('claim_value_blank','blank_text','claims[0].value'),
    ('claim_value_long','text_bound','claims[0].value'),
    ('target_fields','invalid_target_fields','next_hypothesis'),
    ('target_base','invalid_target_source','next_hypothesis.base_source_sha256'),
    ('target_comparator','invalid_target_source','next_hypothesis.comparator_source_sha256'),
    ('target_change','invalid_target_change','next_hypothesis.change'),
    ('target_refs','invalid_reference_count','next_hypothesis.motivating_evidence'),
    ('target_refs_duplicate','duplicate_reference','next_hypothesis.motivating_evidence'),
    ('target_diagnostics','invalid_target_diagnostics','next_hypothesis.diagnostics'),
    ('target_observable','unsupported_target_observable','next_hypothesis.diagnostics'),
    ('target_direction','unsupported_target_direction','next_hypothesis.diagnostics'),
    ('target_duplicate','duplicate_target_observable','next_hypothesis.diagnostics'),
])
def test_fixed_diagnostic_paths_never_echo_model_values(damage,code,path):
    value=proposal();target=value['next_hypothesis']
    if damage=='root_type':value=[value]
    elif damage=='root_fields':value[SECRET]='secret';del value['next_hypothesis']
    elif damage=='prose_type':value['conjecture']={SECRET:True}
    elif damage=='blank_reason':value['next_hypothesis_reason']=' \t\n'
    elif damage=='long_conjecture':value['conjecture']=SECRET*50
    elif damage=='long_limitations':value['limitations']=SECRET*100
    elif damage=='long_reason':value['next_hypothesis_reason']=SECRET*50
    elif damage=='refs_type':value['evidence_refs']=[]
    elif damage=='refs_fields':value['evidence_refs'][SECRET]=[]
    elif damage=='refs_not_array':value['evidence_refs']['conjecture']=True
    elif damage=='refs_empty':value['evidence_refs']['conjecture']=[]
    elif damage=='refs_many':value['evidence_refs']['conjecture']=[str(i) for i in range(5)]
    elif damage=='refs_duplicate':value['evidence_refs']['conjecture']=[SECRET,SECRET]
    elif damage=='refs_invalid':value['evidence_refs']['conjecture']=[SECRET+'\n']
    elif damage=='claims_type':value['claims']={SECRET:False}
    elif damage=='claims_empty':value['claims']=[]
    elif damage=='claim_type':value['claims'][0]=[SECRET]
    elif damage=='claim_fields':value['claims'][0][SECRET]=None
    elif damage=='claim_id':value['claims'][0]['fact_id']=SECRET+'\x7f'
    elif damage=='claim_duplicate':value['claims']*=2
    elif damage=='claim_value_type':value['claims'][0]['value']={SECRET:False}
    elif damage=='claim_value_blank':value['claims'][0]['value']=' '
    elif damage=='claim_value_long':value['claims'][0]['value']=SECRET*50
    elif damage=='target_fields':target[SECRET]=None
    elif damage=='target_base':target['base_source_sha256']=False
    elif damage=='target_comparator':target['comparator_source_sha256']=SECRET
    elif damage=='target_change':target['change']=SECRET*50
    elif damage=='target_refs':target['motivating_evidence']=[]
    elif damage=='target_refs_duplicate':target['motivating_evidence']=[SECRET,SECRET]
    elif damage=='target_diagnostics':target['diagnostics']=[{SECRET:True}]
    elif damage=='target_observable':target['diagnostics']=[dict(observable=SECRET,direction='increase')]
    elif damage=='target_direction':target['diagnostics']=[dict(observable='restart_total_starts',direction=SECRET)]
    else:target['diagnostics']=[dict(observable='restart_total_starts',direction='increase')]*2
    before=copy.deepcopy(value);decoded=review.decode_response(canonical_json(value))
    assert decoded['review'] is None and decoded['error']['code']==code and decoded['error']['path']==path
    assert SECRET not in canonical_json(decoded) and len(canonical_json(decoded))<=256
    assert value==before
    if 'details' in decoded['error']:
        assert all(type(n) is int for n in decoded['error']['details'].values())


@pytest.mark.parametrize('raw,code',[
    (None,'invalid_response_type'), ('','response_bound'), ('é'*32769,'response_bound'),
    ('\ud800','invalid_utf8'), ('{"'+SECRET+'":','invalid_json'),
    ('{"'+SECRET+'":1,"'+SECRET+'":2}','duplicate_json_key'),
    ('{"'+SECRET+'":NaN}','nonfinite_json'), ('{"'+SECRET+'":Infinity}','nonfinite_json'),
    ('```json\n{}\n```','invalid_json'), ('null','invalid_object_type'),
])
def test_json_transport_rejections_are_specific_bounded_and_do_not_echo(raw,code):
    decoded=review.decode_response(raw)
    assert decoded['review'] is None and decoded['error']['code']==code
    assert decoded['error']['path']=='response'
    assert SECRET not in canonical_json(decoded) and len(canonical_json(decoded))<=256


def test_total_character_bound_still_applies_after_valid_individual_fields():
    value=proposal(24)
    for claim in value['claims']:claim['value']='x'*800
    chars=len(canonical_json(value));assert chars>review.MAX_PROPOSAL_CHARS
    assert review.decode_response(canonical_json(value))==dict(review=None,error=dict(
        code='proposal_bound',path='response',details=dict(chars=chars,maximum=16384)))


def test_null_target_and_unknown_named_fact_stay_shape_only():
    value=proposal();value['next_hypothesis']=None
    assert review.decode_response(canonical_json(value))==dict(review=value,error=None)
    # Membership and exact fact matching remain the later card validator's duty.
    assert value['claims'][0]['fact_id']=='synthetic:0'


def test_oversized_response_is_rejected_without_dropping_or_repairing_claims():
    raw = canonical_json(proposal(25))
    before = json.loads(raw)
    assert len(before['claims']) == 25
    assert review.decode_response(raw) == dict(review=None, error=dict(
        code='invalid_claim_count', path='claims', details=dict(count=25, minimum=1, maximum=24)))
    assert json.loads(raw) == before
