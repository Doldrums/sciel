"""Authored offline HOST facts and synthetic interpretation; no dispatch."""
import copy
import hashlib
import json

import pytest

from sciel.hypothesis import evaluated_study_review as review
from sciel.hypothesis import evaluated_study_targets as targets
from sciel.hypothesis import review_eligibility as eligibility
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
import evaluated_study_fixture as adapter


@pytest.fixture(scope='module',autouse=True)
def no_execution():
    def forbidden(*_args,**_kwargs):pytest.fail('Review attempted runtime/provider/candidate execution')
    with pytest.MonkeyPatch.context() as patch:
        for path in ('subprocess.Popen','sciel.eval.repeat_draw.draw','sciel.eval.repeat_draw.adapter_identity',
            'sciel.campaign.budget.BudgetedLLM.generate','sciel.engine.llm.client.EnsembleLLM.generate'):
            patch.setattr(path,forbidden)
        yield


@pytest.fixture(scope='module')
def bundle():
    value=adapter.build()
    return {key:value[key] for key in ('capsule','author_dependencies','author_authority')}


def proposal_for(bundle, *, target=True):
    author=bundle['capsule']['author']
    return dict(claims=[dict(fact_id='quality:verdict',value='REFUTED'),dict(fact_id='author:kind',value='HOST')],
        conjecture='SYNTHETIC fixture conjecture; no causal assertion is established.',
        limitations='SYNTHETIC fixture limitation: this prior loss does not establish future quality.',
        next_hypothesis_reason='SYNTHETIC fixture only: a prospective test is distinct from the saved loss.',
        evidence_refs={field:['quality:verdict'] for field in review.PROSE_FIELDS},
        next_hypothesis=None if not target else dict(base_source_sha256=author['base_source_sha256'],
            comparator_source_sha256=author['comparator_source_sha256'],change='SYNTHETIC bounded edit proposal, not executed.',
            diagnostics=[],motivating_evidence=['quality:verdict']))


@pytest.fixture(scope='module')
def card(bundle):
    proposal=proposal_for(bundle)
    return review.build_card(proposal,bundle,dict(phase='reflection',reference=1,
        raw_response_sha256=hashlib.sha256(canonical_json(proposal).encode()).hexdigest()))


def test_authored_catalog_preserves_quality_unmeasured_HOST_and_both_matched_roles(bundle):
    catalog=review.source_catalog(bundle);facts=catalog['facts']
    assert facts['quality:verdict']['value']=='REFUTED'
    assert facts['coverage:issued']['value']==4 and facts['coverage:unissued']['value']==2
    assert facts['native:observation_status']['value']=='UNMEASURED'
    assert facts['native:forecast_status']['value']==facts['forecast:paired:status']['value']=='NOT_DECLARED'
    forecasts=[v['value'] for k,v in facts.items() if k.startswith('restart:') and k.endswith(':forecast_status')]
    assert forecasts==['NOT_DECLARED']*6
    assert not any(k.startswith('restart:') and k.endswith((':expected',':verdict')) for k in facts)
    assert any(k.startswith('timing:') and v['value'] is not None for k,v in facts.items())
    matches={k.rsplit(':',1)[0] for k in facts if k.startswith('matched:') and k.endswith(':inputs_byte_identical')}
    assert matches
    for stem in matches:
        assert all(stem+':'+key in facts for key in ('comparator_gain','challenger_gain','local_gain_delta'))
    assert any(k.endswith(':prefix_omitted_count') and v['value']>0 for k,v in facts.items() if v['value'] is not None)
    assert 'proposal_time_seconds' in review.RULES and 'not catalogued' in review.RULES


def test_authored_source_tables_are_lossless_and_all_bodies_reconstruct(bundle):
    catalog=review.source_catalog(bundle)
    payload=json.loads(review._source_block(catalog).split(review.PREFIX)[1]);expanded={}
    for table in payload['fact_tables']:
        for row in table['rows']:
            assert len(row['values'])==len(table['columns'])
            for column,value in zip(table['columns'],row['values'],strict=True):
                key=row['prefix']+':'+column['name'];assert key not in expanded
                expanded[key]=dict(kind=column['kind'],unit=column['unit'],value=value)
    assert expanded==catalog['facts'] and payload['catalog_sha256']==canonical_sha(catalog)
    authored=json.loads(review._authoring_block(bundle,catalog).split(review.AUTHORING_PREFIX)[1])
    assert authored['source_omissions']==[]
    assert authored['interface']==builder.interface_description()
    assert {row['source_sha256'] for row in authored['complete_source_bodies']}==set(bundle['capsule']['source_registry'])
    for row in authored['complete_source_bodies']:
        assert builder.source_for(row['body'])==bundle['capsule']['source_registry'][row['source_sha256']]
    assert len(review._source_block(catalog)+review._authoring_block(bundle,catalog))<90000
    reloaded=review.source_catalog(json.loads(canonical_json(bundle)))
    assert review._source_block(reloaded)==review._source_block(catalog)
    assert review._authoring_block(bundle,reloaded)==review._authoring_block(bundle,catalog)


def test_raw_review_unchanged_and_target_uses_unmeasured_immediate_base(bundle,card):
    proposal=proposal_for(bundle)
    assert card['unverified_interpretation']==proposal
    assert card['origin_experiment_id'] is None
    assert card['resolved_target']['base_source_sha256']==bundle['capsule']['author']['base_source_sha256']
    assert card['resolved_target']['base_source_sha256'] not in (
        card['comparator_source_sha256'],card['challenger_source_sha256'])
    assert targets.validate_target(card['resolved_target'],bundle=bundle)==card['resolved_target']
    assert review.validate_card(card,bundle)==card
    assert card['resolved_target']['quality_gate']=='paired_raw_exact_zero_loss_strict_v1'
    assert card['resolved_target']['author_authority_sha256']==canonical_sha(bundle['author_authority'])
    assert card['evidence_bindings']['target_motivation']==card['resolved_target']['motivating_evidence']


def test_source_roles_bind_actual_edit_parent_and_challenger_beside_each_complete_body(bundle):
    before = copy.deepcopy(bundle)
    authored = json.loads(review.authoring_block(bundle).split(review.AUTHORING_PREFIX)[1])
    author = bundle['capsule']['author']
    assert authored['source_roles'] == dict(
        study_edit_parent=author['base_source_sha256'],
        study_challenger=author['candidate_source_sha256'],
        study_comparator=author['comparator_source_sha256'],
        receiving_root=bundle['capsule']['receiving_root']['source_sha256'])
    by_digest = {row['source_sha256']: row for row in authored['complete_source_bodies']}
    parent, challenger, comparator = (by_digest[author[key]] for key in
        ('base_source_sha256', 'candidate_source_sha256', 'comparator_source_sha256'))
    assert parent['roles'] == ['study_edit_parent']
    assert challenger['roles'] == ['study_challenger']
    assert comparator['roles'] == ['study_comparator', 'receiving_root']
    # The observed R2 confusion: these are different input geometries, already
    # visible in the saved complete bodies; role labels do not rewrite either.
    assert 'np.copy(best_params)' in parent['body']
    assert 'ordinary_params' not in parent['body']
    assert 'np.copy(ordinary_params)' in challenger['body']
    for digest, row in by_digest.items():
        assert hashlib.sha256(builder.source_for(row['body']).encode()).hexdigest() == digest
        assert hashlib.sha256(row['body'].encode()).hexdigest() == row['body_sha256']
    assert bundle == before
    assert 'not a new edit' in review.RULES
    assert 'including unissued or failed slots' in authored['source_role_scope']


@pytest.mark.parametrize('field', ['base_source_sha256', 'candidate_source_sha256', 'comparator_source_sha256'])
def test_public_authoring_rejects_relabelled_sources_without_authenticated_derivation(bundle, field):
    altered = copy.deepcopy(bundle)
    altered['capsule']['author'][field] = 'a' * 64
    with pytest.raises(ValueError):
        review.authoring_block(altered)


def test_null_target_remains_valid_abstention(bundle):
    value=proposal_for(bundle,target=False)
    decoded=review.decode_response(canonical_json(value))
    assert decoded==dict(review=value,error=None)
    card=review.build_card(value,bundle,dict(phase='reflection',reference=1,raw_response_sha256='a'*64))
    assert card['resolved_target'] is None and card['evidence_bindings']['target_motivation'] is None


def test_exact_ratio_claim_is_never_float_coerced_or_rounded(bundle):
    catalog=review.source_catalog(bundle)
    ident,entry=next((key,value) for key,value in catalog['facts'].items()
        if value['kind']=='exact_ratio' and value['value']['numerator']!=0)
    proposal=proposal_for(bundle,target=False)
    proposal['claims']=[dict(fact_id=ident,value=canonical_json(entry['value']))]
    writer=dict(phase='reflection',reference=1,raw_response_sha256='a'*64)
    card=review.build_card(proposal,bundle,writer)
    assert card['evidence_bindings']['claims'][0]['value']==entry['value']
    proposal['claims'][0]['value']=str(entry['value']['numerator']/entry['value']['denominator'])
    with pytest.raises(ValueError,match='exact host fact'):review.build_card(proposal,bundle,writer)


@pytest.mark.parametrize('spacing', [False, True])
def test_ratio_member_order_and_whitespace_preserve_synthetic_authored_claim(bundle, spacing):
    # This one token matches the rejected paid response's representational issue.
    # The surrounding review is an explicitly synthetic fixture, not a replay or repair.
    ident = 'pair:3084817015:certified_delta'
    raw = '{"numerator":219306857579,"denominator":72057594037927936}'
    if spacing:
        raw = ' \n { "numerator" : 219306857579,\n "denominator" : 72057594037927936 } \t'
    proposal = proposal_for(bundle, target=False)
    proposal['claims'] = [dict(fact_id=ident, value=raw)]
    before = copy.deepcopy(proposal)
    writer = dict(phase='reflection', reference=1, raw_response_sha256=canonical_sha(proposal))
    card = review.build_card(proposal, bundle, writer)
    assert proposal == before == card['unverified_interpretation']
    assert card['unverified_interpretation']['claims'][0]['value'] == raw
    assert card['evidence_bindings']['claims'][0]['value'] == dict(
        denominator=72057594037927936, numerator=219306857579)
    assert review.validate_card(json.loads(canonical_json(card)), bundle) == card


@pytest.mark.parametrize('raw', [
    '{"numerator":3,"denominator":7,"numerator":3}',
    '{"numerator":2,"denominator":7,"numerator":3}',
    '{"numerator":3,"denominator":7,"denominator":7}',
    '{"numerator":3,"denominator":7,"extra":0}',
    '{"numerator":3}', '{"denominator":7}',
    '{"numerator":3.0,"denominator":7}', '{"numerator":3,"denominator":7.0}',
    '{"numerator":3e0,"denominator":7}', '{"numerator":"3","denominator":7}',
    '{"numerator":6,"denominator":14}', '{"numerator":-3,"denominator":-7}',
    '{"numerator":3,"denominator":0}', '{"numerator":4,"denominator":7}',
    '{"numerator":NaN,"denominator":7}', '{"numerator":Infinity,"denominator":7}',
    '[3,7]', '0.42857142857142855', 'null',
    '{"numerator":3,"denominator":7} trailing',
    '```json\n{"numerator":3,"denominator":7}\n```',
])
def test_ratio_parser_never_repairs_or_normalizes_nonliteral_exact_fields(raw):
    fact = dict(kind='exact_ratio', unit='sum_radii', value=dict(numerator=3, denominator=7))
    before = copy.deepcopy(fact)
    assert not review._claim_matches(raw, fact)
    assert fact == before


@pytest.mark.parametrize('raw,expected', [
    ('{"numerator":true,"denominator":7}', dict(numerator=1, denominator=7)),
    ('{"numerator":false,"denominator":7}', dict(numerator=0, denominator=7)),
    ('{"numerator":3,"denominator":true}', dict(numerator=3, denominator=1)),
])
def test_ratio_boolean_numeric_equality_is_insufficient(raw, expected):
    assert json.loads(raw) == expected  # The exact-int guard must reject this Python equality.
    assert not review._claim_matches(raw, dict(kind='exact_ratio', value=expected))


@pytest.mark.parametrize('raw,fact', [
    ('0', dict(kind='number', value=0.0)),
    ('1.0', dict(kind='integer', value=1)),
    ('true ', dict(kind='boolean', value=True)),
    ('null', dict(kind='unknown', value=None)),
    ('REFUTED ', dict(kind='token', value='REFUTED')),
])
def test_only_exact_ratios_gain_representation_flexibility(raw, fact):
    assert not review._claim_matches(raw, fact)
    assert review._claim_matches(review._token(fact['value']), fact)


def test_claim_mismatch_diagnostic_names_only_bounded_escaped_fact_id(bundle, card):
    proposal = proposal_for(bundle, target=False)
    proposal['claims'] = [dict(fact_id='pair:3084817015:certified_delta', value='private model text')]
    with pytest.raises(ValueError) as caught:
        review._card(proposal, bundle, card['host_facts'], card['writer'])
    assert str(caught.value).endswith('"pair:3084817015:certified_delta"')
    assert 'private model text' not in str(caught.value)
    proposal['claims'][0]['fact_id'] = 'unknown:"\\fact'
    review.validate_proposal(proposal)
    with pytest.raises(ValueError) as caught:
        review._card(proposal, bundle, card['host_facts'], card['writer'])
    assert str(caught.value).endswith(canonical_json('unknown:"\\fact'))
    assert len(str(caught.value)) < 150 and '\n' not in str(caught.value)


def test_authored_coverage_distinguishes_draw_slots_and_seed_pairs(bundle):
    facts = review.source_catalog(bundle)['facts']
    for key, value in dict(planned=6, issued=4, terminal=4, unknown=0, unissued=2).items():
        assert facts['coverage:'+key] == dict(kind='integer', unit='draw_slots', value=value)
    for key, value in dict(planned_pairs=3, issued_pairs=2, partially_issued_pairs=0,
                           completed_pairs=2, valid_pairs=2).items():
        assert facts['coverage:'+key] == dict(kind='integer', unit='seed_pairs', value=value)
    assert 'six slots are three seed pairs' in review.RULES
    assert 'different new source edit can define a new' in review.RULES


def test_pair_coverage_uses_both_role_statuses_and_does_not_credit_unknown_as_complete():
    # All terminal outcomes count as completion; only two valid receipts count as valid.
    # A known slot plus unknown is fully issued but not completed, unlike one unissued slot.
    states = [('valid','valid'), ('candidate_failure','infrastructure'),
              ('valid','unknown'), ('valid','not_dispatched'), ('not_dispatched','not_dispatched')]
    pairs = [dict(comparator_status=a, challenger_status=b) for a,b in states]
    assert review._pair_coverage(pairs) == dict(planned_pairs=5, issued_pairs=3,
        partially_issued_pairs=1, completed_pairs=2, valid_pairs=1)


@pytest.mark.parametrize('damage',['fence','duplicate','extra','nan','list','blank_reason','long_reason','long_claim',
    'duplicate_claim','duplicate_ref','empty_ref','metadata','lessons','missing_target','bad_diagnostic','oversized','invalid_utf8'])
def test_bare_decoder_rejects_without_repair_or_echo(bundle,damage):
    value=proposal_for(bundle);raw=None
    if damage=='fence':raw='```json\n'+canonical_json(value)+'\n```'
    elif damage=='duplicate':raw=canonical_json(value)[:-1]+',"next_hypothesis":null}'
    elif damage=='extra':raw=canonical_json(value)+' trailing secret model prose'
    elif damage=='nan':raw=canonical_json(value).replace('"REFUTED"','NaN')
    elif damage=='list':raw=canonical_json([value])
    elif damage=='blank_reason':value['next_hypothesis_reason']=' '
    elif damage=='long_reason':value['next_hypothesis_reason']='x'*1025
    elif damage=='long_claim':value['claims'][0]['value']='x'*1025
    elif damage=='duplicate_claim':value['claims'].append(copy.deepcopy(value['claims'][0]))
    elif damage=='duplicate_ref':value['evidence_refs']['limitations']*=2
    elif damage=='empty_ref':value['evidence_refs']['limitations']=[]
    elif damage=='metadata':value['comparison_sha256']='a'*64
    elif damage=='lessons':value['lessons']=[]
    elif damage=='missing_target':value.pop('next_hypothesis')
    elif damage=='bad_diagnostic':value['next_hypothesis']['diagnostics']=[dict(observable='causal_gain',direction='increase')]
    elif damage=='oversized':raw='x'*65537
    else:raw='\ud800'
    before=copy.deepcopy(value);decoded=review.decode_response(raw if raw is not None else canonical_json(value))
    assert decoded['review'] is None and {'code','path'} <= set(decoded['error']) <= {'code','path','details'}
    assert value==before and len(canonical_json(decoded))<=256


@pytest.mark.parametrize('damage',['claim_sign','claim_metadata','reference','foreign_source','weak_gate',
    'resolved_target','host_fact','bindings','writer_extra'])
def test_rehashed_cards_cannot_bypass_exact_fact_target_or_shape_checks(bundle,card,damage):
    altered=copy.deepcopy(card)
    if damage=='claim_sign':altered['unverified_interpretation']['claims'][0]['value']='CONFIRMED'
    elif damage=='claim_metadata':altered['unverified_interpretation']['claims'][0]['fact_id']='comparison_id'
    elif damage=='reference':altered['unverified_interpretation']['evidence_refs']['conjecture']=['unknown:fact']
    elif damage=='foreign_source':altered['unverified_interpretation']['next_hypothesis']['base_source_sha256']='a'*64
    elif damage=='weak_gate':altered['resolved_target']['quality_gate']='epsilon'
    elif damage=='resolved_target':altered['resolved_target']['author_authority_sha256']='a'*64
    elif damage=='host_fact':altered['host_facts']['facts']['quality:verdict']['value']='CONFIRMED'
    elif damage=='bindings':altered['evidence_bindings']['conjecture'][0]['value']='CONFIRMED'
    else:altered['writer']['provider_request_id']='invented'
    with pytest.raises(ValueError):review.validate_card(altered,bundle)


def test_new_card_uses_exact_existing_text_eligibility_without_inheriting_approval(bundle,card):
    decision=dict(schema=eligibility.DECISION_SCHEMA,scope=eligibility.SCOPE,decision='accepted',
        card_sha256=canonical_sha(card),raw_response_sha256=card['writer']['raw_response_sha256'],
        **{key:card[key] for key in ('comparison_id','comparison_sha256','comparator_source_sha256','challenger_source_sha256')},
        policy_sha256='a'*64,audit_sha256='b'*64,criteria_sha256='c'*64,issuer='synthetic host test')
    policy=dict(schema=eligibility.POLICY_SCHEMA,decisions=[decision])
    assert eligibility.decision_for(card,policy,policy_sha256='a'*64)['status']=='accepted'
    assert eligibility.decision_for(card,policy,policy_sha256='d'*64)['status']=='stale'
    proposal=proposal_for(bundle);proposal['next_hypothesis_reason']+=' Different authored rationale.'
    changed=review.build_card(proposal,bundle,card['writer'])
    assert eligibility.decision_for(changed,policy,policy_sha256='a'*64)['status']=='pending'


def test_static_small_wire_has_no_component_enums_or_literal_provenance():
    wire=review.response_format();encoded=canonical_json(wire)
    assert len(encoded)<7000
    assert wire['json_schema']['schema']['type']=='object'
    assert set(wire['json_schema']['schema']['required'])==set(review.MODEL_FIELDS)
    assert 'study_sha256' not in encoded and 'quality_gate' not in encoded and 'comparison_id' not in encoded
    assert 'The caller must verify' in targets.instruction()
    assert 'quality_gate must be' not in targets.instruction()
    assert 'at least one exact certified delta must be > 0' in targets.instruction()
