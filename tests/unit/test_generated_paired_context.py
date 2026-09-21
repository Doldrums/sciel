"""Presentation policy tests with an explicitly inert, already-derived study.

Core admission and actual archived-data integration are tested separately. These
tests isolate whole-study budgets, literal authorship and no favorable omission;
no provider, candidate, evaluator, archived Python or optimizer is executed.
"""
import copy
import hashlib
import json

import pytest

from sciel.knowledge_base import generated_paired_context as context
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def study(monkeypatch):
    sources = ['inert control source', 'inert off source', 'inert on source']
    hashes = list(map(sha, sources))
    proposals = []
    for index, name in enumerate(('off', 'on'), 1):
        hypothesis = dict(text='UNVERIFIED-' + name, reasoning='UNVERIFIED-REASON-' + name,
            predicted_low=-.1, predicted_high=.1, confidence=.3,
            restart_predictions=dict(schema='sciel-restart-predictions/1',
                expected=dict(effective_locations_changed=True, result_changed=False,
                              accepted_interval_gain=name == 'on')))
        proposals.append(dict(id='generated:' + sha(name),
            provenance=dict(manifest_sha256=sha('manifest'), run_id='fixture',
                            request_key=name, research_id='r000003a1'),
            request_sha256=sha('request-' + name), response_sha256=sha('response-' + name),
            provider_response_sha256=sha('provider-' + name),
            hypothesis=hypothesis, body='body-' + name, source=sources[index],
            source_sha256=hashes[index], parent_source=sources[0], proof={'fixture': name},
            native_observation=dict(score=None, actual_delta=None, evaluation=None,
                forecast_verdict='INCONCLUSIVE', forecast_scored=False,
                reason='native target unmeasured')))
    receipts = {source: dict(status='valid', fixture=source) for source in hashes}
    comparisons = []
    for name, old, new in [('on_off', 1, 2), ('on_control', 0, 2), ('off_control', 0, 1)]:
        pairs = [dict(seed=seed, comparator=receipts[hashes[old]] if seed == 10 else None,
                      challenger=receipts[hashes[new]] if seed == 10 else None)
                 for seed in (10, 20, 30)]
        group = dict(id='comparison:' + sha(name), branch=name, origin_experiment_id=None,
            comparator_source_sha256=hashes[old], challenger_source_sha256=hashes[new],
            adapter_identity=dict(certificate_policy={'fixture': 'radius decrease'},
                                  certificate_policy_sha256=sha('certificate')),
            limits={'candidate_deadline_seconds': 30}, pairs=pairs,
            source_pair=dict(comparator=sources[old], challenger=sources[new]),
            derived={'outcome': 'incomplete'})
        quality = dict(gate='paired_raw_exact_zero_loss_strict_v1',
            verdict='INCONCLUSIVE' if name == 'on_off' else 'REFUTED',
            reason='incomplete_quality' if name == 'on_off' else 'known_paired_loss',
            pairs=[dict(seed=seed, raw_delta=(0. if name == 'on_off' else -1e-7) if seed == 10 else None,
                        certified_delta=dict(numerator=0 if name == 'on_off' else 1, denominator=1000000)
                        if seed == 10 else None) for seed in (10, 20, 30)])
        forecasts = []
        for proposal in proposals:
            for role, source_index in (('comparator', old), ('challenger', new)):
                if proposal['source_sha256'] != hashes[source_index]:
                    continue
                is_on = proposal['provenance']['request_key'] == 'on'
                observed = dict(effective_locations_changed=True, result_changed=None if is_on else True,
                                accepted_interval_gain=False)
                result = dict(schema='sciel-restart-prediction-result/1',
                    result_sha256=sha('result-' + str(is_on)), status='available', reason=None,
                    spec_sha256=canonical_sha(proposal['hypothesis']['restart_predictions']),
                    observation_sha256=sha('observation-' + str(is_on)),
                    receipt_sha256=canonical_sha(receipts[proposal['source_sha256']]),
                    expected=proposal['hypothesis']['restart_predictions']['expected'], observed=observed,
                    verdicts=dict(effective_locations_changed='CONFIRMED',
                        result_changed='INCONCLUSIVE' if is_on else 'REFUTED',
                        accepted_interval_gain='REFUTED' if is_on else 'CONFIRMED'),
                    coverage=dict(retained=16, accepted=7, unknown_acceptance=0, omitted=424 if is_on else 428),
                    columns=['inert'], cases=[['per-case-details']] * 7)
                forecasts.append(dict(proposal_id=proposal['id'], role=role,
                    rows=[dict(seed=seed, receipt_sha256=result['receipt_sha256'] if seed == 10 else None,
                        result=result if seed == 10 else None,
                        unavailable_reason=None if seed == 10 else 'receipt_not_dispatched')
                        for seed in (10, 20, 30)]))
        comparisons.append(dict(group=group, quality=quality, restart_predictions=forecasts))
    facts = dict(root_parent=dict(id='p1', source_sha256=hashes[0]),
                 proposals=proposals, comparisons=comparisons, scope='INERT fixture, no execution.')
    capsule = dict(schema='inert-derived-context-fixture', root_parent=facts['root_parent'],
        profile='paired_restart_trace/2', metric={'task_name': 'inert'}, policy={'fixture': True},
        proposals=[dict(id=p['id'], request={'messages': [{'content': 'inert prompt ' + str(i)}]})
                   for i, p in enumerate(proposals)])
    bodies = {digest: dict(body=f'# source {i}\n' + 'complete editable line\n' * 250,
                          body_sha256=sha(f'body-{i}')) for i, digest in enumerate(hashes)}
    # Only this test's externally-derived data stage is replaced. Context
    # selection, rendering, hashes, duplicate checks and revalidation are real.
    monkeypatch.setattr(context, '_facts', lambda _: copy.deepcopy(facts))
    monkeypatch.setattr(context, '_body_catalog', lambda *_: (copy.deepcopy(bodies), {'fixture': 'builder'}))
    return capsule, facts, bodies


def test_complete_study_keeps_colliding_local_labels_distinct_and_unmeasured(study):
    capsule, facts, _ = study
    before = canonical_json([capsule, facts])
    value = context.build_context([capsule], branch='p1', max_chars=90000)
    card, = context.expand_context(value)['cards']
    assert len({p['id'] for p in card['authored_proposals']}) == 2
    assert [p['provenance']['research_id'] for p in card['authored_proposals']] == ['r000003a1'] * 2
    for actual, original in zip(card['authored_proposals'], facts['proposals'], strict=True):
        assert actual['unverified_hypothesis'] == original['hypothesis']
        assert actual['native_observation'] == original['native_observation']
        assert actual['native_forecast_status'] == 'UNMEASURED'
        assert actual['native_observation']['forecast_scored'] is False
        assert actual['native_observation']['score'] is actual['native_observation']['actual_delta'] is None
    assert [g['branch'] for g in card['comparisons']] == ['on_off', 'on_control', 'off_control']
    assert all(g['origin_experiment_id'] is None for g in card['comparisons'])
    assert before == canonical_json([capsule, facts])


@pytest.mark.parametrize('limit,wire_sha', [
    (90000, 'ae6f468a59a11702933565c83c4a2570be541cd79da47c00bc755296d00c992c'),
    (16000, '8702d8455d8e04a3bc48ea232f31aa6e160f44ec52e14b7b8869801e5962f951')])
def test_literal_sorted_json_roundtrip_preserves_existing_full_and_compact_wire(study, limit, wire_sha):
    capsule, _, _ = study
    value = context.build_context([capsule], branch='p1', max_chars=limit)
    wire = context.block(value)
    # Captured before the decoder fix: default builder bytes must not change.
    assert sha(wire) == wire_sha
    parsed = json.loads(wire.split(context.PREFIX, 1)[1])
    assert context.expand_context(parsed) == context.expand_context(value)
    assert context.validate_context(parsed, histories=[capsule], branch='p1') == value
    reversed_keys = lambda pairs: dict(reversed(pairs))
    reversed_wire = json.loads(canonical_json(value), object_pairs_hook=reversed_keys)
    assert context.expand_context(reversed_wire) == context.expand_context(value)


@pytest.mark.parametrize('damage', ['reordered_identity', 'unused_identity', 'dropped_identity',
                                  'wrong_columns', 'unused_columns', 'literal_instead_of_reference'])
def test_json_roundtrip_still_rejects_self_rehashed_invalid_codec(study, damage):
    value = json.loads(canonical_json(context.build_context([study[0]], branch='p1', max_chars=90000)))
    if damage == 'reordered_identity':
        value['identity_catalog'][0], value['identity_catalog'][1] = value['identity_catalog'][1], value['identity_catalog'][0]
    elif damage == 'unused_identity':
        value['identity_catalog'].append('unused-' + 'a' * 64)
    elif damage == 'dropped_identity':
        value['identity_catalog'].pop()
    elif damage == 'wrong_columns':
        value['columns']['authored_proposals'][0] = 'invented_field'
    elif damage == 'unused_columns':
        value['columns']['unused'] = ['invented_field']
    else:
        def replace_first(container):
            if type(container) is dict:
                for key, item in container.items():
                    if type(item) is dict and set(item) == {'identity_ref'}:
                        container[key] = value['identity_catalog'][item['identity_ref']]
                        return True
                    if replace_first(item): return True
            elif type(container) is list:
                for index, item in enumerate(container):
                    if type(item) is dict and set(item) == {'identity_ref'}:
                        container[index] = value['identity_catalog'][item['identity_ref']]
                        return True
                    if replace_first(item): return True
            return False
        assert replace_first(value['cards'])
    value['context_sha256'] = canonical_sha({k: v for k, v in value.items() if k != 'context_sha256'})
    with pytest.raises(ValueError):
        context.expand_context(value)


def test_json_roundtrip_keeps_model_objects_opaque(study):
    capsule, facts, _ = study
    marker = {'identity_ref': 0, 'columns': {'invented': ['do not decode']}}
    facts['proposals'][0]['hypothesis']['reasoning'] = marker
    value = context.build_context([capsule], branch='p1', max_chars=90000)
    parsed = json.loads(canonical_json(value))
    original = context.expand_context(value)['cards'][0]['authored_proposals'][0]
    restored = context.expand_context(parsed)['cards'][0]['authored_proposals'][0]
    assert restored['unverified_hypothesis']['reasoning'] == original['unverified_hypothesis']['reasoning'] == marker


def test_quality_preserves_incomplete_known_loss_and_all_missing_pairs(study):
    capsule, facts, _ = study
    card, = context.expand_context(context.build_context([capsule], branch='p1', max_chars=90000))['cards']
    assert [g['strict_quality']['verdict'] for g in card['comparisons']] == ['INCONCLUSIVE', 'REFUTED', 'REFUTED']
    for actual, original in zip(card['comparisons'], facts['comparisons'], strict=True):
        assert actual['stored_quality_outcome'] == 'incomplete'
        assert actual['strict_quality'] == original['quality']
        assert len(actual['strict_quality']['pairs']) == 3
        assert all(p['raw_delta'] is p['certified_delta'] is None for p in actual['strict_quality']['pairs'][1:])


def test_reused_draws_are_not_counted_as_independent_observations(study):
    capsule, _, _ = study
    card, = context.expand_context(context.build_context([capsule], branch='p1', max_chars=90000))['cards']
    coverage = card['draw_coverage']
    assert coverage['planned_count'] == 9
    assert coverage['status_counts'] == {'valid': 3, 'not_dispatched': 6}
    assert coverage['not_dispatched_count'] == 6
    assert len({(r['source_sha256'], r['seed']) for r in coverage['slots']}) == 9
    assert len(card['restart_forecasts']) == 2
    for forecast in card['restart_forecasts']:
        assert len(forecast['rows']) == 3
        assert all(len(row['references']) == 2 for row in forecast['rows'])
        first = forecast['rows'][0]['result']
        assert first['per_case_details_omitted_count'] == 7
        assert 'cases' not in first and 'columns' not in first
        assert all(row['result'] is None and row['unavailable_reason'] == 'receipt_not_dispatched'
                   for row in forecast['rows'][1:])
    # Input group order encounters ON then OFF; inspect by authored identity.
    results = {r['proposal_id']: r['rows'][0]['result'] for r in card['restart_forecasts']}
    assert results['generated:' + sha('on')]['verdicts']['result_changed'] == 'INCONCLUSIVE'
    assert results['generated:' + sha('off')]['verdicts']['result_changed'] == 'REFUTED'


@pytest.mark.parametrize('kwargs', [{'enabled': False}, {'max_items': 0}, {'max_chars': 0}, {'branch': 'unrelated'}])
def test_disabled_or_unrelated_delivery_omits_every_fact_and_model_field(study, kwargs):
    capsule, _, _ = study
    options = dict(branch='p1', max_chars=90000)
    options.update(kwargs)
    value = context.build_context([capsule], **options)
    assert value is None and context.block(value) == ''


def test_budget_counts_rules_and_metadata_not_just_json(study):
    capsule, _, _ = study
    large = context.build_context([capsule], branch='p1', max_chars=90000)
    small = context.build_context([capsule], branch='p1', max_chars=16000)
    assert len(context.block(small)) <= 16000
    assert len(context.block(small)) > len(canonical_json(small))
    assert context.build_context([capsule], branch='p1', max_chars=100) is None
    assert context.build_context([capsule], branch='p1', max_chars=len(context.RULES)) is None
    assert len(large['cards']) == len(small['cards']) == 1


def test_complete_source_bodies_are_optional_without_clipping_facts(study):
    capsule, _, bodies = study
    large = context.build_context([capsule], branch='p1', max_chars=90000)
    small = context.build_context([capsule], branch='p1', max_chars=16000)
    a, b = context.expand_context(large)['cards'][0], context.expand_context(small)['cards'][0]
    assert a['source_bodies'] == bodies and a['body_omissions'] == []
    assert b['body_omissions']
    assert {k: v for k, v in a.items() if k not in ('source_bodies', 'body_omissions')} == {
        k: v for k, v in b.items() if k not in ('source_bodies', 'body_omissions')}
    assert set(b['source_bodies']) | {row['source_sha256'] for row in b['body_omissions']} == set(bodies)
    for ident, value in b['source_bodies'].items():
        assert value == bodies[ident]
    assert all(row['reason'] == 'memory_char_limit' for row in b['body_omissions'])


def test_too_large_evidence_omits_whole_study_not_negative_or_unknown_rows(study):
    capsule, _, _ = study
    value = context.build_context([capsule], branch='p1', max_chars=5000)
    assert value['cards'] == []
    assert value['presented_count'] == 0 and value['omitted_count'] == value['selected_count'] == 1
    assert value['omissions'] == [{'capsule_sha256': canonical_sha(capsule), 'reason': 'memory_char_limit'}]
    assert 'UNVERIFIED-' not in context.block(value)


def test_item_limit_preserves_order_and_never_admits_partial_study(study):
    capsule, _, _ = study
    other = dict(copy.deepcopy(capsule), fixture_second=True)
    value = context.build_context([capsule, other], branch='p1', max_chars=90000, max_items=1)
    assert [card['capsule_sha256'] for card in context.expand_context(value)['cards']] == [canonical_sha(capsule)]
    assert value['omissions'] == [{'capsule_sha256': canonical_sha(other), 'reason': 'item_limit'}]
    with pytest.raises(ValueError, match='duplicate history'):
        context.build_context([capsule, capsule], branch='p1')


@pytest.mark.parametrize('field', ['hypothesis', 'quality', 'forecast', 'native', 'source_body'])
def test_self_rehashed_mutation_does_not_authenticate_context(study, field):
    capsule, _, _ = study
    original = context.build_context([capsule], branch='p1', max_chars=90000)
    value = context.expand_context(original)
    card = value['cards'][0]
    if field == 'hypothesis': card['authored_proposals'][0]['unverified_hypothesis']['reasoning'] = 'invented'
    elif field == 'quality': card['comparisons'][1]['strict_quality']['verdict'] = 'CONFIRMED'
    elif field == 'forecast': card['restart_forecasts'][0]['rows'][0]['result']['verdicts']['result_changed'] = 'CONFIRMED'
    elif field == 'native': card['authored_proposals'][0]['native_observation']['forecast_scored'] = True
    else: card['source_bodies'][next(iter(card['source_bodies']))]['body'] = 'repaired body'
    value = context._seal(value)  # A forger knows the public, deterministic codec.
    with pytest.raises(ValueError, match='bound history'):
        context.validate_context(value, histories=[capsule], branch='p1')
    assert context.validate_context(original, histories=[capsule], branch='p1') == original


def test_absent_context_requires_and_rechecks_explicit_delivery_limits(study):
    capsule, _, _ = study
    with pytest.raises(ValueError, match='explicit delivery limits'):
        context.validate_context(None, histories=[capsule], branch='p1')
    assert context.validate_context(None, histories=[capsule], branch='p1',
        max_chars=16000, max_items=0) is None
    with pytest.raises(ValueError, match='bound history'):
        context.validate_context(None, histories=[capsule], branch='p1', max_chars=16000, max_items=2)


@pytest.mark.parametrize('kwargs', [{'max_chars': True}, {'max_chars': -1}, {'max_chars': 90001},
    {'max_items': True}, {'max_items': -1}, {'max_items': 9}, {'enabled': 1}])
def test_delivery_limits_are_strict(study, kwargs):
    with pytest.raises(ValueError):
        context.build_context([study[0]], branch='p1', **kwargs)


def test_shared_draw_or_prediction_disagreement_fails_closed(study):
    capsule, facts, _ = study
    first = facts['comparisons'][0]['group']['pairs'][0]['challenger']
    # Detach the reused reference so the deliberate forgery affects one group.
    facts['comparisons'][0]['group']['pairs'][0]['challenger'] = dict(first, fixture='changed')
    with pytest.raises(ValueError, match='shared draw'):
        context.build_context([capsule], branch='p1', max_chars=90000)


def test_lossless_reference_codec_never_interprets_model_strings(study):
    capsule, facts, bodies = study
    literal = '{"identity_ref":0} ' + facts['root_parent']['source_sha256']
    facts['proposals'][0]['hypothesis']['reasoning'] = literal
    body_key = next(iter(bodies))
    bodies[body_key]['body'] = literal
    value = context.build_context([capsule], branch='p1', max_chars=90000)
    restored = context.expand_context(value)
    assert canonical_sha(restored) == value['expanded_sha256']
    assert value['identity_catalog'] and value['columns']
    assert restored['cards'][0]['authored_proposals'][0]['unverified_hypothesis']['reasoning'] == literal
    assert restored['cards'][0]['source_bodies'][body_key]['body'] == literal
    assert context._seal(restored) == value


@pytest.mark.parametrize('damage', ['unused_identity', 'short_row', 'wrong_identity'])
def test_rehashed_reference_table_corruption_fails_expansion(study, damage):
    value = context.build_context([study[0]], branch='p1', max_chars=90000)
    if damage == 'unused_identity':
        value['identity_catalog'].append(sha('unused'))
    elif damage == 'short_row':
        value['cards'][0]['authored_proposals'][0].pop()
    else:
        value['identity_catalog'][0] = sha('substituted')
    value['context_sha256'] = canonical_sha({k: v for k, v in value.items() if k != 'context_sha256'})
    with pytest.raises(ValueError):
        context.expand_context(value)
