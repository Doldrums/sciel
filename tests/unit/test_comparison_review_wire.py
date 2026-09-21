"""Opt-in comparison-review wire checks; no provider or candidate execution."""
import hashlib
import json

import pytest

from sciel.hypothesis import auxiliary_schema as schema
from sciel.hypothesis import comparison_review as comparison
from sciel.hypothesis import conclusion_source, mechanism_review
from sciel.hypothesis.models import Reflection
from sciel.hypothesis.parsing import parse_reflection


def review(**changes):
    value = dict(
        comparison_id='pair:branch-a:0', comparison_sha256='b' * 64,
        interpretation='The three saved comparisons have mixed signs.',
        limitations='This comparison cannot isolate the proposed mechanism.',
        next_hypothesis=dict(
            change='Change only the reinsertion order.',
            expected_observation='Every paired delta is nonnegative.',
            falsification='Any negative paired delta refutes this prediction.'))
    value.update(changes)
    return value


def payload(reviews=None):
    return dict(lessons=[], retire_ids=[],
                comparison_reviews=[review()] if reviews is None else reviews)


def parse(value, *, enabled=True):
    return parse_reflection(json.dumps(value, ensure_ascii=False), comparison_reviews=enabled)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def dictionaries(value):
    if type(value) is dict:
        yield value
        for item in value.values():
            yield from dictionaries(item)
    elif type(value) is list:
        for item in value:
            yield from dictionaries(item)


def test_legacy_wire_and_model_serialization_keep_exact_two_fields():
    old = dict(lessons=[], retire_ids=[])
    assert Reflection([], []).to_dict() == old
    parsed = parse_reflection(json.dumps(old))
    assert parsed.comparison_reviews is None and parsed.to_dict() == old
    assert parse(old, enabled=False).to_dict() == old
    assert parse(old) is None
    assert parse(payload(), enabled=False) is None


@pytest.mark.parametrize('reviews', [[], [review()], [review(next_hypothesis=None)]])
@pytest.mark.parametrize('transport', ['bare', 'marker', 'fence', 'marked_fence'])
def test_mixed_comparison_review_is_valid_without_lessons(reviews, transport):
    value = payload(reviews)
    raw = json.dumps(value)
    if transport == 'marker': raw = 'REFLECTION_JSON:\n' + raw
    elif transport == 'fence': raw = '```json\n' + raw + '\n```'
    elif transport == 'marked_fence': raw = 'REFLECTION_JSON:\n```json\n' + raw + '\n```'
    parsed = parse_reflection(raw, comparison_reviews=True)
    assert parsed is not None and parsed.to_dict() == value
    assert parsed.lessons == [] and parsed.retire_ids == []


@pytest.mark.parametrize('value', [None, 0, 1, '', 'true', [], {}])
def test_mode_requires_exact_boolean_at_every_entrypoint(value):
    assert parse_reflection(json.dumps(payload()), comparison_reviews=value) is None
    with pytest.raises(ValueError, match='boolean'):
        schema.response_format('reflection', comparison_reviews=value)
    with pytest.raises(ValueError, match='boolean'):
        schema.identity(comparison_reviews=value)


@pytest.mark.parametrize('damage', ['missing_lessons', 'missing_retire', 'missing_reviews',
    'extra', 'null_reviews', 'bad_lessons', 'bad_retirement'])
def test_new_mode_does_not_relax_reflection_fields(damage):
    value = payload()
    if damage == 'missing_lessons': value.pop('lessons')
    elif damage == 'missing_retire': value.pop('retire_ids')
    elif damage == 'missing_reviews': value.pop('comparison_reviews')
    elif damage == 'extra': value['extra'] = False
    elif damage == 'null_reviews': value['comparison_reviews'] = None
    elif damage == 'bad_lessons': value['lessons'] = [dict(rule='Incomplete')]
    else: value['retire_ids'] = ['same', 'same']
    assert parse(value) is None


@pytest.mark.parametrize('changes', [
    dict(comparison_id=True), dict(comparison_id='bad id'),
    dict(comparison_sha256='A' * 64), dict(comparison_sha256='b' * 63),
    dict(interpretation=' '), dict(interpretation='x' * 2049),
    dict(limitations='x' * 2049), dict(next_hypothesis=False),
    dict(next_hypothesis={}),
    dict(next_hypothesis=dict(change='x', expected_observation='y', falsification='z', extra=0)),
    dict(next_hypothesis=dict(change='x' * 1025, expected_observation='y', falsification='z')),
    dict(next_hypothesis=dict(change='x', expected_observation='y', falsification='')),
])
def test_local_proposal_validation_rejects_malformed_content(changes):
    assert parse(payload([review(**changes)])) is None


@pytest.mark.parametrize('field', ['comparison_id', 'comparison_sha256', 'interpretation',
    'limitations', 'next_hypothesis'])
def test_proposals_require_every_exact_field(field):
    value = review()
    value.pop(field)
    assert parse(payload([value])) is None


def test_extra_fields_duplicates_count_and_escaped_aggregate_are_rejected():
    assert parse(payload([review(extra=True)])) is None
    assert parse(payload([review(), review()])) is None
    assert parse(payload([review(comparison_id=f'pair:{i}') for i in range(5)])) is None
    oversized = review(interpretation='界' * 2048, limitations='界' * 2048)
    assert len(json.dumps([oversized], ensure_ascii=True)) > comparison.MAX_PROPOSAL_CHARS
    assert parse(payload([oversized])) is None


def test_no_transport_repair_or_duplicate_json_keys():
    raw = json.dumps(payload())
    for damaged in ('Prose before bare object. ' + raw, raw + ' trailing prose',
        'CONCLUSION_JSON: ' + raw, 'REFLECTION_JSON: REFLECTION_JSON: ' + raw,
        raw.replace('"comparison_reviews":', '"comparison_reviews":[],"comparison_reviews":')):
        assert parse_reflection(damaged, comparison_reviews=True) is None


def test_review_serialization_is_detached_and_does_not_create_lessons():
    parsed = parse(payload())
    before = parsed.to_dict()
    changed = parsed.to_dict()
    changed['comparison_reviews'][0]['next_hypothesis']['change'] = 'altered'
    changed['comparison_reviews'].clear()
    assert parsed.to_dict() == before
    assert parsed.lessons == []


def test_false_reflection_is_historical_and_comparison_modes_do_not_change_conclusion():
    assert digest(schema.response_format('reflection')) == (
        'e2d06308d0c4bef0348035f5bc6d13a30b9f6954d3e5dbe90ac55f5705347e16')
    for kind in ('conclusion', 'reflection'):
        assert schema.response_format(kind, comparison_reviews=False) == schema.response_format(kind)
    current = schema.response_format('conclusion')
    for version in (1, 2, 3):
        assert schema.response_format('conclusion', comparison_reviews=True,
            comparison_review_version=version) == current

    # The archived C/2 envelope predates C-only full diff/window quotation.
    # Authenticate the entire old envelope after reversing ONLY that explicit
    # enum extension; do not repin the hash or allow comparison-mode drift.
    historical = schema.response_format('conclusion')
    field = historical['json_schema']['schema']['properties']['mechanism_review'][
        'items']['properties']['source_quotes']['items']['properties']['field']
    assert field['enum'] == list(mechanism_review.SOURCE_FIELDS + conclusion_source.SOURCE_FIELDS)
    field['enum'] = list(mechanism_review.SOURCE_FIELDS)
    assert digest(historical) == '7275dd52cbf86be666d945fad37e29e89092c8cb784b226fdfeb036f9f945eee'
    assert schema.response_format('conclusion') == current  # returned envelopes are detached


def test_opt_in_provider_schema_is_closed_nullable_and_uses_supported_keywords():
    wire = schema.response_format('reflection', comparison_reviews=True)
    assert wire['json_schema']['name'] == 'sciel_reflection_v2'
    assert wire['json_schema']['strict'] is True
    root = wire['json_schema']['schema']
    assert set(root['properties']) == {'lessons', 'retire_ids', 'comparison_reviews'}
    item = root['properties']['comparison_reviews']['items']
    assert set(item['properties']) == set(review())
    future = item['properties']['next_hypothesis']
    assert future['type'] == ['object', 'null']
    assert set(future['properties']) == {'change', 'expected_observation', 'falsification'}
    assert str(comparison.MAX_PROPOSALS) in root['properties']['comparison_reviews']['description']
    assert str(comparison.MAX_PROPOSAL_CHARS) in root['properties']['comparison_reviews']['description']
    assert str(comparison.MAX_HYPOTHESIS_CHARS) in future['properties']['falsification']['description']
    for item in dictionaries(root):
        assert not set(item) & {'minLength', 'maxLength', 'maxItems', 'uniqueItems',
            'minimum', 'maximum', 'pattern', 'contains', 'minContains', 'maxContains'}
        if item.get('type') in ('object', ['object', 'null']):
            assert item['additionalProperties'] is False
            assert set(item['required']) == set(item['properties'])


def test_identity_binds_enabled_mode_policy_and_exact_formats():
    old = schema.identity()
    new = schema.identity(comparison_reviews=True)
    assert set(old) == {'version', 'source_sha256', 'response_format_sha256'}
    assert old == schema.identity(comparison_reviews=False)
    assert new['version'] == old['version'] and new['source_sha256'] == old['source_sha256']
    assert new['comparison_reviews'] is True
    assert new['review_policy'] == comparison.identity()
    assert new['response_format_sha256']['conclusion'] == old['response_format_sha256']['conclusion']
    assert new['response_format_sha256']['reflection'] != old['response_format_sha256']['reflection']
    for kind in ('conclusion', 'reflection'):
        assert new['response_format_sha256'][kind] == digest(schema.response_format(kind, comparison_reviews=True))
    new['review_policy'].clear()
    new['response_format_sha256'].clear()
    assert schema.identity(comparison_reviews=True)['review_policy'] == comparison.identity()


def test_standard_validator_accepts_new_payloads_and_local_limits_remain_authoritative():
    jsonschema = pytest.importorskip('jsonschema')
    contract = schema.response_format('reflection', comparison_reviews=True)['json_schema']['schema']
    jsonschema.Draft202012Validator.check_schema(contract)
    validate = jsonschema.Draft202012Validator(contract).validate
    for value in (payload([]), payload(), payload([review(next_hypothesis=None)])):
        validate(value)
    incomplete = payload()
    del incomplete['comparison_reviews'][0]['next_hypothesis']
    with pytest.raises(jsonschema.ValidationError):
        validate(incomplete)
    old = dict(lessons=[], retire_ids=[])
    with pytest.raises(jsonschema.ValidationError):
        validate(old)
    overlong = payload([review(interpretation='x' * 2049)])
    validate(overlong)
    assert parse(overlong) is None
