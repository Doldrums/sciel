"""Versioned reflection transport and prompt compatibility; no live execution."""
import copy
import hashlib
import json

import pytest

from sciel.config import apply_defaults
from sciel.hypothesis import auxiliary_schema, grounded_review, prompts
from sciel.hypothesis.parsing import parse_reflection
from sciel.knowledge_base.catalogs import canonical_json
from tests.unit.test_comparison_review_wire import dictionaries, payload as legacy_payload


def proposal(**changes):
    value = dict(comparison_id='comparison:' + 'a' * 64, comparison_sha256='b' * 64,
        claims=[dict(fact_id='quality:outcome', value='mixed')],
        conjecture='A coarser grid may reduce helper cost; this has not been measured.',
        limitations='One grouped comparison does not establish a causal mechanism.',
        next_hypothesis=None)
    value.update(changes)
    return value


def payload(reviews=None):
    return dict(lessons=[], retire_ids=[], comparison_reviews=[proposal()] if reviews is None else reviews)


def parse(value, version=2):
    return parse_reflection(json.dumps(value), comparison_reviews=True, comparison_review_version=version)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def test_legacy_envelopes_and_prompts_are_byte_identical():
    pins = {
        False: ('e2d06308d0c4bef0348035f5bc6d13a30b9f6954d3e5dbe90ac55f5705347e16',
                'a974a198863503133cfff6c0b108a3cb559a5fc575a0a73b4973bc6fb359497e'),
        True: ('83a1737bd7822c1b4b441a821df3d387e8cd6f86eaad52ff41be37eb2524e334',
               '856ef8cd14dd0e5529da6c5b53e622fb6cafb28ae71cd51ccb0b3712b77ccb45')}
    for enabled, (wire_sha, prompt_sha) in pins.items():
        options = dict(comparison_reviews=enabled)
        wire = auxiliary_schema.response_format('reflection', **options)
        assert digest(wire) == wire_sha
        assert wire == auxiliary_schema.response_format('reflection', comparison_review_version=1, **options)
        if enabled:
            options['comparison_review_groups'] = []
        prompt = prompts.reflection_prompt([], [], structured=True, **options)
        assert hashlib.sha256(prompt.encode()).hexdigest() == prompt_sha
        assert prompt == prompts.reflection_prompt([], [], structured=True, comparison_review_version=1, **options)


def test_v3_is_explicit_and_cannot_reinterpret_legacy_reviews():
    old = legacy_payload()
    # Historical mistakes remain historical text; the new mode never repairs it.
    old['comparison_reviews'][0]['interpretation'] = 'The signs differ for all three seeds.'
    assert parse(old, 1).to_dict() == old
    assert parse(old, 2) is None
    new = payload()
    assert parse(new, 1) is None
    parsed = parse(new, 2)
    assert parsed.to_dict() == new and parsed.lessons == []
    parsed.to_dict()['comparison_reviews'][0]['claims'].clear()
    assert parsed.to_dict() == new
    for version in (1, 2):
        assert parse(payload([]), version).to_dict() == payload([])


@pytest.mark.parametrize('version', [False, True, None, 0, 4, 1.0, '2', [], {}])
def test_versions_require_exact_supported_integers(version):
    assert parse(payload(), version) is None
    for function, args in ((auxiliary_schema.response_format, ('reflection',)),
                           (auxiliary_schema.identity, ())):
        with pytest.raises(ValueError, match='comparison_review_version'):
            function(*args, comparison_reviews=True, comparison_review_version=version)
    with pytest.raises(ValueError, match='comparison_review_version'):
        prompts.reflection_prompt([], [], structured=True, comparison_reviews=True,
            comparison_review_groups=[], comparison_review_version=version)
    with pytest.raises(ValueError, match='version'):
        prompts.comparison_review_source_block([], version=version)


def test_v2_requires_enabled_review_transport():
    assert parse_reflection('{"lessons":[],"retire_ids":[]}', comparison_review_version=2) is None
    for function, args in ((auxiliary_schema.response_format, ('reflection',)),
                           (auxiliary_schema.identity, ()),
                           (prompts.reflection_prompt, ([], []))):
        with pytest.raises(ValueError, match='requires comparison_reviews'):
            function(*args, comparison_review_version=2)


@pytest.mark.parametrize('changes', [
    dict(claims=[]), dict(claims=[dict(fact_id='quality:outcome', value=True)]),
    dict(claims=[dict(fact_id='quality:outcome', value='mixed')] * 2),
    dict(claims=[dict(fact_id=f'fact:{i}', value='unknown') for i in range(25)]),
    dict(conjecture=' '), dict(conjecture='x' * 1025), dict(limitations='x' * 2049),
    dict(next_hypothesis=dict(change='Reduce grid.', expected_observation='Near-zero.', falsification='None.')),
])
def test_v2_local_shape_guards_are_used(changes):
    assert parse(payload([proposal(**changes)])) is None


def test_transport_parser_does_not_claim_source_truth():
    # Shape-valid but false values are rejected later by source-bound admission.
    value = payload([proposal(claims=[dict(fact_id='pair:4186712104:raw_direction', value='positive')])])
    assert parse(value).to_dict() == value
    assert parse(payload([dict(proposal(), interpretation='Legacy prose')])) is None
    raw = json.dumps(payload())
    assert parse_reflection(raw.replace('"claims":', '"claims":[],"claims":'),
        comparison_reviews=True, comparison_review_version=2) is None


def test_typed_hypothesis_roundtrips_and_cannot_substitute_a_weaker_gate():
    from sciel.hypothesis.comparison_targets import SCHEMA, QUALITY_GATE
    target = dict(schema=SCHEMA, base_source_sha256='b' * 64,
        comparator_source_sha256='c' * 64, change='Reduce both lattice loops and denominators from eight to six.',
        quality_gate=QUALITY_GATE,
        diagnostics=[dict(observable='restart_total_starts', direction='increase')])
    value = payload([proposal(next_hypothesis=target)])
    assert parse(value).to_dict() == value
    for bad_gate in ('exact_only', 'near_zero', 'mean_positive'):
        damaged = copy.deepcopy(value)
        damaged['comparison_reviews'][0]['next_hypothesis']['quality_gate'] = bad_gate
        assert parse(damaged) is None
    damaged = copy.deepcopy(value)
    damaged['comparison_reviews'][0]['next_hypothesis']['diagnostics'][0]['observable'] = 'selected_trial_order'
    assert parse(damaged) is None


def test_v3_schema_is_closed_and_preserves_conclusion_wire():
    version_options = dict(comparison_reviews=True, comparison_review_version=2)
    wire = auxiliary_schema.response_format('reflection', **version_options)
    assert wire['json_schema']['name'] == 'sciel_reflection_v3'
    assert wire['json_schema']['strict'] is True
    schema = wire['json_schema']['schema']
    assert set(schema['properties']) == {'lessons', 'retire_ids', 'comparison_reviews'}
    assert set(schema['properties']['comparison_reviews']['items']['properties']) == set(proposal())
    for item in dictionaries(schema):
        assert not set(item) & {'minLength', 'maxLength', 'maxItems', 'uniqueItems',
            'minimum', 'maximum', 'pattern', 'contains', 'minContains', 'maxContains'}
        if item.get('type') in ('object', ['object', 'null']):
            assert item['additionalProperties'] is False
            assert set(item['required']) == set(item['properties'])
    assert auxiliary_schema.response_format('conclusion', **version_options) == auxiliary_schema.response_format('conclusion')
    jsonschema = pytest.importorskip('jsonschema')
    jsonschema.Draft202012Validator.check_schema(schema)
    validate = jsonschema.Draft202012Validator(schema).validate
    validate(payload())
    validate(payload([]))
    with pytest.raises(jsonschema.ValidationError):
        validate(legacy_payload())


def test_v2_identity_pins_new_policy_and_both_exact_wires():
    options = dict(comparison_reviews=True, comparison_review_version=2)
    value = auxiliary_schema.identity(**options)
    assert value['comparison_review_version'] == 2
    assert value['grounded_review_policy'] == grounded_review.identity()
    for kind in ('conclusion', 'reflection'):
        assert value['response_format_sha256'][kind] == digest(auxiliary_schema.response_format(kind, **options))
    old = auxiliary_schema.identity(comparison_reviews=True)
    assert 'comparison_review_version' not in old and 'grounded_review_policy' not in old
    value['grounded_review_policy'].clear()
    assert auxiliary_schema.identity(**options)['grounded_review_policy'] == grounded_review.identity()


def test_v2_prompt_forwards_authenticated_records_and_names_the_new_contract(monkeypatch):
    records, seen = {'origin': object()}, {'origin': 'f' * 64}
    calls = []

    def sources(groups, *, records, seen):
        calls.append((groups, records, seen))
        return 'HOST_FACT_CATALOG_SENTINEL\n'

    monkeypatch.setattr(grounded_review, 'source_block', sources)
    prompt = prompts.reflection_prompt([], [], structured=True, comparison_reviews=True,
        comparison_review_version=2, comparison_review_groups=[],
        comparison_review_records=records, comparison_review_seen=seen)
    assert calls == [([], records, seen)]
    assert 'HOST_FACT_CATALOG_SENTINEL' in prompt
    assert 'exactly comparison_id, comparison_sha256, claims, conjecture, limitations and next_hypothesis' in prompt
    assert 'These two text fields remain unverified' in prompt
    assert 'strict two-metric no-loss gate' in prompt
    assert 'base_source_sha256, comparator_source_sha256, change, quality_gate and diagnostics' in prompt
    assert 'diagnostics=[] proposes no mechanism measurement' in prompt
    assert 'unknown with false or zero' in prompt
    assert 'interpretation and limitations (nonblank' not in prompt


def test_disabled_prompt_cannot_silently_accept_grounded_sources():
    for supplied in ({'comparison_review_records': {}}, {'comparison_review_seen': {}}):
        with pytest.raises(ValueError, match='sources supplied while disabled'):
            prompts.reflection_prompt([], [], **supplied)


def test_config_defaults_are_unchanged_and_explicit_versions_are_preserved():
    raw = {'research': {'comparison_reviews': True, 'structured_auxiliary': True}}
    before = copy.deepcopy(raw)
    legacy = apply_defaults(raw)
    assert raw == before and 'comparison_review_version' not in legacy['research']
    for version in (1, 2):
        configured = apply_defaults({'research': {**raw['research'], 'comparison_review_version': version}})
        assert configured['research']['comparison_review_version'] == version
        assert apply_defaults(configured) == configured
        configured['research'].pop('comparison_review_version')
        assert configured == legacy


@pytest.mark.parametrize('version', [False, True, None, 0, 4, '2', 2.0])
def test_config_rejects_invalid_version(version):
    with pytest.raises(ValueError, match='comparison_review_version'):
        apply_defaults({'research': dict(comparison_reviews=True, structured_auxiliary=True,
            comparison_review_version=version)})


@pytest.mark.parametrize('version', [1, 2])
def test_explicit_config_version_requires_reviews(version):
    with pytest.raises(ValueError, match='comparison_review_version requires comparison_reviews'):
        apply_defaults({'research': dict(comparison_review_version=version)})
