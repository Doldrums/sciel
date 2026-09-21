"""Portable schema/parser contract checks; no model, helper or KB mutation."""
import hashlib
import json
from pathlib import Path

import pytest

from sciel.hypothesis import auxiliary_schema as schema
from sciel.hypothesis import mechanism_review as mechanism
from sciel.hypothesis import parsing


def review(**changes):
    value = dict(claim_quote='A changed initializer may help.',
        source_quotes=[dict(field='change_excerpt', text='return values\n')],
        assessment='directly_supported', direct_effect='The code returns values.',
        untested_downstream='Its effect on quality is unverified.',
        evidence_kind='source_effect', forecast_ref=None)
    value.update(changes)
    return value


def lesson(scope='branch'):
    return dict(rule='One observed change improved this score.', polarity='positive',
        scope=scope, evidence_ids=['r000001a1'], branch='p1' if scope == 'branch' else None)


def payload(kind, scope='branch'):
    if kind == 'conclusion':
        return dict(verdict='REFUTED', explanation='The measured delta exceeded the forecast.',
            lessons=[lesson(scope)], mechanism_review=[review()])
    return dict(lessons=[lesson(scope)], retire_ids=['lesson:old'])


def parse(kind, value):
    raw = json.dumps(value, ensure_ascii=False)
    return (parsing.parse_conclusion(raw, require_mechanism=True, require_typed=True) if kind == 'conclusion'
            else parsing.parse_reflection(raw))


def dictionaries(value):
    if type(value) is dict:
        yield value
        for child in value.values():
            yield from dictionaries(child)
    elif type(value) is list:
        for child in value:
            yield from dictionaries(child)


@pytest.mark.parametrize('kind', ['conclusion', 'reflection'])
def test_native_schema_requires_all_fields_at_every_object_level(kind):
    wire = schema.response_format(kind)
    assert set(wire) == {'type', 'json_schema'} and wire['type'] == 'json_schema'
    envelope = wire['json_schema']
    assert set(envelope) == {'name', 'strict', 'schema'} and envelope['strict'] is True
    root = envelope['schema']
    assert set(root['properties']) == ({'verdict', 'explanation', 'lessons', 'mechanism_review'}
        if kind == 'conclusion' else {'lessons', 'retire_ids'})
    for item in dictionaries(root):
        # Claude's documented subset rejects these standard JSON Schema keys.
        assert not set(item) & {'minLength', 'maxLength', 'maxItems', 'uniqueItems',
                                'minimum', 'maximum', 'pattern', 'contains', 'minContains', 'maxContains'}
        assert 'minItems' not in item or item['minItems'] in (0, 1)
        if item.get('type') == 'object' or item.get('type') == ['object', 'null']:
            assert item['additionalProperties'] is False
            assert set(item['required']) == set(item['properties'])
            assert len(item['required']) == len(set(item['required']))
    branch = root['properties']['lessons']['items']['properties']['branch']
    assert branch['type'] == ['string', 'null']
    assert str(parsing.MAX_ID_CHARS) in branch['description']


def test_described_limits_track_existing_parser_and_mechanism_limits():
    props = schema.response_format('conclusion')['json_schema']['schema']['properties']
    assert str(parsing.MAX_REASONING_CHARS) in props['explanation']['description']
    lessons = props['lessons']
    assert f'At most {parsing.MAX_LESSONS} lessons' in lessons['description']
    item = lessons['items']['properties']
    assert str(parsing.MAX_TEXT_CHARS) in item['rule']['description']
    assert f'1 and {parsing.MAX_EVIDENCE_IDS} distinct' in item['evidence_ids']['description']
    assert item['evidence_ids']['minItems'] == 1
    assert str(parsing.MAX_ID_CHARS) in item['evidence_ids']['items']['description']
    reviews = props['mechanism_review']
    assert reviews['minItems'] == 1
    assert f'One to {mechanism.MAX_ENTRIES} entries' in reviews['description']
    assert str(mechanism.MAX_REVIEW_CHARS) in reviews['description']
    entry = reviews['items']['properties']
    assert entry['assessment']['enum'] == list(mechanism.ASSESSMENTS)
    for field in ('claim_quote', 'direct_effect', 'untested_downstream'):
        assert str(mechanism.MAX_TEXT_CHARS) in entry[field]['description']
    assert entry['claim_quote']['type'] == ['string', 'null']
    assert f'At most {mechanism.MAX_SOURCE_QUOTES} exact quotes' in entry['source_quotes']['description']
    quote = entry['source_quotes']['items']['properties']
    assert quote['field']['enum'] == list(mechanism.SOURCE_FIELDS + mechanism.conclusion_source.SOURCE_FIELDS)
    assert str(mechanism.MAX_TEXT_CHARS) in quote['text']['description']
    retire = schema.response_format('reflection')['json_schema']['schema']['properties']['retire_ids']
    assert f'0 and {parsing.MAX_RETIRE_IDS} distinct' in retire['description']


@pytest.mark.parametrize('kind', ['conclusion', 'reflection'])
@pytest.mark.parametrize('scope', ['branch', 'global'])
def test_native_bare_payloads_roundtrip_through_unchanged_local_parser(kind, scope):
    value = payload(kind, scope)
    assert parse(kind, value).to_dict() == value


def test_unknown_review_without_claim_and_empty_reflection_are_valid():
    value = payload('conclusion')
    value.update(lessons=[], mechanism_review=[review(claim_quote=None,
        assessment='unknown', source_quotes=[], evidence_kind='unavailable')])
    assert parse('conclusion', value).to_dict() == value
    assert parse('reflection', dict(lessons=[], retire_ids=[])).to_dict() == dict(lessons=[], retire_ids=[])


@pytest.mark.parametrize('scope,branch', [('branch', None), ('global', 'p1')])
@pytest.mark.parametrize('kind', ['conclusion', 'reflection'])
def test_local_parser_still_enforces_lesson_scope_branch_relationship(kind, scope, branch):
    value = payload(kind, scope)
    value['lessons'][0]['branch'] = branch
    assert parse(kind, value) is None


@pytest.mark.parametrize('changes', [dict(claim_quote=None), dict(source_quotes=[])])
def test_local_mechanism_cross_field_rules_are_not_replaced(changes):
    value = payload('conclusion')
    value['mechanism_review'] = [review(**changes)]
    assert parse('conclusion', value) is None


def test_aggregate_escaped_review_cap_remains_local():
    value = payload('conclusion')
    value['mechanism_review'] = [review(direct_effect='界' * mechanism.MAX_TEXT_CHARS,
        untested_downstream='界' * mechanism.MAX_TEXT_CHARS)]
    assert len(json.dumps(value['mechanism_review'], ensure_ascii=True)) > mechanism.MAX_REVIEW_CHARS
    assert parse('conclusion', value) is None


def test_shape_valid_review_still_requires_literal_evidence_binding():
    value = payload('conclusion')
    parsed = parse('conclusion', value)
    assert parsed is not None
    evidence = dict(hypothesis=dict(text='A different original claim.'), change_excerpt='return other\n')
    with pytest.raises(mechanism.MechanismReviewError, match='original hypothesis'):
        mechanism.validate_binding(parsed.mechanism_review, evidence)


@pytest.mark.parametrize('kind', ['conclusion', 'reflection'])
@pytest.mark.parametrize('failure', ['rule_length', 'blank_rule', 'lesson_count', 'id_count',
    'duplicate_ids', 'id_length', 'id_whitespace', 'id_control'])
def test_descriptive_lesson_bounds_and_uniqueness_are_enforced_locally(kind, failure):
    value = payload(kind)
    item = value['lessons'][0]
    if failure == 'rule_length': item['rule'] = 'x' * (parsing.MAX_TEXT_CHARS + 1)
    elif failure == 'blank_rule': item['rule'] = '   '
    elif failure == 'lesson_count': value['lessons'] *= parsing.MAX_LESSONS + 1
    elif failure == 'id_count': item['evidence_ids'] = [f'r{i}' for i in range(parsing.MAX_EVIDENCE_IDS + 1)]
    elif failure == 'duplicate_ids': item['evidence_ids'] *= 2
    elif failure == 'id_length': item['evidence_ids'] = ['x' * (parsing.MAX_ID_CHARS + 1)]
    elif failure == 'id_whitespace': item['evidence_ids'] = ['r1\n']
    else: item['evidence_ids'] = ['r\x001']
    assert parse(kind, value) is None


@pytest.mark.parametrize('failure', ['explanation', 'entries', 'quotes', 'effect'])
def test_descriptive_conclusion_limits_remain_strict(failure):
    value = payload('conclusion')
    if failure == 'explanation': value['explanation'] = 'x' * (parsing.MAX_REASONING_CHARS + 1)
    elif failure == 'entries': value['mechanism_review'] *= mechanism.MAX_ENTRIES + 1
    elif failure == 'quotes': value['mechanism_review'][0]['source_quotes'] *= mechanism.MAX_SOURCE_QUOTES + 1
    else: value['mechanism_review'][0]['direct_effect'] = 'x' * (mechanism.MAX_TEXT_CHARS + 1)
    assert parse('conclusion', value) is None


@pytest.mark.parametrize('ids', [['r1', 'r1'], [f'r{i}' for i in range(parsing.MAX_RETIRE_IDS + 1)]])
def test_descriptive_retirement_uniqueness_and_count_remain_strict(ids):
    assert parse('reflection', dict(lessons=[], retire_ids=ids)) is None


class KindString(str):
    pass


@pytest.mark.parametrize('kind', ['', 'CONCLUSION', 'generation', None, False, 1, [], {}, KindString('conclusion')])
def test_kind_is_strict_and_unknown_values_raise_value_error(kind):
    with pytest.raises(ValueError, match='unknown structured auxiliary'):
        schema.response_format(kind)


def test_returned_formats_and_identity_do_not_share_mutable_state():
    before = schema.identity()
    value = schema.response_format('conclusion')
    value['json_schema']['schema']['properties']['lessons']['items']['required'].remove('branch')
    assert 'branch' in schema.response_format('conclusion')['json_schema']['schema']['properties']['lessons']['items']['required']
    changed = schema.identity()
    changed['response_format_sha256']['conclusion'] = 'altered'
    assert schema.identity() == before


def test_identity_binds_module_and_each_full_provider_payload(monkeypatch):
    before = schema.identity()
    assert before['version'] == schema.VERSION
    assert before['source_sha256'] == hashlib.sha256(Path(schema.__file__).read_bytes()).hexdigest()
    for kind in ('conclusion', 'reflection'):
        raw = json.dumps(schema.response_format(kind), sort_keys=True, separators=(',', ':'),
                         ensure_ascii=True, allow_nan=False).encode()
        assert before['response_format_sha256'][kind] == hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(parsing, 'MAX_TEXT_CHARS', parsing.MAX_TEXT_CHARS - 1)
    after = schema.identity()
    assert after['source_sha256'] == before['source_sha256']
    assert all(after['response_format_sha256'][kind] != before['response_format_sha256'][kind]
               for kind in ('conclusion', 'reflection'))


def test_identity_detects_module_byte_drift(monkeypatch):
    before = schema.identity()
    read = Path.read_bytes
    def changed(path):
        raw = read(path)
        return raw + b'\n# drift' if path.resolve() == Path(schema.__file__).resolve() else raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    after = schema.identity()
    assert after['source_sha256'] != before['source_sha256']
    assert after['response_format_sha256'] == before['response_format_sha256']


def test_standard_json_schema_validator_when_available():
    jsonschema = pytest.importorskip('jsonschema')
    for kind in ('conclusion', 'reflection'):
        contract = schema.response_format(kind)['json_schema']['schema']
        jsonschema.Draft202012Validator.check_schema(contract)
        validate = jsonschema.Draft202012Validator(contract).validate
        for scope in ('branch', 'global'):
            validate(payload(kind, scope))
        value = payload(kind)
        del value['lessons'][0]['branch']
        with pytest.raises(jsonschema.ValidationError):
            validate(value)
        value = payload(kind)
        value['lessons'][0]['extra'] = True
        with pytest.raises(jsonschema.ValidationError):
            validate(value)
        value = payload(kind)
        value['lessons'][0]['rule'] = 'x' * (parsing.MAX_TEXT_CHARS + 1)
        validate(value)  # Supported provider wire intentionally leaves this to local validation.
        assert parse(kind, value) is None
        value = payload(kind)
        value['lessons'][0]['evidence_ids'] *= 2
        validate(value)
        assert parse(kind, value) is None
