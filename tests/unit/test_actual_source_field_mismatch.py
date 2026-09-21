"""A real rejected review quotes the operator body but names the wrapper field."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.hypothesis.auxiliary_schema import response_format
from sciel.hypothesis.mechanism_review import (
    SOURCE_QUOTE_RULES, source_quote_identity, validate_binding, MechanismReviewError)
from sciel.hypothesis.parsing import decode_conclusion
from sciel.hypothesis.prompts import conclusion_prompt
from sciel.hypothesis.rendering import presentation_identity

FIXTURE = Path(__file__).parents[1] / 'fixtures/auxiliary_source_field_mismatch.json'


def actual():
    value = json.loads(FIXTURE.read_text())
    assert hashlib.sha256(value['response_text'].encode()).hexdigest() == value['provenance']['response_text_sha256']
    return value


def test_paid_json_was_valid_but_exact_source_field_binding_still_rejects():
    data = actual()
    parsed = decode_conclusion(data['response_text'], require_mechanism=True)['conclusion']
    assert parsed is not None
    review = parsed.mechanism_review
    quote = review[1]['source_quotes'][0]
    assert quote['field'] == 'controlled_change.after.text'
    assert quote['text'].count('\n') == 1 and '\\' not in quote['text']
    assert quote['text'] in data['evidence']['controlled_change']['target_body']
    assert quote['text'] not in data['evidence']['controlled_change']['after']['text']
    before = copy.deepcopy(data)
    with pytest.raises(MechanismReviewError) as rejected:
        validate_binding(review, data['evidence'])
    assert rejected.value.diagnostic == {
        'code': 'source_quote_mismatch', 'path': 'mechanism_review[1].source_quotes[0].text'}
    assert data == before


@pytest.mark.parametrize('synthetic_case', ['correct_field', 'single_line'])
def test_synthetic_examples_bind_without_normalizing_any_source(synthetic_case):
    data = actual()
    review = json.loads(data['response_text'])['mechanism_review']
    quote = review[1]['source_quotes'][0]
    if synthetic_case == 'correct_field':
        quote['field'] = 'controlled_change.target_body'
    else:
        quote['text'] = quote['text'].split('\n')[0]
    before = copy.deepcopy(data)
    assert validate_binding(review, data['evidence']) == review
    assert data == before
    # These are new inert examples; the paid response remains rejected above.
    assert json.loads(data['response_text'])['mechanism_review'] != review


def test_prompt_and_native_schema_explain_exact_field_choice_and_identity():
    evidence = actual()['evidence']
    assert SOURCE_QUOTE_RULES in conclusion_prompt(evidence, structured=True)
    schema = response_format('conclusion')['json_schema']['schema']
    description = schema['properties']['mechanism_review']['items']['properties']['source_quotes']['description']
    assert SOURCE_QUOTE_RULES in description
    assert presentation_identity()['mechanism_source_quotation'] == source_quote_identity()
    assert source_quote_identity()['rules_sha256'] == hashlib.sha256(SOURCE_QUOTE_RULES.encode()).hexdigest()
