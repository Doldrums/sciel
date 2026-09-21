"""Reflection errors are bounded host diagnostics, never repaired model replies."""
import copy
import hashlib
import json

import pytest

from sciel.hypothesis import parsing, reasoned_review
from sciel.hypothesis.parsing import decode_reflection, parse_reflection


def proposal(version=3, index=0, **changes):
    value = dict(comparison_id=f'comparison:{index}', comparison_sha256='a' * 64,
                 limitations='A retained paired group does not establish causality.',
                 next_hypothesis=None)
    if version == 1:
        value['interpretation'] = 'A possible explanation remains unverified.'
    else:
        value.update(claims=[dict(fact_id='quality:strict_verdict', value='INCONCLUSIVE')],
                     conjecture='The proposed explanation remains unverified.')
    if version == 3:
        value['next_hypothesis_reason'] = 'No concrete bounded source change has been identified.'
    value.update(changes)
    return value


def payload(version=3):
    value = dict(lessons=[dict(rule=' An unchanged legacy lesson. ', polarity='negative',
                             scope='branch', branch='p1', evidence_ids=['r1'])],
                 retire_ids=['l1'])
    if version:
        value['comparison_reviews'] = [proposal(version)]
    return value


def options(version):
    return dict(comparison_reviews=bool(version), comparison_review_version=version or 1)


def decoded(value, version=3):
    return decode_reflection(json.dumps(value), **options(version))


def assert_rejected(response, expected, **kwargs):
    before = copy.deepcopy(response)
    result = decode_reflection(response, **kwargs)
    assert result == dict(reflection=None, error=expected)
    assert parse_reflection(response, **kwargs) is None
    assert response == before
    return result


@pytest.mark.parametrize('version', [0, 1, 2, 3])
@pytest.mark.parametrize('envelope', [
    '{}', 'REFLECTION_JSON: {}', 'Prior prose.\n  REFLECTION_JSON:\n{}',
    '```json\n{}\n```', 'REFLECTION_JSON:\n```json\n{}\n```'])
def test_all_legacy_envelopes_and_review_versions_preserve_exact_values(version, envelope):
    value = payload(version)
    raw = envelope.format(json.dumps(value))
    result = decode_reflection(raw, **options(version))
    assert result['error'] is None
    assert result['reflection'].to_dict() == value
    assert parse_reflection(raw, **options(version)).to_dict() == value
    result['reflection'].to_dict()['lessons'].clear()
    assert result['reflection'].to_dict() == value


@pytest.mark.parametrize('response', [
    None, False, 12, [], {}, '', 'not JSON', '[]',
    '{"lessons":[],"retire_ids":[],"retire_ids":[]}',
    '{"lessons":[],"retire_ids":[],"unknown":NaN}',
    '```JSON\n{"lessons":[],"retire_ids":[]}\n```',
    '```python\n{"lessons":[],"retire_ids":[]}\n```',
    '```json\n{"lessons":[],"retire_ids":[]}\n```\nextra',
    'CONCLUSION_JSON:\n```json\n{"lessons":[],"retire_ids":[]}\n```',
    'REFLECTION_JSON: {}\nREFLECTION_JSON: {}',
    '{"lessons":[],"retire_ids":[]} {}',
    '[' * 2000 + ']' * 2000,
    'x' * (parsing.MAX_RESPONSE_CHARS + 1),
    '{"lessons":[],"retire_ids":[],"x":"' + 'x' * parsing.MAX_BLOCK_CHARS + '"}',
])
def test_invalid_duplicate_fenced_and_oversized_transport_is_never_repaired(response):
    assert_rejected(response, dict(code='invalid_reflection_transport', path='response'))


@pytest.mark.parametrize(('kwargs', 'code', 'path'), [
    ({'comparison_reviews': 1}, 'invalid_comparison_reviews_requirement', 'comparison_reviews'),
    ({'comparison_reviews': None}, 'invalid_comparison_reviews_requirement', 'comparison_reviews'),
    ({'comparison_review_version': True}, 'invalid_comparison_review_version', 'comparison_review_version'),
    ({'comparison_review_version': 4}, 'invalid_comparison_review_version', 'comparison_review_version'),
    ({'comparison_review_version': '3'}, 'invalid_comparison_review_version', 'comparison_review_version'),
    ({'comparison_review_version': 3}, 'comparison_reviews_required', 'comparison_reviews'),
])
def test_option_failures_have_fixed_paths(kwargs, code, path):
    assert_rejected(json.dumps(payload(0)), dict(code=code, path=path), **kwargs)


@pytest.mark.parametrize(('change', 'code', 'path'), [
    ({'model_secret_unexpected_field': 'DO_NOT_ECHO'}, 'invalid_reflection_fields', 'reflection'),
    ({'lessons': [{'rule': 'DO_NOT_ECHO'}]}, 'invalid_reflection_lessons', 'lessons'),
    ({'retire_ids': ['DO_NOT_ECHO with spaces']}, 'invalid_reflection_retire_ids', 'retire_ids'),
    ({'comparison_reviews': [{'DO_NOT_ECHO': 'private text'}]}, 'invalid_comparison_reviews', 'comparison_reviews'),
])
def test_other_shape_errors_use_deterministic_fallback_without_rejected_values(change, code, path):
    value = payload()
    value.update(change)
    result = assert_rejected(json.dumps(value), dict(code=code, path=path), **options(3))
    assert 'DO_NOT_ECHO' not in json.dumps(result)


@pytest.mark.parametrize('reason', [None, False, 7, [], {}, '', ' \n\t', 'SECRET' * 172])
def test_reason_diagnostic_reports_only_safe_index_and_length(reason):
    value = payload()
    value['comparison_reviews'] = [proposal(index=index) for index in range(3)]
    value['comparison_reviews'][2]['next_hypothesis_reason'] = reason
    expected = dict(code='text_too_long' if isinstance(reason, str) and len(reason) > 1024 else 'invalid_text',
                    path='comparison_reviews[2].next_hypothesis_reason', maximum=1024)
    if isinstance(reason, str):
        expected['actual'] = len(reason)
    result = assert_rejected(json.dumps(value), expected, **options(3))
    assert 'SECRET' not in json.dumps(result)
    with pytest.raises(reasoned_review.ReasonedReviewError) as caught:
        reasoned_review.validate_proposals(value['comparison_reviews'])
    assert caught.value.diagnostic == expected
    assert str(caught.value) == 'reasoned comparison review: invalid next hypothesis reason'
    detached = caught.value.diagnostic
    detached['path'] = 'tampered'
    detached['maximum'] = -1
    assert caught.value.diagnostic == expected
    result['error'].clear()
    assert decoded(value)['error'] == expected


def test_reason_and_proposal_bounds_are_unchanged_and_content_is_not_normalized():
    value = payload()
    value['comparison_reviews'][0]['next_hypothesis_reason'] = ' ' + 'x' * 1022 + ' '
    assert decoded(value)['reflection'].to_dict() == value
    value['comparison_reviews'][0]['next_hypothesis_reason'] += 'x'
    assert decoded(value)['error'] == dict(code='text_too_long',
        path='comparison_reviews[0].next_hypothesis_reason', actual=1025, maximum=1024)
    value['comparison_reviews'] = [proposal(index=index) for index in range(4)]
    assert decoded(value)['reflection'].to_dict() == value
    value['comparison_reviews'].append(proposal(index=4))
    assert decoded(value)['error'] == dict(code='invalid_comparison_reviews', path='comparison_reviews')


@pytest.mark.parametrize('version', [1, 2, 3])
def test_semantically_unbound_claims_remain_shape_only_and_versions_do_not_convert(version):
    value = payload(version)
    assert decoded(value, version)['reflection'].to_dict() == value
    for other in {1, 2, 3} - {version}:
        assert decoded(value, other) == dict(reflection=None,
            error=dict(code='invalid_comparison_reviews', path='comparison_reviews'))
    if version >= 2:
        value['comparison_reviews'][0]['claims'] = [dict(fact_id='not_a_delivered_fact', value='false')]
        assert decoded(value, version)['reflection'].to_dict() == value


def test_authored_answer_remains_byte_identical_and_reports_third_reason_limit():
    value = payload()
    value['comparison_reviews'] = [proposal(index=i, next_hypothesis_reason='x' * size)
                                   for i, size in enumerate((826, 905, 1027))]
    raw = json.dumps(value)
    provider = {'choices': [{'message': {'content': raw}}]}
    before = hashlib.sha256(raw.encode()).hexdigest()
    assert_rejected(raw, dict(code='text_too_long',
        path='comparison_reviews[2].next_hypothesis_reason', actual=1027, maximum=1024), **options(3))
    assert provider['choices'][0]['message']['content'] == raw
    assert hashlib.sha256(raw.encode()).hexdigest() == before
