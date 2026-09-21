"""Reasoned decisions are versioned wire data, never automatic experiment authority."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.config import apply_defaults
from sciel.hypothesis import auxiliary_schema, grounded_review, prompts, reasoned_review
from sciel.hypothesis.parsing import parse_reflection
from sciel.knowledge_base.catalogs import canonical_json
from test_comparison_review_wire import dictionaries, payload as legacy_payload
from test_grounded_review_wire import payload as grounded_payload, proposal as grounded_proposal
from test_external_history_v2_actual import actual_history


OPTIONS = dict(comparison_reviews=True, comparison_review_version=3)


def proposal(**changes):
    value = dict(grounded_proposal(), next_hypothesis_reason=(
        'No bounded change can be specified from the supplied source contrast without '
        'an available implementation of the proposed alternate selection criterion.'))
    value.update(changes)
    return value


def payload(**changes):
    return dict(lessons=[], retire_ids=[], comparison_reviews=[proposal(**changes)])


def parse(value, version=3):
    return parse_reflection(json.dumps(value), comparison_reviews=True,
                            comparison_review_version=version)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def target():
    from sciel.hypothesis.comparison_targets import SCHEMA, QUALITY_GATE
    return dict(schema=SCHEMA, base_source_sha256='b' * 64,
        comparator_source_sha256='c' * 64, change='Reduce both grid loops and denominators from eight to six.',
        quality_gate=QUALITY_GATE, diagnostics=[])


@pytest.mark.parametrize('next_target', [None, target()])
def test_null_and_proposal_require_and_preserve_their_unverified_reason(next_target):
    value = payload(next_hypothesis=next_target,
        next_hypothesis_reason='  An unverified rationale, preserved exactly.  ')
    parsed = parse(value)
    assert parsed.to_dict() == value
    assert parsed.lessons == [] and parsed.retire_ids == []
    detached = parsed.to_dict()
    detached['comparison_reviews'][0]['next_hypothesis_reason'] = 'changed'
    assert parsed.to_dict() == value


@pytest.mark.parametrize('reason', [None, False, 0, [], {}, '', ' \n\t', 'x' * 1025])
@pytest.mark.parametrize('next_target', [None, target()])
def test_reason_is_required_nonblank_and_bounded_for_both_decisions(reason, next_target):
    assert parse(payload(next_hypothesis=next_target, next_hypothesis_reason=reason)) is None


def test_legacy_and_grounded_payloads_are_never_silently_upgraded_or_downgraded():
    values = [legacy_payload(), grounded_payload(), payload()]
    for version, value in enumerate(values, 1):
        before = copy.deepcopy(value)
        assert parse(value, version).to_dict() == value
        for other in ({1, 2, 3} - {version}):
            assert parse(value, other) is None
        assert value == before
    no_reviews = dict(lessons=[], retire_ids=[], comparison_reviews=[])
    for version in (1, 2, 3):
        assert parse(no_reviews, version).to_dict() == no_reviews


def test_duplicate_reason_extra_decision_and_weaker_target_are_rejected():
    value = payload()
    raw = json.dumps(value).replace('"next_hypothesis_reason":',
        '"next_hypothesis_reason":"first","next_hypothesis_reason":')
    assert parse_reflection(raw, **OPTIONS) is None
    assert parse(payload(decision='abstain')) is None
    assert parse(payload(next_hypothesis={**target(), 'quality_gate': 'exact_only'})) is None
    assert parse(payload(next_hypothesis_reason='x' * 1024)).to_dict() == payload(next_hypothesis_reason='x' * 1024)


def test_reason_parser_does_not_establish_truth_or_authority():
    # Pure shape parsing must not silently turn the reason into a checked fact.
    value = payload(next_hypothesis_reason='A knowingly unsupported claim about missing data.',
                    claims=[dict(fact_id='quality:outcome', value='improved')])
    assert parse(value).to_dict() == value
    assert parse(value).lessons == []


@pytest.mark.parametrize('version', [True, False, None, 0, 4, 3.0, '3', [], {}])
def test_version_dispatch_requires_exact_supported_integer(version):
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
    with pytest.raises(ValueError, match='comparison_review_version'):
        apply_defaults({'research': dict(structured_auxiliary=True, comparison_reviews=True,
                                       comparison_review_version=version)})


def test_v3_is_explicit_and_requires_enabled_structured_review_channel():
    assert parse_reflection('{"lessons":[],"retire_ids":[]}', comparison_review_version=3) is None
    for function, args in ((auxiliary_schema.response_format, ('reflection',)),
                           (auxiliary_schema.identity, ()),
                           (prompts.reflection_prompt, ([], []))):
        with pytest.raises(ValueError, match='requires comparison_reviews'):
            function(*args, comparison_review_version=3)
    with pytest.raises(ValueError, match='structured'):
        prompts.reflection_prompt([], [], comparison_review_groups=[], **OPTIONS)
    with pytest.raises(ValueError, match='requires comparison_reviews'):
        apply_defaults({'research': {'comparison_review_version': 3}})
    configured = apply_defaults({'research': dict(structured_auxiliary=True, **OPTIONS)})
    assert configured['research']['comparison_review_version'] == 3
    assert apply_defaults(configured) == configured
    assert 'comparison_review_version' not in apply_defaults({'research': {}})['research']


def test_v4_wire_is_closed_fresh_and_preserves_conclusion_schema():
    wire = auxiliary_schema.response_format('reflection', **OPTIONS)
    assert wire['json_schema']['name'] == 'sciel_reflection_v4'
    assert wire['json_schema']['strict'] is True
    schema = wire['json_schema']['schema']
    review = schema['properties']['comparison_reviews']['items']
    assert set(review['properties']) == set(proposal())
    assert 'next_hypothesis_reason' in review['required']
    assert review['properties']['next_hypothesis']['type'] == ['object', 'null']
    for item in dictionaries(schema):
        assert not set(item) & {'minLength', 'maxLength', 'maxItems', 'uniqueItems',
            'minimum', 'maximum', 'pattern', 'contains', 'minContains', 'maxContains'}
        if item.get('type') in ('object', ['object', 'null']):
            assert item['additionalProperties'] is False
            assert set(item['required']) == set(item['properties'])
    review['properties'].clear()
    assert auxiliary_schema.response_format('reflection', **OPTIONS)['json_schema']['schema'][
        'properties']['comparison_reviews']['items']['properties']
    assert auxiliary_schema.response_format('conclusion', **OPTIONS) == auxiliary_schema.response_format('conclusion')


def test_v3_identity_pins_new_policy_and_exact_envelopes_without_mutating_v2():
    old = auxiliary_schema.identity(comparison_reviews=True, comparison_review_version=2)
    identity = auxiliary_schema.identity(**OPTIONS)
    assert identity['comparison_review_version'] == 3
    assert identity['reasoned_review_policy'] == reasoned_review.identity()
    assert 'reasoned_review_policy' not in old
    assert old['grounded_review_policy'] == grounded_review.identity()
    for kind in ('conclusion', 'reflection'):
        assert identity['response_format_sha256'][kind] == digest(auxiliary_schema.response_format(kind, **OPTIONS))
    identity['reasoned_review_policy'].clear()
    assert auxiliary_schema.identity(**OPTIONS)['reasoned_review_policy'] == reasoned_review.identity()


def test_new_prompt_delivers_sources_and_reasoned_abstention_without_forcing_a_win(monkeypatch):
    records, seen, calls = {'origin': object()}, {'origin': 'a' * 64}, []

    def sources(groups, *, records, seen):
        calls.append((groups, records, seen))
        return 'REASONED_SOURCE_SENTINEL\n'

    monkeypatch.setattr(reasoned_review, 'source_block', sources)
    prompt = prompts.reflection_prompt([], [], structured=True, comparison_review_groups=[],
        comparison_review_records=records, comparison_review_seen=seen, **OPTIONS)
    assert calls == [([], records, seen)]
    for text in ('REASONED_SOURCE_SENTINEL', 'next_hypothesis and next_hypothesis_reason',
        'Mixed or refuted evidence may motivate a test without supporting a signed lesson',
        'proposing a test does not claim it will pass the quality gate',
        'Otherwise return null', 'concrete obstacle or missing input or measurement',
        'Do not invent a change merely to avoid null', 'reason is required, nonblank and at most 1024',
        'remain unverified even when every claim passes', 'strict two-metric no-loss gate',
        'diagnostics=[] proposes no mechanism measurement'):
        assert text in prompt


@pytest.mark.parametrize('enabled,version,wire_sha,prompt_sha', [
    (False, 1, 'e2d06308d0c4bef0348035f5bc6d13a30b9f6954d3e5dbe90ac55f5705347e16',
     'a974a198863503133cfff6c0b108a3cb559a5fc575a0a73b4973bc6fb359497e'),
    (True, 1, '83a1737bd7822c1b4b441a821df3d387e8cd6f86eaad52ff41be37eb2524e334',
     '856ef8cd14dd0e5529da6c5b53e622fb6cafb28ae71cd51ccb0b3712b77ccb45'),
    (True, 2, 'f324735bd1689e2145fd73bad395764b3a8b62f6dd27b406d2138d396b994214',
     '453feb0b79b601416c5b34b2ff80d2b5480e22a9e348798a3b5503aa8e7b71de')])
def test_legacy_wire_and_prompt_bytes_remain_exact(enabled, version, wire_sha, prompt_sha):
    options = dict(comparison_reviews=enabled, comparison_review_version=version)
    assert digest(auxiliary_schema.response_format('reflection', **options)) == wire_sha
    if enabled:
        options['comparison_review_groups'] = []
    prompt = prompts.reflection_prompt([], [], structured=True, **options)
    assert hashlib.sha256(prompt.encode()).hexdigest() == prompt_sha
    assert 'next_hypothesis_reason' not in prompt


def test_full_authored_v2_prompt_and_wire_match_the_external_plan(actual_history):
    from sciel.engine.external_reflection import ExternalReflectionJob
    from sciel.engine.reflection_context import prepare_reflection_context
    from sciel.knowledge_base import KnowledgeBase
    from sciel.knowledge_base.catalogs import canonical_sha

    capsule, final = actual_history
    config = {**final['config'], 'comparison_reviews': True, 'comparison_review_version': 2,
              'comparison_review_text': True, 'batch_only_lessons': True, 'max_aux_calls': 1,
              'external_history_sha256': canonical_sha(capsule)}
    job = ExternalReflectionJob(capsule, expected_sha256=config['external_history_sha256'], config=config)
    plan = job.prepare(run_id='authored-v2-wire', segment_id='fixture')
    kb = KnowledgeBase(**{key: config[key] for key in ('max_lessons', 'max_experiments', 'max_fragments')})
    kb.load_state(final['knowledge'])
    before = kb.state()
    prepared = prepare_reflection_context(kb, final['conclusions'], config=config, interval=1, through=2)
    assert prepared['prompt'] == plan['prompt']
    assert prepared['response_format'] == plan['response_format']
    assert prepared['response_format']['json_schema']['name'] == 'sciel_reflection_v3'
    assert 'next_hypothesis_reason' not in prepared['prompt']
    assert kb.state() == before
