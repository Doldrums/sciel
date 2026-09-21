"""Reasoned R over an authored v2 history using inert replies.

The seven preceding journals and v2 card remain immutable. A new in-memory
derivative adds one explicit v3 R without a provider, candidate or solver."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.engine.external_reflection import ExternalReflectionJob, ExternalReflectionRejected
from sciel.engine.external_research_v2 import policy_identity, validate
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_comparison_review_text_flow import common_review
from test_external_history_v2_actual import construct, import_history
from test_reasoned_review_flow import ABSTENTION, RATIONALE

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def source():
    from external_review_fixture import grounded_history
    authored = grounded_history()
    capsule, config, terminal = authored.capsule, authored.config, authored.terminal
    original = canonical_json(capsule)
    capsule['policy'] = policy_identity()
    config.update(comparison_review_version=3, external_history_sha256=canonical_sha(capsule))
    replay = validate(capsule, config=config, metric=capsule['metric'])
    assert replay['knowledge'].state() == terminal['knowledge']
    assert replay['comparison_reviews'] == terminal['comparison_reviews']
    assert len(capsule['journals']) == 7 and len(capsule['operations']) == 10
    old_card, = terminal['comparison_reviews'].values()
    assert old_card['schema'] == 'sciel-comparison-review-card/2'
    assert 'next_hypothesis_reason' not in old_card['unverified_interpretation']
    proposal = dict(comparison_id=old_card['comparison_id'], comparison_sha256=old_card['comparison_sha256'],
                    **copy.deepcopy(old_card['unverified_interpretation']), next_hypothesis_reason=ABSTENTION)
    job = ExternalReflectionJob(capsule, expected_sha256=canonical_sha(capsule), config=config)
    plan = job.prepare(run_id='inert-reasoned-review', segment_id='inert-reasoned-r1')
    return NS(capsule=capsule, original=original, config=config, terminal=terminal, proposal=proposal, job=job, plan=plan)


def admit(source, proposal):
    response = dict(text=json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=[proposal])),
                    model=source.config['model'], cost=0., prompt_tokens=1, completion_tokens=1,
                    finish_reason='stop', refusal=None)
    return source.job.admit(source.plan, response, manifest_sha256='a' * 64,
                            journal_prefix=source.plan['journal_prefix'])


@pytest.fixture(scope='module', params=['abstain', 'propose'])
def admitted(source, request):
    proposal = copy.deepcopy(source.proposal)
    if request.param == 'propose':
        group = source.capsule['comparison_catalog'][proposal['comparison_id']]
        proposal.update(next_hypothesis_reason=RATIONALE, next_hypothesis=dict(
            schema='sciel-comparison-target/1', base_source_sha256=group['challenger_source_sha256'],
            comparator_source_sha256=group['challenger_source_sha256'],
            change='Change only the reinsertion grid from 8 by 8 to 6 by 6.',
            quality_gate='paired_raw_exact_zero_loss_strict_v1', diagnostics=[]))
    return proposal, admit(source, proposal)


def test_actual_old_history_and_new_decision_import_resume_without_dispatch(source, admitted, tmp_path):
    proposal, result = admitted
    assert source.plan['response_format']['json_schema']['name'] == 'sciel_reflection_v4'
    extended = result['extended_capsule']
    assert extended['journals'][:-1] == source.capsule['journals']
    assert extended['operations'][:-1] == source.capsule['operations']
    assert result['terminal']['knowledge'] == source.terminal['knowledge']
    assert result['terminal']['conclusions'] == source.terminal['conclusions']
    assert result['provider_dispatch_verified'] is False
    final = dict(config=source.config, metric=source.capsule['metric'], **result['terminal'])
    f = construct(extended, final, tmp_path / 'import')
    import_history(f, extended)
    card = f.c._comparison_reviews[proposal['comparison_id']]
    assert card['schema'] == 'sciel-comparison-review-card/3'
    assert card['unverified_interpretation']['next_hypothesis_reason'] == proposal['next_hypothesis_reason']
    assert card['unverified_interpretation']['next_hypothesis'] == proposal['next_hypothesis']
    clone = construct(extended, final, tmp_path / 'clone')
    clone.c.load_state(f.c.state())
    clone.worker.load_state(f.worker.state())
    clone.worker.assert_journal(f.log.path, f.log.anchor())
    for key in ('attempts', 'outcomes', 'aux_calls', 'aux_cost', 'reflections', 'reflected_through'):
        assert getattr(f.c, key) == 0
    assert f.worker.calls_started == f.worker.calls_returned == f.worker.evaluations == 0
    assert f.model.calls == f.charges == []
    contexts = []
    for mode in (True, False):
        f.c.config['comparison_review_text'] = mode
        contexts.append(f.c.context(NS(_all={'p1': f.parent}), f.parent, 0, 1))
    a, b = copy.deepcopy(contexts)
    on, off = a.pop('comparison_reviews'), b.pop('comparison_reviews')
    assert on['presented_count'] == off['presented_count'] == 1
    assert common_review(on) == common_review(off)
    assert canonical_json(a) == canonical_json(b)
    assert on['cards'][0]['unverified_interpretation']['next_hypothesis_reason'] == proposal['next_hypothesis_reason']
    assert off['cards'][0]['unverified_interpretation'] is None
    assert 'origin_snapshot' not in on['cards'][0]


@pytest.mark.parametrize('receiving', [1, 2])
def test_new_historical_wire_requires_receiving_version_three(source, admitted, receiving):
    # Receiver1 encounters the immutable real v2 R first; receiver2 must reach
    # and reject the new v3 R rather than reinterpret it as an old envelope.
    with pytest.raises(ValueError, match='receiving version ' + str(receiving + 1)):
        validate(admitted[1]['extended_capsule'],
                 config={**source.config, 'comparison_review_version': receiving}, metric=source.capsule['metric'])


def test_empty_new_wire_cannot_bypass_version_gate(source):
    response = dict(text=json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=[])),
                    model=source.config['model'], cost=0., prompt_tokens=1, completion_tokens=1,
                    finish_reason='stop', refusal=None)
    result = source.job.admit(source.plan, response, manifest_sha256='a' * 64,
                              journal_prefix=source.plan['journal_prefix'])
    assert result['accepted_review_ids'] == []
    assert result['terminal']['comparison_reviews'] == source.terminal['comparison_reviews']
    with pytest.raises(ValueError, match='receiving version 3'):
        validate(result['extended_capsule'], config={**source.config, 'comparison_review_version': 2},
                 metric=source.capsule['metric'])


def test_missing_reason_rejected_without_overwriting_old_v2_card(source):
    proposal = copy.deepcopy(source.proposal)
    proposal.pop('next_hypothesis_reason')
    before = canonical_json(source.capsule)
    with pytest.raises(ExternalReflectionRejected) as rejected:
        admit(source, proposal)
    rows = list(map(json.loads, rejected.value.journal.splitlines()))
    assert sum(row.get('phase') == 'aux_result' for row in rows) == 1
    assert rows[-1]['comparison_review_ids'] == []
    assert rows[-1]['comparison_reviews_changed'] is False and rows[-1]['changed'] is False
    assert canonical_json(source.capsule) == before
