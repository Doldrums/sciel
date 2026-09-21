"""Saved paired evidence -> synthetic R -> public G preparation, without draws."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine.external_research_v2 import policy_identity
from sciel.knowledge_base.catalogs import canonical_sha, canonical_json
from sciel.knowledge_base.generated_paired_review_context import PREFIX, block
from test_generated_paired_integration import (pilot_artifacts, component, construct,
    import_component, prepare, rows)


@pytest.fixture(autouse=True)
def no_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Review delivery must not dispatch a model or numerical draw.')
    for name in ('sciel.eval.repeat_draw.draw', 'sciel.eval.repeat_draw.adapter_identity',
                 'sciel.engine.llm.client.EnsembleLLM.generate'):
        monkeypatch.setattr(name, forbidden)


@pytest.fixture(scope='module')
def review_sidecar(component):
    from sciel.engine.generated_paired_reflection import GeneratedPairedReflectionJob
    from test_generated_paired_reflection import response
    config = dict(external_component_sha256=canonical_sha(component),
        restart_predictions=True, structured_auxiliary=True, comparison_reviews=True,
        comparison_review_version=3, inject_lessons=True, max_aux_calls=1,
        model='anthropic/claude-sonnet-4.5', batch_only_lessons=True,
        comparison_review_text=True,
        comparison_review_text_policy={'schema': 'sciel-review-text-policy/1', 'decisions': []})
    job = GeneratedPairedReflectionJob(component, expected_sha256=canonical_sha(component), config=config)
    plan = job.prepare(run_id='synthetic-review-delivery', segment_id='test')
    returned = response(NS(config=config, component=component), prospective=True)
    return job.admit(plan, returned, manifest_sha256='a' * 64,
                     journal_prefix=plan['journal_prefix'])['extended_capsule']


def policy_for(sidecar, status):
    decisions = []
    if status != 'pending':
        for card in sidecar['review_cards'].values():
            decisions.append(dict(schema='sciel-review-text-decision/1',
                scope='comparison_review_text', decision='rejected' if status == 'rejected' else 'accepted',
                issuer='synthetic-unit-test-only', card_sha256=canonical_sha(card),
                raw_response_sha256=card['writer']['raw_response_sha256'],
                policy_sha256='0' * 64 if status == 'stale' else canonical_sha(policy_identity()),
                audit_sha256='b' * 64, criteria_sha256='c' * 64,
                **{k: card[k] for k in ('comparison_id', 'comparison_sha256',
                    'comparator_source_sha256', 'challenger_source_sha256')}))
    return dict(schema='sciel-review-text-policy/1', decisions=decisions)


def receiving(pilot, component, sidecar, path, *, status='accepted', text=True):
    f = construct(pilot, component, path,
        external_component_review_sha256=canonical_sha(sidecar),
        comparison_reviews=True, comparison_review_version=3, comparison_review_text=text,
        comparison_review_text_policy=policy_for(sidecar, status), batch_only_lessons=True)
    import_component(f, component)
    assert f.worker.import_external_component_review(sidecar,
        expected_sha256=canonical_sha(sidecar))
    return f


@pytest.mark.parametrize('status,text', [('accepted', True), ('accepted', False),
    ('pending', True), ('rejected', True), ('stale', True)])
def test_whole_review_reaches_public_G_only_under_exact_host_decision(
        component, pilot_artifacts, review_sidecar, tmp_path, status, text):
    f = receiving(pilot_artifacts, component, review_sidecar, tmp_path, status=status, text=text)
    try:
        ctx = f.c.context(f.db, f.parent, 0, 1)
        shown = ctx['generated_paired_reviews']
        card, = shown['cards']
        original, = review_sidecar['review_cards'].values()
        assert card['branch'] == original['branch'] != ctx['branch']
        visible = status == 'accepted' and text
        assert (card['unverified_interpretation'] is not None) == visible
        assert card['text_eligibility']['status'] == status
        ident, prompt = prepare(f, ctx)
        delivered, _ = json.JSONDecoder().raw_decode(prompt.split(PREFIX, 1)[1])
        assert delivered == shown
        assert len(block(shown)) == shown['rendered_chars'] <= shown['reserved_chars'] <= 8000
        for key in ('conjecture', 'limitations', 'next_hypothesis_reason'):
            assert (original['unverified_interpretation'][key] in prompt) == visible
        assert rows(f.log)[-1]['generated_paired_reviews'] == shown
        assert f.c.knowledge.comparisons == f.c.knowledge.experiments == f.c.knowledge.lessons == []
        assert f.c.outcomes == f.c.aux_calls == f.worker.evaluations == 0
        assert f.model.calls == f.charges == [] and ident in f.c.pending
    finally:
        f.log.close()


def test_text_eligibility_cannot_release_capacity(component, pilot_artifacts, review_sidecar, tmp_path):
    contexts = []
    for index, (status, text) in enumerate([('accepted', True), ('accepted', False), ('pending', True),
                                          ('rejected', True), ('stale', True)]):
        f = receiving(pilot_artifacts, component, review_sidecar, tmp_path / str(index),
                      status=status, text=text)
        try:
            ctx = f.c.context(f.db, f.parent, 0, 1)
            contexts.append(ctx)
            prepare(f, ctx)
        finally:
            f.log.close()
    assert len({c['generated_paired_reviews']['reserved_chars'] for c in contexts}) == 1
    keys = ('generated_paired_history', 'lesson_texts', 'paired_comparisons', 'comparison_facts',
            'restart_predictions', 'conclusion_feedback', 'comparison_review_delivery')
    assert len({canonical_json({k: c.get(k) for k in keys}) for c in contexts}) == 1


@pytest.mark.parametrize('damage', ['remove', 'reason', 'branch', 'reservation'])
def test_snapshot_cannot_forge_or_drop_review(component, pilot_artifacts, review_sidecar, tmp_path, damage):
    f = receiving(pilot_artifacts, component, review_sidecar, tmp_path)
    try:
        ctx = f.c.context(f.db, f.parent, 0, 1)
        shown = ctx['generated_paired_reviews']
        if damage == 'remove':
            del ctx['generated_paired_reviews']
        else:
            if damage == 'reason':
                shown['cards'][0]['unverified_interpretation']['next_hypothesis_reason'] = 'Invented.'
            elif damage == 'branch':
                shown['cards'][0]['branch'] = ctx['branch']
            else:
                shown['reserved_chars'] = 0
            shown['context_sha256'] = canonical_sha({k: v for k, v in shown.items() if k != 'context_sha256'})
        before, events = f.c.state(), rows(f.log)
        with pytest.raises(ValueError, match='authoritative'):
            prepare(f, ctx)
        assert f.c.state() == before and rows(f.log) == events
        assert f.model.calls == f.charges == []
    finally:
        f.log.close()


@pytest.mark.parametrize('key,value', [('external_component_review_sha256', '0' * 64),
    ('comparison_review_text', False), ('comparison_review_text_policy',
                                      {'schema': 'sciel-review-text-policy/1', 'decisions': []})])
@pytest.mark.parametrize('boundary', ['context', 'aux', 'complete_step'])
def test_changed_trust_cannot_prepare_or_consume_work(component, pilot_artifacts,
        review_sidecar, tmp_path, key, value, boundary):
    f = receiving(pilot_artifacts, component, review_sidecar, tmp_path)
    try:
        before, events = f.c.state(), rows(f.log)
        original = copy.deepcopy(f.c.config[key])
        f.c.config[key] = value
        try:
            with pytest.raises(ValueError):
                if boundary == 'context':
                    f.c.context(f.db, f.parent, 0, 1)
                elif boundary == 'aux':
                    f.c._aux('reflection', 'inert', 1)
                else:
                    f.c.complete_step(reflect=False)
        finally:
            f.c.config[key] = original
        assert f.c.state() == before and rows(f.log) == events
        assert f.model.calls == f.charges == []
    finally:
        f.log.close()
