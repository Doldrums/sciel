"""Authored versioned R stages over current G/C/R journals; no provider calls."""
from copy import deepcopy
from types import SimpleNamespace

from sciel.engine.external_reflection import ExternalReflectionJob
from sciel.hypothesis import grounded_review
from sciel.knowledge_base.base import _snapshot
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from research_history_fixture import build_history


def append_review(capsule, config, proposal, *, version):
    config = dict(config, comparison_reviews=True, comparison_review_version=version,
        comparison_review_text=True, batch_only_lessons=True, max_aux_calls=1,
        external_history_sha256=canonical_sha(capsule))
    job = ExternalReflectionJob(capsule, expected_sha256=canonical_sha(capsule), config=config)
    plan = job.prepare(run_id=f'authored-review-v{version}', segment_id=f'review-v{version}')
    response = dict(text=canonical_json(dict(lessons=[], retire_ids=[], comparison_reviews=[proposal])),
        model=config['model'], cost=0., prompt_tokens=1, completion_tokens=1,
        finish_reason='stop', refusal=None)
    result = job.admit(plan, response, manifest_sha256=str(version)*64,
                       journal_prefix=plan['journal_prefix'])
    return SimpleNamespace(capsule=result['extended_capsule'], config=config,
        terminal=result['terminal'], response=response, proposal=proposal, job=job, plan=plan)


def grounded_history():
    capsule, final = build_history()
    group = list(capsule['comparison_catalog'].values())[-1]
    records = {row['id']: _snapshot(row) for row in final['knowledge']['experiments']}
    catalog = grounded_review.source_catalog(group, records=records,
        seen=final['knowledge']['seen_experiments'])
    proposal = dict(comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        claims=[dict(fact_id=key, value=value) for key, value in list(catalog['facts'].items())[:10]],
        conjecture='An authored unverified explanation.',
        limitations='The observed signs do not establish a general causal effect.', next_hypothesis=None)
    return append_review(capsule, final['config'], proposal, version=2)


def reasoned_history():
    prior = grounded_history()
    proposal = deepcopy(prior.proposal)
    proposal.update(next_hypothesis_reason='No bounded edit is justified by this synthetic evidence.',
        limitations='The pairs are all showing disagreement between raw and certified directions.')
    return append_review(prior.capsule, prior.config, proposal, version=3)
