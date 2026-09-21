"""Current pure admission plus an authored SDK reply and durable cost ledger."""
from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

from sciel.campaign.budget import BudgetedLLM, CostLedger
from sciel.engine.external_reflection import ExternalReflectionRejected
from sciel.engine.generated_paired_reflection import GeneratedPairedReflectionJob
from sciel.hypothesis.comparison_targets import QUALITY_GATE, SCHEMA as TARGET_SCHEMA
from sciel.knowledge_base.catalogs import canonical_sha
from generated_paired_integration_fixture import component, pilot_artifacts

CATALOG_MARKER = 'GENERATED_PAIRED_REVIEW_SOURCES_JSON:\n'


def sha(raw):
    return hashlib.sha256(raw.encode() if isinstance(raw, str) else raw).hexdigest()


def catalogs(plan):
    values, _ = json.JSONDecoder().raw_decode(plan['prompt'].split(CATALOG_MARKER, 1)[1])
    return {value['comparison_id']: value for value in values}


def build(path, kind):
    capsule = component.__wrapped__(pilot_artifacts.__wrapped__())
    config = dict(external_component_sha256=canonical_sha(capsule),
        restart_predictions=True, structured_auxiliary=True, comparison_reviews=True,
        comparison_review_version=3, inject_lessons=True, max_aux_calls=1,
        model='fixture/model', batch_only_lessons=True, comparison_review_text=True,
        comparison_review_text_policy={'schema': 'sciel-review-text-policy/1', 'decisions': []})
    job = GeneratedPairedReflectionJob(capsule, expected_sha256=canonical_sha(capsule), config=config)
    plan = job.prepare(run_id='authored-rejection-' + kind, segment_id='fixture')
    catalog = catalogs(plan)
    rows = []
    lengths = (826, 905, 1027) if kind == 'overlong' else (489, 370, 358)
    for index, value in enumerate(catalog.values()):
        count = 9 if kind == 'overlong' else (10 if index < 2 else 9)
        claims = [dict(fact_id=key, value=value['facts'][key]) for key in sorted(value['facts'])[:count]]
        if kind == 'overlong':
            claims += [dict(fact_id=key, value=value[key]) for key in ('comparison_id', 'comparison_sha256')]
        rows.append(dict(comparison_id=value['comparison_id'], comparison_sha256=value['comparison_sha256'],
            claims=claims, conjecture='Authored unverified explanation.',
            limitations='Fixed observations do not establish causal benefit.', next_hypothesis=None,
            next_hypothesis_reason='x' * lengths[index]))
    if kind == 'foreign_target':
        first = next(iter(catalog.values()))
        rows[0]['next_hypothesis'] = dict(schema=TARGET_SCHEMA,
            base_source_sha256=first['comparator_source_sha256'],
            comparator_source_sha256=capsule['root_parent']['source_sha256'],
            change='Authored bounded change with an intentionally unavailable local comparator.',
            quality_gate=QUALITY_GATE, diagnostics=[])
    raw = json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=rows))
    client = Mock()
    provider = NS(id='fixture-' + kind, usage=NS(cost=.01, prompt_tokens=100, completion_tokens=100),
        choices=[NS(index=0, finish_reason='stop', message=NS(content=raw, refusal=None))])
    client.chat.completions.create.return_value = provider
    ledger = CostLedger(path / 'cost.json', 10)
    model = config['model']
    llm = BudgetedLLM(dict(models=[model], weights=[1], max_tokens=4096,
        api_base='https://openrouter.ai/api/v1'),
        dict(models=[dict(id=model, pricing=dict(prompt='.000001', completion='.000002'))]),
        ledger, plan['run_id'], 'inert-test-key', client)
    returned = asdict(llm.generate(plan['prompt'], model, response_format=plan['response_format']))
    response = {key: returned[key] for key in
        ('text', 'model', 'cost', 'prompt_tokens', 'completion_tokens', 'finish_reason', 'refusal')}
    try:
        job.admit(plan, response, manifest_sha256='a' * 64, journal_prefix=plan['journal_prefix'])
    except ExternalReflectionRejected as failure:
        journal = failure.journal
    else:
        raise AssertionError('The deliberately invalid reflection was admitted')
    return NS(component=capsule, config=config, job=job, plan=plan, catalogs=catalog,
        response=response, raw=raw, raw_sha=sha(raw), provider=provider, client=client,
        llm=llm, ledger=ledger, events=[json.loads(line) for line in journal.splitlines()])
