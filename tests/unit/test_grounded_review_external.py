"""Versioned external reflection replay with saved history and inert replies."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine.external_reflection import ExternalReflectionJob, ExternalReflectionRejected
from sciel.engine.external_research_v2 import validate
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_external_research_v2_reducer import actual
from test_external_history_v2_actual import construct, import_history
from test_grounded_review_flow import grounded_proposal
from test_comparison_review_text_flow import common_review


@pytest.fixture(scope='module')
def source():
    capsule, final = actual.__wrapped__()
    config = {**final['config'], 'comparison_reviews': True, 'comparison_review_version': 2,
        'max_aux_calls': 1, 'external_history_sha256': canonical_sha(capsule)}
    restored = validate(capsule, config=config, metric=capsule['metric'])
    group = restored['knowledge'].comparisons[-1]
    proposal = grounded_proposal(group, restored['knowledge'])
    job = ExternalReflectionJob(capsule, expected_sha256=canonical_sha(capsule), config=config)
    plan = job.prepare(run_id='inert-grounded-reflection', segment_id='inert-grounded-r1')
    return capsule, final, config, proposal, job, plan


def response(source, proposal=None):
    return dict(text=json.dumps(dict(lessons=[], retire_ids=[],
        comparison_reviews=[source[3] if proposal is None else proposal])),
        model=source[2]['model'], cost=0., prompt_tokens=1, completion_tokens=1,
        finish_reason='stop', refusal=None)


def admit(source, proposal=None):
    _, _, _, _, job, plan = source
    return job.admit(plan, response(source, proposal), manifest_sha256='a' * 64,
                     journal_prefix=plan['journal_prefix'])


def test_grounded_external_r_replays_imports_resumes_and_keeps_counters_zero(source, tmp_path):
    capsule, final, config, proposal, _, plan = source
    before = canonical_json(capsule)
    assert plan['response_format']['json_schema']['name'] == 'sciel_reflection_v3'
    result = admit(source)
    assert result['provider_dispatch_verified'] is False
    assert result['terminal']['knowledge'] == final['knowledge']
    extended = result['extended_capsule']
    replay = validate(extended, config=config, metric=capsule['metric'])
    card = replay['comparison_reviews'][proposal['comparison_id']]
    assert card['schema'] == 'sciel-comparison-review-card/2'
    assert card['unverified_interpretation']['next_hypothesis'] == proposal['next_hypothesis']
    final2 = {**final, 'config': config}
    f = construct(extended, final2, tmp_path / 'import')
    import_history(f, extended)
    assert f.c._comparison_reviews == result['terminal']['comparison_reviews']
    assert f.c.knowledge.state() == final['knowledge']
    for key in ('attempts', 'outcomes', 'aux_calls', 'aux_cost', 'reflections', 'reflected_through'):
        assert getattr(f.c, key) == 0
    clone = construct(extended, final2, tmp_path / 'clone')
    clone.c.load_state(f.c.state())
    clone.worker.load_state(f.worker.state())
    clone.worker.assert_journal(f.log.path, f.log.anchor())
    assert f.model.calls == f.charges == []
    assert canonical_json(capsule) == before
    contexts = []
    for mode in (True, False):
        f.c.config['comparison_review_text'] = mode
        contexts.append(f.c.context(NS(_all={'p1': f.parent}), f.parent, 0, 1))
    a, b = copy.deepcopy(contexts)
    on, off = a.pop('comparison_reviews'), b.pop('comparison_reviews')
    assert on['presented_count'] == off['presented_count'] == 1
    assert common_review(on) == common_review(off)
    assert canonical_json(a) == canonical_json(b)
    assert 'origin_snapshot' not in on['cards'][0]


@pytest.mark.parametrize('damage', ['value', 'target'])
def test_external_wrong_fact_or_gate_keeps_rejection_journal_without_new_memory(source, damage):
    proposal = copy.deepcopy(source[3])
    if damage == 'value': proposal['claims'][0]['value'] = 'improved'
    else: proposal['next_hypothesis']['quality_gate'] = 'near_zero_exact_only'
    with pytest.raises(ExternalReflectionRejected) as caught:
        admit(source, proposal)
    rows = list(map(json.loads, caught.value.journal.splitlines()))
    assert sum(row.get('phase') == 'aux_result' for row in rows) == 1
    assert rows[-1]['phase'] == 'reflection'
    assert rows[-1]['comparison_review_ids'] == []
    assert rows[-1]['changed'] is False


def test_receiving_legacy_policy_cannot_silently_accept_new_card_contract(source):
    result = admit(source)
    with pytest.raises(ValueError, match='receiving version 2'):
        validate(result['extended_capsule'], config={**source[2], 'comparison_review_version': 1},
                 metric=source[0]['metric'])
