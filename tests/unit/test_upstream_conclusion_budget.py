"""Exact complete C prompts preserve known costs and measured evidence."""
from copy import deepcopy
from hashlib import sha256
import pytest
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis import conclusion_source
from test_conclusion_format_repair import setup as conclusion, response
IDENTITY = dict(schema='inert-exact-counter/1', revision='fixed')

def count(prompt):
    return len(prompt.encode('utf-8')) + 7


def local(model):
    model.local_inference = True
    model.max_tokens = 512
    model.prompt_tokenizer_identity = deepcopy(IDENTITY)
    model.count_prompt_tokens = count
    return model


def bound_conclusion(items, cap, *, source=False, **options):
    old, model, events, charges = conclusion(items, **options)
    local(model)
    # Reconstruct the controller with the explicit option, not just its private
    # runtime field. The actual measured queue/record are unchanged fixture data.
    from sciel.engine.research import ResearchController
    cfg = {**old.config, 'generation_context_tokens': cap}
    c = ResearchController(cfg, model, old.log, charges.append)
    for row in old.knowledge.experiments:
        c.knowledge.record(deepcopy(row))
    c.queue = deepcopy(c.knowledge.experiments)
    c.attempts = c.outcomes = c.valid_hypotheses = 1
    if source:
        parent = ''.join(f'old_{i} = {i}\n' for i in range(180))
        child = ''.join(f'new_{i} = {i+1}\n' for i in range(180))
        record = c.queue[0]
        record['parent_source_sha256'] = sha256(parent.encode()).hexdigest()
        record['source_sha256'] = record['proposed_sha256'] = sha256(child.encode()).hexdigest()
        c._conclusion_sources[record['id']] = (parent, child)
    return c, model, events, charges


def test_irreducible_conclusion_skips_before_reservation_without_losing_measurement():
    c, model, events, charges = bound_conclusion([], 1024)
    measured = deepcopy(c.knowledge.experiments)
    c.complete_step()
    assert model.calls == charges == [] and c.aux_calls == 0 and not c._aux_requests
    assert c.queue == [] and c.stopped_reason is None and not c.knowledge.lessons
    assert c.knowledge.experiments == measured
    saved = c.conclusions['r000001a1']
    assert saved['status'] == 'skipped' and saved['observed_comparison'] == observed_comparison(measured[0])
    skip = next(e for e in events if e['phase'] == 'aux_skipped')
    assert skip['generation_dispatch'] == 'not_dispatched' and skip['current_call_cost_unknown'] is False
    assert skip['token_budget']['total_reserved_tokens'] > 1024
    assert not any(e['phase'] == 'aux_start' for e in events)


def test_conclusion_boundary_and_actual_output_allowance_are_bound():
    c, model, _, _ = bound_conclusion([], 100000, conclusion_max_tokens=3072)
    prompt = c.prepare_conclusions()[0]['prompt']
    cap = count(prompt) + 3072
    c, model, events, charges = bound_conclusion([response()], cap, conclusion_max_tokens=3072)
    c.complete_step()
    assert len(model.calls) == 1 and model.calls[0][1]['max_tokens'] == 3072
    assert charges == [.03] and c.conclusions['r000001a1']['status'] == 'parsed'
    facts = next(e['token_budget'] for e in events if e['phase'] == 'aux_start')
    assert facts['total_reserved_tokens'] == cap and facts['fits']
    c, model, _, charges = bound_conclusion([], cap - 1, conclusion_max_tokens=3072)
    c.complete_step()
    assert model.calls == charges == []


def test_source_diff_shrinks_with_exact_binding_while_measured_core_is_unchanged():
    wide, _, _, _ = bound_conclusion([], 100000, source=True)
    full = wide.prepare_conclusions()[0]
    cap = count(full['prompt']) + 512 - 1
    c, model, events, _ = bound_conclusion([], cap, source=True)
    job = c.prepare_conclusions()[0]
    conclusion_source.validate_job(job)
    assert job['source_evidence']['projection']['mode'] == 'windows'
    assert count(job['prompt']) + 512 <= cap
    assert {k:v for k,v in job['evidence'].items() if k != 'conclusion_source'} == {
        k:v for k,v in full['evidence'].items() if k != 'conclusion_source'}
    candidates = next(e['candidates'] for e in events if e['phase'] == 'conclusion_token_projection')
    assert not candidates[0]['fits'] and candidates[-1]['fits'] and len(candidates) > 1
    assert not model.calls
    bad = deepcopy(dict(job)); bad['source_evidence'] = deepcopy(dict(job['source_evidence']))
    bad['source_evidence']['candidate_source'] += '# forged\n'
    with pytest.raises(ValueError): conclusion_source.validate_job(bad)


def test_oversized_format_correction_retains_first_known_charge_and_never_retries():
    c, _, _, _ = bound_conclusion([], 100000)
    cap = count(c.prepare_conclusions()[0]['prompt']) + 512
    c, model, events, charges = bound_conclusion([response(bad_ref=True)], cap)
    c.complete_step()
    assert len(model.calls) == c.aux_calls == 1 and charges == [.03]
    assert c.conclusions['r000001a1']['status'] == 'invalid' and not c.knowledge.lessons
    assert any(e['phase'] == 'aux_skipped' and e['reason'] == 'context_token_limit' for e in events)
    assert c.stopped_reason is None
    # No pending correction/unknown outcome is manufactured by the local skip.
    assert not c._aux_requests and c.aux_cost == pytest.approx(.03)
