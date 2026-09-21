"""Deterministic search, draw and journal fixtures for repeat integration."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import pytest
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.hypothesis.rendering import decode_evidence


ROOT = Path(__file__).resolve().parents[2]


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def journal(directory):
    return [json.loads(line) for line in (Path(directory) / 'sciel_events.jsonl').read_text().splitlines()]


def fixture_code(step):
    return (f'# fixture_step = {step}\n'
            f'def construct_packing():\n    return {step}\n'
            'def run_packing():\n    return construct_packing()\n')


class FakeLLM:
    models = ['fixture']
    weights = [1.]
    max_retries = 1

    def __init__(self, start_step=0):
        self.mutation_count = start_step
        self.calls = []

    def choose_model(self, rng):
        return 'fixture'

    def generate(self, prompt, model, temperature=None):
        if 'Review this one experiment.' in prompt:
            from _mechanism_fixtures import unknown_mechanism
            raw, _ = json.JSONDecoder().raw_decode(prompt.split('EVIDENCE_JSON:', 1)[1].lstrip())
            evidence = decode_evidence(raw)
            kind = 'conclusion'
            lessons = ([dict(rule=f"Synthetic observed improvement {evidence['id']}",
                polarity='positive', scope='branch', evidence_ids=[evidence['id']])]
                if evidence['status'] == 'evaluated' and evidence['actual_delta'] > 0 else [])
            text = 'CONCLUSION_JSON: ' + json.dumps(dict(
                verdict=evidence['forecast_verdict'], explanation='Inert observed fixture only.',
                lessons=lessons, mechanism_review=unknown_mechanism(evidence)))
        else:
            assert 'Review existing experimental evidence and lessons.' not in prompt
            kind = 'mutation'
            self.mutation_count += 1
            text = 'HYPOTHESIS_JSON: ' + json.dumps(dict(
                hypothesis='Change the inert fixture step.', predicted_low=-1., predicted_high=1.,
                reasoning='Synthetic measured outcome, never executed.', confidence=.5))
            text += '\n```python\n' + fixture_code(self.mutation_count) + '```'
        self.calls.append(dict(kind=kind, prompt=prompt, model=model, temperature=temperature, text=text))
        return LLMResponse(text=text, model=model, cost=.001, prompt_tokens=10, completion_tokens=10)


@pytest.fixture
def task():
    return load_task(str(ROOT / 'tasks/oe/circle_packing'))


def config(repeat=True, *, research=True, **changes):
    value = dict(task='oe/circle_packing', budget=2, max_iterations=2,
        dup_cap=1, random_seed=29, num_islands=1, inflight_cap=1,
        schedule_policy='round_robin', milestone_fracs=[], op_fixed='full_rewrite',
        policy={'enabled': False, 'plateau_shift': False}, llm={'mock': True},
        portfolio_seed=False, idea_island=False, dynamic_islands=False,
        crossover=False, embed_grid=False, embed_niche=False, three_head=False,
        confirm_records=False)
    if research:
        value['research'] = dict(enabled=True, retrieve_k=2,
            reflection_interval=0, max_aux_calls=2)
    if repeat is False:
        value['repeat_promotion'] = {'enabled': False}
    elif repeat is not None:
        value['repeat_promotion'] = dict(enabled=repeat, draws_per_source=3,
            seed_master=157, max_comparisons=2, max_draw_attempts=15,
            max_wall_seconds=1000., timeout_seconds=30., cpu_seconds=30,
            mem_bytes=2 * 1024**3, certification_seconds=5.)
    value.update(copy.deepcopy(changes))
    return value


def construct(task, directory, *, settings=None, start_step=0):
    llm = FakeLLM(start_step)
    engine = Engine(settings or config(), task, str(directory), llm, python=sys.executable)
    return engine, llm


@pytest.fixture
def search_observations(monkeypatch):
    """Read candidate bytes as data; never import any evaluator or candidate."""
    calls = []

    def evaluate(task, path, *, record_candidate, python):
        code = Path(path).read_text()
        step_match = re.search(r'fixture_step = (\d+)', code)
        step = int(step_match.group(1)) if step_match else 0
        raw = [.5, 1., 1.1][step]
        score = raw / 2.635
        cases = dict(sum_radii=raw, validity=1., combined_score=score)
        calls.append(dict(code=code, score=score, cases=cases, step=step))
        return CascadeOutcome(EvalResult(ok=True, score=score, cases=cases,
            artifacts={'fixture_only': True}, timing=.01), score, 'full', False)

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    return calls


def semantic_search_state(engine):
    """Only search state; UUIDs, elapsed time and separate H15 files are excluded."""
    return {
        'db': engine.db.state(), 'schedule': engine.schedule.state(),
        'operators': engine.op_bandit.state(), 'reward': engine.reward.state(),
        'selection': engine.selection.state(), 'rng': engine.rng.state(),
        'B_evals': engine.B_evals, 'iteration_no': engine.iteration_no,
        'cost_total': engine.cost_total,
        'knowledge': None if engine.research is None else engine.research.knowledge.state(),
    }


class FakeDraws:
    """Adapter-shaped receipts from fixed numeric data, without subprocesses.

    The arrays are certified by pure host arithmetic once per fixture value.
    This verifies transport to the coordinator, not candidate execution.
    """
    identity = {'schema': 'fake-repeat-adapter/1', 'runtime': 'inert-fixture'}

    def __init__(self, monkeypatch):
        self.calls = []
        self.identity_calls = []
        self.expected_runtime = {'backend': 'native'}
        self.on_draw = None
        self.failure_at = None
        self.failure_status = 'candidate_failure'
        self.numeric = {}
        monkeypatch.setattr('sciel.eval.repeat_draw.adapter_identity', self.adapter_identity)
        monkeypatch.setattr('sciel.eval.repeat_draw.draw', self.draw)

    def adapter_identity(self, *, python=None, runtime=None):
        assert runtime == self.expected_runtime
        self.identity_calls.append(dict(python=python, runtime=copy.deepcopy(runtime)))
        return copy.deepcopy(self.identity)

    def draw(self, source, *, seed, timeout, cpu_seconds, mem_bytes,
             certification_seconds, python=None, runtime=None, expected_identity=None):
        from sciel.eval.packing_geometry import validate_packing, certify_packing

        assert expected_identity == self.identity
        assert runtime == self.expected_runtime

        step_match = re.search(r'fixture_step = (\d+)', source)
        step = int(step_match.group(1)) if step_match else 0
        call = dict(source=source, source_sha256=sha(source), seed=seed,
                    timeout=timeout, cpu_seconds=cpu_seconds, mem_bytes=mem_bytes,
                    certification_seconds=certification_seconds, step=step,
                    runtime=copy.deepcopy(runtime))
        self.calls.append(call)
        if self.on_draw is not None:
            self.on_draw(call)
        # Child 2 wins single-score search but loses repeats against child 1.
        total = [.4, .6, .5][step]
        if total not in self.numeric:
            arrays = {'centers': [[(i % 6 + .5) / 6, (i // 6 + .5) / 5]
                                  for i in range(26)], 'radii': [total / 26] * 26}
            self.numeric[total] = (validate_packing(arrays), certify_packing(arrays))
        capture, certificate = copy.deepcopy(self.numeric[total])
        assert capture['valid'] and certificate['status'] == 'certified'
        status = self.failure_status if len(self.calls) == self.failure_at else 'valid'
        valid = status == 'valid'
        return dict(schema='sciel-repeat-draw/1', source_sha256=sha(source), seed=seed,
            adapter_identity=copy.deepcopy(self.identity), status=status,
            seed_applied=dict(python_random=seed, numpy_legacy=seed,
                python_hash_seed=str(seed), before_candidate_import=True,
                determinism_proven=False),
            limits=dict(candidate_wall_seconds=timeout,
                certification_wall_seconds=certification_seconds,
                candidate_cpu_seconds=cpu_seconds, candidate_mem_bytes=mem_bytes,
                certification_cpu_seconds=cpu_seconds, certification_mem_bytes=mem_bytes),
            raw_sum=capture['sum_radii'] if valid else None,
            certified_sum=certificate['sum_exact'] if valid else None,
            capture=capture if valid else None, certificate=certificate if valid else None,
            reason=None if valid else 'inert injected failure',
            started_at='2026-09-16T00:00:00+00:00',
            completed_at='2026-09-16T00:00:00.010000+00:00',
            elapsed_seconds=.01, candidate_elapsed_seconds=.006,
            certification_elapsed_seconds=.003, identity_elapsed_seconds=.001)


@pytest.fixture
def draws(monkeypatch):
    return FakeDraws(monkeypatch)


def batches(engine):
    return engine.repeat_promotion.state()['data']['batches']


def _split_observed_feedback(prompt):
    """Remove exactly the added observation block, preserving every other byte."""
    from sciel.engine.observed_changes import _PREFIX, _SUFFIX
    if '<OBSERVED_CHANGES_UNTRUSTED_JSON>' not in prompt:
        assert _SUFFIX not in prompt
        return prompt, []
    assert prompt.count(_PREFIX) == prompt.count(_SUFFIX) == 1
    before, tail = prompt.split(_PREFIX, 1)
    shown, end = json.JSONDecoder().raw_decode(tail)
    assert type(shown) is list and tail[end:].startswith(_SUFFIX)
    return before + tail[end + len(_SUFFIX):], shown


def _observed_feedback(calls, records, groups, programs):
    """Check source-bound facts before normalizing only independent-run IDs."""
    from sciel.engine.observed_paired import paired_record
    full = {(record['scope'], record['id']): record for record in records}
    paired = {group['id']: paired_record(group) for group in groups}
    batch_ids = {group['id']: group['batch_id'] for group in groups}
    result = []
    for call in calls:
        _, shown = _split_observed_feedback(call['prompt'])
        normalized = []
        for projected in shown:
            record = full[(projected['scope'], projected['id'])]
            assert projected['record_sha256'] == record['record_sha256']
            if record['scope'] == 'paired':
                assert record == paired[record['id']]
            else:
                assert record['child_source_sha256'] == sha(programs[record['child_program_id']].code)
                if record['parent_program_id'] is not None:
                    assert record['parent_source_sha256'] == sha(programs[record['parent_program_id']].code)
            for key, value in projected.items():
                if key in ('parent_observation', 'child_observation') and value is not None:
                    assert value == {field: record[key][field] for field in value}
                elif key == 'source_diff' and value is not None and value.get('omitted_from_prompt'):
                    assert value == {**record[key], 'text': '', 'chars_omitted': record[key]['chars_total'],
                                     'omitted_from_prompt': True}
                elif key == 'pairs_omitted_from_prompt':
                    assert value == len(record['pairs']) and 'pairs' not in projected
                elif key == 'details_omitted_from_prompt':
                    assert value is True
                elif key == 'edited_units' and projected.get('details_omitted_from_prompt'):
                    assert value == []
                elif key == 'edited_units_omitted' and projected.get('details_omitted_from_prompt'):
                    assert value == record[key] + len(record['edited_units'])
                else:
                    assert key in record and value == record[key]
            item = copy.deepcopy(projected)
            item.pop('record_sha256')
            if item['scope'] == 'paired':
                item['id'] = batch_ids[item['id']]
            normalized.append(item)
        result.append(normalized)
    return result


def _comparison_normalized_calls(calls, groups=None):
    """Normalize only declared run identities after checking complete card views."""
    from sciel.knowledge_base.presentation import comparison_card
    from sciel.hypothesis.prompts import PAIRED_COMPARISON_RULES
    from sciel.engine.research import MAX_MEMORY_CHARS
    from sciel.knowledge_base.catalogs import canonical_json
    from sciel.knowledge_base.comparison_facts import validate_comparison_facts
    expected = {g['id']: comparison_card(g) for g in groups or []}
    batch_ids = {g['id']: g['batch_id'] for g in groups or []}
    ordinary = {}
    for call in calls:
        if call['kind'] == 'conclusion':
            raw, _ = json.JSONDecoder().raw_decode(call['prompt'].split('EVIDENCE_JSON:', 1)[1].lstrip())
            record = decode_evidence(raw)
            ordinary[record['id']] = record
    result = copy.deepcopy(calls)
    for call in result:
        call['prompt'], _ = _split_observed_feedback(call['prompt'])
        marker = 'RESEARCH_CONTEXT_JSON:\n'
        if marker not in call['prompt']:
            continue
        prefix, suffix = call['prompt'].split(marker, 1)
        raw, end = json.JSONDecoder().raw_decode(suffix)
        context = decode_evidence(raw)
        remainder = suffix[end:]
        block = context.get('paired_comparisons')
        if block is not None:
            facts = context.get('comparison_facts')
            if facts is not None:
                records, seen = [], set()
                for card in block['cards']:
                    for record in (card, ordinary.get(card['origin_experiment_id'])):
                        if record is not None and record['id'] not in seen:
                            records.append(record)
                            seen.add(record['id'])
                # Check all projected measurements, signs, origins, omissions
                # and the exact wire budget before normalizing run identities.
                assert facts == validate_comparison_facts(facts, records=records, max_chars=facts['max_chars'])
            if groups is None:
                feedback = context.get('conclusion_feedback')
                if feedback is not None:
                    # Restore exactly the reservation for the removed whole
                    # paired block. Full equality below still checks review
                    # content, omissions, and the resulting available budget.
                    assert feedback['max_chars'] > 0  # No clipped-zero budget in this fixture.
                    feedback['max_chars'] += len(canonical_json(block))
                    if facts is not None:
                        feedback['max_chars'] += len(canonical_json(facts))
                    assert feedback['max_chars'] <= MAX_MEMORY_CHARS
                context.pop('paired_comparisons')
                context.pop('comparison_facts', None)
                assert PAIRED_COMPARISON_RULES in remainder
                remainder = remainder.replace(PAIRED_COMPARISON_RULES, '', 1)
            else:
                for card in block['cards']:
                    assert card == expected[card['id']]
                    card['id'] = batch_ids[card['id']]
                    for field in ('namespace', 'evidence_sha256', 'view_sha256'):
                        card.pop(field)
                for field in ('presented_ids', 'omitted_ids'):
                    block[field] = [batch_ids[ident] for ident in block[field]]
                if facts is not None:
                    facts['paired'] = {batch_ids[ident]: row for ident, row in facts['paired'].items()}
                    for field in ('presented_ids', 'omitted_ids'):
                        facts[field] = [batch_ids.get(ident, ident) for ident in facts[field]]
        call['prompt'] = prefix + marker + json.dumps(context, sort_keys=True) + remainder
    return result
