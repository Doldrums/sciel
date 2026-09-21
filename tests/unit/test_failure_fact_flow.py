"""Actual fake-provider requests distinguish engineering failures from regressions.

Real ordinary/Circle workers, Research and KB; every measurement is an inert
callback. No candidate, evaluator, optimizer, subprocess or external model runs.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from sciel.engine.circle_generation import CircleGeneration
from sciel.engine.llm.client import LLMResponse
from sciel.engine.types import NoChild, Program, Result, Snapshot
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis.prompts import PRIOR_EXPERIMENT_RULES
from sciel.hypothesis.rendering import decode_evidence
from test_circle_generation_engine import BODIES, response
from test_research_measurement_state import controller, circle_metric


PARENT_SCORE = .9988377503677311
CHILD_SCORE = .998471926481528
RAW_SUM = 2.6309735262788263
REASON = 'The numerical effect remains uncertain.'


class Provider:
    models, max_retries = ['fake'], 1

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def generate(self, prompt, model, temperature=None, *, response_format=None):
        self.calls.append(dict(prompt=prompt, model=model, temperature=temperature,
                               response_format=copy.deepcopy(response_format)))
        assert self.texts, 'unexpected extra generation or auxiliary request'
        return LLMResponse(self.texts.pop(0), model, .002, 11, 7, finish_reason='stop')


def setup(mode, *, inject=True, continuation=False):
    hypothesis = dict(hypothesis='Change the initialization.', predicted_low=.0001,
                      predicted_high=.001, reasoning=REASON, confidence=.7)
    sources = ['def run_search(:\n    pass\n', 'def run_search():\n    return (\n',
               'def run_search():\n    return 0.998471926481528\n',
               'def run_search():\n    return 0.997\n']
    if mode == 'circle':
        texts = [response('return incumbent._params\n', **hypothesis),
                 response('centers = []\ncenters.extend([1])\nreturn None\n', **hypothesis),
                 response(BODIES[0], **hypothesis), response(BODIES[1], **hypothesis)]
        parent_code = body.source_for()
    else:
        texts = ['HYPOTHESIS_JSON: ' + json.dumps(hypothesis) + '\n```python\n' + source + '```'
                 for source in sources]
        parent_code = f'def run_search():\n    return {PARENT_SCORE}\n'
    c, parent, db, events = controller(metric=circle_metric(), max_experiments=8,
        inject_lessons=inject, max_aux_calls=0, reflection_interval=0)
    llm = Provider(texts[3:] if continuation else texts)
    c.llm = llm
    parent.code, parent.score = parent_code, PARENT_SCORE
    parent.cases = dict(validity=1., sum_radii=2.6319374722189712, combined_score=PARENT_SCORE)
    other = Program('p2', parent_code, PARENT_SCORE, copy.deepcopy(parent.cases), 1, 'seed')
    db._all[other.id] = other
    return SimpleNamespace(mode=mode, research=c, llm=llm, parent=parent, other=other, db=db,
        events=events, worker=CircleGeneration(c) if mode == 'circle' else None,
        evaluations=[], sources=sources)


def step(f):
    c, p = f.research, f.parent
    ctx = c.context(f.db, p, 0, c.attempts + 1)
    op = 'circle_generate' if f.mode == 'circle' else 'full_rewrite'
    snap = Snapshot(0, op, 'fake', c.attempts, p.id, p.code, p.score,
        required_entrypoints=['run_packing' if f.mode == 'circle' else 'run_search'],
        research_context=ctx, kb_lessons=ctx['lesson_texts'])

    def evaluate(source):
        expected = [body.source_for(s) for s in BODIES] if f.mode == 'circle' else f.sources[2:]
        assert source in expected, 'an engineering rejection must never reach evaluation'
        f.evaluations.append(source)
        # Evaluator transport success and task validity are distinct explicit facts.
        cases = dict(validity=1., sum_radii=RAW_SUM, combined_score=CHILD_SCORE)
        return CascadeOutcome(EvalResult(True, CHILD_SCORE, cases, timing=.001), CHILD_SCORE, 'full', False)

    result = run_worker(op, snap, f.llm, evaluate, dup_cap=1, base_temp=.2,
                        research=c, circle_generation=f.worker)
    c.complete_step()
    return result


def prefix(f):
    assert isinstance(step(f), NoChild)
    assert isinstance(step(f), NoChild)
    assert isinstance(step(f), Result)
    records = f.research.knowledge.experiments
    assert [r['status'] for r in records] == ['no_child', 'no_child', 'evaluated']
    assert len(f.evaluations) == 1 and len(f.llm.calls) == 3
    return records


def context_from_call(call):
    payload, _ = json.JSONDecoder().raw_decode(call['prompt'].split('RESEARCH_CONTEXT_JSON:', 1)[1].lstrip())
    return decode_evidence(payload)


def assert_history(call, records):
    ctx = context_from_call(call)
    failures = ctx['failed_hypotheses']
    assert len(failures) == 3 and PRIOR_EXPERIMENT_RULES in call['prompt']
    by_id = {r['id']: next(s for s in failures if f"id={r['id']};" in s) for r in records}
    for record in records:
        assert f"source_sha256={record['source_sha256']}" in by_id[record['id']]
    for record in records[:2]:
        text = by_id[record['id']]
        assert 'Unmeasured (no_child)' in text and 'quality not measured' in text
        assert 'diagnostic (untrusted)' in text
        assert 'measured_facts=' not in text and 'regression' not in text
        assert record['evaluation'] is record['score'] is record['actual_delta'] is None
    measured = by_id[records[2]['id']]
    facts, _ = json.JSONDecoder().raw_decode(measured.split('measured_facts=', 1)[1])
    assert facts == dict(status='evaluated', evaluator_ok=True,
        task_validity=dict(case='validity', value=1., valid_value=1.),
        parent_score=PARENT_SCORE, child_score=CHILD_SCORE, actual_delta=CHILD_SCORE-PARENT_SCORE)
    assert 'measured regression' in measured and 'Forecast refuted' in measured
    assert 'quality not measured' not in measured and 'null=unknown' in measured
    assert records[2]['forecast_verdict'] == 'REFUTED'
    assert records[2]['evaluation']['ok'] is True
    assert records[2]['evaluation']['cases']['validity'] == 1.
    assert ctx['parent_score'] == PARENT_SCORE
    return ctx


@pytest.mark.parametrize('mode', ['ordinary', 'circle'])
@pytest.mark.parametrize('inject', [True, False])
def test_actual_next_request_keeps_engineering_and_measured_facts_distinct(mode, inject):
    f = setup(mode, inject=inject)
    records = prefix(f)
    before = copy.deepcopy(f.research.knowledge.experiments)
    sibling = f.research.context(f.db, f.other, 1, 4)
    assert sibling['failed_hypotheses'] == []
    assert f.research.knowledge.experiments == before
    assert isinstance(step(f), Result)
    context = assert_history(f.llm.calls[3], records)
    assert context['fragments'] == []  # No positive source observation is fabricated.
    assert f.research.knowledge.experiments[:3] == records
    assert all(r['injected_ids'] == [] for r in f.research.knowledge.experiments)
    assert f.research.attempts == f.research.outcomes == 4
    assert f.research.aux_calls == f.research.reflections == 0
    assert len(f.llm.calls) == 4 and len(f.evaluations) == 2
    assert f.research.knowledge.lessons == []


@pytest.mark.parametrize('mode', ['ordinary', 'circle'])
def test_split_restore_preserves_exact_next_wire_and_rejects_old_identity_atomically(mode):
    uninterrupted = setup(mode, inject=False)
    records = prefix(uninterrupted)
    saved = json.loads(json.dumps(uninterrupted.research.state(), allow_nan=False))
    assert saved['presentation']['failure_context'] == 'sciel-failure-context/2'
    resumed = setup(mode, inject=False, continuation=True)
    before = resumed.research.state()
    legacy = copy.deepcopy(saved)
    legacy['presentation']['failure_context'] = 'sciel-failure-context/1'
    with pytest.raises(ValueError, match='presentation'):
        resumed.research.load_state(legacy)
    assert resumed.research.state() == before and resumed.llm.calls == []
    resumed.research.load_state(saved)
    if resumed.worker is not None:
        resumed.worker.load_state(copy.deepcopy(uninterrupted.worker.state()))
    assert resumed.research.state() == uninterrupted.research.state()
    assert isinstance(step(uninterrupted), Result) and isinstance(step(resumed), Result)
    assert resumed.llm.calls == uninterrupted.llm.calls[3:]
    assert_history(resumed.llm.calls[0], records)
    assert resumed.research.state() == uninterrupted.research.state()
    assert len(resumed.evaluations) == 1 and len(uninterrupted.evaluations) == 2
