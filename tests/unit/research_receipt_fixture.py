"""Authored task-bound returned receipts; no candidate or evaluator execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.engine.llm.client import LLMResponse
from sciel.engine.parallel_calls import _encode, digest as wire_digest
from sciel.engine.research import ResearchController, digest
from sciel.engine.research_cells_context import compact_context
from sciel.engine.types import Program, Snapshot
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.output_identity import canonical
from sciel.eval.registry import load_task
from sciel.hypothesis.measurement import metric_context
from test_llm_subscription_accounting import unpriced
from test_research_cell_context import card


ROOT = Path(__file__).resolve().parents[2]


def non_circle_receipt(task_name):
    task_path = {'heilbronn': 'heilbronn_triangle', 'oracle': 'function_minimization_oracle'}[task_name]
    task = load_task(str(ROOT / 'tasks' / 'oe' / task_path))
    parent_code = Path(task.initial_program_path).read_text()
    if task_name == 'heilbronn':
        source = 'def heilbronn_triangle11():\n    return [[0.0, 0.0] for _ in range(11)]\n'
        score, cases = 0., {'min_area': 0., 'min_area_normalized': 0., 'validity': 1.}
        witness = dict(schema='sciel-full-output/1', source_sha256=digest(source),
            evaluator_sha256=hashlib.sha256(Path(task.evaluator_path).read_bytes()).hexdigest(),
            invocation_id='authored-heilbronn', stage='full', confirmation=None,
            input_sha256=None, seed_sha256=None, output={'points': [[0., 0.] for _ in range(11)]})
        artifacts = {'full_output': canonical(witness)}
    else:
        source = 'def run_search(oracle):\n    oracle.query(0.0, 0.0)\n'
        score = .25
        cases = {'mean_best_query_score': score, 'validity': 1.,
                 'reliability_score': 1., 'objective_queries': 1.}
        artifacts = {}
    config = dict(model=None, max_lessons=10, max_experiments=20, max_fragments=5,
        retrieve_k=3, max_aux_calls=0, reflection_interval=0)
    research = ResearchController(config, NS(models=['authored']), NS(research=lambda **_: None),
        lambda _: pytest.fail('fixture must not charge an auxiliary call'), metric=metric_context(task))
    parent = Program('p1', parent_code, 0., {}, 0, 'seed')
    context = research.context(NS(_all={parent.id: parent}), parent, 0, 1)
    context['research_cells_context'] = compact_context([card()])
    snap = Snapshot(0, 'full_rewrite', 'authored', 1, parent.id, parent_code, parent.score,
        research_context=context)
    ident, _ = research.begin(snap, 0, 'full_rewrite', 'Authored offline proposal.')
    hypothesis = dict(hypothesis='Test a fixed proposed starting state.', predicted_low=0.,
        predicted_high=.3, reasoning='This inert fixture exercises evidence transport.', confidence=.5)
    reply = unpriced(LLMResponse('HYPOTHESIS_JSON: ' + json.dumps(hypothesis) +
        '\n```python\n' + source + '```', 'authored', 0., 1, 1))
    assert research.proposal(ident, reply, source, NS(code=source))
    request = {'source': source, 'task_hash': task.task_hash}
    request_sha = wire_digest({'kind': 'evaluation', 'request': request})
    outcome = CascadeOutcome(EvalResult(ok=True, score=score, cases=cases, artifacts=artifacts), score, 'full')
    payload = dict(status='returned', request_sha256=request_sha, value=_encode('evaluation', outcome))
    return research, task, dict(parent_code=parent_code,
        evaluation_intent={'request': request, 'request_sha256': request_sha},
        evaluation_receipt={'payload': payload, 'sha256': wire_digest(payload)},
        hypothesis_record=deepcopy(research.pending[ident]))
