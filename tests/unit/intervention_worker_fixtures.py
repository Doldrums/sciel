"""Inert intervention models, snapshots and worker execution helpers."""
import copy
import hashlib
import json
from types import SimpleNamespace
import pytest
from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.config import apply_defaults
from sciel.engine.db import Database, normalized_hash
from sciel.engine.intervention import CircleIntervention
from sciel.engine.llm.client import LLMResponse
from sciel.engine.prompt.sources import source_digest
from sciel.engine.research import ResearchController
from sciel.engine.types import NoChild, Result, Snapshot
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_interventions as edits


def plan(policy='incumbent', **changes):
    value = {'hypothesis': f'{edits.OPERATION}:{policy}', 'predicted_low': -.2,
             'predicted_high': .2, 'reasoning': 'Refinement may trade away diversity.', 'confidence': .5}
    value.update(changes)
    return 'HYPOTHESIS_JSON: ' + json.dumps(value)


def response(text=None, cost=.03, **changes):
    value = dict(text=plan() if text is None else text, cost=cost, model='fake',
                 prompt_tokens=32, completion_tokens=16)
    value.update(changes)
    return SimpleNamespace(**value)


class Log:
    def __init__(self):
        self.events = []

    def research(self, **event):
        self.events.append(copy.deepcopy(event))


class Model:
    models, weights, max_retries = ['fake'], [1.0], 1

    def __init__(self, rows, log):
        self.rows, self.calls, self.log = list(rows), [], log

    def generate(self, prompt, model, temperature):
        assert self.log.events[-1]['phase'] == 'intervention'
        assert self.log.events[-1]['action'] == 'request_start'
        assert self.log.events[-1]['prompt_sha256'] == source_digest(prompt)
        self.calls.append((prompt, model, temperature))
        value = self.rows.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def setup(rows, *, parent='initial'):
    log, charged = Log(), []
    model = Model(rows, log)
    cfg = apply_defaults({'research': {'enabled': True, 'max_aux_calls': 0,
                                      'reflection_interval': 0}})['research']
    research = ResearchController(cfg, model, log, charged.append)
    db = Database(1)
    code = edits.source_for(parent)
    program = db.add(code, .8, {'validity': 1., 'sum_radii': 2.108}, 0, 'seed')
    snapshot = Snapshot(0, 'intervention', 'fake', 0, program.id, code, .8,
        code_hashes=db.code_hashes(0), required_entrypoints=['run_packing'], task_prompt='Pack 26 circles.')
    snapshot.research_context = research.context(db, program, 0, 1)
    worker = CircleIntervention(research)
    evaluated = []

    def evaluate(source):
        # Inspect source, never import or execute it.
        evaluated.append(source)
        proof_events = [e for e in log.events if e.get('action') == 'source_binding']
        assert proof_events[-1]['proof']['child_source_sha256'] == source_digest(source)
        assert proof_events[-1]['source_origin'] == 'deterministic_builder'
        hypothesis_events = [e for e in log.events if e['phase'] == 'hypothesis']
        assert hypothesis_events[-1]['evidence']['source_sha256'] == source_digest(source)
        assert log.events.index(proof_events[-1]) < log.events.index(hypothesis_events[-1])
        return CascadeOutcome(EvalResult(True, .9, {'validity': 1., 'sum_radii': 2.3715},
            timing=.012, artifacts={'test_only': {'candidate_executed': False}}), .88, 'full', False)

    return SimpleNamespace(worker=worker, research=research, model=model, log=log, charged=charged,
        snapshot=snapshot, db=db, parent=program, evaluated=evaluated, evaluate=evaluate)


def run(s, evaluate=None, **kwargs):
    return s.worker.run(s.snapshot, s.model, evaluate or s.evaluate, **kwargs)


def journal(path, events):
    data = ''.join(json.dumps({'event': 'research', **event}) + '\n' for event in events).encode()
    path.write_bytes(data)
    return {'byte_offset': len(data), 'prefix_sha256': hashlib.sha256(data).hexdigest()}


def move_parent(s, policy):
    """Make only this measured source active, as after same-cell replacement."""
    program = s.db.add(edits.source_for(policy), .95, {'validity': 1.}, 0, 'gen',
                       parent_id=s.parent.id)
    s.parent = program
    s.snapshot.parent_id, s.snapshot.parent_code = program.id, program.code
    s.snapshot.parent_score = program.score
    s.snapshot.research_context = s.research.context(s.db, program, 0, 1)
    s.snapshot.code_hashes = {normalized_hash(program.code)}
