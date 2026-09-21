"""Stopped mock engines and bound checkpoint fixtures for provider recovery."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from sciel.engine import checkpoint
from sciel.engine.recovery import (ProviderRecoveryError, apply_provider_recovery,
    inspect_provider_stop, prepare_provider_recovery, validate_recoveries)
from sciel.run_log import RunLog


class State:
    def __init__(self, value):
        self.value = deepcopy(value)
        self._counter = value.get('counter', 0)

    def state(self):
        return deepcopy(self.value)

    def load_state(self, value):
        self.value = deepcopy(value)
        self._counter = value.get('counter', 0)


class Research(State):
    @property
    def stopped_reason(self):
        return self.value['stopped_reason']

    @stopped_reason.setter
    def stopped_reason(self, reason):
        self.value['stopped_reason'] = reason


def make_engine(folder, monkeypatch):
    monkeypatch.setattr(checkpoint, '_identity', lambda engine: {'fixture': True})
    monkeypatch.setattr(checkpoint, 'check_identity', lambda engine, state: [])
    folder.mkdir(exist_ok=True)
    engine = SimpleNamespace(run_dir=str(folder), log=RunLog(str(folder / 'sciel_events.jsonl')),
        cfg={'budget': None, 'max_iterations': None, 'research': {'enabled': True, 'max_aux_calls': None}},
        milestones_fired=set(), lineage_fail={}, eval_times={}, noop_parents=set(),
        kb_on=False, c2_on=False, plateau_shadow=False, oc_monitors=None,
        llm=SimpleNamespace(generate=lambda *a, **k: pytest.fail('provider called')),
        research=Research({'attempts': 1, 'outcomes': 1, 'aux_calls': 1, 'aux_cost': .2,
                           'pending': {}, 'queue': [], 'knowledge': {'kept': True},
                           'stopped_reason': 'research auxiliary request stopped: LLMError'}))
    for name in checkpoint._SCALARS:
        setattr(engine, name, 1)
    engine.budget = None
    engine.cost_total = 1.25
    for name in checkpoint._COMPONENTS:
        setattr(engine, name, State({'counter': 2, 'sentinel': name}))
    engine.close = engine.log.close
    return engine


def stopped(folder, monkeypatch, *, generation=False, metadata='default', configure=None):
    engine = make_engine(folder, monkeypatch)
    if generation:
        engine.research.stopped_reason = 'generation outcome unknown'
    if configure:
        configure(engine)
    engine.log.run_start(config=engine.cfg)
    diagnostics = {'stage': 'request', 'reason': 'connection_error', 'cause_type': 'APIConnectionError'}
    if metadata != 'default':
        diagnostics = metadata
    facts = dict(generation_returned=False, generation_dispatch='unknown', cost=None,
                 current_call_cost_unknown=True, llm_error=diagnostics)
    if generation:
        engine.log.research(phase='attempt', experiment_id='r1', evidence={'status': 'pending'})
        if diagnostics is not None:
            engine.log.research(phase='generation_error', experiment_id='r1', **facts)
        engine.log.research(phase='outcome', experiment_id='r1', evidence={
            'status': 'generation_stopped', 'failure': 'generation outcome unknown',
            'generation_dispatch': 'unknown', 'generation_cost': None, 'score': None})
    else:
        ref = dict(kind='conclusion', reference='r1', call_index=1)
        engine.log.research(phase='aux_start', **ref)
        engine.log.research(phase='aux_stopped', stage='request', error_type='LLMError', **ref, **facts)
    path = Path(checkpoint.save(engine, 'final'))
    engine.log.checkpoint(reason='research_stopped', iteration_no=1, B_evals=1, **engine._saved_checkpoint)
    engine.log.run_end(reason='research_stopped', iteration_no=1, B_evals=1,
                       checkpoint_id=engine._saved_checkpoint['checkpoint_id'],
                       checkpoint_sha256=engine._saved_checkpoint['checkpoint_sha256'])
    engine.close()
    return path


def restored(path, monkeypatch):
    engine = make_engine(path.parent, monkeypatch)
    checkpoint.load(engine, str(path))
    return engine


def idea_stop(folder, stage, *, error=None):
    """Actual two-stage request/abort/final-checkpoint path; all calls are local."""
    import sys
    from sciel.campaign.budget import CampaignStopped
    from sciel.engine.factory import build_engine
    from sciel.engine.llm.client import LLMError, LLMResponse
    from sciel.engine.llm.mock import MockLLM
    from sciel.eval.cascade import CascadeOutcome
    from sciel.eval.facade import EvalResult

    class Provider(MockLLM):
        def generate(self, prompt, *args, **kwargs):
            if stage == 'code' and prompt.startswith('IDEA STAGE 1:'):
                return LLMResponse(json.dumps({'family': 'grid', 'strategy': 'Adjust spacing',
                    'mechanism': 'Place objects closer'}), 'mock-model', .03, 12, 10)
            raise error if error is not None else LLMError(cause=ConnectionError())

    cfg = dict(task='test/toy_quadratic', budget=None, max_iterations=None,
        schedule_horizon=10, checkpoint_interval=10, num_islands=1, inflight_cap=1,
        dup_cap=1, random_seed=23, schedule_policy='round_robin', milestone_fracs=[],
        embed_grid=False, embed_niche=False, portfolio_seed=False, dynamic_islands=False,
        idea_island=False, crossover=False, op_fixed='idea', idea_two_stage={'enabled': True},
        policy={'enabled': False, 'plateau_shift': False}, llm={'mock': True},
        research={'enabled': True, 'max_aux_calls': None})
    tasks = str(Path(__file__).resolve().parents[2] / 'tasks')
    engine = build_engine(cfg, tasks, str(folder), python=sys.executable, llm=Provider())
    engine._evaluate = lambda code: CascadeOutcome(
        result=EvalResult(ok=True, score=.1, cases={}, timing=.01),
        raw_first_score=.1, reached='full', confirmed=False)
    with pytest.raises(CampaignStopped, match='two-stage idea stopped'):
        engine.run()
    return folder / 'checkpoint.json'


def rewrite_bound_fixture(path, mutate):
    """Keep physical provenance valid so rejection tests exercise semantic checks."""
    state = json.loads(path.read_bytes())
    log = path.with_name('sciel_events.jsonl')
    anchor = state['provenance']['journal_anchor']
    rows = [json.loads(line) for line in log.read_bytes()[:anchor['byte_offset']].splitlines()]
    tail = [json.loads(line) for line in log.read_bytes()[anchor['byte_offset']:].splitlines()]
    mutate(state, rows)
    prefix = b''.join((json.dumps(r) + '\n').encode() for r in rows)
    anchor.update(byte_offset=len(prefix), prefix_sha256=hashlib.sha256(prefix).hexdigest())
    original = json.dumps(state).encode()
    path.write_bytes(original)
    for row in tail:
        row['checkpoint_sha256'] = hashlib.sha256(original).hexdigest()
        if row['event'] == 'checkpoint':
            row['provenance'] = state['provenance']
    log.write_bytes(prefix + b''.join((json.dumps(r) + '\n').encode() for r in tail))
