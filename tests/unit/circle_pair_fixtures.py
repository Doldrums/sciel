"""Pair responses, inert providers and checkpoint helpers for engine tests."""
import copy
import json
from pathlib import Path
import pytest
from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_proposal_pair as pair
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis.rendering import decode_evidence
from sciel.hypothesis.parsing import parse_hypothesis_value
from test_circle_generation_engine import BODIES, COST, config as single_config, response
from test_circle_intervention_engine import digest, rows, task_copy


POLICY = 'first_eligible_of_two'


def candidate(target=BODIES[0], **forecast):
    return json.loads(response(target, **forecast))


def paired(first=None, second=None):
    return json.dumps({'candidates': [candidate() if first is None else first,
        candidate(BODIES[1]) if second is None else second]}, ensure_ascii=False,
        allow_nan=False)


def config(**changes):
    value = single_config(circle_generation={'enabled': True, 'proposal_policy': POLICY})
    value.update(copy.deepcopy(changes))
    return value


class PairModel:
    models = ['fixture']
    weights = [1.]
    max_retries = 1

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def choose_model(self, rng):
        return 'fixture'

    def generate(self, prompt, model, temperature=None, *, response_format=None):
        assert json.dumps(response_format, sort_keys=True) == json.dumps(pair.response_format(), sort_keys=True)
        assert 'Review this one experiment.' not in prompt
        self.calls.append(dict(prompt=prompt, model=model, temperature=temperature,
                               response_format=copy.deepcopy(response_format)))
        assert self.replies, 'unexpected second provider call or auxiliary call'
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if not isinstance(reply, str):
            return reply
        return LLMResponse(reply, model, COST, 17, 11, finish_reason='stop')


def build(path, replies, *, cfg=None, scores=None):
    task = task_copy(path.parent/'task', initial=body.source_for())
    model = PairModel(replies)
    engine = Engine(config() if cfg is None else cfg, task, str(path), model)
    evaluated = []
    scoremap = {body.DEFAULT_BODY: .5, BODIES[0]: .6, BODIES[1]: .7}
    scoremap.update(scores or {})

    def evaluate(source):
        parsed = body.body_of(source)  # Exact source parsing; no source import.
        if source != body.source_for():
            events = rows(engine)
            bound = next(e for e in reversed(events) if e.get('phase') == 'circle_generation'
                         and e.get('action') == 'source_binding')
            forecast = next(e for e in reversed(events) if e.get('phase') == 'hypothesis')
            assert bound['experiment_id'] == forecast['experiment_id']
            assert bound['proof']['child_source_sha256'] == digest(source)
            assert forecast['evidence']['source_sha256'] == digest(source)
            assert events.index(bound) < events.index(forecast)
        evaluated.append(source)
        score = scoremap[parsed]
        cases = dict(sum_radii=score*2.635, target_ratio=score,
                     validity=1., combined_score=score)
        return CascadeOutcome(EvalResult(True, score, cases, timing=.01), score, 'full', False)

    engine._evaluate = evaluate
    engine.fixture_evaluated = evaluated
    return engine


def observations(engine):
    return engine.research.knowledge.experiments


def actions(engine, action):
    return [e for e in rows(engine) if e.get('phase') == 'circle_generation'
            and e.get('action') == action]


def context(prompt):
    raw, _ = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])
    return decode_evidence(raw)


def assert_one_selected(engine, raw, target, forecast, *, selected_index=0):
    child = body.source_for(target)
    assert engine.fixture_evaluated == [body.source_for(), child]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    assert engine.research.attempts == engine.research.outcomes == 1
    assert engine.research.aux_calls == 0
    record, = observations(engine)
    assert record['status'] == 'evaluated'
    assert record['hypothesis'] == parse_hypothesis_value(forecast['hypothesis']).to_dict()
    assert record['source_sha256'] == record['proposed_sha256'] == digest(child)
    assert record['parent_source_sha256'] == digest(body.source_for())
    assert record['controlled_change'] == body.apply(body.source_for(), target)['proof']
    assert record['source_repaired'] is False
    assert record['generation_cost'] == COST and record['generation_dispatch'] == 'returned'
    assert engine.circle_generation.evaluations == len(engine.circle_generation.seen_proofs) == 1
    start, = actions(engine, 'request_start')
    returned, = actions(engine, 'request_result')
    bound, = actions(engine, 'source_binding')
    assert start['response_format'] == pair.response_format()
    assert returned['response'] == raw and returned['response_truncated'] is False
    assert returned['response_sha256'] == digest(raw) and returned['cost'] == COST
    assert bound['proof'] == record['controlled_change']
    assert bound['experiment_id'] == record['id']
    assert len([p for p in engine.db._all.values() if p.origin == 'gen']) == 1
    group = record['proposal_group']
    assert group['request_id'] == record['id'] and group['response_sha256'] == digest(raw)
    assert group['parent_source_sha256'] == record['parent_source_sha256']
    assert group['assessment_complete'] is True
    assert group['selected_index'] == selected_index
    assert group['selected_source_sha256'] == record['source_sha256']
    assert [e['index'] for e in group['entries']] == [0, 1]
    selected = group['entries'][selected_index]
    assert selected['disposition'] == 'selected' and selected['eligible'] is True
    assert selected['hypothesis'] == record['hypothesis'] and selected['body'] == target
    assert all(not {'score', 'actual_delta', 'evaluation', 'forecast_verdict'} & set(e)
               for e in group['entries'])
    return record


def _rewrite_checkpoint(directory, state, *, change_journal=None, refresh_pair_history=True):
    """Rehash outer artifact bindings so semantic joins, not stale hashes, fail."""
    import hashlib
    from sciel.knowledge_base.base import _digest, _snapshot

    knowledge = state['research']['knowledge']
    for record in knowledge['experiments']:
        snapshot = _snapshot(record)
        knowledge['seen_experiments'][record['id']] = _digest(snapshot)
        for fragment in knowledge['fragments']:
            if fragment['experiment_id'] == record['id']:
                fragment['evidence'] = copy.deepcopy(snapshot)
    groups = {r['id']: r['proposal_group'] for r in knowledge['experiments'] if 'proposal_group' in r}
    if refresh_pair_history:
        for item in state['circle_generation']['proposal_pair_history']:
            group = groups[item['experiment_id']]
            item.update(group_sha256=_digest(group), response_sha256=group['response_sha256'],
                        selected_index=group['selected_index'],
                        selected_source_sha256=group['selected_source_sha256'])
    journal = directory/'sciel_events.jsonl'
    saved = journal.read_bytes()
    anchor = state['provenance']['journal_anchor']
    prefix = [json.loads(line) for line in saved[:anchor['byte_offset']].splitlines()]
    tail = [json.loads(line) for line in saved[anchor['byte_offset']:].splitlines()]
    if change_journal is not None:
        change_journal(prefix)
    encoded = ''.join(json.dumps(row, allow_nan=False)+'\n' for row in prefix).encode()
    anchor.update(byte_offset=len(encoded), prefix_sha256=hashlib.sha256(encoded).hexdigest(),
                  event_seq=prefix[-1]['event_seq'])
    target = directory/'tampered.json'
    checkpoint_bytes = json.dumps(state, allow_nan=False).encode()
    target.write_bytes(checkpoint_bytes)
    for row in tail:
        if row.get('checkpoint_id') == state['provenance']['checkpoint_id']:
            row['checkpoint_sha256'] = hashlib.sha256(checkpoint_bytes).hexdigest()
            if 'provenance' in row:
                row['provenance'] = copy.deepcopy(state['provenance'])
    journal.write_bytes(encoded + ''.join(json.dumps(row, allow_nan=False)+'\n' for row in tail).encode())
    return target


def _novelty_config(**changes):
    return config(container={'kind': 'pool_novelty', 'selection': 'shared',
                             'novelty_threshold': .98}, pop_cap=2, **changes)
