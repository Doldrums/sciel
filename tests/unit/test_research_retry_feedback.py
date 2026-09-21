"""Retry feedback uses fake model/cascade only; never executes source or an API."""
from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.db import Database, normalized_hash
from sciel.engine.llm.client import LLMResponse
from sciel.engine.prompt.build import build
from sciel.engine.research import ResearchController
from sciel.engine.types import NoChild, Result, Snapshot
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis.rendering import decode_evidence

PARENT = 'def run_search():\n    return 0.5\n'
CHILD = 'def run_search():\n    return 0.6\n'


def text(code=CHILD, *, hypothesis=True):
    head = 'HYPOTHESIS_JSON: '+json.dumps({'hypothesis':'Test this source change.',
        'predicted_low':.01,'predicted_high':.2,'reasoning':'A local forecast.','confidence':.5})+'\n' if hypothesis else ''
    return head+'```python\n'+code+'```\n'


class Model:
    models = ['fake']
    max_retries = 1

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def generate(self, prompt, model, temperature=None):
        self.calls.append((prompt,model,temperature))
        return LLMResponse(self.responses.pop(0),model,.01,10,10)


def setup(responses, *, research_on=True):
    model, events, evaluated = Model(responses), [], []
    cfg = apply_defaults({'research':{'enabled':True,'max_aux_calls':0,'reflection_interval':0}})['research']
    research = ResearchController(cfg,model,SimpleNamespace(research=lambda **event:events.append(copy.deepcopy(event))),
                                  lambda _cost:pytest.fail('successful returned worker owns known costs'))
    db = Database(1)
    parent = db.add(PARENT,.5,{},0,'seed')
    snap = Snapshot(0,'diff','fake',0,parent.id,PARENT,.5,
        code_hashes={normalized_hash(PARENT)},required_entrypoints=['run_search'])
    if research_on:
        snap.research_context = research.context(db,parent,0,1)

    def evaluate(code):
        evaluated.append(code)
        return CascadeOutcome(EvalResult(True,.6,{}),.6,'full')

    return model,research,snap,events,evaluated,evaluate


def context(prompt):
    raw = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n',1)[1].lstrip())[0]
    return decode_evidence(raw)


def assert_unmeasured(row):
    assert row['quality_measured'] is False
    assert row['score'] is row['actual_delta'] is None


@pytest.mark.parametrize('first,category,status',[
    (text(PARENT),'duplicate','no_child'),
    (text(hypothesis=False),'hypothesis_invalid','hypothesis_invalid'),
    (text('def run_search(:\n    return 0.4\n'),'syntax','no_child'),
])
def test_rejection_reaches_next_prompt_with_identity_and_original_baseline(first,category,status):
    model,research,snap,events,evaluated,evaluate = setup([first,text()])
    before = copy.deepcopy(snap)
    result = run_worker('diff',snap,model,evaluate,dup_cap=2,research=research)
    assert isinstance(result,Result) and result.ok
    assert len(model.calls) == 2 and evaluated == [CHILD]
    assert result.cost == pytest.approx(.02)
    assert snap == before
    initial, following = map(lambda c:context(c[0]),model.calls)
    assert 'retry_feedback' not in initial
    feedback = following['retry_feedback']
    assert feedback['schema'] == 'sciel-research-retry-feedback/1'
    assert feedback['total_rejections'] == 1 and feedback['omitted_count'] == 0
    assert len(feedback['rejections']) == 1
    row = feedback['rejections'][0]
    assert row['status'] == status and row['category'] == category
    assert row['experiment_id'] == research.knowledge.experiments[0]['id']
    trace = result.meta['validation_trace']['attempts'][0]
    assert row['proposed_sha256'] == trace['extracted_code_sha256']
    assert row['source_sha256'] == trace['candidate_code_sha256']
    assert_unmeasured(row)
    assert following['parent_score'] == initial['parent_score'] == .5
    assert following['failed_hypotheses'] == initial['failed_hypotheses'] == []
    assert 'unknown' in model.calls[1][0] and 'untrusted' in model.calls[1][0]
    assert research.attempts == research.outcomes == 2
    assert research.knowledge.lessons == []
    assert [e['phase'] for e in events].count('attempt') == 2


def test_latest_three_rejections_preserve_count_without_accumulating_prompt_payload():
    model,research,snap,_,evaluated,evaluate = setup([text(PARENT)]*6)
    before = copy.deepcopy(snap)
    result = run_worker('diff',snap,model,evaluate,dup_cap=6,research=research)
    assert isinstance(result,NoChild) and result.reason == 'dup_cap'
    assert len(model.calls) == 6 and evaluated == [] and result.cost == pytest.approx(.06)
    feedback = context(model.calls[-1][0])['retry_feedback']
    assert feedback['total_rejections'] == 5 and feedback['omitted_count'] == 2
    assert [r['experiment_id'] for r in feedback['rejections']] == [e['id'] for e in research.knowledge.experiments[2:5]]
    assert all(r['diagnostic'] == 'duplicate: normalized source already present on the island'
               for r in feedback['rejections'])
    for row in feedback['rejections']:
        assert_unmeasured(row)
    assert snap == before


def test_repaired_proposed_and_candidate_sources_remain_distinct_in_retry_feedback():
    proposed = 'def helper():\n    return 0.7\n'
    model,research,snap,_,evaluated,evaluate = setup([text(proposed),text()])
    calls = []
    def novelty(code):
        calls.append(code)
        return {'accept':len(calls)>1}
    result = run_worker('diff',snap,model,evaluate,dup_cap=2,research=research,novelty_check=novelty)
    assert isinstance(result,Result) and evaluated == [CHILD]
    row = context(model.calls[1][0])['retry_feedback']['rejections'][0]
    assert row['category'] == 'code_novelty'
    assert row['proposed_sha256'] == hashlib.sha256(proposed.encode()).hexdigest()
    assert row['source_sha256'] == hashlib.sha256(calls[0].encode()).hexdigest()
    assert row['proposed_sha256'] != row['source_sha256']
    assert_unmeasured(row)


def test_research_off_retry_prompts_and_costs_are_unchanged():
    model,_,snap,_,evaluated,evaluate = setup([text(PARENT),text()],research_on=False)
    expected = [build('diff',snap),build('full_rewrite',snap)]
    result = run_worker('diff',snap,model,evaluate,dup_cap=2)
    assert [call[0] for call in model.calls] == expected
    assert 'retry_feedback' not in ''.join(call[0] for call in model.calls)
    assert isinstance(result,Result) and result.cost == pytest.approx(.02)
    assert evaluated == [CHILD]


def test_feedback_is_local_to_one_worker_dispatch():
    model,research,snap,_,_,evaluate = setup([text(PARENT),text(),text()])
    run_worker('diff',snap,model,evaluate,dup_cap=2,research=research)
    run_worker('diff',snap,model,evaluate,dup_cap=1,research=research)
    assert 'retry_feedback' in context(model.calls[1][0])
    assert 'retry_feedback' not in context(model.calls[2][0])


def test_postresponse_validation_exception_closes_unknown_quality_and_charges_once():
    from sciel.campaign.budget import CampaignStopped
    model,research,snap,_,evaluated,evaluate = setup([text(PARENT),text()])
    charged = []
    research.charge = charged.append
    def broken_novelty(_code):
        raise RuntimeError('fixture internal validation error')
    with pytest.raises(CampaignStopped,match='response processing stopped: RuntimeError'):
        run_worker('diff',snap,model,evaluate,dup_cap=4,research=research,novelty_check=broken_novelty)
    assert len(model.calls) == 2 and evaluated == []
    assert charged == [pytest.approx(.02)]
    assert research.attempts == research.outcomes == 2 and not research.pending
    record = research.knowledge.experiments[-1]
    assert record['status'] == 'generation_stopped'
    assert record['generation_dispatch'] == 'returned' and record['generation_cost'] == .01
    assert record['score'] is record['actual_delta'] is record['evaluation'] is None
    assert record['forecast_verdict'] == 'INCONCLUSIVE'


def test_malformed_returned_evaluation_stops_and_keeps_attempt_as_unmeasured():
    from sciel.campaign.budget import CampaignStopped
    model,research,snap,_,evaluated,_evaluate = setup([text()])
    charged = []
    research.charge = charged.append
    def malformed(code):
        evaluated.append(code)
        return CascadeOutcome(EvalResult(True,.6,{'bad_case':float('nan')}),.6,'full')
    with pytest.raises(CampaignStopped,match='response processing stopped: ValueError'):
        run_worker('diff',snap,model,malformed,dup_cap=4,research=research)
    assert len(model.calls) == 1 and evaluated == [CHILD] and charged == [.01]
    record = research.knowledge.experiments[-1]
    # The evaluator returned; malformed metadata is a processing stop, not a
    # fabricated evaluator failure eligible for a later conclusion request.
    assert record['status'] == 'generation_stopped' and record['failure'] == 'ValueError'
    assert record['source_sha256'] == hashlib.sha256(CHILD.encode()).hexdigest()
    assert record['score'] is record['actual_delta'] is record['evaluation'] is None
    assert not research.pending
    assert research.queue == []


@pytest.mark.parametrize('cost',[None,float('nan'),float('inf'),-.01,True])
def test_invalid_returned_cost_keeps_prior_known_cost_without_inventing_a_charge(cost):
    from sciel.campaign.budget import CampaignStopped
    model,research,snap,_,evaluated,evaluate = setup([text(PARENT),text()])
    charged = []
    research.charge = charged.append
    original = model.generate
    def generate(*args,**kwargs):
        response = original(*args,**kwargs)
        if len(model.calls) == 2:
            response.cost = cost
        return response
    model.generate = generate
    with pytest.raises(CampaignStopped,match='response processing stopped: ValueError'):
        run_worker('diff',snap,model,evaluate,dup_cap=4,research=research)
    assert len(model.calls) == 2 and evaluated == []
    assert charged == [.01]
    record = research.knowledge.experiments[-1]
    assert record['generation_cost'] is None and record['generation_dispatch'] == 'returned'
    assert record['status'] == 'generation_stopped'
    assert record['score'] is record['actual_delta'] is None


def test_malformed_response_text_retains_known_cost_without_retry():
    from sciel.campaign.budget import CampaignStopped
    model,research,snap,_,evaluated,evaluate = setup([None])
    charged = []
    research.charge = charged.append
    with pytest.raises(CampaignStopped):
        run_worker('diff',snap,model,evaluate,dup_cap=4,research=research)
    assert len(model.calls) == 1 and evaluated == [] and charged == [.01]
    assert not research.pending and research.outcomes == 1


def test_cleanup_failure_retains_pending_but_still_charges_once(monkeypatch):
    from sciel.campaign.budget import CampaignStopped
    model,research,snap,_,evaluated,evaluate = setup([text()])
    charged = []
    research.charge = charged.append
    def broken_record(_record):
        raise RuntimeError('fixture persistence error')
    monkeypatch.setattr(research.knowledge,'record',broken_record)
    with pytest.raises(CampaignStopped,match='research bookkeeping failed: RuntimeError'):
        run_worker('diff',snap,model,evaluate,dup_cap=4,research=research)
    assert len(model.calls) == 1 and evaluated == [CHILD] and charged == [.01]
    assert research.pending and research.attempts == 1 and research.outcomes == 0
    assert research.stopped_reason
    with pytest.raises(ValueError,match='unfinished work'):
        research.load_state(research.state())


def test_research_off_internal_errors_keep_original_exception():
    model,_,snap,_,_,evaluate = setup([text()],research_on=False)
    def broken_novelty(_code):
        raise RuntimeError('fixture original exception')
    with pytest.raises(RuntimeError,match='fixture original exception'):
        run_worker('diff',snap,model,evaluate,dup_cap=4,novelty_check=broken_novelty)
    assert len(model.calls) == 1
