"""Live controller/Engine wiring with inert providers and synthetic helpers only.

One module fixture executes 32 tiny initializer calls, never a packing solver.
All subsequent dispatch tests reuse that receipt; no historical helpers or API.
"""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.config import apply_defaults
from sciel.engine import initializer_research as policy
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController
from sciel.engine.types import Snapshot
from sciel.hypothesis import circle_initializer_witness as witness
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis.prompts import conclusion_prompt, reflection_prompt
from sciel.knowledge_base.catalogs import canonical_json

BODY = 'return best_params + uniform_noise * 0.25\n'
CLAIM = 'A different initializer will improve score.'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope='module')
def receipt():
    sources = {'parent': body.source_for(), 'candidate': body.source_for(BODY)}
    result = witness.witness_sources(sources,
        expected_source_sha256={k:digest(v) for k,v in sources.items()})
    assert result['status'] == 'observed'
    assert sum(len(s['observations']) for s in result['sources']) == 32
    return result


class Model:
    models = ['fixture']
    max_retries = 1

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, model, temperature):
        self.prompts.append(prompt)
        if 'Review this one experiment.' in prompt:
            value = dict(verdict='REFUTED', explanation='The recorded score decreased.', lessons=[],
                mechanism_review=[dict(claim_quote=CLAIM, source_quotes=[], assessment='unknown',
                    direct_effect='Finite fixture behavior is shown separately.',
                    untested_downstream='No downstream quality effect is established.')])
        else:
            value = dict(lessons=[], retire_ids=[])
        return LLMResponse(json.dumps(value), model, .001, 10, 10)


def controller(*, enabled=True, auxiliary=2, reflection=1):
    cfg = apply_defaults({'research': {'enabled': True, 'model': 'fixture',
        'max_aux_calls': auxiliary, 'reflection_interval': reflection}})['research']
    if enabled:
        cfg['initializer_witness'] = True
    model, events = Model(), []
    c = ResearchController(cfg, model, SimpleNamespace(research=lambda **x:events.append(x)), lambda _:None)
    source, parent = body.source_for(BODY), body.source_for()
    r = dict(id='r000001a1', branch='p1', parent_id='p1', status='evaluated',
        parent_score=1., score=.75, actual_delta=-.25, source_sha256=digest(source),
        proposed_sha256=digest(source), parent_source_sha256=digest(parent), source_repaired=False,
        controlled_change=body.apply(parent, BODY)['proof'],
        hypothesis=dict(text=CLAIM, reasoning='An untested starting point may alter refinement.',
            predicted_low=.1, predicted_high=.2, confidence=.5),
        metric=c.metric, parent_cases={}, evaluation=dict(ok=True, score=.75, cases={},
            reached='full', confirmed=False, raw_first_score=.75))
    c.knowledge.record(r)
    c.attempts = c.outcomes = c.valid_hypotheses = 1
    c.queue = copy.deepcopy(c.knowledge.experiments)
    return c, model, events


def install_receipt(monkeypatch, receipt):
    calls = []
    def returned(sources, fixtures, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        assert kwargs['expected_plan_sha256'] == receipt['plan']['plan_sha256']
        assert {k:digest(v) for k,v in sources.items()} == kwargs['expected_source_sha256']
        return copy.deepcopy(receipt)
    monkeypatch.setattr(witness, 'witness_sources', returned)
    return calls


def test_committed_measurement_survives_witness_and_reaches_both_aux_prompts(monkeypatch, receipt):
    calls = install_receipt(monkeypatch, receipt)
    c, model, events = controller()
    before, seen = c.knowledge.experiments, copy.deepcopy(c.knowledge._seen)
    c.complete_step()
    assert len(calls) == 1 and len(model.prompts) == 2
    assert all('INITIALIZER_EVIDENCE_JSON:' in p for p in model.prompts)
    assert c.knowledge.experiments == before and c.knowledge._seen == seen
    assert c.outcomes == c.attempts == 1 and c.aux_calls == 2 and c.knowledge.lessons == []
    assert c.conclusions['r000001a1']['status'] == 'parsed'
    assert c._initializer_attempts['r000001a1']['status'] == 'completed'
    phases = [e['phase'] for e in events]
    assert phases.index('initializer_witness_intent') < phases.index('initializer_witness_result') < phases.index('initializer_witness_registered')
    presentations = [e for e in events if e['phase'] == 'initializer_witness_presentation']
    assert [e['role'] for e in presentations] == ['conclusion', 'reflection']
    assert [e['prompt_sha256'] for e in presentations] == [digest(p) for p in model.prompts]
    assert all('data_base64' not in p and 'raw_stdout_base64' not in p for p in model.prompts)


def test_registry_fact_reaches_next_generation_without_lesson_or_exposure_credit(monkeypatch, receipt):
    install_receipt(monkeypatch, receipt)
    c, _, events = controller(); c.complete_step()
    parent = SimpleNamespace(id='p1', parent_id=None, code=body.source_for(), card='control', score=1., cases={})
    ctx = c.context(SimpleNamespace(_all={'p1':parent}), parent, 0, 2)
    assert ctx['injected_ids'] == [] and 'initializer_evidence' in ctx
    snapshot = Snapshot(0, 'circle_generate', 'fixture', 2, parent.id, parent.code,
                        parent.score, research_context=ctx)
    ident, prompt = c.begin(snapshot, 0, 'circle_generate', 'inert source instruction',
        response_instruction='Return a structured initializer.', context_scope='circle_restart_body/1')
    assert 'initializer_evidence' in prompt and c.pending[ident]['injected_ids'] == []
    shown = [e for e in events if e['phase'] == 'initializer_witness_presentation' and e['role'] == 'generation']
    assert len(shown) == 1 and shown[0]['prompt_sha256'] == digest(prompt)
    assert shown[0]['experiment_id'] == ident


def test_saved_attempt_restores_and_requeued_observation_cannot_reexecute(monkeypatch, receipt):
    calls = install_receipt(monkeypatch, receipt)
    c, _, _ = controller(); c.complete_step(); saved = c.state()
    fresh, _, _ = controller(); fresh.queue = []; fresh.load_state(saved)
    assert fresh.state() == saved
    fresh.queue = copy.deepcopy(fresh.knowledge.experiments)
    fresh.complete_step()
    assert len(calls) == 1 and fresh.knowledge.experiments == c.knowledge.experiments


def test_internal_diagnostic_error_stops_auxiliary_without_reclassifying_quality(monkeypatch):
    c, model, _ = controller(); before = c.knowledge.state(); count = []
    def failed(*args, **kwargs):
        count.append(True); raise ValueError('fixture internal integrity error')
    monkeypatch.setattr(witness, 'witness_sources', failed)
    with pytest.raises(CampaignStopped, match='Initializer diagnostics stopped'):
        c.complete_step()
    assert c.knowledge.state() == before and model.prompts == [] and c.aux_calls == 0
    assert c.knowledge.experiments[0]['status'] == 'evaluated'
    assert c._initializer_attempts['r000001a1']['status'] == 'started'
    c.queue = copy.deepcopy(c.knowledge.experiments)
    with pytest.raises(CampaignStopped, match='cannot be replayed'):
        c.complete_step()
    assert len(count) == 1


def test_disabled_policy_keeps_old_state_and_never_executes(monkeypatch):
    c, _, _ = controller(enabled=False)
    monkeypatch.setattr(witness, 'witness_sources', lambda *a,**k:pytest.fail('disabled dispatch'))
    c.complete_step()
    assert 'initializer_witness' not in c.state() and 'initializer_evidence' not in c.knowledge.state()


def test_known_incomplete_receipt_keeps_missing_facts_and_continues_ordinary_review(monkeypatch):
    sources = {'parent': body.source_for(), 'candidate': body.source_for(BODY)}
    def unavailable(*args, **kwargs):
        raise OSError('synthetic process unavailable')
    monkeypatch.setattr(witness, '_launch', unavailable)
    incomplete = witness.witness_sources(sources,
        expected_source_sha256={k:digest(v) for k,v in sources.items()})
    assert incomplete['status'] == 'incomplete'
    calls = install_receipt(monkeypatch, incomplete)
    c, model, _ = controller(); before = c.knowledge.experiments
    c.complete_step()
    assert len(calls) == 1 and len(model.prompts) == 2
    assert c.knowledge.experiments == before and c.stopped_reason is None
    binding = c.knowledge.initializer_evidence_for('r000001a1')
    assert all(source['status'] == 'incomplete' for source in binding['evidence']['sources'].values())
    assert all(o['effective_output'] is None for source in binding['evidence']['sources'].values()
               for o in source['observations'])


@pytest.mark.parametrize('damage', ['missing', 'unfinished', 'source', 'evidence', 'identity',
                                  'removed_attempt', 'removed_registry', 'removed_both', 'plan'])
def test_restore_damaged_attempt_is_atomic(monkeypatch, receipt, damage):
    install_receipt(monkeypatch, receipt)
    c, _, _ = controller(); c.complete_step(); saved = c.state(); bad = copy.deepcopy(saved)
    if damage == 'missing': bad.pop('initializer_witness')
    elif damage == 'identity': bad['initializer_witness']['identity']['policy'] = 'changed'
    elif damage == 'removed_attempt': bad['initializer_witness']['attempts'].clear()
    elif damage == 'removed_registry': bad['knowledge'].pop('initializer_evidence')
    elif damage == 'removed_both':
        bad['initializer_witness']['attempts'].clear()
        bad['knowledge'].pop('initializer_evidence')
    else:
        attempt = bad['initializer_witness']['attempts']['r000001a1']
        if damage == 'unfinished': attempt['status'] = 'started'
        elif damage == 'source': attempt['source_sha256'] = 'a'*64
        elif damage == 'plan': attempt['plan_sha256'] = 'c'*64
        else: attempt['evidence_sha256'] = 'b'*64
    with pytest.raises(ValueError): c.load_state(bad)
    assert c.state() == saved


def test_orphaned_existing_registry_stops_before_any_helper_dispatch(monkeypatch, receipt):
    calls = install_receipt(monkeypatch, receipt)
    c, _, _ = controller(); c.complete_step()
    c._initializer_attempts.clear()
    c.queue = copy.deepcopy(c.knowledge.experiments)
    with pytest.raises(CampaignStopped): c.complete_step()
    assert len(calls) == 1


def test_different_receipt_plan_is_not_registered(monkeypatch, receipt):
    returned = copy.deepcopy(receipt)
    returned['plan']['plan_sha256'] = 'c'*64
    monkeypatch.setattr(witness, 'witness_sources', lambda *a, **kw: returned)
    c, model, _ = controller(); before = c.knowledge.state()
    with pytest.raises(CampaignStopped): c.complete_step()
    assert c.knowledge.state() == before and model.prompts == []


def test_unsupported_unknown_source_is_persisted_without_execution(monkeypatch):
    from sciel.knowledge_base.base import KnowledgeBase
    c, _, _ = controller(auxiliary=0, reflection=0)
    raw = dict(id='r000001a1', branch='p1', parent_id='p1', status='eval_failed',
               hypothesis=None, source_sha256=None, parent_source_sha256=None, metric=c.metric,
               parent_cases={}, evaluation=None)
    c.knowledge = KnowledgeBase(max_lessons=c.config['max_lessons'],
        max_experiments=c.config['max_experiments'], max_fragments=c.config['max_fragments'])
    c.knowledge.record(raw); c.queue = copy.deepcopy(c.knowledge.experiments)
    monkeypatch.setattr(witness, 'witness_sources', lambda *a, **kw: pytest.fail('unsupported dispatch'))
    c.complete_step(); saved = c.state()
    assert c._initializer_attempts['r000001a1']['status'] == 'unavailable'
    c.load_state(saved)
    assert c.state() == saved


def test_integer_quality_fields_match_normalized_committed_observation(monkeypatch, receipt):
    calls = install_receipt(monkeypatch, receipt)
    c, _, _ = controller(auxiliary=0, reflection=0)
    # Numeric scalar normalization is a KB contract, not a different measurement.
    c.queue[0]['score'] = 1; c.queue[0]['actual_delta'] = 0
    from sciel.knowledge_base.base import KnowledgeBase
    c.knowledge = KnowledgeBase(max_lessons=c.config['max_lessons'],
        max_experiments=c.config['max_experiments'], max_fragments=c.config['max_fragments'])
    c.queue[0]['evaluation'].update(score=1, raw_first_score=1)
    c.knowledge.record(c.queue[0])
    c.complete_step()
    assert len(calls) == 1 and c.knowledge.experiments[0]['score'] == 1.


def test_generation_injection_switch_omits_initializer_context(monkeypatch, receipt):
    install_receipt(monkeypatch, receipt)
    c, _, _ = controller(); c.complete_step(); c.config['inject_lessons'] = False
    parent = SimpleNamespace(id='p1', parent_id=None, code=body.source_for(), card='control', score=1., cases={})
    ctx = c.context(SimpleNamespace(_all={'p1': parent}), parent, 0, 2)
    assert 'initializer_evidence' not in ctx


def test_generation_journals_whole_context_omission(monkeypatch, receipt):
    install_receipt(monkeypatch, receipt)
    c, _, events = controller(); c.complete_step()
    monkeypatch.setattr(c, '_initializer_context', lambda ids, available: None)
    parent = SimpleNamespace(id='p1', parent_id=None, code=body.source_for(), card='control', score=1., cases={})
    ctx = c.context(SimpleNamespace(_all={'p1': parent}), parent, 0, 2)
    snapshot = Snapshot(0, 'circle_generate', 'fixture', 2, parent.id, parent.code,
                        parent.score, research_context=ctx)
    ident, prompt = c.begin(snapshot, 0, 'circle_generate', 'inert instruction')
    shown = [e for e in events if e['phase'] == 'initializer_witness_presentation' and e['role'] == 'generation']
    assert len(shown) == 1 and shown[0]['experiment_id'] == ident
    assert shown[0]['selected_experiment_ids'] == ['r000001a1']
    assert shown[0]['context'] is None and shown[0]['prompt_sha256'] == digest(prompt)
    assert 'initializer_evidence' not in prompt


def test_tiny_prompt_budget_omits_whole_context(monkeypatch, receipt):
    install_receipt(monkeypatch, receipt)
    c, _, _ = controller(); c.complete_step(); before = c.knowledge.state()
    assert c._initializer_context(['r000001a1'], 0) is None
    assert c.knowledge.state() == before


def test_new_config_requires_boolean_and_circle_generation():
    from test_circle_generation_engine import config
    cfg = config(); cfg['research']['initializer_witness'] = True
    assert apply_defaults(cfg)['research']['initializer_witness'] is True
    for invalid in (1, 'yes', None):
        bad = copy.deepcopy(cfg); bad['research']['initializer_witness'] = invalid
        with pytest.raises(ValueError, match='initializer_witness'): apply_defaults(bad)
    bad = copy.deepcopy(cfg); bad.pop('circle_generation'); bad['op_fixed'] = 'full_rewrite'
    with pytest.raises(ValueError, match='initializer_witness'): apply_defaults(bad)


def test_real_engine_orders_diagnostic_after_outcome_without_new_evaluation(monkeypatch, receipt, tmp_path):
    from test_circle_generation_engine import config, build, response, rows
    calls = install_receipt(monkeypatch, receipt)
    cfg = config(); cfg['research']['initializer_witness'] = True
    engine = build(tmp_path/'run', [response()], cfg=cfg)
    result = engine.run()
    assert result['B_evals'] == 1 and len(engine.fixture_evaluated) == 2 and len(calls) == 1
    events = rows(engine); phases = [e.get('phase') for e in events]
    assert phases.index('outcome') < phases.index('initializer_witness_intent')
    assert engine.research.knowledge.experiments[0]['score'] == .6
    assert engine.research.aux_calls == 0 and len(engine.llm.calls) == 1


def test_rejected_generation_restores_without_fabricated_diagnostic_slot(monkeypatch, tmp_path):
    from test_circle_generation_engine import config, build, response
    cfg = config(); cfg['research']['initializer_witness'] = True
    monkeypatch.setattr(witness, 'witness_sources', lambda *a, **kw: pytest.fail('rejected dispatch'))
    engine = build(tmp_path/'run', [response(target='return forbidden_name\n')], cfg=cfg)
    engine.run()
    saved = engine.research.state()
    assert saved['outcomes'] == 1 and saved['initializer_witness']['attempts'] == {}
    assert saved['knowledge']['experiments'][0]['source_sha256'] is None
    engine.research.load_state(saved)
    assert engine.research.state() == saved
