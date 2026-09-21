"""Unparsed generation failures stay engineering feedback, never quality evidence.

Only inert strings, fake provider responses and real bookkeeping are used.
"""
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController
from sciel.engine.types import Program, Snapshot
from sciel.engine.worker import run_worker
from sciel.hypothesis.rendering import decode_evidence, presentation_identity
from sciel.knowledge_base import KnowledgeBase


PARENT = 'def run_search():\n    return 0.5\n'
CHILD = 'def run_search():\n    return 0.6\n'
SOURCE_SHA = hashlib.sha256(CHILD.encode()).hexdigest()


def record(ident, *, branch='p1', status='hypothesis_invalid', failure='Malformed forecast', hypothesis=None):
    measured = status == 'evaluated'
    return dict(id=ident, branch=branch, parent_score=.5, status=status,
        score=.4 if measured else None, actual_delta=.4-.5 if measured else None,
        source_sha256=SOURCE_SHA if measured else None,
        proposed_sha256=SOURCE_SHA if measured else None, hypothesis=hypothesis, failure=failure)


class FakeLLM:
    models = ['fake']

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, model, temperature=None):
        self.prompts.append(prompt)
        # Structurally valid inert code, but no valid five-field forecast.
        return LLMResponse('HYPOTHESIS_JSON: {"broken":true}\n```python\n'+CHILD+'```',
                           model, .002, 10, 10)


def controller(**settings):
    config = apply_defaults({'research': {'enabled': True, 'max_aux_calls': 0,
        'reflection_interval': 0, **settings}})['research']
    llm, events = FakeLLM(), []
    research = ResearchController(config, llm,
        NS(research=lambda **event: events.append(copy.deepcopy(event))),
        lambda _: pytest.fail('no auxiliary calls'))
    parent = Program('p1', PARENT, .5, {}, 0, 'seed')
    other = Program('p2', PARENT, .5, {}, 1, 'seed')
    return research, llm, NS(_all={'p1': parent, 'p2': other}), parent, other, events


def snapshot(research, db, parent, iteration):
    context = research.context(db, parent, parent.island, iteration)
    return Snapshot(parent.island, 'full_rewrite', 'fake', iteration, parent.id, parent.code,
        parent.score, required_entrypoints=['run_search'], research_context=context)


@pytest.mark.parametrize('status', ['hypothesis_invalid', 'no_child', 'generation_stopped', 'eval_failed'])
def test_missing_forecast_retains_unmeasured_diagnostic_without_synthetic_quality(status):
    kb = KnowledgeBase()
    kb.record(record('e1', status=status, failure='SyntaxError: unexpected indent'))
    before = kb.state()
    text, = kb.failed_hypotheses('p1')
    assert f'Unmeasured ({status})' in text and 'quality not measured' in text
    assert 'No valid forecast.' in text and 'id=e1; source_sha256=None' in text
    assert 'diagnostic (untrusted): SyntaxError: unexpected indent' in text
    assert 'regression' not in text and 'refuted' not in text.lower()
    assert kb.state() == before
    stored, = kb.experiments
    assert stored['hypothesis'] is stored['score'] is stored['actual_delta'] is None
    assert stored['forecast_verdict'] == 'INCONCLUSIVE'


def test_scoping_order_zero_limit_and_older_evaluated_without_forecast_unchanged():
    kb = KnowledgeBase()
    for value in [record('old'), record('foreign', branch='p2'), record('middle'),
                  record('measured', status='evaluated'), record('new')]:
        kb.record(value)
    assert ['id=new;' in kb.failed_hypotheses('p1', 2)[0],
            'id=middle;' in kb.failed_hypotheses('p1', 2)[1]] == [True, True]
    assert len(kb.failed_hypotheses('p1', 100)) == 3
    assert 'id=foreign;' in kb.failed_hypotheses('p2')[0]
    assert kb.failed_hypotheses('unrelated') == kb.failed_hypotheses('p1', 0) == []
    for limit in (True, -1, 1.5):
        with pytest.raises(ValueError, match='limit'):
            kb.failed_hypotheses('p1', limit)


def test_diagnostic_bounded_no_null_text_and_restore_preserves_record_digest():
    kb = KnowledgeBase(max_experiments=2)
    kb.record(record('pruned'))
    kb.record(record('no_diagnostic', failure=None))
    kb.record(record('long', failure='X'*700+'MUST_NOT_REACH_PROMPT'))
    before = kb.state()
    restored = KnowledgeBase(max_experiments=2)
    restored.load_state(copy.deepcopy(before))
    texts = restored.failed_hypotheses('p1')
    assert texts == kb.failed_hypotheses('p1') and restored.state() == before
    assert texts[0].endswith('X'*700) and 'MUST_NOT_REACH_PROMPT' not in texts[0]
    assert len(texts[0]) < 1000 and 'diagnostic (untrusted)' not in texts[1]
    assert all('id=pruned;' not in text for text in texts)
    texts[0] = 'caller mutation'
    assert kb.state() == restored.state() == before


def test_valid_hypothesis_measured_feedback_keeps_exact_scores_in_string_context():
    hypothesis = dict(text='Change initialization', predicted_low=.01, predicted_high=.03,
                      reasoning='Uncertain local effect', confidence=.5)
    kb = KnowledgeBase()
    kb.record(record('e1', status='evaluated', hypothesis=hypothesis, failure=None))
    assert kb.failed_hypotheses('p1') == [
        'Forecast refuted; measured regression (forecast REFUTED) '
        f'[id=e1; source_sha256={SOURCE_SHA}]; measured_facts='
        '{"status":"evaluated","evaluator_ok":null,"task_validity":{"case":null,"value":null,"valid_value":null},'
        '"parent_score":0.5,"child_score":0.4,"actual_delta":-0.09999999999999998}'
        ' (null=unknown; evaluator_ok is not task validity): Change initialization']


@pytest.mark.parametrize('polarity', ['positive', 'negative'])
def test_unparsed_failure_cannot_create_signed_lesson_or_reflection_replacement(polarity):
    kb = KnowledgeBase()
    kb.record(record('e1'))
    lesson = {'rule': 'A supposed quality lesson', 'polarity': polarity,
              'scope': 'branch', 'evidence_ids': ['e1']}
    before = kb.state()
    with pytest.raises(ValueError, match='measured'):
        kb.add_lessons([lesson], allowed_evidence_ids=['e1'])
    with pytest.raises(ValueError, match='measured'):
        kb.reflect([lesson], [], allowed_evidence_ids=['e1'])
    assert kb.state() == before and kb.lessons == kb.fragments == []


@pytest.mark.parametrize('inject', [True, False])
def test_real_worker_failure_arrives_in_next_branch_prompt_and_restores_without_calls(inject):
    research, llm, db, parent, other, events = controller(inject_lessons=inject)
    def forbidden(_):
        pytest.fail('unparsed forecast must not execute a candidate')
    run_worker('full_rewrite', snapshot(research, db, parent, 1), llm, forbidden,
               dup_cap=1, research=research)
    research.complete_step()
    observed, = research.knowledge.experiments
    assert observed['status'] == 'hypothesis_invalid' and observed['hypothesis'] is None
    assert observed['score'] is observed['actual_delta'] is None
    saved = research.state()
    restored, restored_llm, _, _, _, _ = controller(inject_lessons=inject)
    restored.load_state(copy.deepcopy(saved))
    assert restored.state() == saved and restored_llm.prompts == []
    context = research.context(db, parent, 0, 2)
    assert restored.context(db, parent, 0, 2) == context
    assert research.context(db, other, 1, 2)['failed_hypotheses'] == []
    assert context['injected_ids'] == context['lesson_texts'] == context['fragments'] == []
    run_worker('full_rewrite', snapshot(restored, db, parent, 2), restored_llm, forbidden,
               dup_cap=1, research=restored)
    payload = json.JSONDecoder().raw_decode(restored_llm.prompts[0].split('RESEARCH_CONTEXT_JSON:', 1)[1].lstrip())[0]
    feedback, = decode_evidence(payload)['failed_hypotheses']
    assert 'No valid forecast.' in feedback and 'quality not measured' in feedback
    assert observed['id'] in feedback and 'missing or malformed pre-evaluation hypothesis' in feedback
    assert len(llm.prompts) == len(restored_llm.prompts) == 1
    assert restored.aux_calls == 0 and restored.knowledge.lessons == []
    assert all(e['phase'] not in ('aux_request', 'reflection') for e in events)


@pytest.mark.parametrize('legacy', [True, False])
def test_presentation_identity_rejects_missing_or_drift_before_controller_mutation(legacy):
    research, _, _, _, _, _ = controller()
    saved = research.state()
    assert saved['presentation']['failure_context'] == 'sciel-failure-context/2'
    if legacy:
        del saved['presentation']['failure_context']
    else:
        saved['presentation']['failure_context'] = 'sciel-failure-context/1'
    before = research.state()
    with pytest.raises(ValueError, match='presentation'):
        research.load_state(saved)
    assert research.state() == before
    detached = presentation_identity()
    detached['failure_context'] = 'caller mutation'
    assert presentation_identity()['failure_context'] == 'sciel-failure-context/2'


@pytest.mark.parametrize('force', [False, True])
def test_research_only_checkpoint_cannot_force_old_failure_presentation(tmp_path, force):
    research, _, _, _, _, _ = controller()
    journal = tmp_path/'events.jsonl'
    journal.write_bytes(b'{"event":"run_start"}\n')
    engine = NS(task=None, cfg={'research': research.config}, research=research,
                log=NS(path=str(journal)))
    identity = checkpoint._identity(engine)
    saved = research.state()
    del saved['presentation']['failure_context']
    del identity['research']['presentation']['failure_context']
    state = {'identity': identity, 'research': saved, 'provenance': {
        'schema': checkpoint.PROVENANCE_SCHEMA, 'counters': {'db_counter': None},
        'journal_anchor': {'byte_offset': journal.stat().st_size,
                           'prefix_sha256': hashlib.sha256(journal.read_bytes()).hexdigest()}}}
    file = tmp_path/'checkpoint.json'
    file.write_text(json.dumps(state))
    before = research.state()
    with pytest.raises(checkpoint.IdentityMismatch, match='presentation' if force else 'research'):
        checkpoint.load(engine, str(file), allow_identity_mismatch=force)
    assert research.state() == before
