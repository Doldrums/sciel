"""Final-source novelty/retry integration; no network or candidate execution."""
import json
from pathlib import Path

import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.engine import checkpoint
from sciel.engine.idea import TwoStageIdea
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.engine.novelty import CandidateNoveltyRegistry
from sciel.engine.recovery import inspect_provider_stop, prepare_provider_recovery
from sciel.engine.types import NoChild, Result
from sciel.engine.validation import source_sha256, validate_candidate
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from test_research_retry_feedback import setup, text, PARENT, CHILD
from test_two_stage_idea import setup as idea_setup, response, code_response, CARD


def callback(registry, retries=2):
    def inspect(code):
        return registry.inspect(code)
    inspect.candidate_registry = True
    inspect.max_resamples = retries
    return inspect


def registry_with(code=PARENT, *, ok=True):
    registry = CandidateNoveltyRegistry()
    registry.register(code, 'evaluation-old', island=2, evaluation_ok=ok)
    if ok:
        registry.bind_program(code, 'p-archived', 2)
    return registry


def previous(prompt):
    raw = prompt.split('PREVIOUS_RESPONSE_JSON (exact previous generator response):\n', 1)[1]
    return json.JSONDecoder().raw_decode(raw.lstrip())[0]


@pytest.mark.parametrize('research_on', [False, True])
def test_exact_then_ast_duplicates_receive_full_feedback_and_only_new_final_source_evaluates(research_on):
    original = text(PARENT)
    reformatted = text('# This comment changes bytes only.\n' + PARENT)
    model, research, snap, events, evaluated, evaluate = setup([original, reformatted, text()], research_on=research_on)
    result = run_worker('diff', snap, model, evaluate, dup_cap=1,
                        novelty_check=callback(registry_with()), research=research if research_on else None)
    assert isinstance(result, Result) and evaluated == [CHILD]
    assert len(model.calls) == 3 and result.cost == pytest.approx(.03)
    assert previous(model.calls[1][0]) == original
    assert previous(model.calls[2][0]) == reformatted
    assert all('p-archived' in call[0] for call in model.calls[1:])
    assert [r['match_kind'] for r in result.meta['novelty_checks'][:2]] == ['sha256', 'ast']
    assert len(result.meta['validation_trace']['attempts']) == 3
    if research_on:
        rejected = research.knowledge.experiments[:2]
        assert all(r['score'] is r['actual_delta'] is r['evaluation'] is None for r in rejected)
        assert [r['id'] for r in research.queue] == [research.knowledge.experiments[-1]['id']]


@pytest.mark.parametrize('retries', [0, 1, 2])
def test_registry_retry_cap_overrides_legacy_dup_cap_and_never_evaluates(retries):
    model, research, snap, _, evaluated, evaluate = setup([text(PARENT)] * 8)
    result = run_worker('full_rewrite', snap, model, evaluate, dup_cap=8,
                        novelty_check=callback(registry_with(), retries), research=research)
    assert isinstance(result, NoChild) and result.reason == 'dup_cap'
    assert len(model.calls) == retries + 1 and not evaluated
    assert result.cost == pytest.approx(.01 * (retries + 1))
    assert research.queue == [] and research.outcomes == retries + 1
    assert len(result.meta['validation_trace']['attempts']) == retries + 1
    assert 'resample' not in result.meta['validation_trace']['attempts'][-1]


def test_complete_previous_response_is_not_truncated_in_retry_prompt():
    first = text(PARENT) + '\n' + ('long explanation ' * 5000)
    model, _, snap, _, evaluated, evaluate = setup([first, text()], research_on=False)
    result = run_worker('full_rewrite', snap, model, evaluate, novelty_check=callback(registry_with()))
    assert isinstance(result, Result) and evaluated == [CHILD]
    assert len(first) > 65536 and previous(model.calls[1][0]) == first


def test_actual_archived_ast_duplicate_is_rejected_after_validation_without_evaluation():
    fixtures = Path(__file__).resolve().parents[1] / 'fixtures/candidate_novelty'
    parent, duplicate = ((fixtures / name).read_text() for name in ('p143.py', 'p175.py'))
    reply = text(duplicate)
    model, _, snap, _, evaluated, evaluate = setup([reply] * 3, research_on=False)
    snap.parent_id, snap.parent_code, snap.required_entrypoints = 'p143', parent, []
    registry = CandidateNoveltyRegistry()
    registry.register(parent, 'eval-p143', evaluation_ok=True)
    registry.bind_program(parent, 'p143')
    result = run_worker('full_rewrite', snap, model, evaluate, novelty_check=callback(registry))
    assert isinstance(result, NoChild) and evaluated == []
    assert len(model.calls) == 3 and result.cost == pytest.approx(.03)
    assert all(row['match_kind'] == 'ast' for row in result.meta['novelty_checks'])
    for prompt, _, _ in model.calls[1:]:
        assert 'p143' in prompt and 'structural_ast_duplicate' in prompt
        assert previous(prompt) == reply


def test_constant_change_evaluates_even_with_high_embedding_cosine():
    fixtures = Path(__file__).resolve().parents[1] / 'fixtures/candidate_novelty'
    parent = (fixtures / 'p143.py').read_text()
    changed = parent.replace('TIME_LIMIT = 28', 'TIME_LIMIT = 29', 1)
    assert changed != parent
    model, _, snap, _, evaluated, evaluate = setup([text(changed)], research_on=False)
    snap.parent_id, snap.parent_code, snap.required_entrypoints = 'p143', parent, []
    result = run_worker('full_rewrite', snap, model, evaluate,
                        novelty_check=callback(registry_with(parent)))
    assert isinstance(result, Result) and evaluated == [changed]
    assert result.meta['novelty']['match_kind'] == 'embedding'
    assert result.meta['novelty']['neighbors'][0]['similarity'] > .99
    assert result.meta['novelty']['neighbors'][0]['ast_diff']


def test_final_repair_and_graft_are_inspected_once_with_exact_final_hash():
    parent = 'def solve(): return 7\n'
    proposed = 'A proposed algorithm.\n"""\ndef helper(): return 2\n'
    calls = []
    def inspect(code):
        calls.append(code)
        return {'accept': True, 'match_kind': 'new', 'remeasurement': False}
    inspect.candidate_registry = True
    result = validate_candidate(proposed, parent_code=parent, required_entrypoints=['solve'],
                                code_hashes=set(), novelty_check=inspect)
    assert len(calls) == 1 and calls[0] == result.code
    assert result.code.startswith('"""A proposed algorithm.') and parent.strip() in result.code
    assert result.decisions[-1]['input_code_sha256'] == source_sha256(result.code)
    assert result.decisions[-2]['stage'] == 'interface'


def test_registry_replaces_normalized_hash_that_erases_blank_lines_inside_literals():
    before = 'def run_search():\n    return """a\n\nb"""\n'
    after = 'def run_search():\n    return """a\nb"""\n'
    from sciel.engine.db import normalized_hash
    assert normalized_hash(before) == normalized_hash(after)
    result = validate_candidate(after, parent_code=before, required_entrypoints=['run_search'],
        code_hashes={normalized_hash(before)}, novelty_check=callback(registry_with(before)))
    assert result.violation is None and result.novelty['accept'] is True


@pytest.mark.parametrize('failed', [False, True])
def test_known_failed_source_is_remeasurement_without_research_efficacy_or_conclusion(failed):
    model, research, snap, _, evaluated, evaluate = setup([text()])
    if failed:
        evaluate = lambda code: CascadeOutcome(EvalResult(False, 0., {}), 0., 'smoke')
    result = run_worker('full_rewrite', snap, model, evaluate,
        research=research, novelty_check=callback(registry_with(CHILD, ok=False)))
    assert isinstance(result, Result) and result.meta['remeasurement'] is True
    assert result.meta['novelty']['matched_evaluation_id'] == 'evaluation-old'
    record, = research.knowledge.experiments
    assert record['status'] == 'no_child'
    assert record['score'] is record['actual_delta'] is record['evaluation'] is None
    assert research.queue == []


def test_raised_remeasurement_does_not_queue_efficacy_conclusion_and_keeps_cost():
    model, research, snap, _, _, _ = setup([text()])
    charges = []
    research.charge = charges.append
    def evaluate(code):
        raise RuntimeError('private evaluator detail')
    with pytest.raises(CampaignStopped):
        run_worker('full_rewrite', snap, model, evaluate, research=research,
                   novelty_check=callback(registry_with(CHILD, ok=False)))
    assert charges == [.01] and research.queue == []
    assert research.knowledge.experiments[0]['status'] == 'no_child'


@pytest.mark.parametrize('failure', ['evaluator', 'not_dispatched'])
def test_nonresearch_prior_paid_retry_costs_survive_later_stop(failure):
    model, _, snap, _, _, _ = setup([text(PARENT), text()], research_on=False)
    generate = model.generate
    def dispatch(*args, **kwargs):
        if failure == 'not_dispatched' and model.calls:
            raise RequestNotDispatched('local stop')
        return generate(*args, **kwargs)
    model.generate = dispatch
    def evaluate(code):
        raise RuntimeError('private evaluator detail')
    with pytest.raises(CampaignStopped) as stopped:
        run_worker('full_rewrite', snap, model, evaluate, novelty_check=callback(registry_with()))
    assert stopped.value.unbooked_known_cost == pytest.approx(.01 if failure == 'not_dispatched' else .02)
    assert stopped.value.current_call_cost_unknown is False
    assert 'private evaluator detail' not in str(stopped.value)


@pytest.mark.parametrize('malformed', [None, 42, {'unexpected': 'response'}])
def test_enabled_malformed_response_preserves_paid_cost(malformed):
    model, _, snap, _, evaluated, evaluate = setup([malformed], research_on=False)
    with pytest.raises(CampaignStopped) as stopped:
        run_worker('full_rewrite', snap, model, evaluate, novelty_check=callback(registry_with()))
    assert stopped.value.unbooked_known_cost == pytest.approx(.01)
    assert stopped.value.current_call_cost_unknown is False
    assert stopped.value.generation_returned is True
    assert stopped.value.stage == 'response_processing' and evaluated == []


def test_idea_retries_only_code_under_one_card_and_preserves_all_costs_and_state():
    duplicate = code_response('def run_search():\n    return 1.0\n')
    research, idea, snap, model, log, charged, evaluated, run = idea_setup([
        response(json.dumps(CARD)), response(duplicate), response(duplicate), response(code_response())])
    registry = registry_with(snap.parent_code)
    result = run(novelty_check=callback(registry))
    assert isinstance(result, Result) and len(model.calls) == 4
    assert sum('IDEA STAGE 1:' in p for p in model.calls) == 1
    assert previous(model.calls[2]) == previous(model.calls[3]) == duplicate
    assert len(evaluated) == 1 and result.cost == pytest.approx(.12) and charged == []
    assert idea.code_retries == 2 and len(research.queue) == 1
    restored = TwoStageIdea(research)
    restored.load_state(json.loads(json.dumps(idea.state())))
    assert restored.state() == idea.state()


@pytest.mark.parametrize('raises', [False, True])
def test_idea_remeasurement_never_queues_efficacy_conclusion(raises):
    from test_two_stage_idea import CHILD as idea_child
    research, idea, snap, model, log, charged, evaluated, run = idea_setup([
        response(json.dumps(CARD)), response(code_response())])
    registry = registry_with(idea_child, ok=False)
    def evaluate(code):
        if raises:
            raise RuntimeError('private evaluator detail')
        return CascadeOutcome(EvalResult(True, 1.5, {}), 1.5, 'full')
    if raises:
        with pytest.raises(CampaignStopped):
            run(evaluate, novelty_check=callback(registry))
        assert charged == [pytest.approx(.06)]
    else:
        result = run(evaluate, novelty_check=callback(registry))
        assert result.meta['remeasurement'] is True and result.cost == pytest.approx(.06)
    assert research.queue == []
    evidence, = research.knowledge.experiments
    assert evidence['status'] == 'no_child'
    assert evidence['score'] is evidence['actual_delta'] is evidence['evaluation'] is None


def test_idea_retry_provider_recovery_accounts_every_returned_rejection(tmp_path, monkeypatch):
    from test_engine_recovery_review_fixes import construct, settings, outcome
    seed = (Path(__file__).resolve().parents[2] / 'tasks/test/toy_quadratic/initial_program.py').read_text()
    class Provider(MockLLM):
        calls = 0
        def generate(self, prompt, model, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                value = json.dumps(CARD)
            elif self.calls <= 3:
                value = code_response(seed)
            else:
                raise LLMError(cause=ConnectionError())
            return LLMResponse(value, model, .03, 10, 10)
    provider = Provider()
    cfg = settings(op_fixed='idea', idea_two_stage={'enabled': True},
        candidate_novelty={'enabled': True, 'max_resamples': 2},
        research={'enabled': True, 'max_aux_calls': None})
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', lambda *args, **kwargs: outcome())
    engine = construct(tmp_path, cfg, provider)
    with pytest.raises(CampaignStopped, match='two-stage idea stopped'):
        engine.run()
    path = tmp_path / 'checkpoint.json'
    receipt = inspect_provider_stop(path, allow_legacy=False)
    assert receipt['known_attempt_cost'] == pytest.approx(.09)
    assert receipt['request']['call_index'] == provider.calls == 4
    saved = json.loads(path.read_bytes())
    assert saved['idea_two_stage']['code_retries'] == 2 and saved['cost_total'] == pytest.approx(.09)
    restored = construct(tmp_path, saved['config'], provider)
    checkpoint.load(restored, str(path))
    prepare_provider_recovery(restored, path, allow_legacy=False)
    restored.request_stop()
    assert restored.run()['reason'] == 'paused'
    assert provider.calls == 4 and restored.cost_total == pytest.approx(.09)
    assert restored.idea.code_retries == 2
