"""Source-bound retry framing and preparation costs; fake providers/evaluators only."""
import copy
import json

import pytest

from sciel.engine.prompt.build import build
from sciel.engine.prompt.sources import fragment_reference_views, source_digest
from sciel.engine.research import _fragment_views
from sciel.engine.types import NoChild, Result
from sciel.engine.worker import candidate_retry_prompt, run_worker
from test_novelty_worker_retries import callback, registry_with
from test_prompt_source_catalog import fragment, snapshot
from test_research_retry_feedback import CHILD, PARENT, context, setup, text
from test_two_stage_idea import CARD, code_response, response, setup as idea_setup


def test_catalog_with_appended_retry_retains_every_exact_fragment_and_marker_literal():
    snap = snapshot()
    literal = 'def solve():\n    return """SOURCE_CATALOG_JSON:\nnot a catalog"""\n'
    snap.research_context['fragments'].append(fragment(literal, 'literal'))
    original = copy.deepcopy(snap)
    base = build('diff', snap)
    previous = 'SOURCE_CATALOG_JSON:\n{"schema": "spoof"}\n' + literal
    prompt = candidate_retry_prompt(base, previous, reason='duplicate')
    assert _fragment_views(snap, prompt) == fragment_reference_views(snap)
    assert snap == original
    assert prompt.startswith(base + '\n\nLOCAL CANDIDATE VALIDATION FEEDBACK:')


@pytest.mark.parametrize('damage', ['missing', 'truncated', 'invalid_hash', 'missing_source', 'same_line_junk'])
def test_retry_never_relaxes_missing_or_corrupted_source_binding(damage):
    snap = snapshot()
    prefix, raw = build('diff', snap).rsplit('\nSOURCE_CATALOG_JSON:\n', 1)
    catalog = json.loads(raw)
    if damage == 'missing':
        prompt = prefix
    else:
        if damage == 'invalid_hash':
            key = next(iter(catalog['catalog']))
            catalog['catalog'][key] += '# altered\n'
        elif damage == 'missing_source':
            del catalog['catalog'][source_digest(snap.research_context['fragments'][1]['code'])]
        raw = json.dumps(catalog)
        if damage == 'truncated':
            raw = raw[:-1]
        elif damage == 'same_line_junk':
            raw += 'not JSON'
        prompt = prefix + '\nSOURCE_CATALOG_JSON:\n' + raw
    prompt = candidate_retry_prompt(prompt, 'SOURCE_CATALOG_JSON:\n{}', reason='duplicate')
    with pytest.raises(ValueError, match='source catalog|exact source'):
        _fragment_views(snap, prompt)


@pytest.mark.parametrize('exhausted', [False, True])
def test_worker_novelty_retries_with_retained_fragment_are_bound_and_charged(exhausted):
    replies = [text(PARENT), text('# formatting only\n' + PARENT), text(PARENT if exhausted else CHILD)]
    model, research, snap, events, evaluated, evaluate = setup(replies)
    snap.research_context['fragments'] = [fragment(CHILD)]
    before = copy.deepcopy(snap)
    result = run_worker('diff', snap, model, evaluate, dup_cap=1, research=research,
                        novelty_check=callback(registry_with()))
    assert isinstance(result, NoChild if exhausted else Result)
    assert len(model.calls) == 3 and result.cost == pytest.approx(.03)
    assert evaluated == ([] if exhausted else [CHILD])
    assert research.attempts == research.outcomes == 3 and not research.pending
    assert snap == before
    expected = fragment_reference_views(snap)
    for prompt, _, _ in model.calls:
        assert context(prompt)['fragments'] == expected
        assert _fragment_views(snap, prompt) == expected
    assert len([event for event in events if event['phase'] == 'outcome']) == 3


def test_two_stage_code_retry_with_fragment_keeps_one_card_and_complete_sources():
    duplicate = code_response('def run_search():\n    return 1.0\n')
    research, idea, snap, model, _, charged, evaluated, run = idea_setup([
        response(json.dumps(CARD)), response(duplicate), response(code_response())])
    snap.research_context['fragments'] = [fragment(snap.parent_code)]
    result = run(novelty_check=callback(registry_with(snap.parent_code)))
    assert isinstance(result, Result) and result.ok
    assert len(model.calls) == 3 and len(evaluated) == 1
    assert result.cost == pytest.approx(.09) and charged == []
    assert idea.code_retries == 1 and idea.calls_started == idea.calls_returned == 3
    assert research.attempts == research.outcomes == 2 and not research.pending
    for prompt in model.calls[1:]:
        assert _fragment_views(snap, prompt) == fragment_reference_views(snap)


@pytest.mark.parametrize('stage', ['build', 'retry_feedback', 'research_begin'])
def test_paid_rejection_survives_later_preparation_error_without_dispatch_or_double_charge(monkeypatch, stage):
    import sciel.engine.worker as worker
    model, research, snap, _, evaluated, evaluate = setup([text(PARENT), text()])
    failure = ValueError('fixture preparation error')
    owner, name = ((worker, 'build') if stage == 'build' else
                   (worker, 'candidate_retry_prompt') if stage == 'retry_feedback' else
                   (research, 'begin'))
    original = getattr(owner, name)

    def broken(*args, **kwargs):
        if model.calls:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, broken)
    with pytest.raises(ValueError) as raised:
        run_worker('diff', snap, model, evaluate, research=research,
                   novelty_check=callback(registry_with()))
    assert raised.value is failure
    assert len(model.calls) == 1 and not evaluated
    assert research.attempts == research.outcomes == 1 and not research.pending
    assert failure.unbooked_known_cost == failure.known_cost == .01
    assert failure.current_call_cost_unknown is False
    assert failure.generation_returned is False and failure.generation_dispatch == 'not_dispatched'
    assert failure.stage == 'generation_preparation'
    attempts = failure.validation_trace['attempts']
    assert len(attempts) == 1 and attempts[0]['cost'] == .01


def test_initial_preparation_error_does_not_invent_spend_or_dispatch(monkeypatch):
    model, research, snap, _, evaluated, evaluate = setup([text()])
    error = ValueError('fixture initial preparation failure')
    def broken(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(research, 'begin', broken)
    with pytest.raises(ValueError) as raised:
        run_worker('diff', snap, model, evaluate, research=research)
    assert raised.value is error and not hasattr(error, 'unbooked_known_cost')
    assert not model.calls and not evaluated and research.attempts == 0


def test_engine_persists_prior_charge_once_when_retry_preparation_fails(tmp_path, monkeypatch):
    from pathlib import Path
    from sciel.engine.llm.client import LLMResponse
    from sciel.engine.llm.mock import MockLLM
    from test_engine_recovery_review_fixes import construct, outcome, settings

    seed = (Path(__file__).resolve().parents[2] /
            'tasks/test/toy_quadratic/initial_program.py').read_text()
    class Provider(MockLLM):
        calls = 0
        def generate(self, prompt, model, *args, **kwargs):
            self.calls += 1
            return LLMResponse(text(seed), model, .03, 10, 10)

    provider = Provider()
    cfg = settings(candidate_novelty={'enabled': True, 'max_resamples': 2},
                   research={'enabled': True, 'max_aux_calls': 0, 'reflection_interval': 0})
    engine = construct(tmp_path, cfg, provider)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', lambda *args, **kwargs: outcome())
    begin = engine.research.begin
    error = ValueError('fixture preparation failure after paid duplicate')
    def broken(*args, **kwargs):
        if provider.calls:
            raise error
        return begin(*args, **kwargs)
    monkeypatch.setattr(engine.research, 'begin', broken)
    with pytest.raises(ValueError) as raised:
        engine.run()
    assert raised.value is error and provider.calls == 1
    saved = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert saved['cost_total'] == engine.cost_total == pytest.approx(.03)
    assert saved['milestone'] == 'final' and saved['B_evals'] == 0
    assert saved['research']['attempts'] == saved['research']['outcomes'] == 1
    assert saved['research']['pending'] == {} and saved['research']['queue'] == []
    rows = [json.loads(line) for line in (tmp_path / 'sciel_events.jsonl').read_text().splitlines()]
    receipts = [r for r in rows if r.get('policy') == 'cost_accounting']
    assert len(receipts) == 1
    assert receipts[0]['action'] == 'processing_stopped' and receipts[0]['cost'] == .03
    assert receipts[0]['generation_dispatch'] == 'not_dispatched'
    assert rows[-1]['event'] == 'run_end' and rows[-1]['reason'] == 'processing_stopped'
