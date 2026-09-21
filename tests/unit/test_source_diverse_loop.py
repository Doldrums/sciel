"""Engine propagation and resume tests with inert source and fake observations.

The evaluator module deliberately raises if imported. No candidate, evaluator,
provider or network request is executed by these fixtures.
"""
import copy
import json
import re
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.hypothesis import CONTRACT as RESEARCH_CONTRACT
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base import retrieval


RULES = {
    'A': 'INERT ADVICE ALPHA: preserve the recorded source observation.',
    'B': 'INERT ADVICE BETA: retain the recorded source observation.',
    'C': 'INERT ADVICE GAMMA: consider the independent source observation.',
}
POLICY = {'mode': 'source_diverse', 'diversity_weight': .5}


def decoded_block(prompt, marker):
    raw, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())
    return decode_evidence(raw)


def cards(prompt):
    result = []
    for suffix in prompt.split('RESEARCH_LESSON_JSON:\n')[1:]:
        raw, _ = json.JSONDecoder().raw_decode(suffix)
        result.append(decode_evidence(raw))
    return result


class FakeLLM:
    models = ['fixture']
    weights = [1.0]
    max_retries = 1

    def __init__(self):
        self.calls = []

    def choose_model(self, rng):
        return 'fixture'

    def generate(self, prompt, model, temperature=None):
        assert model == 'fixture'
        if 'Review this one experiment.' in prompt:
            from _mechanism_fixtures import unknown_mechanism
            kind = 'conclusion'
            evidence = decoded_block(prompt, 'EVIDENCE_JSON:')
            labels = {'r000001a1': ['A', 'B'], 'r000002a1': ['C'],
                      'r000003a1': []}[evidence['id']]
            text = 'CONCLUSION_JSON: ' + json.dumps({
                'verdict': evidence['forecast_verdict'],
                'mechanism_review': unknown_mechanism(evidence),
                'explanation': 'Inert observations only; no causal inference.',
                'lessons': [{'rule': RULES[label], 'polarity': 'positive',
                    'scope': 'branch', 'evidence_ids': [evidence['id']]}
                    for label in labels]})
        else:
            assert 'Review existing experimental evidence and lessons.' not in prompt
            kind = 'mutation'
            # Derive a unique inert response solely from the complete sent source
            # catalog. The fake has no hidden sequence state to reset on resume.
            catalog = json.loads(prompt.rsplit('SOURCE_CATALOG_JSON:\n', 1)[1])['catalog']
            previous = [int(n) for code in catalog.values()
                        for n in re.findall(r'fixture_step = (\d+)', code)]
            step = max(previous, default=0) + 1
            score = (step + 1) / 8
            text = 'HYPOTHESIS_JSON: ' + json.dumps({
                'hypothesis': 'Inert response increments the fixture value.',
                'predicted_low': 0., 'predicted_high': 1.,
                'reasoning': 'Synthetic transport test, not candidate execution.',
                'confidence': .5}) + (
                    f'\n```python\ndef run_search():\n'
                    f'    fixture_step = {step}\n    return {score!r}\n```')
        cost = .002 if kind == 'mutation' else .003
        call = dict(kind=kind, prompt=prompt, text=text, model=model,
                    temperature=temperature, cost=cost)
        self.calls.append(call)
        return LLMResponse(text=text, model=model, cost=cost,
                           prompt_tokens=10, completion_tokens=10)


@pytest.fixture
def task(tmp_path):
    folder = tmp_path / 'task'
    folder.mkdir()
    (folder / 'task.yaml').write_text(
        'suite: test\nname: source_diverse\norigin: math\nprimary_metric: score\n'
        'stochastic: false\nsmoke_timeout: 1\nfull_timeout: 1\n')
    (folder / 'initial_program.py').write_text(
        'def run_search():\n    fixture_step = 0\n    return 0.125\n')
    (folder / 'evaluator.py').write_text(
        'raise AssertionError("real evaluator must never be imported")\n')
    return load_task(str(folder))


def engine(task, directory, policy=POLICY):
    research = dict(enabled=True, retrieve_k=2, reflection_interval=0, max_aux_calls=3)
    if policy is not None:
        research['lesson_selection'] = copy.deepcopy(policy)
    cfg = dict(task='test/source_diverse', budget=3, max_iterations=3,
        dup_cap=1, random_seed=21, num_islands=1, schedule_policy='round_robin',
        milestone_fracs=[], op_fixed='full_rewrite',
        policy={'enabled': False, 'plateau_shift': False}, llm={'mock': True},
        portfolio_seed=False, idea_island=False, dynamic_islands=False,
        crossover=False, embed_grid=False, embed_niche=False, three_head=False,
        confirm_records=False, research=research)
    llm = FakeLLM()
    instance = Engine(cfg, task, str(directory), llm, python=sys.executable)
    observed_sources = []

    def text_score(code):
        observed_sources.append(code)
        score = float(re.search(r'return ([0-9.]+)', code).group(1))
        return CascadeOutcome(EvalResult(ok=True, score=score,
            cases={'value': score}, timing=.01), score, 'full', False)

    instance._evaluate = text_score
    return instance, llm, observed_sources


def journal(directory):
    return [json.loads(line) for line in (directory / 'sciel_events.jsonl').read_text().splitlines()]


def mutation_calls(llm):
    return [call for call in llm.calls if call['kind'] == 'mutation']


def test_full_engine_selects_independent_source_in_third_paid_stub_request(task, tmp_path):
    outcomes = {}
    for mode, expected in [('ranked', ['l000001', 'l000002']),
                           ('source_diverse', ['l000001', 'l000003']),
                           ('source_diverse_guarded', ['l000001', 'l000002'])]:
        directory = tmp_path / mode
        instance, llm, observed = engine(task, directory, {'mode': mode, 'diversity_weight': .5})
        result = instance.run()
        assert result['B_evals'] == result['iteration_no'] == 3
        assert len(observed) == 4  # Seed plus three fake observations.
        assert instance.research.reflections == 0
        mutations = mutation_calls(llm)
        assert len(mutations) == 3
        assert [card['lesson_id'] for card in cards(mutations[2]['prompt'])] == expected
        attempts = [event for event in journal(directory) if event.get('phase') == 'attempt']
        assert attempts[2]['evidence']['injected_ids'] == expected
        assert len(attempts[2]['lesson_presentation']['presented']) == 2
        if mode == 'source_diverse_guarded':
            selection = attempts[2]['lesson_presentation']['selection']
            assert selection['selected'] == 'baseline'
            assert 'whole_card_anchors' in selection['failed_checks']
            assert selection['candidate']['ids'] == ['l000001', 'l000003']
            assert selection['candidate_decisions_are_exposure_credit'] is False
        lessons = {lesson['id']: lesson for lesson in instance.research.knowledge.lessons}
        for ident in ('l000001', 'l000002', 'l000003'):
            assert ('r000003a1' in lessons[ident]['exposure_ids']) is (ident in expected)
        assert lessons['l000001']['source_support_ids'] == ['r000001a1']
        assert lessons['l000002']['source_support_ids'] == ['r000001a1']
        assert lessons['l000003']['source_support_ids'] == ['r000002a1']
        outcomes[mode] = result, llm
    assert outcomes['ranked'][0]['cost_total'] == outcomes['source_diverse'][0]['cost_total']
    assert outcomes['ranked'][0]['cost_total'] == outcomes['source_diverse_guarded'][0]['cost_total']
    assert outcomes['ranked'][0]['cost_total'] == pytest.approx(.015)
    assert ([call['kind'] for call in outcomes['ranked'][1].calls]
            == [call['kind'] for call in outcomes['source_diverse'][1].calls]
            == ['mutation', 'conclusion'] * 3)
    assert [call['prompt'] for call in mutation_calls(outcomes['ranked'][1])[:2]] == [
        call['prompt'] for call in mutation_calls(outcomes['source_diverse'][1])[:2]]


@pytest.mark.parametrize('mode', ['source_diverse', 'source_diverse_guarded'])
def test_engine_checkpoint_resume_matches_continuous_prompts_cost_rng_and_exposures(task, tmp_path, mode):
    policy = {'mode': mode, 'diversity_weight': .5}
    whole, whole_llm, whole_observed = engine(task, tmp_path / 'whole', policy)
    whole_result = whole.run()
    split = tmp_path / 'split'
    first, first_llm, first_observed = engine(task, split, policy)
    first._stop_at = 2
    assert first.run()['reason'] == 'paused'
    saved = json.loads((split / 'checkpoint.json').read_text())
    assert saved['research']['lesson_selection'] == retrieval.selection_identity(first.cfg['research'])
    assert saved['identity']['research']['lesson_selection'] == saved['research']['lesson_selection']
    resumed, resumed_llm, resumed_observed = engine(task, split, policy)
    checkpoint.load(resumed, str(split / 'checkpoint.json'))
    assert resumed.research.state() == first.research.state()
    assert resumed.run() == whole_result
    assert first_llm.calls + resumed_llm.calls == whole_llm.calls
    assert first_observed + resumed_observed == whole_observed
    assert resumed.research.state() == whole.research.state()
    assert resumed.db.state() == whole.db.state()
    assert resumed.rng.state() == whole.rng.state()
    assert resumed.research.knowledge.lessons == whole.research.knowledge.lessons


@pytest.mark.parametrize('drift', ['weight', 'version', 'missing_identity'])
def test_force_identity_cannot_resume_changed_or_unbound_selection(task, tmp_path, monkeypatch, drift):
    directory = tmp_path / 'paused'
    first, _, _ = engine(task, directory)
    first._stop_at = 2
    assert first.run()['reason'] == 'paused'
    path = directory / 'checkpoint.json'
    policy = copy.deepcopy(POLICY)
    if drift == 'weight':
        policy['diversity_weight'] = .75
    elif drift == 'version':
        monkeypatch.setattr(retrieval, 'CONTRACT', 'sciel-lesson-selection/future')
    else:
        saved = json.loads(path.read_text())
        del saved['research']['lesson_selection']
        del saved['identity']['research']['lesson_selection']
        path.write_text(json.dumps(saved))
    changed, llm, observed = engine(task, directory, policy)
    before = changed.research.state()
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match='cannot resume research'):
            checkpoint.load(changed, str(path), allow_identity_mismatch=True)
        assert llm.calls == observed == []
        assert changed.research.state() == before
    finally:
        changed.log.close()


def test_default_research_engine_keeps_selection_identity_absent(task, tmp_path):
    instance, llm, observed = engine(task, tmp_path / 'legacy', None)
    assert RESEARCH_CONTRACT == 'sciel-research/12'
    assert instance.run()['B_evals'] == 3
    assert 'lesson_selection' not in instance.cfg['research']
    assert 'lesson_selection' not in instance.research.summary()
    saved = json.loads((tmp_path / 'legacy' / 'checkpoint.json').read_text())
    assert 'lesson_selection' not in saved['research']
    assert 'lesson_selection' not in saved['identity']['research']
    assert [card['lesson_id'] for card in cards(mutation_calls(llm)[2]['prompt'])] == [
        'l000001', 'l000002']


@pytest.mark.parametrize('drift', ['guard_version', 'mode', 'missing_identity'])
def test_guarded_resume_refuses_drift_even_with_force(task, tmp_path, monkeypatch, drift):
    policy = {'mode': 'source_diverse_guarded', 'diversity_weight': .5}
    directory = tmp_path / 'guarded'
    first, _, _ = engine(task, directory, policy)
    first._stop_at = 2
    assert first.run()['reason'] == 'paused'
    path = directory / 'checkpoint.json'
    if drift == 'guard_version':
        monkeypatch.setattr(retrieval, 'GUARD_VERSION', 'future')
    elif drift == 'mode':
        policy['mode'] = 'source_diverse'
    else:
        saved = json.loads(path.read_text())
        del saved['research']['lesson_selection']
        path.write_text(json.dumps(saved))
    changed, llm, observed = engine(task, directory, policy)
    before = changed.research.state()
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match='cannot resume research'):
            checkpoint.load(changed, str(path), allow_identity_mismatch=True)
        assert changed.research.state() == before
        assert llm.calls == observed == []
    finally:
        changed.log.close()
