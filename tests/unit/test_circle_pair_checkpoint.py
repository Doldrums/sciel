"""Behavioral checks for circle pair checkpoint."""
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
from circle_pair_fixtures import (
    POLICY, candidate, paired, config,
    PairModel, build, observations, actions,
    context, assert_one_selected, _rewrite_checkpoint, _novelty_config,
)


def test_continuous_and_resumed_pair_histories_have_identical_actual_next_prompt(tmp_path):
    raw = paired(candidate('return np.random.choice(26)\n'), candidate('return ordinary_params\n'))
    cfg = config(budget=1, max_iterations=2, pop_cap=1, grid_bins=1)
    whole = build(tmp_path/'whole', [raw, paired()], cfg=cfg)
    expected = whole.run()
    split = build(tmp_path/'split', [raw], cfg=cfg)
    original_step = split._step

    def pause_after_paid_pair(island):
        original_step(island)
        if split.circle_generation.finished == 1:
            split._stop_at = split.B_evals

    split._step = pause_after_paid_pair
    assert split.run()['reason'] == 'paused'
    saved_first = copy.deepcopy(observations(split))
    resumed = build(tmp_path/'split', [paired()], cfg=cfg)
    checkpoint.load(resumed, str(tmp_path/'split/checkpoint.json'))
    assert observations(resumed) == saved_first
    assert resumed.run() == expected
    assert resumed.llm.calls == whole.llm.calls[1:]
    assert resumed.research.state() == whole.research.state()
    assert resumed.circle_generation.state() == whole.circle_generation.state()
    assert resumed.fixture_evaluated == [body.source_for(BODIES[0])]
    assert resumed.cost_total == 2*COST
    next_prompt = resumed.llm.calls[0]['prompt']
    assert 'np.random.choice' in next_prompt and 'baseline_initializer' in next_prompt
    assert context(next_prompt)['parent_score'] == .5


def test_pair_checkpoint_cannot_resume_under_single_policy_even_with_force(tmp_path):
    directory = tmp_path/'run'
    engine = build(directory, [paired()])
    engine.run()
    from test_circle_generation_engine import build as build_single
    restored = build_single(directory, [], cfg=single_config())
    try:
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(restored, str(directory/'checkpoint.json'), allow_identity_mismatch=True)
        assert not restored.llm.calls and not restored.fixture_evaluated
    finally:
        restored.log.close()


@pytest.mark.parametrize('change', [None, 'selected_index', 'selected_bool',
    'eligible_counter', 'counter_bool', 'raw_substitution', 'body_substitution'])
def test_rehashed_pair_checkpoint_still_validates_selection_counts_and_raw_group_binding(tmp_path, change):
    from sciel.hypothesis.circle_proposal_group import build_group
    directory = tmp_path/'run'
    raw = paired()
    engine = build(directory, [raw])
    engine.run()
    state = json.loads((directory/'checkpoint.json').read_text())
    record = state['research']['knowledge']['experiments'][0]
    group = record['proposal_group']
    if change == 'selected_index':
        group['selected_index'] = 1
    elif change == 'selected_bool':
        group['selected_index'] = False  # Python equality must not alias index0.
    elif change == 'eligible_counter':
        state['circle_generation']['proposal_pair_counts']['eligible_entries'] = 1
    elif change == 'counter_bool':
        state['circle_generation']['proposal_pair_counts']['selected_entries'] = True
    elif change in ('raw_substitution', 'body_substitution'):
        altered = json.loads(raw)
        if change == 'raw_substitution':
            altered['candidates'][1]['hypothesis']['reasoning'] = 'Different unmeasured alternative reasoning.'
        else:
            altered['candidates'][1]['body'] = 'return best_params + uniform_noise * 0.75\n'
        record['proposal_group'] = build_group(request_id=record['id'],
            raw_response=json.dumps(altered), parent_source_sha256=record['parent_source_sha256'],
            gates=[e['gate'] for e in group['entries']], selected_proof=record['controlled_change'])
    path = _rewrite_checkpoint(directory, state)
    restored = build(directory, [])
    try:
        if change is None:
            checkpoint.load(restored, str(path), allow_identity_mismatch=True)
            assert restored.circle_generation.state() == engine.circle_generation.state()
            assert observations(restored) == observations(engine)
        else:
            with pytest.raises(checkpoint.IdentityMismatch):
                checkpoint.load(restored, str(path), allow_identity_mismatch=True)
        assert not restored.llm.calls and not restored.fixture_evaluated
    finally:
        restored.log.close()


@pytest.mark.parametrize('change', ['drop_group', 'drop_record', 'research_forecast_count'])
def test_pair_checkpoint_cannot_erase_retained_group_record_or_selected_forecast_count(tmp_path, change):
    directory = tmp_path/'run'
    engine = build(directory, [paired()])
    engine.run()
    state = json.loads((directory/'checkpoint.json').read_text())
    knowledge = state['research']['knowledge']
    original, = copy.deepcopy(knowledge['experiments'])
    history = copy.deepcopy(state['circle_generation']['proposal_pair_history'])
    if change == 'drop_group':
        del knowledge['experiments'][0]['proposal_group']
    elif change == 'drop_record':
        knowledge['experiments'] = []
        knowledge['fragments'] = []
    else:
        assert state['research']['valid_hypotheses'] == 1
        state['research']['valid_hypotheses'] = 0
    # Keep the complete worker/journal history. Refresh the remaining KB record
    # digest (including fragment snapshots) and outer byte hashes, so a missing
    # optional-looking field cannot fail only because its old digest remains.
    path = _rewrite_checkpoint(directory, state, refresh_pair_history=False)
    assert state['circle_generation']['proposal_pair_history'] == history
    assert original['id'] in state['research']['knowledge']['seen_experiments']
    if change == 'drop_group':
        from sciel.knowledge_base.base import _digest, _snapshot
        assert knowledge['seen_experiments'][original['id']] == _digest(_snapshot(knowledge['experiments'][0]))
    restored = build(directory, [])
    try:
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(restored, str(path), allow_identity_mismatch=True)
        assert not restored.llm.calls and not restored.fixture_evaluated
    finally:
        restored.log.close()


@pytest.mark.parametrize('change', ['consumed_sources', 'history_fingerprint', 'evidence_source', 'response_body'])
def test_rehashed_journal_cannot_redefine_pair_selection_snapshot_or_receipts(tmp_path, change):
    from sciel.engine.db import normalized_hash
    directory = tmp_path/'run'
    engine = build(directory, [paired()])
    engine.run()
    state = json.loads((directory/'checkpoint.json').read_text())

    def mutate(events):
        group = next(e for e in events if e.get('phase') == 'circle_generation'
                     and e.get('action') == 'proposal_group')
        if change == 'consumed_sources':
            group['selection_snapshot']['consumed_sources'] = ['f'*64]
        elif change == 'history_fingerprint':
            group['selection_snapshot']['code_hashes'] = sorted(set(
                group['selection_snapshot']['code_hashes'] + [normalized_hash(body.source_for(BODIES[0]))]))
        elif change == 'evidence_source':
            group['gate_evidence'][0]['source_sha256'] = '0'*64
        else:
            receipt = next(e for e in events if e.get('phase') == 'circle_generation'
                           and e.get('action') == 'request_result')
            value = json.loads(receipt['response'])
            value['candidates'][1]['body'] = 'return best_params + uniform_noise * 0.75\n'
            receipt['response'] = json.dumps(value)
            receipt['response_sha256'] = digest(receipt['response'])

    path = _rewrite_checkpoint(directory, state, change_journal=mutate)
    restored = build(directory, [])
    try:
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(restored, str(path), allow_identity_mismatch=True)
        assert not restored.llm.calls and not restored.fixture_evaluated
    finally:
        restored.log.close()
