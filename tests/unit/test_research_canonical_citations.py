"""Canonical evaluator evidence has a typed, run-bound citation path.

All execution leaves are fixed fixtures. No saved/model-authored source runs.
"""
from copy import deepcopy
import json

import pytest

from sciel.research_tools import canonical_evidence, shared_evidence
from sciel.research_tools.controller import parse_action
from sciel.research_tools.experiment_protocol import validate_dispatch
from sciel.research_tools.research_steps import new_state
from test_research_experimental_protocol import protocol
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, current_receipt, decision, finish

SOURCE = 'def run_search():\n    return 19\n'


def designed(action, **changes):
    spec = protocol()
    spec.update(changes)
    return dict(action, experimental_protocol=spec)


def persistent(prompt):
    return json.JSONDecoder().raw_decode(prompt.split('PERSISTENT RESEARCH DECISION JSON:\n', 1)[1])[0]


def fixture(tmp_path, *, status='evaluated', budget=64000, observations=None, final=None):
    tools, backend, events, costs = build(tmp_path,
        [result({'candidate_source': SOURCE, **(observations or {})})],
        experimental_protocol=True, shared_observations=True, share_between_islands=False, max_prompt_chars=budget)
    current = snapshot()
    evaluated = []
    ids = {}
    def evaluate(source, provenance):
        evaluated.append((source, provenance))
        return {'status': status, 'score': .7 if status == 'evaluated' else None,
            'evaluation_id': 'evaluation:fixture-run:wave:1:research:0:0',
            'source_sha256': provenance['candidate_source_sha256'], 'timing_seconds': .025,
            'cases': {'native_metric': 7., 'n_trials': 3}, 'efficacy': 'unknown'}
    def submit(prompt):
        ids['tool'] = current_receipt(prompt)
        return designed(decision(action='submit_candidate', candidate_receipt_id=ids['tool']),
            kind='closure', budget_action='none', prior_receipt_ids=[ids['tool']])
    def close(prompt):
        ids['canonical'] = persistent(prompt)['canonical_evaluations'][-1]['artifact_id']
        if final:
            return final(prompt, ids)
        return designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none',
            prior_receipt_ids=[ids['tool'], ids['canonical']])
    llm = LLM([designed(decision()), submit, close])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO(), candidate_evaluator=evaluate)
    return tools, backend, events, costs, evaluated, ids, episode, llm


@pytest.mark.parametrize('status', ['evaluated', 'rejected', 'eval_failed', 'output_repeat_rejected'])
def test_fresh_canonical_result_is_citable_with_exact_status_and_separate_receipt_type(tmp_path, status):
    tools, backend, events, costs, evaluated, ids, episode, llm = fixture(tmp_path, status=status)
    assert episode['research_state']['status'] == 'closed'
    assert [s['status'] for s in episode['research_state']['steps']] == ['success', status, 'closed']
    assert len(backend.calls) == len(evaluated) == 1 and costs == [.03] * 3
    assert episode['research_state']['known_llm_cost'] == pytest.approx(.09)
    proposal = tools._read_artifact(episode['terminal_decision_proposal_id'], 'proposal')
    assert proposal['evidence_receipt_ids'] == [ids['tool']]
    assert proposal['canonical_evaluation_ids'] == [ids['canonical']]
    assert not any(e['phase'] == 'rejected' for e in events)
    assert shared_evidence.collect(tools.lifecycle, tools.library, 1, tools.task_identity)['canonical_evaluations'] == []
    tools.commit(episode)
    view = shared_evidence.collect(tools.lifecycle, tools.library, 1, tools.task_identity)
    row = view['canonical_evaluations'][0]
    assert row['artifact_id'] == ids['canonical'] and row['status'] == status
    assert row['research_target_id'] == 'p0' and 'candidate_id' not in row
    assert row['cases'] == {'native_metric': 7., 'n_trials': 3}
    assert row['tool_receipt_id'] == ids['tool']
    assert ids['canonical'] not in {r['receipt_id'] for r in view['check_index']}
    saved = json.loads(json.dumps(tools.state()))
    tools.load_state(saved, expected_run_id='fixture-run')
    assert tools.state() == saved
    assert shared_evidence.collect(tools.lifecycle, tools.library, 1, tools.task_identity) == view


def test_committed_foreign_canonical_citation_allowed_but_not_experiment_or_stage_input(tmp_path):
    tools, backend, _, _, _, ids, episode, _ = fixture(tmp_path)
    tools.commit(episode)
    foreign = snapshot(1, 1)
    plan = prepare(tools, foreign)
    close = designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none',
        prior_receipt_ids=[ids['canonical']])
    parsed = parse_action(json.dumps(close), tools.config)
    audit = validate_dispatch(parsed, plan, foreign.research_targets['selected']['source_sha256'], default_limits=tools.config['limits'])
    assert audit['adherence'] == 'not_applicable'
    with pytest.raises(ValueError, match='successful visible invocation'):
        tools._stage_receipt(ids['canonical'], plan, new_state(), [])
    supported = {**close, 'conclusion': 'supported'}
    done = tools.execute(plan, foreign, LLM([supported, close]), io=IO())
    assert [r['status'] for r in done['research_state']['steps']] == ['proposal_invalid', 'closed']
    assert 'requires recorded experiment evidence' in done['research_state']['steps'][0]['error']
    assert len(backend.calls) == 1
    tools.commit(done)
    tools.load_state(deepcopy(tools.state()), expected_run_id='fixture-run')


@pytest.mark.parametrize('change', ['source', 'target', 'run', 'task', 'cell', 'evaluation_run'])
def test_forged_canonical_binding_rejected_even_with_a_new_content_hash(tmp_path, change):
    tools, _, _, _, _, ids, _, _ = fixture(tmp_path)
    raw = tools._read_artifact(ids['canonical'], 'candidate_evaluation')
    if change == 'source': raw['source'] = 'def run_search(): return 400\n'
    if change == 'target': raw['provenance']['target_binding']['candidate_id'] = 'p_wrong'
    if change == 'run': raw['provenance']['run_id'] = 'other-run'
    if change == 'task': raw['provenance']['task_identity'] = {'task': 'other'}
    if change == 'cell': raw['provenance']['cell_id'] = 'f' * 64
    if change == 'evaluation_run': raw['evaluation']['evaluation_id'] = 'evaluation:other-run:0'
    forged = tools._artifact('candidate_evaluation', raw)
    with pytest.raises(ValueError, match='binding|another run'):
        canonical_evidence.authenticate(tools.lifecycle, tools.library, forged)


def test_unknown_self_and_uncommitted_canonical_artifacts_do_not_gain_citation_rights(tmp_path):
    tools, _, _, _, _, ids, episode, _ = fixture(tmp_path)
    foreign = snapshot(1, 1)
    plan = prepare(tools, foreign)  # Author episode has not committed.
    for ident in (ids['canonical'], episode['terminal_decision_proposal_id'], 'f' * 64):
        action = designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none', prior_receipt_ids=[ident])
        with pytest.raises(ValueError, match='authenticated tool observations or canonical'):
            validate_dispatch(parse_action(json.dumps(action), tools.config), plan,
                foreign.research_targets['selected']['source_sha256'], default_limits=tools.config['limits'])


def test_checkpoint_and_shared_catalogue_reject_foreign_run_and_action_membership(tmp_path):
    tools, _, _, _, _, _, episode, _ = fixture(tmp_path)
    tools.commit(episode)
    state = deepcopy(tools.state())
    with pytest.raises(ValueError, match='another run'):
        tools.load_state(state, expected_run_id='other-run')
    changed = deepcopy(state); changed['run_id'] = 'other-run'
    with pytest.raises(ValueError, match='run/task/cell/source binding'):
        tools.load_state(changed, expected_run_id='other-run')
    changed = deepcopy(state); changed['committed_episodes'] = ['f' * 64]
    with pytest.raises(ValueError, match='not a committed research episode'):
        tools.load_state(changed)
    tools.lifecycle.run_id = 'other-run'
    tools.lifecycle._shared_evidence_index = None
    with pytest.raises(ValueError, match='run/task/cell/source binding'):
        shared_evidence.collect(tools.lifecycle, tools.library, 1, tools.task_identity)


def test_changed_snapshot_run_cannot_rebind_controller(tmp_path):
    tools, _, _, _, _, _, _, _ = fixture(tmp_path)
    other = snapshot(1, 2)
    from sciel.methods.models import content_hash
    other.research_targets['run_id'] = 'other-run'
    other.research_targets['sha256'] = content_hash({k: v for k, v in other.research_targets.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='another run'):
        prepare(tools, other)
    assert tools.state()['run_id'] == 'fixture-run'


@pytest.mark.parametrize('budget', [8000, 64000])
def test_recent_shared_canonical_summary_survives_large_measurement_in_real_prompt(tmp_path, budget):
    tools, _, _, _, _, ids, episode, _ = fixture(tmp_path, budget=budget, observations={'large': list(range(4000))})
    tools.commit(episode)
    foreign = snapshot(1, 1); plan = prepare(tools, foreign)
    prompt = tools._prompt(plan, foreign, [], [], 0, research_state=new_state())
    assert len(prompt) <= budget
    view = json.JSONDecoder().raw_decode(prompt.split(shared_evidence.MARKER, 1)[1])[0]
    assert view['canonical_evaluations'][0]['artifact_id'] == ids['canonical']
    assert view['canonical_evaluations'][0]['tool_receipt_id'] == ids['tool']
    assert view['canonical_evaluations'][0]['cases']['native_metric'] == 7.
    assert view['canonical_evaluations_omitted'] == 0
    assert len(view['entries']) + view['omitted_count'] == 1


def test_cold_shared_index_accepts_earlier_recipient_citing_later_created_cell(tmp_path):
    tools, backend, _, _ = build(tmp_path, [result({'candidate_source': SOURCE})],
        experimental_protocol=True, shared_observations=True, share_between_islands=False)
    earlier = snapshot(0)
    empty_close = designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none')
    first = tools.execute(prepare(tools, earlier), earlier, LLM([empty_close]), io=IO())
    tools.commit(first)
    later = snapshot(1, 1)
    ids = {}
    def submit(prompt):
        return designed(decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt)),
            kind='closure', budget_action='none')
    def record(prompt):
        ids['canonical'] = persistent(prompt)['canonical_evaluations'][-1]['artifact_id']
        return designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none',
            prior_receipt_ids=[ids['canonical']])
    measured = tools.execute(prepare(tools, later), later, LLM([designed(decision()), submit, record]),
        io=IO(), candidate_evaluator=lambda code, provenance: {'status':'evaluated','score':.7})
    tools.commit(measured)
    recipient = snapshot(0, 2)
    plan = prepare(tools, recipient)
    assert plan['cell_id'] == first['plan']['cell_id']
    citation = designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none',
        prior_receipt_ids=[ids['canonical']])
    tools.commit(tools.execute(plan, recipient, LLM([citation]), io=IO()))
    tools.lifecycle._shared_evidence_index = None
    cold = shared_evidence.collect(tools.lifecycle, tools.library, 0, tools.task_identity)
    assert [row['artifact_id'] for row in cold['canonical_evaluations']] == [ids['canonical']]
    tools.load_state(deepcopy(tools.state()), expected_run_id='fixture-run')
    assert shared_evidence.collect(tools.lifecycle, tools.library, 0, tools.task_identity) == cold
    assert len(backend.calls) == 1


def test_rehashed_proposal_citation_metadata_must_match_saved_response(tmp_path):
    tools, _, _, _, _, _, episode, _ = fixture(tmp_path)
    tools.commit(episode)
    original = tools._read_artifact(episode['terminal_decision_proposal_id'], 'proposal')
    changed = deepcopy(original)
    changed['canonical_evaluation_ids'] = []  # Omit the actually cited typed result.
    forged = tools._artifact('proposal', changed)
    tools.lifecycle.cells.revise(episode['plan']['cell_id'], phase='research',
        evidence=[{'kind':'research_decision','id':forged}], summary='Tampered metadata fixture')
    with pytest.raises(ValueError, match='citation metadata differs'):
        shared_evidence.collect(tools.lifecycle, tools.library, 1, tools.task_identity)
    with pytest.raises(ValueError, match='citation metadata differs'):
        tools.load_state(deepcopy(tools.state()), expected_run_id='fixture-run')


def test_trimmed_decision_steps_do_not_erase_committed_canonical_history(tmp_path):
    tools, _, _, _, _, ids, episode, _ = fixture(tmp_path)
    tools.commit(episode)
    cell_id = episode['plan']['cell_id']
    tools.lifecycle._state['research_state'][cell_id]['steps'] = []
    tools.lifecycle._shared_evidence_index = None
    view = shared_evidence.collect(tools.lifecycle, tools.library, 1, tools.task_identity)
    assert view['canonical_evaluation_index'][0]['artifact_id'] == ids['canonical']
    tools.load_state(deepcopy(tools.state()), expected_run_id='fixture-run')
    assert shared_evidence.collect(tools.lifecycle, tools.library, 1, tools.task_identity) == view


def test_canonical_projection_order_is_stable_for_interleaved_cells_and_resume(tmp_path):
    tools, _, _, _ = build(tmp_path, [result({'candidate_source': SOURCE})] * 3,
        experimental_protocol=True, shared_observations=True, share_between_islands=False)
    produced = []
    for iteration, island in enumerate([0, 1, 0]):
        current = snapshot(island, iteration)
        def submit(prompt):
            return designed(decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt)),
                kind='closure', budget_action='none')
        def close(prompt):
            ident = persistent(prompt)['canonical_evaluations'][0]['artifact_id']
            produced.append(ident)
            return designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none',
                prior_receipt_ids=[ident])
        create = designed(decision(question=f'Does measured setting {iteration} help?'), question_key=f'setting_{iteration}')
        episode = tools.execute(prepare(tools, current), current, LLM([create, submit, close]), io=IO(),
            candidate_evaluator=lambda source, provenance: {'status':'evaluated','score':.7})
        tools.commit(episode)
        warm = shared_evidence.collect(tools.lifecycle, tools.library, 3, tools.task_identity)
    assert [r['artifact_id'] for r in warm['canonical_evaluations']] == produced
    tools.lifecycle._shared_evidence_index = None
    assert shared_evidence.collect(tools.lifecycle, tools.library, 3, tools.task_identity) == warm
    tools.load_state(deepcopy(tools.state()), expected_run_id='fixture-run')
    assert shared_evidence.collect(tools.lifecycle, tools.library, 3, tools.task_identity) == warm
