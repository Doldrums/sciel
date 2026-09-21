"""Exact producing evidence for admitted research candidates; authored code is inert."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from research_target_fixtures import bind_snapshot
from sciel.engine.research_candidate import candidate_binding as bind_result
from sciel.engine.validation import source_sha256
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, current_receipt, decision, finish, lineage

SOURCE = 'def run_search():\n    return 19\n'
LATER_SOURCE = 'def run_search():\n    return 23\n'


def candidate_case(tmp_path, *, later=True, status='evaluated'):
    tools, backend, events, costs = build(tmp_path, [result({'candidate_source': SOURCE}),
        result({'candidate_source': LATER_SOURCE})], yield_after_success=True)
    first_snapshot = lineage(snapshot())
    first = tools.execute(prepare(tools, first_snapshot), first_snapshot,
        LLM([decision()]), io=IO())
    producing_receipt = first['receipt_ids'][0]
    tools.commit(first)
    tools.load_state(deepcopy(tools.state()))
    second_snapshot = lineage(snapshot(iteration=1))
    measured = []

    def evaluate(source, provenance):
        record = {'status': status, 'source_sha256': source_sha256(source),
            'evaluation_id': 'evaluation:' + provenance['run_id'] + ':candidate:1',
            'score': .5 if status == 'evaluated' else None,
            'cases': {'validity': 1 if status == 'evaluated' else 0}, 'efficacy': 'unknown'}
        measured.append(SimpleNamespace(child_code=source, origin='research_tool',
            ok=status == 'evaluated', score=record['score'], cases=record['cases'],
            meta={'research_candidate': deepcopy(provenance), 'evaluation_id': record['evaluation_id']}))
        return record

    actions = [decision(action='submit_candidate', candidate_receipt_id=producing_receipt),
        decision(code=decision()['code'] + '# a later unrelated tool\n') if later else finish(conclusion='inconclusive')]
    episode = tools.execute(prepare(tools, second_snapshot), second_snapshot,
        LLM(actions), io=IO(), candidate_evaluator=evaluate)
    return tools, backend, events, episode, measured[-1], producing_receipt


@pytest.mark.parametrize('later', [False, True])
@pytest.mark.parametrize('link_kind', ['mutation', 'remeasurement'])
def test_old_receipt_produces_child_even_without_new_invocation_or_after_later_tool(tmp_path, later, link_kind):
    tools, backend, events, episode, measured, receipt = candidate_case(tmp_path, later=later)
    binding = bind_result(tools, episode, measured)
    assert episode['receipt_ids'] != [receipt]  # The production regression.
    assert bool(episode['receipt_ids']) is later
    tools.commit(episode, child_program_id='p_measured', link_kind=link_kind,
        candidate_binding=binding, child_source=SOURCE)
    link = [e for e in events if e['phase'] == 'child_link'][-1]
    assert link['link_basis'] == 'canonical_candidate_source'
    assert link['link_kind'] == link_kind
    assert ('remeasurement of an existing program' in link['scope']) is (link_kind == 'remeasurement')
    assert link['receipt_ids'] == [receipt]
    assert link['tool_ids'] == [measured.meta['research_candidate']['tool_id']]
    assert link['candidate_source_sha256'] == source_sha256(SOURCE)
    assert link['evaluation_id'] == measured.meta['evaluation_id']
    canonical = tools._read_artifact(link['canonical_evaluation_id'], 'candidate_evaluation')
    assert canonical['source'] == SOURCE
    assert canonical['provenance']['receipt_id'] == receipt
    assert link['candidate_invocation_id'] == canonical['provenance']['invocation_id']
    assert len(backend.calls) == 1 + int(later)
    saved = deepcopy(tools.state())
    tools.load_state(saved)
    tools.commit(episode, child_program_id='p_measured', link_kind=link_kind,
        candidate_binding=binding, child_source=SOURCE)
    assert tools.state() == saved
    assert len([e for e in events if e['phase'] == 'child_link']) == 1


@pytest.mark.parametrize('field', ['child_code', 'score', 'cases', 'evaluation_id', 'missing_provenance', 'empty_provenance'])
def test_changed_result_is_rejected_before_admission_binding(tmp_path, field):
    tools, _, events, episode, measured, _ = candidate_case(tmp_path)
    before = deepcopy(tools.state())
    if field == 'evaluation_id':
        measured.meta[field] += ':wrong'
    elif field == 'missing_provenance':
        del measured.meta['research_candidate']
    elif field == 'empty_provenance':
        measured.meta['research_candidate'] = {}
    else:
        setattr(measured, field, {'child_code': LATER_SOURCE, 'score': .9, 'cases': {'validity': 0}}[field])
    with pytest.raises(ValueError):
        bind_result(tools, episode, measured)
    assert tools.state() == before
    assert not any(e['phase'] == 'child_link' for e in events)


@pytest.mark.parametrize('field', ['child_source', 'receipt_ids', 'tool_ids', 'candidate_source_sha256', 'candidate_invocation_id'])
def test_admitted_source_and_link_fields_are_reauthenticated_before_commit(tmp_path, field):
    tools, _, events, episode, measured, _ = candidate_case(tmp_path)
    binding = bind_result(tools, episode, measured)
    before = deepcopy(tools.state())
    source = SOURCE
    if field == 'child_source':
        source = LATER_SOURCE
    elif field.endswith('_ids'):
        binding[field] = ['a' * 64]
    else:
        binding[field] = 'a' * 64
    with pytest.raises(ValueError):
        tools.commit(episode, child_program_id='p_not_admitted', candidate_binding=binding, child_source=source)
    assert tools.state() == before
    assert not any(e['phase'] == 'child_link' for e in events)


@pytest.mark.parametrize('field', ['run_id', 'cell_id', 'invocation_id', 'receipt_id', 'candidate_source_sha256'])
def test_resealed_canonical_provenance_still_must_match_original_invocation(tmp_path, field):
    tools, _, events, episode, measured, _ = candidate_case(tmp_path)
    submission = episode['candidate_submissions'][0]
    old_id = tools._artifact('candidate_evaluation', submission)
    forged = deepcopy(submission)
    forged['provenance'][field] = 'a' * 64
    new_id = tools._artifact('candidate_evaluation', forged)
    episode['candidate_submissions'][0] = forged
    episode['candidate_submission'] = forged
    measured.meta['research_candidate'] = deepcopy(forged['provenance'])
    for step in episode['research_state']['steps']:
        if step.get('canonical_evaluation_id') == old_id:
            step['canonical_evaluation_id'] = new_id
    with pytest.raises((ValueError, FileNotFoundError)):
        bind_result(tools, episode, measured)
    assert not any(e['phase'] == 'child_link' for e in events)


@pytest.mark.parametrize('status', ['eval_failed', 'rejected', 'output_repeat_rejected'])
def test_nonadmitted_result_never_gets_candidate_or_observation_link(tmp_path, status):
    tools, _, events, episode, measured, _ = candidate_case(tmp_path, later=False, status=status)
    assert bind_result(tools, episode, measured) is None
    tools.commit(episode)
    assert not any(e['phase'] == 'child_link' for e in events)


def test_canonical_episode_cannot_fall_back_to_observation_link_without_binding(tmp_path):
    tools, _, events, episode, _, _ = candidate_case(tmp_path)
    with pytest.raises(ValueError, match='producing evidence binding'):
        tools.commit(episode, child_program_id='p_unbound')
    assert not any(e['phase'] == 'child_link' for e in events)


def test_ordinary_generated_after_observations_keeps_distinct_scope(tmp_path):
    tools, _, events, _ = build(tmp_path, [result({'negative_check': True})], yield_after_success=True)
    current = lineage(snapshot())
    episode = tools.execute(prepare(tools, current), current, LLM([decision()]), io=IO())
    tools.commit(episode, child_program_id='p_ordinary')
    link = [e for e in events if e['phase'] == 'child_link'][-1]
    assert link['link_basis'] == 'observations_before_generation'
    assert link['tool_ids'] == episode['tool_ids'] and link['receipt_ids'] == episode['receipt_ids']
    assert 'canonical_evaluation_id' not in link
    assert 'causal benefit is not established' in link['scope']


def test_independent_branch_link_binds_replayed_producer_and_new_canonical_source(tmp_path):
    tools, _, events, _ = build(tmp_path, [result({'candidate_source': SOURCE}),
        result({'candidate_source': LATER_SOURCE})])
    current = lineage(snapshot(), ancestors=[('root', 'a' * 64)])
    count = 0
    def evaluate(source, provenance):
        nonlocal count
        count += 1
        return {'status':'evaluated','source_sha256':source_sha256(source),
            'evaluation_id':'evaluation:' + provenance['run_id'] + ':trial:' + str(count),
            'score': .5, 'cases': {'validity': 1}, 'efficacy':'unknown'}
    first = tools.execute(prepare(tools, current), current, LLM([decision(),
        lambda prompt: decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt)),
        finish(conclusion='inconclusive')]),
        io=IO(), candidate_evaluator=evaluate)
    tools.commit(first)
    other = snapshot(iteration=1);other.parent_id='sibling';other.parent_code='def run_search(): return 7\n'
    bind_snapshot(other);lineage(other, ancestors=[('root','a'*64)])
    episode = tools.execute(prepare(tools, other), other, LLM([finish(conclusion='inconclusive')]),
        io=IO(), candidate_evaluator=evaluate)
    saved = episode['candidate_submission'];assert saved['provenance']['independent_branch_trial']
    binding = tools.candidate_binding(episode, source=saved['source'], provenance=saved['provenance'],
        evaluation_id=saved['evaluation']['evaluation_id'], score=.5, cases={'validity':1})
    tools.commit(episode, child_program_id='p_branch', candidate_binding=binding, child_source=LATER_SOURCE)
    link = [e for e in events if e['phase']=='child_link'][-1]
    assert link['receipt_ids'] == [saved['provenance']['receipt_id']]
    assert link['receipt_ids'] != [first['candidate_submission']['provenance']['receipt_id']]
    assert link['candidate_source_sha256'] == source_sha256(LATER_SOURCE)
    tools.load_state(deepcopy(tools.state()))


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('tamper', [None, 'research_candidate', 'research_tools', 'score'])
def test_inline_engine_commit_step_authenticates_before_admission(
        tmp_path, monkeypatch, workers, tamper):
    import sys
    from sciel.campaign.budget import CampaignStopped
    from sciel.engine.factory import build_engine
    from test_parallel_engine import config, evaluator, events as run_events, source, task_fixture
    from test_research_tool_controller import IMAGE
    from test_research_tool_result_flow import SavedExecutor
    from test_research_yield_engine_integration import YieldProvider

    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluations, _ = evaluator(monkeypatch)
    candidate = source('run_search', 19)
    backend = SavedExecutor([result({'candidate_source': candidate}), result({'unrelated_check': 0})])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)
    producer = {}

    def submit(prompt):
        producer['receipt'] = current_receipt(prompt)
        return decision(action='submit_candidate', candidate_receipt_id=producer['receipt'])

    provider = YieldProvider([decision(), submit,
        decision(code=decision()['code'] + '# unrelated later check\n'),
        finish(conclusion='inconclusive')])
    cfg = config(task, tmp_path / 'capacity', workers=workers, num_islands=1,
        max_iterations=1, research={'enabled': False},
        research_tools={'enabled': True, 'image_id': IMAGE, 'interval': 1,
            'proactive_every_attempts': 1, 'max_scientific_steps': 3,
            'max_requests': 2, 'share_between_islands': False})
    directory = tmp_path / 'run'
    engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
    execute = engine._execute_step
    admissions = []
    on_result = engine._on_result

    def capture_admission(res, *args, **kwargs):
        admissions.append(res)
        return on_result(res, *args, **kwargs)

    def execute_with_possible_corruption(*args, **kwargs):
        res = execute(*args, **kwargs)
        assert res.origin == 'research_tool'
        if tamper == 'score':
            res.score += .1
        elif tamper:
            del res.meta[tamper]
        return res

    monkeypatch.setattr(engine, '_on_result', capture_admission)
    monkeypatch.setattr(engine, '_execute_step', execute_with_possible_corruption)
    if tamper:
        with pytest.raises(CampaignStopped) as stopped:
            engine.run()
        assert isinstance(stopped.value.__cause__, ValueError)
        assert not admissions
        assert engine.iteration_no == engine.B_evals == 0
        assert not any(p.origin == 'research_tool' for p in engine.db._all.values())
        assert not any(row.get('phase') == 'child_link' for row in run_events(directory))
    else:
        completed = engine.run()
        assert completed['iteration_no'] == completed['B_evals'] == 1
        assert len(admissions) == 1 and len(backend.calls) == 2
        assert evaluations.count(candidate) == 1
        assert completed['cost_total'] == pytest.approx(.12)
        child = next(p for p in engine.db._all.values() if p.origin == 'research_tool')
        link = next(row for row in run_events(directory) if row.get('phase') == 'child_link')
        assert child.code == candidate and link['child_program_id'] == child.id
        assert link['receipt_ids'] == [producer['receipt']]
        assert link['receipt_ids'] != admissions[0].meta['research_tools']['receipt_ids'][-1:]
        assert link['link_basis'] == 'canonical_candidate_source'
