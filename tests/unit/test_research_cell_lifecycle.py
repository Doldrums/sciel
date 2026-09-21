"""Cell/method transitions with real pinned packages and upstream receipt fixtures.

No Docker or model call is made. Execution authentication and exact replay-output
comparison belong to ResearchTools; these fixtures model its authenticated rows.
"""
from copy import deepcopy
import json
from types import SimpleNamespace
from research_target_fixtures import bind_snapshot, binding
from sciel.engine.research_targets import tool_inputs
import pytest
from sciel.methods.models import canonical_bytes, content_hash
from sciel.research_tools.config import validate_config
from sciel.research_tools.library import ToolLibrary
from sciel.research_tools.result_contract import assess_receipt, replay_evidence
from sciel.research_tools.replay_feedback import build_feedback
from test_research_tool_result_contract import result as tool_result
from sciel.research_tools.lifecycle import CellLifecycle
from research_cell_lifecycle_fixtures import (
    IMAGE, CODE, snapshot, signals,
    lab, package, prepare, episode,
    local_success, released,
)


def test_only_recorded_enabled_stagnation_opens_dynamic_cell(lab):
    life = lab.lifecycle
    assert life.prepare(snapshot(), None) is None
    assert life.prepare(snapshot(), {'population_stall': {'threshold_reached': True}}) is None
    assert life.prepare(snapshot(), signals(active=False)) is None
    assert life.cells.all_cells() == []
    plan = prepare(lab)
    cell = life.cells.get(plan['cell_id'])
    assert plan['stage'] == 'research'
    assert plan['trigger_sources'] == ['population_stall']
    assert cell['trigger']['evidence'] == signals()
    assert cell['question'] == plan['question']
    assert cell['origin_island'] == 0
    assert any(event['phase'] == 'cell_created' for event in lab.events)
    life.commit(episode(lab, plan))
    reused = prepare(lab, parent='p_after_eviction')
    assert reused['cell_id'] == plan['cell_id']
    assert len(life.cells.all_cells()) == 1


def test_shadow_and_disabled_detectors_do_not_silently_create_cells(lab):
    cfg = {**lab.config, 'trigger_sources': ['opportunity_cost'], 'allow_shadow_triggers': False}
    life = CellLifecycle(lab.run, cfg, lambda **_: None)
    probe = signals(active=False, opportunity_cost={'enabled': True, 'current_phase': True,
                                                     'detected': True, 'mode': 'shadow'})
    assert life.prepare(snapshot(), probe) is None
    probe['opportunity_cost']['mode'] = 'active'
    assert life.prepare(snapshot(), probe)['stage'] == 'research'
    # Disabled input evidence cannot bypass the configured active/shadow gate.
    separate = CellLifecycle(lab.run, cfg, lambda **_: None)
    probe['opportunity_cost']['enabled'] = False
    assert separate.prepare(snapshot(), probe) is None


def test_local_execution_pins_exact_version_but_does_not_release_or_adopt(lab):
    bundle, plan, record = local_success(lab)
    method = lab.lifecycle.state()['methods'][bundle['id']]
    assert method['stage'] == 'locally_tested'
    assert method['local_receipt_id'] == record['invocations'][0]['receipt_id']
    assert method['author_cell_id'] == plan['cell_id']
    assert method['replay_attempts'] == method['adoptions'] == method['used_again'] == {}
    cell = lab.lifecycle.cells.get(plan['cell_id'])
    assert bundle['id'] in cell['tool_ids']
    assert cell['revision']['evidence'][0]['tool_id'] == bundle['id']
    assert cell['active'] is False
    assert not any(event['phase'] == 'adopted' for event in lab.events)


def test_known_failure_and_predispatch_refusal_are_retained_without_release(lab):
    for index, dispatch in enumerate(('invoked', 'not_dispatched')):
        bundle = package(lab, name=f'failed-{index}')
        plan = prepare(lab, anchor=index)
        record = episode(lab, plan, bundle, status='failed', dispatch=dispatch)
        assert lab.lifecycle.commit(record) == []
        assert bundle['id'] not in lab.lifecycle.state()['methods']
        cell = lab.lifecycle.cells.get(plan['cell_id'])
        assert cell['revision']['evidence'][0]['id'] == record['invocations'][0]['receipt_id']
        assert cell['active'] is False


def test_independent_replay_reuses_identical_version_and_requires_explicit_match(lab):
    bundle, author, local = local_success(lab)
    first = prepare(lab, 1)
    assert first['stage'] == 'independent_replay'
    assert first['assigned_tool_id'] == bundle['id']
    assert first['reference_receipt_id'] == local['invocations'][0]['receipt_id']
    assert lab.lifecycle.commit(episode(lab, first, bundle, matched=False)) == []
    assert lab.lifecycle.state()['methods'][bundle['id']]['stage'] == 'locally_tested'
    # A failed replay is not retried as if it were never measured on this cell.
    assert prepare(lab, 1)['stage'] == 'research'
    lab.lifecycle.cells.deactivate(first['cell_id'])
    second = prepare(lab, 2)
    assert second['stage'] == 'independent_replay'
    assert lab.lifecycle.commit(episode(lab, second, bundle, matched=True)) == [bundle['id']]
    method = lab.lifecycle.state()['methods'][bundle['id']]
    assert method['stage'] == 'released'
    assert set(method['replay_attempts']) == {first['cell_id'], second['cell_id']}
    assert lab.lifecycle.cells.get(author['cell_id'])['roles']['replayer'] == 2
    assert len(list(lab.library.objects.glob('*.json'))) == 1
    assert lab.library.get(bundle['id'])['files']['tool.py'] == CODE


def test_recipient_trial_adoption_and_later_use_have_distinct_revisions_and_receipts(lab):
    bundle, author, replay_plan, _, _ = released(lab)
    recipient = prepare(lab, 1, active=False)
    assert recipient['stage'] == 'adopt_trial'
    before = lab.lifecycle.cells.get(recipient['cell_id'])['head_revision']
    trial = episode(lab, recipient, bundle, ordinal=2)
    assert lab.lifecycle.commit(trial) == []
    method = lab.lifecycle.state()['methods'][bundle['id']]
    adopted_revision = method['adoptions'][recipient['cell_id']]
    adopted = lab.lifecycle.cells.get(recipient['cell_id'])
    assert adopted_revision != before
    assert adopted['head_revision'] == adopted_revision
    assert adopted['revision']['phase'] == 'adopt'
    assert adopted['revision']['decision']['benefit'] == 'not_established'
    assert method['used_again'] == {}
    assert lab.lifecycle.cells.get(author['cell_id'])['roles']['recipient'] == 1
    subsequent = prepare(lab, 1, active=False, parent='p_new_parent')
    assert subsequent['stage'] == 'use_again'
    later = episode(lab, subsequent, bundle, ordinal=3)
    assert lab.lifecycle.commit(later) == []
    method = lab.lifecycle.state()['methods'][bundle['id']]
    assert method['used_again'][recipient['cell_id']] == later['invocations'][0]['receipt_id']
    assert method['used_again'][recipient['cell_id']] != trial['invocations'][0]['receipt_id']
    assert lab.lifecycle.cells.get(recipient['cell_id'])['head_revision'] != adopted_revision
    events = [event for event in lab.events if event['phase'] == 'post_adoption_use']
    assert events[-1]['adoption_revision_id'] == adopted_revision
    assert len(list(lab.library.objects.glob('*.json'))) == 1
    assert prepare(lab, 1, active=False) is None
    final_state = lab.lifecycle.state()
    restored = CellLifecycle(lab.run, lab.config, lambda **_: None)
    restored.load_state(final_state)
    assert restored.state() == final_state


def test_recipient_contract_failure_does_not_adopt_or_claim_use_again(lab):
    bundle, _, _, _, _ = released(lab)
    recipient = prepare(lab, 1, active=False)
    failed = episode(lab, recipient, bundle, status='failed', ordinal=2)
    lab.lifecycle.commit(failed)
    method = lab.lifecycle.state()['methods'][bundle['id']]
    assert method['adoption_attempts'][recipient['cell_id']] == failed['invocations'][0]['receipt_id']
    assert method['adoptions'] == method['used_again'] == {}
    assert prepare(lab, 1, active=False) is None
    assert not any(event['phase'] == 'adopted' for event in lab.events)


def test_new_cell_origin_can_receive_durable_tool_after_author_island_disappears(lab):
    bundle, origin, _ = local_success(lab)
    # There is deliberately no population, candidate storage, or island object to
    # retain: only the genesis, revision and exact method package persist.
    checkpoint = lab.lifecycle.state()
    resumed = CellLifecycle(lab.run, lab.config, lambda **_: None)
    resumed.load_state(checkpoint)
    lab.lifecycle = resumed
    foreign = prepare(lab, 7, parent='p_after_old_island_retirement')
    assert foreign['stage'] == 'independent_replay'
    assert resumed.cells.get(origin['cell_id'])['question'] == origin['question']
    assert resumed.commit(episode(lab, foreign, bundle, matched=True)) == [bundle['id']]
    assert resumed.cells.get(origin['cell_id'])['roles']['replayer'] == 7


def test_active_work_is_bounded_but_new_questions_remain_durable_and_reactivatable(lab):
    first, second = prepare(lab, 0), prepare(lab, 1)
    assert first and second
    assert prepare(lab, 2) is None
    assert len(lab.lifecycle.cells.all_cells()) == 3
    assert sum(cell['active'] for cell in lab.lifecycle.cells.all_cells()) == 2
    lab.lifecycle.commit(episode(lab, first))
    third = prepare(lab, 2)
    assert third is not None
    assert len(lab.lifecycle.cells.all_cells()) == 3
    assert sum(cell['active'] for cell in lab.lifecycle.cells.all_cells()) == 2


def test_resume_preserves_both_lineages_and_context_without_promoting_orphans(lab):
    bundle, origin, replay, _, _ = released(lab)
    recipient = prepare(lab, 1, active=False)
    trial = episode(lab, recipient, bundle)
    source = trial['invocations'][0]
    card = {'schema': 'sciel-research-check/3', 'replay_comparison': None, 'cell_id': recipient['cell_id'],
            'revision_id': recipient['cell_revision_id'], 'question': recipient['question'],
            'source': {'tool_id': bundle['id'], 'invocation_id': source['invocation_id'],
                       'receipt_id': source['receipt_id'], **source['target_binding']},
            'observations': {'status': 'success', 'output_excerpt': 'observed', 'resources': source['resources']},
            'interpretation': 'No benefit claim.', 'next_check': 'Measure canonical task behavior.',
            'limitations': ['Fixture observation only.']}
    trial['cards'] = [card]
    lab.lifecycle.commit(trial)
    state = lab.lifecycle.state()
    lab.lifecycle.cells.ensure_cell(99, 'An uncommitted question', {'kind': 'stagnation', 'evidence': {'n': 1}})
    resumed = CellLifecycle(lab.run, lab.config, lambda **_: None)
    resumed.load_state(state)
    assert resumed.state() == state
    assert len(resumed.cells.all_cells()) == 2
    assert resumed.context(snapshot(1))[0]['revision_id'] == state['methods'][bundle['id']]['adoptions'][recipient['cell_id']]
    returned = resumed.context(snapshot(1)); returned[0]['question'] = 'mutated'
    assert resumed.context(snapshot(1))[0]['question'] == recipient['question']
    assert resumed.prepare(snapshot(1), signals(active=False))['stage'] == 'use_again'


@pytest.mark.parametrize('change', ['author_island', 'receipt_value', 'release_without_replay', 'adoption_revision'])
def test_resume_rejects_impossible_method_index_bindings_atomically(lab, change):
    bundle, _, _ = local_success(lab)
    good = lab.lifecycle.state()
    bad = deepcopy(good)
    method = bad['methods'][bundle['id']]
    cell_id = method['author_cell_id']
    if change == 'author_island':
        method['author_island_id'] = 999
    elif change == 'receipt_value':
        method['replay_attempts'][cell_id] = {'not': 'an immutable receipt id'}
    elif change == 'release_without_replay':
        method['stage'] = 'released'
    else:
        method['adoptions'][cell_id] = 'f' * 64  # absent and not an adoption revision
    with pytest.raises(ValueError):
        lab.lifecycle.load_state(bad)
    assert lab.lifecycle.state() == good


def test_same_active_cell_cannot_receive_two_parallel_episodes(lab):
    first = prepare(lab)
    assert first is not None
    assert prepare(lab) is None
    assert prepare(lab, anchor=99) is None
    assert len(lab.lifecycle.cells.all_cells()) == 1
    lab.lifecycle.commit(episode(lab, first))
    next_plan = prepare(lab, anchor=99)
    assert next_plan is not None and next_plan['cell_id'] != first['cell_id']


def test_method_read_is_detached_and_foreign_visibility_requires_local_adoption(lab):
    bundle, author, replay, _, _ = released(lab)
    assert lab.lifecycle.adopted_tool_ids(author['cell_id']) == []
    assert lab.lifecycle.adopted_tool_ids(replay['cell_id']) == []
    viewed = lab.lifecycle.method(bundle['id']); viewed['stage'] = 'forged'
    assert lab.lifecycle.method(bundle['id'])['stage'] == 'released'
    plan = prepare(lab, 1, active=False)
    lab.lifecycle.commit(episode(lab, plan, bundle))
    assert lab.lifecycle.adopted_tool_ids(plan['cell_id']) == [bundle['id']]
    assert lab.lifecycle.adopted_tool_ids(author['cell_id']) == []


def test_untrusted_checkpoint_cannot_claim_release_after_a_nonmatching_replay(lab):
    bundle, _, _ = local_success(lab)
    replay = prepare(lab, 1)
    trial = episode(lab, replay, bundle, matched=False)
    lab.lifecycle.commit(trial)
    state = lab.lifecycle.state()
    bad = deepcopy(state)
    method = bad['methods'][bundle['id']]
    method['stage'] = 'released'
    method['release_record'] = {'cell_id': replay['cell_id'],
        'receipt_id': trial['invocations'][0]['receipt_id'],
        'reference_receipt_id': method['local_receipt_id'], 'matched': 'true'}
    with pytest.raises(ValueError, match='bound independent replay evidence'): lab.lifecycle.load_state(bad)
    assert lab.lifecycle.state() == state


def test_resume_reads_saved_receipts_and_rejects_modified_source_and_outputs(lab):
    bundle, _, _, local, replay = released(lab)
    state = lab.lifecycle.state()
    receipt_file = lab.library.path / 'artifacts' / (replay['invocations'][0]['receipt_id'] + '.json')
    raw = receipt_file.read_bytes()
    receipt_file.write_bytes(raw.replace(b'"finding":1', b'"finding":9'))
    with pytest.raises(ValueError, match='artifact'): lab.lifecycle.load_state(state)
    assert lab.lifecycle.state() == state
    receipt_file.write_bytes(raw)
    source_file = lab.library.objects / (bundle['id'] + '.json')
    source_file.write_bytes(source_file.read_bytes().replace(b'payload.get', b'payload.pop'))
    with pytest.raises(ValueError, match='hash mismatch'): lab.lifecycle.load_state(state)
    assert lab.lifecycle.state() == state


def test_bounded_context_selects_recent_cards_before_copying_all_retained_cells(lab, monkeypatch):
    # Leave room for mandatory receipt-bound assessments so this fixture tests
    # the visible-cell selection/copy bound, rather than character eviction.
    lab.config['max_visible_tools'] = lab.lifecycle.config['max_visible_tools'] = 6
    from sciel.engine import research_cells_context
    import sciel.research_tools.lifecycle as lifecycle_module

    measured_ids = []
    for index in range(15):
        plan = prepare(lab, anchor=index)
        bundle = package(lab, name=f'measured-{index}')
        record = episode(lab, plan, bundle, question='What was measured?')
        observed = record['invocations'][0]
        card = {'schema': 'sciel-research-check/3', 'replay_comparison': None, 'cell_id': plan['cell_id'],
            'revision_id': plan['cell_revision_id'], 'question': 'What was measured?',
            'source': {key:observed[key] for key in ('tool_id','invocation_id','receipt_id')},
            'observations': {'status': 'success', 'output_excerpt': '{"finding":1}',
                             'resources':observed['resources']},
            'interpretation': '', 'next_check': '', 'limitations': ['Auxiliary evidence.']}
        card['source'].update(binding(snapshot()))
        record['cards'] = [card]
        lab.lifecycle.commit(record)
        measured_ids.append(plan['cell_id'])
    # Newer unsupported cards must be filtered before the visible-cell cap.
    # These old /3 records remain loadable; they are not measured memory.
    for index in range(2):
        plan = prepare(lab, anchor=100+index)
        card = {'schema': 'sciel-research-check/3', 'replay_comparison': None, 'cell_id': plan['cell_id'],
            'revision_id': plan['cell_revision_id'], 'question': plan['question'],
            'source': {'tool_id': None, 'invocation_id': None, 'receipt_id': None, **binding(snapshot())},
            'observations': {'status': 'skipped', 'output_excerpt': '', 'resources': {}},
            'interpretation': 'No executable evidence.', 'next_check': 'Form a testable question.',
            'limitations': ['No invocation.']}
        lab.lifecycle.commit(episode(lab, plan, cards=[card]))
    all_state = lab.lifecycle.state()
    assert len(all_state['cards']) == len(all_state['cells']['cells']) == 17
    expected_ids = list(reversed(measured_ids))[:lab.config['max_visible_tools']]
    restored = CellLifecycle(lab.run, lab.config, lambda **_: None)
    restored.load_state(all_state)
    contexts = []
    for lifecycle in (lab.lifecycle, restored):
        copied = []
        def bounded_copy(value, *args, **kwargs):
            if type(value) is dict and value.get('schema') in ('sciel-research-check/3','sciel-research-check/4','sciel-research-check/5'):
                copied.append(value['cell_id'])
                assert value['cell_id'] in expected_ids  # No hidden or rejected payload copied.
            return deepcopy(value, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(lifecycle_module, 'deepcopy', bounded_copy)
            patch.setattr(research_cells_context, 'deepcopy', bounded_copy)
            view = lifecycle.context_view(snapshot())
        assert [card['cell_id'] for card in view['cards']] == expected_ids
        assert set(copied) == set(expected_ids)
        assert view['omission_reasons'] == {'unmeasured_interpretation':2, 'cell_limit':len(measured_ids) - len(expected_ids)}
        assert view['omitted_count'] == 2 + len(measured_ids) - len(expected_ids)
        contexts.append(deepcopy(view))
        view['cards'][0]['observations']['resources']['cpu_seconds'] = 999
        assert lifecycle.state() == all_state  # Every returned payload is detached.
    assert contexts[0] == contexts[1]


def test_resume_recomputes_release_comparison_from_actual_saved_outputs(lab):
    bundle, _, _ = local_success(lab)
    replay = prepare(lab, 1)
    # Simulate a corrupt upstream matched flag; persisted bodies remain available
    # to reject this impossible release during checkpoint authentication.
    trial = episode(lab, replay, bundle, matched=True, observed_output={'finding': 99})
    lab.lifecycle.commit(trial)
    fresh = CellLifecycle(lab.run, lab.config, lambda **_: None)
    with pytest.raises(ValueError, match='immutable policy'):
        fresh.load_state(lab.lifecycle.state())
    assert fresh.state()['methods'] == {}


def test_replay_card_binds_specific_question_and_investigated_candidate_not_current_parent(lab):
    bundle, _, _ = local_success(lab)
    replay = prepare(lab, 1, parent='p_current_caller')
    question = 'Does this exact check reproduce the saved observation?'
    trial = episode(lab, replay, bundle, matched=True, question=question)
    source = trial['invocations'][0]
    artifacts = lab.library.path / 'artifacts'
    reference = json.loads((artifacts / (replay['reference_receipt_id'] + '.json')).read_text())['payload']
    repeated = json.loads((artifacts / (source['receipt_id'] + '.json')).read_text())['payload']
    comparison = build_feedback(reference, repeated, tool_id=bundle['id'],
        reference_receipt_id=replay['reference_receipt_id'], replay_receipt_id=source['receipt_id'])
    card = {'schema': 'sciel-research-check/3', 'replay_comparison': comparison, 'cell_id': replay['cell_id'],
        'revision_id': replay['cell_revision_id'], 'question': question,
        'source': {'tool_id': bundle['id'], 'invocation_id': source['invocation_id'],
                   'receipt_id': source['receipt_id'], **source['target_binding']},
        'observations': {'status': 'success', 'output_excerpt': 'observed', 'resources': source['resources']},
        'interpretation': 'Exact observation repeated.', 'next_check': 'Try recipient inputs.',
        'limitations': ['No task benefit established.']}
    trial['cards'] = [card]
    lab.lifecycle.commit(trial)
    saved = lab.lifecycle.state()
    fresh = CellLifecycle(lab.run, lab.config, lambda **_: None)
    fresh.load_state(saved)
    assert fresh.context(snapshot(1))[0]['question'] != replay['question']
    assert fresh.context(snapshot(1))[0]['source']['candidate_id'] == 'p_original'
    for key in ('candidate', 'question'):
        bad = deepcopy(saved)
        target = bad['cards'][replay['cell_id']]
        if key == 'candidate': target['source']['candidate_id'] = 'p_current_caller'
        else: target['question'] = 'An unsupported different experimental question'
        with pytest.raises(ValueError, match='source differs'):
            fresh.load_state(bad)
        assert fresh.state() == saved


def test_refused_post_adoption_followup_does_not_monopolize_future_cell_work(lab):
    bundle, _, _, _, _ = released(lab)
    recipient = prepare(lab, 1, active=False)
    lab.lifecycle.commit(episode(lab, recipient, bundle))
    later = prepare(lab, 1, active=False)
    assert later['stage'] == 'use_again'
    refused = episode(lab, later, bundle, status='failed', dispatch='not_dispatched', ordinal=2)
    lab.lifecycle.commit(refused)
    method = lab.lifecycle.method(bundle['id'])
    assert method['use_attempts'][later['cell_id']] == refused['invocations'][0]['receipt_id']
    assert method['used_again'] == {}
    assert prepare(lab, 1, active=False) is None
    saved = lab.lifecycle.state()
    resumed = CellLifecycle(lab.run, lab.config, lambda **_: None)
    resumed.load_state(saved)
    assert resumed.prepare(snapshot(1), signals(active=False)) is None
    # A fresh research opportunity may deliberately try again. An actual failed
    # invocation is still an actual use, with no success/benefit claim.
    lab.lifecycle = resumed
    new_research = prepare(lab, 1, active=True)
    assert new_research['stage'] == 'research'
    actual = episode(lab, new_research, bundle, status='failed', ordinal=3)
    resumed.commit(actual)
    assert resumed.method(bundle['id'])['used_again'][later['cell_id']] == actual['invocations'][0]['receipt_id']
    resumed.load_state(resumed.state())


def test_use_attempt_checkpoint_cannot_relabel_adoption_trial_as_later_attempt(lab):
    bundle, _, _, _, _ = released(lab)
    recipient = prepare(lab, 1, active=False)
    lab.lifecycle.commit(episode(lab, recipient, bundle))
    saved = lab.lifecycle.state()
    bad = deepcopy(saved)
    method = bad['methods'][bundle['id']]
    method['use_attempts'][recipient['cell_id']] = method['adoption_attempts'][recipient['cell_id']]
    with pytest.raises(ValueError, match='earlier adoption'):
        lab.lifecycle.load_state(bad)
    assert lab.lifecycle.state() == saved


@pytest.mark.parametrize('own_priority', [False, True])
def test_visible_cell_limit_keeps_its_distinct_measurement_and_decision(tmp_path, own_priority):
    from test_research_retained_observations import later_close
    from test_research_cell_context import card as foreign_card
    tools, _, _, _, original, last = later_close(tmp_path, max_visible_tools=1)
    terminal = next(iter(tools.lifecycle._state['cards'].values()))
    own = last['plan']['cell_id'] if own_priority else None
    # A filled distinct-cell limit must still admit another card of its same
    # cell. The exact own cell takes priority even if a foreign card came first.
    cards = [foreign_card(), original] if own_priority else [original, foreign_card(), terminal]
    view = tools.lifecycle.delivery_context(cards, recipient_cell_id=own, max_cells=1)
    assert len(view['cards']) == 2
    assert view['cards'][0] == original
    assert view['cards'][1] == terminal
    assert view['omission_reasons'] == {'cell_limit':1}
