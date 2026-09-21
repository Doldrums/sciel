"""Authored receipt values, inert executors: no generated source executes."""
from copy import deepcopy
from retained_observation_fixtures import retained_fixture
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.engine.research_cells_context import SCHEMA, compact_context
from sciel.research_tools.decision_projection import authenticate_projection, proposal_decision
from sciel.research_tools.observation_cards import format_context
from sciel.research_tools.retained_observations import retained_bundle
from sciel.research_tools.result_projection import PROJECTION_SCHEMAS, verify_projection, projection_fields


SUMMARY_FIELDS = ('prefix_query_counts', 'mean_window_gains', 'mean_prefix_scores')


def encoded(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))


def sha(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def frozen_case():
    """Importable production retained_bundle input; no live filesystem dependency."""
    fixture = retained_fixture()
    for row in fixture['records'] + [fixture['proposal']]:
        assert sha(row['payload']) == row['payload_sha256']
        assert row['artifact_canonical_sha256'] == row['artifact_id']
    assert sha(fixture['terminal_card']) == fixture['terminal_card_sha256']
    records = [(row['artifact_id'], row['payload']) for row in fixture['records']]
    lookup = dict(records)
    revisions = {card['source']['receipt_id']: card['revision_id']
        for card in fixture['actual_g3']['context']['cards'] if card['source']['receipt_id']}

    class SavedIndex:
        records = lookup
        receipt_revisions = revisions

        def receipt(self, ident, tool_id):
            record = lookup[ident]
            assert record['tool_id'] == tool_id
            return deepcopy(record)

    lifecycle = SimpleNamespace(config={'max_context_chars': fixture['budget']}, _state={'methods': {}})
    evidence = {'records': records, 'index': SavedIndex(), 'proposal': fixture['proposal']['payload']}
    return fixture, lifecycle, deepcopy(fixture['terminal_card']), evidence


def result_wrapper(record):
    receipt = record['receipt']
    return {'output': receipt['output'], 'result': record['result'],
        'replay_policy': record['replay_policy'], 'reason': receipt.get('reason'),
        'diagnostic': receipt.get('diagnostic')}


def delivered_values(card, original):
    value = json.loads(card['observations']['output_excerpt'])
    if value.get('schema') in PROJECTION_SCHEMAS:
        verify_projection(value, original)
        return {row['path']: row['value'] for row in projection_fields(value)}
    assert value == original
    return {'/output/observations/' + key: item
            for key, item in value['output']['observations'].items()}


def assert_bounds(cards, omissions, budget):
    envelope = {'schema': SCHEMA, 'cards': cards, 'omitted_count': sum(omissions.values()),
                'omission_reasons': omissions}
    assert len(encoded(envelope)) <= budget
    assert len(format_context(cards)) <= budget
    assert compact_context(envelope, max_chars=budget) == envelope


def test_actual_three_receipts_and_terminal_keep_substantive_measurements():
    fixture, lifecycle, terminal, evidence = frozen_case()
    before = deepcopy(fixture)
    assert len(encoded(result_wrapper(evidence['records'][-1][1]))) > 10000
    cards, omissions = retained_bundle(lifecycle, None, terminal, evidence, budget=10000)
    assert_bounds(cards, omissions, 10000)
    lookup = dict(evidence['records'])
    measured = [card for card in cards if card['source']['receipt_id']]
    assert len(measured) == 3 and len({card['source']['receipt_id'] for card in measured}) == 3
    for card in measured:
        ident = card['source']['receipt_id']
        record = lookup[ident]
        assert card['source'] == {'tool_id': record['tool_id'],
            'invocation_id': record['receipt']['invocation_id'], 'receipt_id': ident,
            **record['target_binding']}
        assert card['observations']['resources'] == record['receipt']['resources']
        assert card['observations']['status'] == record['result']['status']
        assert card['question'] == record['question']
        assert card['revision_id'] == evidence['index'].receipt_revisions[ident]
        if record['result']['status'] == 'host_failed':
            assert 'ValueError: method output exceeds max_output_bytes' in card['observations']['output_excerpt']
    latest_id, latest = evidence['records'][-1]
    card = next(card for card in measured if card['source']['receipt_id'] == latest_id)
    values = delivered_values(card, result_wrapper(latest))
    for key in SUMMARY_FIELDS:
        assert values.get('/output/observations/' + key) == latest['receipt']['output']['observations'][key]
    closed = next(card for card in cards if card['source']['receipt_id'] is None)
    authenticate_projection(closed['decision'], evidence['proposal'], fixture['proposal']['artifact_id'])
    assert closed['source'] == terminal['source'] and closed['observations']['resources'] == {}
    for field in ('interpretation', 'next_decision'):
        assert closed['decision']['omitted_chars'][field] < closed['decision']['original_chars'][field]
    assert closed['decision']['next_decision'] == proposal_decision(evidence['proposal'])['next_decision']
    # Receipt presentation never turns a subscription-unpriced return into free USD.
    response = fixture['proposal']['payload']['response']
    assert response['cost'] is None
    assert response['cost_provenance']['source'] == 'subscription_unpriced'
    assert fixture == before


def test_historical_actual_context_explains_failure_without_rewriting_receipts():
    fixture, _, _, evidence = frozen_case()
    historical = fixture['actual_g3']['context']
    before_size = len(format_context(historical['cards']))
    assert before_size < 10000
    ident, record = evidence['records'][-1]
    old = next(card for card in historical['cards'] if card['source']['receipt_id'] == ident)
    values = delivered_values(old, result_wrapper(record))
    assert not any('/output/observations/' + key in values for key in SUMMARY_FIELDS)
    # All three exact vectors fit together without evicting any of the four cards.
    cards = deepcopy(historical['cards'])
    changed = next(card for card in cards if card['source']['receipt_id'] == ident)
    projection = json.loads(changed['observations']['output_excerpt'])
    for key in SUMMARY_FIELDS:
        projection['fields'].append({'path': '/output/observations/' + key,
            'value': record['receipt']['output']['observations'][key]})
    for key in ('omitted_fields', 'omitted_observation_fields', 'omitted_paths_omitted'):
        projection[key] -= len(SUMMARY_FIELDS)
    verify_projection(projection, result_wrapper(record))
    changed['observations']['output_excerpt'] = encoded(projection)
    assert_bounds(cards, historical['omission_reasons'], 10000)
    assert before_size < len(format_context(cards)) <= 10000


def committed_case(tmp_path, *, second_cell=False):
    """Cold lifecycle around authored observations with inert fake leaves."""
    from sciel.methods.models import content_hash
    from test_research_tool_controller import IO, LLM, prepare, snapshot
    from test_research_tool_result_flow import controller, result
    from test_sequential_research_episode import decision, finish

    fixture, _, _, evidence = frozen_case()
    outputs = [deepcopy(record['receipt']['output']) for _, record in evidence['records']]
    if second_cell:
        outputs.append(result({'control_values': [7.25, 8.5], 'seed': 41}))
    tools, backend, _, costs = controller(tmp_path, outputs,
        max_context_chars=10000, max_scientific_steps=4, yield_after_success=True,
        shared_observations=True, share_between_islands=False, max_visible_tools=2)
    saved_execute = backend.execute

    def inert_execute(*args, **kwargs):
        receipt = saved_execute(*args, **kwargs)
        if receipt['output'] is None:
            # Recreate the saved host output-cap failure, not a malformed success.
            original = evidence['records'][0][1]['receipt']
            for key in ('status', 'reason', 'stderr'):
                receipt[key] = original[key]
            receipt.pop('receipt_sha256')
            receipt['receipt_sha256'] = content_hash(receipt)
            backend.receipts[-1] = deepcopy(receipt)
        return receipt

    backend.execute = inert_execute
    episodes = []
    for iteration, actions in enumerate(([decision(), decision()], [decision()])):
        current = snapshot(iteration=iteration)
        episode = tools.execute(prepare(tools, current), current, LLM(actions), io=IO())
        tools.commit(episode)
        episodes.append(episode)
    raw = proposal_decision(evidence['proposal'])
    current = snapshot(iteration=2)
    episode = tools.execute(prepare(tools, current), current, LLM([finish(
        question=raw['question'], reason=raw['interpretation'],
        next_decision=raw['next_decision'], conclusion=raw['conclusion'])]), io=IO())
    tools.commit(episode)
    episodes.append(episode)
    if second_cell:
        current = snapshot(1, 3)
        other = tools.execute(prepare(tools, current), current, LLM([decision()]), io=IO())
        tools.commit(other)
        episodes.append(other)
    return fixture, tools, backend, costs, episodes


@pytest.mark.parametrize('second_cell', [False, True])
def test_cold_lifecycle_preserves_summary_decision_and_bounded_two_cell_delivery(tmp_path, second_cell):
    from test_research_tool_controller import snapshot

    fixture, tools, backend, costs, episodes = committed_case(tmp_path, second_cell=second_cell)
    saved = json.loads(json.dumps(tools.state()))
    latest_id = episodes[1]['receipt_ids'][-1]
    source = json.loads((tools.path / (latest_id + '.json')).read_text())['payload']
    terminal_proposal_id = episodes[2]['terminal_decision_proposal_id']
    proposal = tools._read_artifact(terminal_proposal_id, 'proposal')
    before = [path.read_bytes() for path in sorted(tools.path.glob('*.json'))]
    shown = None
    for cold in (False, True):
        if cold:
            tools.load_state(deepcopy(saved))
        view = tools.context_view(snapshot(0, 4))
        assert_bounds(view['cards'], view['omission_reasons'], 10000)
        if shown is not None:
            assert view == shown
        shown = deepcopy(view)
        card = next(c for c in view['cards'] if c['source']['receipt_id'] == latest_id)
        values = delivered_values(card, result_wrapper(source))
        for key in SUMMARY_FIELDS:
            assert values.get('/output/observations/' + key) == source['receipt']['output']['observations'][key]
        terminal = next(c for c in view['cards'] if c['source']['receipt_id'] is None)
        authenticate_projection(terminal['decision'], proposal, terminal_proposal_id)
        for key in ('interpretation', 'next_decision'):
            assert terminal['decision']['omitted_chars'][key] < terminal['decision']['original_chars'][key]
        receipts = [c['source']['receipt_id'] for c in view['cards'] if c['source']['receipt_id']]
        assert len(receipts) == len(set(receipts))
        if second_cell:
            other_id = episodes[-1]['receipt_ids'][0]
            other_cards = [c for c in view['cards'] if c['source']['receipt_id'] == other_id]
            if other_cards:
                assert other_cards[0]['cell_id'] == episodes[-1]['plan']['cell_id']
                assert other_cards[0]['source']['candidate_id'] == 'p1'
                other_source = tools._read_artifact(other_id, 'invocation')
                other_values = delivered_values(other_cards[0], result_wrapper(other_source))
                assert other_values['/output/observations/control_values'] == [7.25, 8.5]
            else:
                # A second complete bundle is not promised at ten thousand chars.
                assert view['omission_reasons']['character_limit'] >= 1
        assert tools.state() == saved
    assert len(backend.calls) == (4 if second_cell else 3)
    assert costs == [.03] * (5 if second_cell else 4)
    assert [path.read_bytes() for path in sorted(tools.path.glob('*.json'))] == before


def test_growing_into_paired_projection_cannot_drop_existing_scalar_fact():
    from sciel.research_tools.retained_observations import _preserves_measurements
    from sciel.research_tools.result_projection import _paired_context
    from test_paired_result_projection import witness

    original = witness()
    paired = json.loads(_paired_context(original, 3900))
    verify_projection(paired, original)
    assert paired['comparison_projection']['shown_rows'] == 3
    # Historical /1 and /2 are still legitimate receipt projections.
    # Same source, different legitimate projections. More paired rows alone
    # do not justify removing a previously visible exact summary.
    old = json.loads(_paired_context(original, 1200))
    verify_projection(old, original)
    missing = [row for row in old['fields'] if row not in paired['fields']]
    assert missing
    assert not _preserves_measurements(old, paired)


@pytest.mark.parametrize('replacement,expected', [
    ({'value': 1, 'valid': True}, True),
    ({'value': True, 'valid': True}, False),
    ({'value': 1}, False),
])
def test_growth_compares_whole_rows_with_json_types_and_escaped_paths(replacement, expected):
    from sciel.research_tools.retained_observations import _preserves_measurements

    original_row = {'value': 1, 'valid': True}
    prior = {'schema': 'sciel-tool-result-projection/3', 'fields': [],
        'observation_rows': {'arrays': [{'path': '/output/observations/a~1b~0c',
            'rows': [{'index': 0, 'value': original_row}]}]}}
    complete = {'output': {'observations': {'a/b~c': [replacement]}}}
    assert _preserves_measurements(prior, complete) is expected
    prior_array = {'schema': 'sciel-tool-result-projection/1', 'fields': [
        {'path': '/output/observations/a~1b~0c', 'value': [original_row]}]}
    assert _preserves_measurements(prior_array, complete) is expected
    # Individual row delivery cannot masquerade as the original whole array.
    assert not _preserves_measurements(prior_array, prior)


@pytest.mark.parametrize('malformed', [[1], {'observations': [1]}])
def test_compact_failed_output_keeps_cause_and_resources(malformed):
    from sciel.research_tools.retained_observations import _measurement

    _, lifecycle, _, evidence = frozen_case()
    ident, record = evidence['records'][0]
    record['receipt']['output'] = deepcopy(malformed)
    record['result'] = {'status': 'invalid_output', 'error': 'Output observations must be an object.'}
    card = _measurement(lifecycle, None, evidence['index'], ident, record, 3000, compact=True)
    assert card['observations']['status'] == 'invalid_output'
    assert 'Output observations must be an object.' in card['observations']['output_excerpt']
    assert card['observations']['resources'] == record['receipt']['resources']
    assert card['source']['receipt_id'] == ident
    assert len(format_context([card])) <= 3000


@pytest.mark.parametrize('research_enabled', [False, True])
def test_two_closed_cells_reach_actual_generation_with_one_shared_budget(
        tmp_path, monkeypatch, research_enabled):
    import sys
    from research_target_fixtures import bind_snapshot
    from sciel.engine.factory import build_engine
    from sciel.methods.models import content_hash
    from test_parallel_engine import ResearchProvider, config, evaluator, task_fixture
    from test_research_terminal_delivery_engine import delivered_cards
    from test_research_tool_controller import IMAGE, IO, LLM, prepare, snapshot
    from test_research_tool_result_flow import SavedExecutor, result
    from test_sequential_research_episode import decision, finish

    fixture, _, _, evidence = frozen_case()
    original = evidence['records'][-1][1]['receipt']['output']
    backend = SavedExecutor([deepcopy(original), result({'control_values': [7.25, 8.5], 'seed': 41})])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluator(monkeypatch)
    provider = ResearchProvider(concurrent=1)
    cfg = config(task, tmp_path / 'capacity', workers=1, num_islands=2, max_iterations=1,
        research={'enabled': research_enabled, 'reflection_interval': 0, 'max_aux_calls': None},
        research_tools={'enabled': True, 'image_id': IMAGE, 'interval': 1,
            'max_scientific_steps': 3,
            'yield_after_success': True, 'max_requests': 2, 'max_context_chars': 10000,
            'shared_observations': True, 'share_between_islands': False, 'max_visible_tools': 2})
    engine = build_engine(cfg, str(tasks), str(tmp_path / 'run'), python=sys.executable, llm=provider)
    tools = engine.research_tools
    episodes, before = [], {}
    original_start = engine.log.run_start

    def prepare_closed_cells(*args, **kwargs):
        original_start(*args, **kwargs)
        raw = proposal_decision(evidence['proposal'])
        def current_snapshot(island, iteration):
            current = snapshot(island, iteration)
            current.parent_id = 'saved-original-' + str(island)
            bind_snapshot(current, task_hash=engine.task.task_hash)
            current.research_targets['run_id'] = engine.log.run_id
            current.research_targets['sha256'] = content_hash({key:value
                for key,value in current.research_targets.items() if key != 'sha256'})
            return current
        for island in (0, 1):
            current = current_snapshot(island, island * 2)
            measured = tools.execute(prepare(tools, current), current, LLM([decision()]), io=IO())
            tools.commit(measured)
            current = current_snapshot(island, island * 2 + 1)
            closed = tools.execute(prepare(tools, current), current, LLM([finish(
                question=raw['question'], reason=raw['interpretation'] if island == 0 else 'Control result is available.',
                next_decision=raw['next_decision'], conclusion='inconclusive')]), io=IO())
            tools.commit(closed)
            episodes.extend((measured, closed))
        assert len(tools.lifecycle._state['cards']) == 2
        assert all(card['source']['receipt_id'] is None for card in tools.lifecycle._state['cards'].values())
        snapshot_state = json.loads(json.dumps(tools.state()))
        tools.load_state(snapshot_state)
        before.update((path.name, path.read_bytes()) for path in tools.path.glob('*.json'))
        monkeypatch.setattr(tools, 'prepare', lambda *args, **kwargs: None)

    monkeypatch.setattr(engine.log, 'run_start', prepare_closed_cells)
    done = engine.run()
    prompts = [prompt for prompt, role in zip(provider.calls, provider.roles) if role == 'G']
    assert done['iteration_no'] == 1 and len(prompts) == 1 and len(backend.calls) == 2
    cards = delivered_cards(prompts[0], research_enabled)
    assert len(format_context(cards)) <= 10000
    assert len(encoded(compact_context(cards))) <= 10000
    latest_id = episodes[0]['receipt_ids'][0]
    measured = next(c for c in cards if c['source']['receipt_id'] == latest_id)
    record = tools._read_artifact(latest_id, 'invocation')
    values = delivered_values(measured, result_wrapper(record))
    for key in SUMMARY_FIELDS:
        assert values.get('/output/observations/' + key) == original['observations'][key]
    closed = next(c for c in cards if c['cell_id'] == measured['cell_id'] and c['source']['receipt_id'] is None)
    proposal_id = episodes[1]['terminal_decision_proposal_id']
    authenticate_projection(closed['decision'], tools._read_artifact(proposal_id, 'proposal'), proposal_id)
    # The primary measured bundle must include a readable terminal decision.
    for key in ('interpretation', 'next_decision'):
        assert closed['decision']['omitted_chars'][key] < closed['decision']['original_chars'][key]
    assert closed['decision']['conclusion'] == 'inconclusive'
    assert closed['decision']['original_chars']['interpretation'] == len(raw_text := proposal_decision(evidence['proposal'])['interpretation'])
    assert closed['decision']['original_text_sha256']['interpretation'] == hashlib.sha256(raw_text.encode()).hexdigest()
    assert len({(c['cell_id'], c['source']['receipt_id']) for c in cards}) == len(cards)
    assert all((tools.path / name).read_bytes() == raw for name, raw in before.items())
    if research_enabled:
        followups = engine.research_tools.state()['followups']['records']
        assert len(followups) == 1
        exposed = next(card for card in followups[0]['exposures']['cards']
                       if card['source']['receipt_id'] == latest_id)
        assert exposed['card_sha256'] == content_hash(measured)
        assert exposed['source'] == measured['source']
        assert followups[0]['causal_benefit'] == followups[0]['model_use'] == 'unknown'
