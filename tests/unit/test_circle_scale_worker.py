"""Operation routing and prompt isolation; no optimizer, task eval, or API calls."""
import copy
import json
import sys

import pytest

from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.db import Database, normalized_hash
from sciel.engine.intervention import CircleIntervention, builder_for, parse_target
from sciel.engine.loop import Engine
from sciel.engine.prompt.sources import source_digest
from sciel.engine.types import NoChild, Result, Snapshot
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_interventions as references
from sciel.hypothesis import circle_restart_scales as scales

from test_intervention_worker import setup as reference_setup, response
from test_circle_intervention_engine import Model, config, task_copy, rows, COST


def plan(target='0.03', *, operation=scales.OPERATION):
    return 'HYPOTHESIS_JSON: ' + json.dumps({
        'hypothesis': f'{operation}:{target}', 'predicted_low': -.2,
        'predicted_high': .2, 'reasoning': 'The scale effect is uncertain.', 'confidence': .5})


def setup(texts=(), parent='0.06'):
    fixture = reference_setup([response(text) for text in texts])
    fixture.worker = CircleIntervention(fixture.research, operation=scales.OPERATION)
    fixture.db = Database(1)
    source = scales.source_for(parent)
    fixture.parent = fixture.db.add(source, .8, {'validity': 1., 'sum_radii': 2.108}, 0, 'seed')
    fixture.snapshot = Snapshot(0, 'intervention', 'fake', 0, fixture.parent.id, source, .8,
        code_hashes=fixture.db.code_hashes(0), required_entrypoints=['run_packing'],
        task_prompt='Pack 26 circles.')
    fixture.snapshot.research_context = fixture.research.context(fixture.db, fixture.parent, 0, 1)
    return fixture


def block(prompt, label):
    return json.JSONDecoder().raw_decode(prompt.split(label + ':\n', 1)[1])[0]


@pytest.mark.parametrize('parent,expected_sha', [
    # Captured from the committed /2 formatter, before adding operation routing.
    ('initial', '7ed30e885bcb02fd7124cb2f6f07395a7a395ae7d11161af04339331f929343a'),
    ('incumbent', 'ecd836ac01930467c98a899a2bd894683c5dce7dba1bbff0b8a1f500e9cc8690'),
    ('alternating', '03bb3981f12932c4f19775b2757e3b2c34649d03a6f61c9960ed7307d7f747d5'),
])
def test_default_reference_prompt_and_identity_have_no_drift(parent, expected_sha):
    fixture = reference_setup([], parent=parent)
    explicit = CircleIntervention(fixture.research, operation=references.OPERATION)
    assert fixture.worker.identity == explicit.identity
    prepared = fixture.worker.prepare_prompt(fixture.snapshot)
    assert prepared == explicit.prepare_prompt(fixture.snapshot)
    assert source_digest(prepared['prompt']) == expected_sha
    assert fixture.worker.identity['schema'] == 'sciel-circle-intervention-worker/2'


def test_preparing_a_subset_is_pure_and_does_not_consume_other_choices():
    fixture = setup()
    before = copy.deepcopy((fixture.worker.state(), fixture.research.state(),
                            fixture.snapshot, fixture.log.events))
    restricted = fixture.worker.prepare_prompt(fixture.snapshot, ['0.12', '0.015'])
    assert restricted['allowed'] == ['0.015', '0.12']
    assert [row['policy'] for row in block(restricted['prompt'], 'ALLOWED_INTERVENTIONS_JSON')] == restricted['allowed']
    unrestricted = fixture.worker.prepare_prompt(fixture.snapshot)
    assert unrestricted['allowed'] == ['0.015', '0.03', '0.12']
    assert fixture.worker.prepare_prompt(fixture.snapshot, [])['prompt'] is None
    assert fixture.worker.state() == before[0]
    assert fixture.research.state() == before[1]
    assert vars(fixture.snapshot) == vars(before[2])
    assert fixture.log.events == before[3] and not fixture.model.calls and not fixture.evaluated
    # Detached proofs cannot corrupt the next dispatch.
    restricted['proofs']['0.015']['before']['text'] = 'tampered'
    assert fixture.worker.prepare_prompt(fixture.snapshot) == unrestricted


@pytest.mark.parametrize('targets', [['0.06'], ['0.03', '0.03'], ['unknown'], [.03], '0.03', {'0.03'}])
def test_preparation_cannot_make_a_noop_or_unknown_target_eligible(targets):
    fixture = setup()
    before = fixture.worker.state()
    with pytest.raises(ValueError, match='eligible'):
        fixture.worker.prepare_prompt(fixture.snapshot, targets)
    assert fixture.worker.state() == before


@pytest.mark.parametrize('parent,target', [('0.06', '0.03'), ('0.03', '0.12')])
def test_scale_prompt_names_exact_measured_bounds_and_transition(parent, target):
    fixture = setup([plan(target)], parent=parent)
    fixture.snapshot.parent_card = 'Claim: the baseline uses initial-grid reference and bound 0.06.'
    prepared = fixture.worker.prepare_prompt(fixture.snapshot)
    prompt = prepared['prompt']
    current = block(prompt, 'CURRENT_PARENT_POLICY_JSON')
    assert current['policy'] == parent
    assert current['source_sha256'] == source_digest(scales.source_for(parent))
    assert current['assignment'] == scales.apply(scales.source_for(parent), target)['proof']['before']
    assert 'both centers and radii' in prompt
    assert 'The incumbent reference' in prompt
    assert 'unverified interpretations, not measured benefits' in prompt
    allowed = block(prompt, 'ALLOWED_INTERVENTIONS_JSON')
    chosen = next(row for row in allowed if row['policy'] == target)
    assert chosen['hypothesis'] == f'{scales.OPERATION}:{target}'
    assert chosen['from_policy'] == parent and chosen['to_policy'] == target
    result = fixture.worker.run(fixture.snapshot, fixture.model, fixture.evaluate)
    assert isinstance(result, Result) and result.child_code == scales.source_for(target)
    assert result.cost == .03 and len(fixture.model.calls) == len(fixture.evaluated) == 1
    assert fixture.model.calls[0][0].endswith(prompt)
    record, = fixture.research.knowledge.experiments
    assert record['parent_source_sha256'] == current['source_sha256']
    assert record['source_sha256'] == source_digest(scales.source_for(target))
    assert record['actual_delta'] == .9 - .8


@pytest.mark.parametrize('operation,target', [(scales.OPERATION, '0.015'), (references.OPERATION, 'incumbent')])
def test_public_parser_binds_only_selected_operation(operation, target):
    text = plan(target, operation=operation)
    assert parse_target(text, operation) == target
    other = references.OPERATION if operation == scales.OPERATION else scales.OPERATION
    assert parse_target(text, other) is None
    assert parse_target(text + '\n```python\nraise RuntimeError()\n```', operation) is None


@pytest.mark.parametrize('text', [plan('incumbent', operation=references.OPERATION), plan('0.030'), plan('0.06')])
def test_wrong_operation_alias_or_noop_never_evaluates_or_invents_quality(text):
    fixture = setup([text])
    result = fixture.worker.run(fixture.snapshot, fixture.model, fixture.evaluate)
    assert isinstance(result, NoChild) and result.reason == 'intervention_plan_invalid'
    assert result.cost == .03 and len(fixture.model.calls) == 1 and not fixture.evaluated
    record, = fixture.research.knowledge.experiments
    assert record['score'] is record['actual_delta'] is record['evaluation'] is None


def test_scale_resume_keeps_consumed_source_closure_even_with_new_parent():
    fixture = setup([plan('0.03')])
    fixture.worker.run(fixture.snapshot, fixture.model, fixture.evaluate)
    saved = json.loads(json.dumps(fixture.worker.state()))
    restored = CircleIntervention(fixture.research, operation=scales.OPERATION)
    restored.load_state(saved)
    assert restored.state() == saved
    source = scales.source_for('0.03')
    parent = fixture.db.add(source, .9, {'validity': 1.}, 0, 'gen', parent_id=fixture.parent.id)
    snapshot = copy.deepcopy(fixture.snapshot)
    snapshot.parent_id, snapshot.parent_code, snapshot.parent_score = parent.id, source, .9
    snapshot.code_hashes = {normalized_hash(source)}  # Old seed has been evicted.
    snapshot.research_context = fixture.research.context(fixture.db, parent, 0, 2)
    assert restored.prepare_prompt(snapshot)['allowed'] == ['0.015', '0.12']
    with pytest.raises(ValueError, match='eligible'):
        restored.prepare_prompt(snapshot, ['0.06'])
    foreign = CircleIntervention(fixture.research)
    before = foreign.state()
    with pytest.raises(ValueError, match='checkpoint'):
        foreign.load_state(saved)
    assert foreign.state() == before


@pytest.mark.parametrize('operation', [None, True, 1, [], {}, 'unknown', 'circle_restart_scale/2'])
def test_unknown_operation_is_rejected_by_config_builder_and_parser(operation):
    with pytest.raises(ValueError):
        apply_defaults({'circle_intervention': {'enabled': False, 'operation': operation}})
    with pytest.raises(ValueError):
        builder_for(operation)
    with pytest.raises(ValueError):
        parse_target(plan(), operation)


def test_default_and_disabled_config_shapes_remain_unchanged():
    assert 'circle_intervention' not in apply_defaults({})
    assert apply_defaults({'circle_intervention': {'enabled': False}})['circle_intervention'] == {'enabled': False}
    assert apply_defaults(config())['circle_intervention'] == {'enabled': True}
    assert apply_defaults(config(circle_intervention={'enabled': True, 'operation': scales.OPERATION}))['circle_intervention'] == {
        'enabled': True, 'operation': scales.OPERATION}


def engine_fixture(path, texts, *, task=None, cfg=None):
    task = task or task_copy(path.parent / 'task', initial=scales.baseline_source())
    cfg = cfg or config(circle_intervention={'enabled': True, 'operation': scales.OPERATION})
    engine = Engine(cfg, task, str(path), Model(texts), python=sys.executable)
    evaluated = []
    def evaluate(source):
        policy = scales.policy_of(source)  # Exact bytes only; never import source.
        evaluated.append(policy)
        score = {'0.06': .8, '0.03': .9, '0.015': .95, '0.12': .96}[policy]
        return CascadeOutcome(EvalResult(True, score, {'validity': 1., 'sum_radii': 2.635 * score}),
                              score, 'full', False)
    engine._evaluate = evaluate
    engine.fixture_evaluated = evaluated
    return engine


def test_real_engine_routes_scale_and_binds_selected_builder_in_checkpoint(tmp_path):
    engine = engine_fixture(tmp_path / 'run', [plan()])
    result = engine.run()
    assert result['B_evals'] == 1 and result['reason'] == 'budget'
    assert engine.fixture_evaluated == ['0.06', '0.03']
    assert engine.cost_total == COST and len(engine.llm.calls) == 1
    saved = json.loads((tmp_path / 'run/checkpoint.json').read_text())
    assert saved['circle_intervention']['identity']['operation'] == scales.OPERATION
    assert saved['circle_intervention']['identity']['builder_contract'] == scales.CONTRACT
    bindings = [row for row in rows(engine) if row.get('phase') == 'intervention' and row.get('action') == 'source_binding']
    assert len(bindings) == 1
    assert bindings[0]['proof'] == scales.apply(scales.baseline_source(), '0.03')['proof']


def test_engine_scale_resume_matches_continuous_without_replaying_seed(tmp_path):
    cfg = config(budget=2, max_iterations=2, pop_cap=1, grid_bins=1,
                 circle_intervention={'enabled': True, 'operation': scales.OPERATION})
    whole = engine_fixture(tmp_path / 'whole', [plan(), plan('0.12')], cfg=cfg)
    expected = whole.run()
    split = engine_fixture(tmp_path / 'split', [plan()], cfg=cfg)
    split._stop_at = 1
    assert split.run()['reason'] == 'paused'
    resumed = engine_fixture(tmp_path / 'split', [plan('0.12')], cfg=cfg)
    checkpoint.load(resumed, str(tmp_path / 'split/checkpoint.json'))
    assert resumed.run() == expected
    assert resumed.fixture_evaluated == ['0.12']
    assert resumed.llm.calls == whole.llm.calls[1:]
    for name in ('intervention', 'research', 'db', 'rng', 'selection', 'schedule'):
        assert getattr(resumed, name).state() == getattr(whole, name).state()


def test_force_cannot_load_scale_checkpoint_into_reference_even_at_shared_source(tmp_path):
    task = task_copy(tmp_path / 'task', initial=scales.baseline_source())
    scale = engine_fixture(tmp_path / 'run', [plan()], task=task)
    scale.run()
    # The 0.06 scale source equals reference/incumbent, so task identity is equal.
    reference = Engine(config(), task, str(tmp_path / 'other'), Model([]), python=sys.executable)
    before = reference.intervention.state()
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match='circle intervention'):
            checkpoint.load(reference, str(tmp_path / 'run/checkpoint.json'), allow_identity_mismatch=True)
        assert reference.intervention.state() == before and not reference.llm.calls
    finally:
        reference.log.close()


def test_scale_seed_preflight_rejects_reference_initial_before_run_directory(tmp_path):
    task = task_copy(tmp_path / 'task', initial=references.baseline_source())
    with pytest.raises(ValueError, match='scale family'):
        engine_fixture(tmp_path / 'run', [], task=task)
    assert not (tmp_path / 'run').exists()
