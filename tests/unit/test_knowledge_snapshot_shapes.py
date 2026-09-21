"""Exact historical field-shape compatibility; all records are inert fixtures."""
from copy import deepcopy
import hashlib
from itertools import combinations

import pytest

from sciel.knowledge_base import base, records


# Frozen pre-optimization vocabulary. Do not derive the oracle from the new
# allowed/required constants: additions must update this compatibility witness.
REQUIRED = frozenset({
    'id', 'branch', 'parent_id', 'parent_score', 'child_id', 'source_sha256',
    'proposed_sha256', 'hypothesis', 'status', 'actual_delta', 'score',
    'forecast_verdict', 'injected_ids', 'source_repaired', 'error', 'island',
    'iteration', 'parent_source_sha256', 'attempt', 'operator', 'generation_cost',
    'failure', 'change_excerpt', 'change_excerpt_may_be_truncated',
    'code_fragment_sha256',
})
MEASUREMENT = frozenset({'metric', 'parent_cases', 'evaluation'})
COST = frozenset({'generation_cost_accounting', 'generation_cost_provenance'})
METHOD = frozenset({'method_exposures', 'generation_response_sha256'})
OPTIONAL = frozenset({
    'generation_dispatch', 'controlled_change', 'proposal_group',
    'restart_prediction_binding', 'output_novelty', 'generation_model_identity',
}) | MEASUREMENT | COST | METHOD


def old_shapes():
    """Literal old expansion, computed once by the regression oracle."""
    fields = REQUIRED | MEASUREMENT | {'generation_dispatch'}
    shapes = (fields, fields - MEASUREMENT, fields - {'generation_dispatch'},
              fields - MEASUREMENT - {'generation_dispatch'})
    for group in ({'controlled_change'}, {'proposal_group'},
                  {'restart_prediction_binding'}, {'method_exposures'}, METHOD,
                  {'output_novelty'}, COST, {'generation_model_identity'}):
        shapes += tuple(shape | group for shape in shapes)
    assert len(shapes) == 1024
    accepted = set(shapes)
    assert len(accepted) == 768
    return accepted


def test_all_historical_shapes_and_malformed_partial_groups_are_equivalent():
    assert len(REQUIRED) == 25 and len(OPTIONAL) == 13
    assert type(records._SNAPSHOT_FIELDS) is frozenset
    assert type(records._SNAPSHOT_REQUIRED_FIELDS) is frozenset
    assert records._SNAPSHOT_REQUIRED_FIELDS == REQUIRED
    assert records._SNAPSHOT_FIELDS == REQUIRED | OPTIONAL
    accepted = old_shapes()
    checks = accepted_subsets = 0
    optional = sorted(OPTIONAL)
    for mask in range(1 << len(optional)):
        keys = REQUIRED | {key for bit, key in enumerate(optional) if mask & (1 << bit)}
        variants = [keys, keys | {'unknown_snapshot_field'}]
        variants.extend(keys - {key} for key in REQUIRED)
        for candidate in variants:
            expected = candidate in accepted
            assert records._valid_snapshot_fields(set(candidate)) is expected, sorted(candidate)
            checks += 1
        accepted_subsets += keys in accepted
    assert checks == 221184 and accepted_subsets == 768
    # Method-exposure-only snapshots predate the saved response digest.
    assert records._valid_snapshot_fields(set(REQUIRED | {'method_exposures'}))
    assert not records._valid_snapshot_fields(set(REQUIRED | {'generation_response_sha256'}))
    for group in (MEASUREMENT, COST):
        for size in range(1, len(group)):
            for subset in combinations(group, size):
                assert not records._valid_snapshot_fields(set(REQUIRED | set(subset)))


def record(ident='r000001a1', *, model=False, methods=False):
    source = hashlib.sha256(b'value = 1\n').hexdigest()
    result = dict(id=ident, branch='p1', status='evaluated', parent_score=1.,
        score=.75, actual_delta=-.25, source_sha256=source, proposed_sha256=source,
        parent_source_sha256='a' * 64, change_excerpt='value = 1\n',
        hypothesis=dict(text='A concrete but untested improvement.', reasoning='Test the change.',
                        predicted_low=.1, predicted_high=.2, confidence=.5))
    if model or methods:
        result['generation_dispatch'] = 'returned'
    if methods:
        from sciel.engine.research_cells_context import SCHEMA
        from sciel.research_tools.followups import capture_exposures
        result['method_exposures'] = capture_exposures(
            dict(schema=SCHEMA, cards=[], omitted_count=0, omission_reasons={}),
            prompt_sha256=hashlib.sha256(b'inert saved prompt').hexdigest())
    if model:
        result['generation_model_identity'] = dict(schema='sciel-llm-model-identity/1',
            requested_model='provider/model-latest', resolved_model='provider/model-2026-09-21')
    return result


def retained_state(*, model=False, methods=False):
    kb = base.KnowledgeBase(max_lessons=10, max_experiments=2, max_fragments=2)
    kb.record(record(model=model, methods=methods))
    kb.add_lessons([dict(rule='This measured child regressed.', polarity='negative',
        scope='branch', evidence_ids=['r000001a1'])], allowed_evidence_ids=['r000001a1'])
    # The source survives only in its bound lesson snapshot and lifetime index.
    kb.record(record('r000002a1'))
    kb.record(record('r000003a1'))
    state = kb.state()
    assert {r['id'] for r in state['experiments']} == {'r000002a1', 'r000003a1'}
    return state


@pytest.mark.parametrize('model,methods', [(False, False), (False, True), (True, True)])
def test_cold_restore_keeps_legacy_and_model_evidence_exact_after_source_eviction(model, methods):
    state = retained_state(model=model, methods=methods)
    before = deepcopy(state)
    source = state['lessons'][0]['evidence_snapshots'][0]
    assert ('generation_model_identity' in source) is model
    assert ('method_exposures' in source) is methods
    assert 'generation_response_sha256' not in source
    restored = base.KnowledgeBase(**state['limits'])
    restored.load_state(state)
    assert restored.state() == state == before


@pytest.mark.parametrize('damage', [
    'partial_measurement', 'partial_cost', 'digest_without_exposure', 'missing_required',
    'unknown', 'list', 'invalid_number', 'inconsistent_delta', 'fragment_hash',
    'lifetime_digest', 'model_without_returned_dispatch',
])
def test_invalid_shape_or_semantics_still_reject_atomically(damage):
    state = retained_state()
    snap = state['lessons'][0]['evidence_snapshots'][0]
    if damage == 'partial_measurement':
        snap['metric'] = None
    elif damage == 'partial_cost':
        snap['generation_cost_accounting'] = {'known_usd': 0, 'unpriced_calls': 1}
    elif damage == 'digest_without_exposure':
        snap['generation_response_sha256'] = 'b' * 64
    elif damage == 'missing_required':
        del snap['code_fragment_sha256']
    elif damage == 'unknown':
        snap['unknown_snapshot_field'] = None
    elif damage == 'list':
        state['lessons'][0]['evidence_snapshots'][0] = []
    elif damage == 'invalid_number':
        snap['parent_score'] = True
    elif damage == 'inconsistent_delta':
        snap['score'] = .875
    elif damage == 'fragment_hash':
        snap['code_fragment_sha256'] = 'f' * 64
    elif damage == 'lifetime_digest':
        snap['change_excerpt'] = 'changed without its lifetime digest'
    elif damage == 'model_without_returned_dispatch':
        snap.update(generation_model_identity=record(model=True)['generation_model_identity'],
                    generation_dispatch='unknown')
    # For semantic violations, even a self-consistent incoming digest cannot
    # substitute for the unchanged experiment/source validators.
    if damage not in {'lifetime_digest', 'list'}:
        state['seen_experiments']['r000001a1'] = base._digest(snap)
    target = base.KnowledgeBase(**state['limits'])
    target.record(record('r000099a1'))
    before = target.state()
    with pytest.raises(ValueError):
        target.load_state(state)
    assert target.state() == before
