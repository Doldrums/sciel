"""Pure source/profile joins: compile/inspect only, no candidate invocation."""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.engine import circle_pair_selection as selection
from sciel.hypothesis import circle_proposal_group as groups
from sciel.hypothesis import circle_restart_body as legacy
from sciel.hypothesis import circle_traced_body as traced
from sciel.hypothesis import circle_optimizer_traced_body as optimizer_traced
from sciel.hypothesis.circle_source_family import (
    TRACED_PROFILE, OPTIMIZER_TRACED_PROFILE,
    builder_for_profile, builder_for_source, builder_for_operation)
from sciel.hypothesis.controlled_changes import validate_controlled_change

FIRST = 'x = ordinary_params.copy()\nx[52:] *= 0.99\nreturn x\n'
SECOND = 'x = ordinary_params.copy()\nx[:52] += uniform_noise[:52] * 0.2\nreturn x\n'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def wire(first=FIRST, second=SECOND):
    return json.dumps({'candidates': [dict(body=body, hypothesis={
        'hypothesis': 'Alternative ' + str(i), 'predicted_low': -.02,
        'predicted_high': .03, 'reasoning': 'Unmeasured initializer change.',
        'confidence': .6}) for i, body in enumerate((first, second))]})


def make(*, family=traced, selected=True):
    parent = family.source_for()
    built = family.apply(parent, FIRST)
    gates = [groups.gate('passed' if selected else 'rejected',
        **({} if selected else {'reason': 'fixture_rejection'})) for _ in range(2)]
    group = groups.build_group(request_id='r000001a1', raw_response=wire(),
        parent_source_sha256=sha(parent), gates=gates,
        selected_proof=built['proof'] if selected else None,
        **({'execution_profile': TRACED_PROFILE} if family is traced else {}))
    return parent, built, group


def record(parent, built, group):
    entry = group['entries'][group['selected_index']]
    return dict(id=group['request_id'], parent_source_sha256=sha(parent),
        status='evaluated', generation_dispatch='returned', generation_cost=.003,
        hypothesis=entry['hypothesis'], proposed_sha256=sha(built['code']),
        source_sha256=sha(built['code']), source_repaired=False, score=.6,
        actual_delta=.1, child_id='p2', code_fragment='',
        controlled_change=deepcopy(built['proof']))


def test_exact_family_resolution_and_legacy_profile_absence():
    assert builder_for_profile() is legacy
    assert builder_for_profile(TRACED_PROFILE) is traced
    assert builder_for_profile(OPTIMIZER_TRACED_PROFILE) is optimizer_traced
    for family in (legacy, traced, optimizer_traced):
        source = family.source_for(FIRST)
        assert builder_for_source(source) is family
        assert builder_for_operation(family.OPERATION) is family
        assert family.source_for(family.body_of(source)) == source
        with pytest.raises(ValueError):
            builder_for_source(source + '\n# outside pinned scaffold\n')
    with pytest.raises(ValueError):
        builder_for_source({'source': legacy.source_for()})


@pytest.mark.parametrize('profile', [False, 1, '', 'paired_restart_trace/999'])
def test_unknown_profile_is_not_legacy_or_an_automatic_upgrade(profile):
    with pytest.raises(ValueError):
        builder_for_profile(profile)


def test_traced_controlled_proof_binds_full_actual_sources_and_detaches():
    parent, built, _ = make()
    args = dict(parent_source_sha256=sha(parent), proposed_sha256=sha(built['code']),
        source_sha256=sha(built['code']), source_repaired=False,
        parent_code=parent, proposed_code=built['code'], source_code=built['code'])
    proof = validate_controlled_change(built['proof'], **args)
    assert proof['operation'] == traced.OPERATION
    assert proof['execution_profile'] == TRACED_PROFILE
    assert proof == built['proof'] and proof is not built['proof']
    hash_only = {k:v for k,v in args.items() if not k.endswith('_code')}
    assert validate_controlled_change(proof, **hash_only) == proof
    for key, replacement in (('parent_code', legacy.source_for()),
            ('source_code', legacy.source_for(FIRST)), ('source_repaired', True)):
        with pytest.raises(ValueError):
            validate_controlled_change(proof, **{**args, key: replacement})
    for key, replacement in (('operation', legacy.OPERATION),
            ('execution_profile', None), ('common_scaffold_sha256', '0'*64)):
        with pytest.raises(ValueError):
            validate_controlled_change({**proof, key: replacement}, **args)


def test_legacy_group_and_snapshot_keep_exact_original_field_sets():
    parent, _, group = make(family=legacy)
    snap = selection.snapshot(parent, [], [], ['run_packing'], novelty_enabled=False)
    assert set(snap) == {'parent_body', 'parent_source_sha256', 'code_hashes',
        'consumed_sources', 'required_entrypoints', 'novelty_enabled'}
    assert set(group) == groups._FIELDS
    assert 'execution_profile' not in groups.group_card(group)
    assert [e['candidate_source_sha256'] for e in group['entries']] == [
        sha(legacy.source_for(b)) for b in (FIRST, SECOND)]
    assert groups.validate_group(group, raw_response=wire()) == group


def test_pair_keeps_profile_and_traced_hash_for_both_selected_and_unselected():
    parent, built, group = make()
    original = groups.canonical_json(group)
    assert group['execution_profile'] == TRACED_PROFILE
    assert [e['candidate_source_sha256'] for e in group['entries']] == [
        sha(traced.source_for(b)) for b in (FIRST, SECOND)]
    assert group['selected_source_sha256'] == sha(built['code'])
    assert group['entries'][1]['disposition'] == 'eligible_unselected'
    assert groups.group_card(group)['execution_profile'] == TRACED_PROFILE
    checked = groups.validate_group(group, raw_response=wire(), experiment=record(parent,built,group))
    checked['entries'][1]['body'] = 'return None\n'
    assert groups.canonical_json(group) == original


@pytest.mark.parametrize('damage', ['drop', 'null', 'unselected_source', 'legacy_proof'])
def test_group_rejects_profile_erasure_and_untraced_substitution(damage):
    parent, built, group = make()
    experiment = record(parent,built,group)
    if damage == 'drop': del group['execution_profile']
    elif damage == 'null': group['execution_profile'] = None
    elif damage == 'unselected_source':
        group['entries'][1]['candidate_source_sha256'] = sha(legacy.source_for(SECOND))
    else:
        experiment['controlled_change'] = legacy.apply(legacy.source_for(),FIRST)['proof']
    with pytest.raises(ValueError):
        groups.validate_group(group, raw_response=wire(), experiment=experiment)


def test_rejected_pair_still_records_traced_unmeasured_alternative_hashes():
    _, _, group = make(selected=False)
    assert group['selected_index'] is None and group['selected_source_sha256'] is None
    assert group['selected_proof_sha256'] is None
    assert [e['candidate_source_sha256'] for e in group['entries']] == [
        sha(traced.source_for(b)) for b in (FIRST, SECOND)]
    assert all(e['disposition']=='gate_rejected' for e in group['entries'])
    assert groups.validate_group(group, raw_response=wire()) == group


def test_selection_replay_rebuilds_same_traced_sources_without_legacy_equivalence(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy equivalence is not traced-family evidence')
    monkeypatch.setattr(selection.equivalence, 'verify', forbidden)
    parent = traced.source_for()
    snap = selection.snapshot(parent, [], [], ['run_packing'], novelty_enabled=False)
    assert snap['execution_profile'] == TRACED_PROFILE
    before = deepcopy(snap)
    assessed = selection.assess(wire(), snap)
    assert assessed['selected']['index'] == 0
    assert assessed['selected']['source'] == traced.source_for(FIRST)
    assert [row['source_sha256'] for row in assessed['evidence']] == [
        sha(traced.source_for(b)) for b in (FIRST,SECOND)]
    assert all(row['initializer_equivalence_proof'] is None for row in assessed['evidence'])
    replay = selection.assess(wire(), snap, replay_evidence=assessed['evidence'])
    assert replay['evidence'] == assessed['evidence']
    assert replay['selected']['built'] == assessed['selected']['built']
    assert snap == before
    for profile in (None, 'missing'):
        damaged = deepcopy(snap)
        if profile == 'missing': del damaged['execution_profile']
        else: damaged['execution_profile'] = profile
        with pytest.raises(ValueError):
            selection.assess(wire(), damaged, replay_evidence=assessed['evidence'])


def test_traced_history_consumes_exact_family_only():
    parent = traced.source_for()
    snap = selection.snapshot(parent, [], [sha(traced.source_for(FIRST))],
                              ['run_packing'], novelty_enabled=False)
    assessed = selection.assess(wire(), snap)
    assert assessed['evidence'][0]['match_kind'] == 'previous_source'
    assert assessed['selected']['index'] == 1
    assert assessed['selected']['source'] == traced.source_for(SECOND)
    legacy_history = selection.snapshot(parent, [], [sha(legacy.source_for(FIRST))],
                                       ['run_packing'], novelty_enabled=False)
    assert selection.assess(wire(), legacy_history)['selected']['index'] == 0
