"""Offline preparation regressions: no provider, Docker or candidate execution.

The deliberately synthetic TEST_ONLY prices below never become a real pilot
artifact: each recipe lives exclusively in pytest's temporary directory.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from sciel.methods.models import canonical_bytes, content_hash

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location('methods_pilot_recipe', ROOT/'scripts/prepare_methods_pilot.py')
recipe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recipe)

MODEL = 'TEST_ONLY/mock'
IMAGE = 'sha256:' + 'a'*64
PRICING = {'models': [{'id': MODEL, 'pricing': {'prompt': '0.000001', 'completion': '0.000002'}}]}


def profile(task):
    return json.loads((ROOT/f'configs/methods/pilot-{task}.json').read_bytes())


def prepare(tmp_path, task='sort', **kwargs):
    return recipe.prepare(profile(task), deepcopy(PRICING), MODEL, IMAGE, tmp_path,
                          repeats=2, controls=['unreplayed', 'solution_only'], **kwargs)


def json_at(root, relative):
    return json.loads((root/relative).read_bytes())


@pytest.mark.parametrize('task', ['sort', 'circle26'])
def test_arm_equality_actual_bundle_identity_and_heldout_exclusion(tmp_path, task):
    manifest = prepare(tmp_path, task)
    verified = recipe.verify(tmp_path/'manifest.json')
    assert verified['units'] == 10
    assert verified['paid_dispatches'] == 0
    assert verified['heldout_status'] == 'unexecuted'
    assert manifest['prepared_only'] and not manifest['scientific_claims']
    assert manifest['execution_scope'] == 'development_only'

    bundles = json_at(tmp_path, 'toolbox.json')
    for bundle in bundles:
        assert bundle['id'] == content_hash(dict(schema='sciel-method/1', manifest=bundle['manifest'], files=bundle['files']))
        assert bundle['files']
        assert bundle['manifest']['runtime_lock'] == {'image_id': IMAGE}
    expected_names = {'trusted_sort_control'} if task == 'sort' else {
        'fixed_centers_lp', 'single_circle_relocation', 'supplied_state_slsqp',
        'fixed_centers_greedy_control', 'strict_feasibility_certificate'}
    assert {bundle['manifest']['name'] for bundle in bundles} == expected_names

    campaign = json_at(tmp_path, 'campaign.json')
    assert set(campaign['common']['toolbox']) == {bundle['id'] for bundle in bundles}
    development = json_at(tmp_path, 'development.json')['cases']
    heldout = json_at(tmp_path, 'heldout.json')['cases']
    assert {case['id'] for case in development}.isdisjoint(case['id'] for case in heldout)
    for case in [*development, *heldout]:
        assert case['input_sha256'] == content_hash(case['input'])
    assert campaign['common']['splits'] == {
        'development': [case['id'] for case in development],
        'heldout': [case['id'] for case in heldout]}

    shared = None
    output_paths = set()
    for unit in manifest['units']:
        cfg = json_at(tmp_path, unit['config'])
        frozen = json_at(tmp_path, unit['campaign_unit'])
        assert cfg['common_methods'] == bundles
        assert cfg['budget'] == campaign['common']['budget']
        assert len(cfg['labs']) == campaign['common']['lab_count'] == 3
        assert cfg['routing']['transfer_inputs'] == [development[1]['input']]
        assert set(cfg['routing']['control']) <= {bundle['id'] for bundle in bundles}
        assert cfg['model'] == MODEL and cfg['pricing'] == PRICING
        for case in heldout:
            assert case['id'] not in canonical_bytes(cfg).decode()
            for lab in cfg['labs']:
                assert canonical_bytes(case['input']).decode() not in lab['agenda']
        assert 'heldout' not in canonical_bytes(cfg).decode()
        policy = {key: cfg.pop(key) for key in ('mode', 'library_policy', 'transfer_policy')}
        assert policy == {key: frozen[key] for key in policy}
        if shared is None:
            shared = cfg
        else:
            assert cfg == shared  # Only feature policies differ, including repeats.
        output_paths.add(unit['output'])
    assert len(output_paths) == 10
    assert all(path.endswith('/run') for path in output_paths)
    prep = json_at(tmp_path, 'preparation.json')
    assert prep['resources']['dollars'] == prep['resources']['gpu_seconds'] == 0
    assert prep['exclusions']
    assert campaign['common']['upfront_costs'][0]['id'] == prep['id']
    assert campaign['common']['upfront_costs'][0]['resources']['cpu_seconds'] != '0'


@pytest.mark.parametrize('task', ['sort', 'circle26'])
def test_existing_frozen_recipe_is_not_overwritten(tmp_path, task):
    prepare(tmp_path, task)
    before = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='never overwrite'):
        prepare(tmp_path, task)
    after = {str(path.relative_to(tmp_path)): path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    assert before == after


@pytest.mark.parametrize('task', ['sort', 'circle26'])
def test_modified_cli_configuration_is_rejected(tmp_path, task):
    manifest = prepare(tmp_path, task)
    path = tmp_path/manifest['units'][0]['config']
    changed = json.loads(path.read_bytes())
    changed['budget']['dollars'] = '1000'
    path.write_bytes(canonical_bytes(changed))
    with pytest.raises(ValueError, match='artifact drift'):
        recipe.verify(tmp_path/'manifest.json')


@pytest.mark.parametrize('task', ['sort', 'circle26'])
@pytest.mark.parametrize('prices', [
    {'models': []},
    {'models': [{'id': 'different/model', 'pricing': {'prompt': '1', 'completion': '1'}}]},
])
def test_missing_explicit_selected_model_prices_fails_before_output(tmp_path, task, prices):
    output = tmp_path/'must-not-exist'
    with pytest.raises(ValueError):
        recipe.prepare(profile(task), prices, MODEL, IMAGE, output)
    assert not output.exists()


@pytest.mark.parametrize('task', ['sort', 'circle26'])
def test_source_drift_fails_closed_without_modifying_real_sources(tmp_path, monkeypatch, task):
    prepare(tmp_path, task)
    changed = recipe.source_identity()
    changed[next(iter(changed))] = 'f'*64
    monkeypatch.setattr(recipe, 'source_identity', lambda: changed)
    with pytest.raises(ValueError, match='implementation drift'):
        recipe.verify(tmp_path/'manifest.json')
