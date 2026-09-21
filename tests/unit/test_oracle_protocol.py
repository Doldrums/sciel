"""Pure protocol checks; no Docker, candidate execution or old task imports."""

import json
import importlib.util
from pathlib import Path
import sys

import pytest

_path = Path(__file__).resolve().parents[2] / "tasks/oe/function_minimization_oracle/assets/oracle_runtime.py"
_spec = importlib.util.spec_from_file_location("oracle_protocol_test_runtime", _path)
_runtime = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _runtime
_spec.loader.exec_module(_runtime)
OracleSession, ProtocolError, decode_frame = _runtime.OracleSession, _runtime.ProtocolError, _runtime.decode_frame


def query(x=0.0, y=0.0):
    return json.dumps({"op": "query", "point": [x, y]}).encode()


def test_exact_budget_is_server_owned_and_duplicates_cost_queries():
    calls = []
    session = OracleSession(lambda x, y: calls.append((x, y)) or 1.)
    for number in range(1, 51):
        assert session.handle(query())["query"] == number
    with pytest.raises(ProtocolError, match="budget"):
        session.handle(query())
    assert len(calls) == session.queries == 50
    with pytest.raises(ProtocolError, match="closed"):
        session.handle(b'{"op":"finish"}')


def test_finish_uses_best_observed_point_without_free_final_query():
    session = OracleSession(lambda x, y: x*x+y*y)
    session.handle(query(.5, .5))
    session.handle(query(.1, .2))
    session.handle(query(.7, .7))
    session.handle(b'{"op":"finish"}')
    assert session.finished and not session.failed
    assert session.queries == 3
    assert session.best_point == (.1, .2)
    assert session.best_value == pytest.approx(.05)


@pytest.mark.parametrize("data", [
    b'{"op":"reset"}', b'{"op":"finish","score":1}',
    b'{"op":"query","point":[0,0],"count":0}',
    b'{"op":"query","point":[true,0]}',
    b'{"op":"query","point":[NaN,0]}',
    b'{"op":"query","point":[1e999,0]}',
    b'{"op":"query","point":[2,0]}',
    b'{"op":"query","point":[0,0,0]}',
    b'{"op":"query","point":"0,0"}',
    b'{"op":"query","op":"finish","point":[0,0]}',
    b'[]', b'null', b'{', b'x'*4097,
])
def test_bad_frames_fail_closed_before_objective(data):
    calls=[]
    session=OracleSession(lambda x,y: calls.append((x,y)) or 0.)
    with pytest.raises(ProtocolError):
        session.handle(data)
    assert session.failed and not calls
    with pytest.raises(ProtocolError, match="closed"):
        session.handle(query())


def test_empty_finish_is_not_a_score():
    session=OracleSession(lambda x,y: 0.)
    with pytest.raises(ProtocolError, match="finish"):
        session.handle(b'{"op":"finish"}')


def test_objective_failure_is_charged_without_leaking_private_message():
    def objective(x,y):
        raise RuntimeError("secret-parameters")
    session=OracleSession(objective)
    with pytest.raises(ProtocolError, match="^trusted objective failed$"):
        session.handle(query())
    assert session.queries == 1 and session.failed
    assert "secret" not in session.failure


def test_no_float_or_shape_coercion_in_wire_decoder():
    assert decode_frame(b'{"op":"query","point":[0,1]}')["point"] == [0,1]


def test_random_landscapes_have_an_analytic_zero_and_nonnegative_values():
    for seed in range(20):
        instance = _runtime.instance_from_seed(seed.to_bytes(32, "big"))
        objective = _runtime.objective_for(instance)
        assert objective(*instance["centre"]) == 0.
        assert all(objective(x, y) >= 0 for x in (-1., -.5, 0., .5, 1.)
                   for y in (-1., -.5, 0., .5, 1.))
    assert _runtime.instance_from_seed(b'a'*32) != _runtime.instance_from_seed(b'b'*32)


def test_private_batch_is_fresh_committed_and_not_exposed(tmp_path):
    first, public = _runtime.create_instances(2, private_dir=tmp_path/'private',
                                             commitment_dir=tmp_path/'public')
    second, another = _runtime.create_instances(2, private_dir=tmp_path/'private',
                                               commitment_dir=tmp_path/'public')
    assert first != second and public['commitment_sha256'] != another['commitment_sha256']
    private = tmp_path/'private'/(public['batch']+'.json')
    assert private.stat().st_mode & 0o777 == 0o600
    assert (tmp_path/'private').stat().st_mode & 0o777 == 0o700
    assert 'seed' not in public and 'instances' not in public
    assert Path(public['commitment_path']).is_file()
    with pytest.raises(ValueError, match='role mismatch'):
        _runtime.load_sealed_instances(private, public['commitment_sha256'], n=2)


def test_sealed_instances_require_correct_commitment_count_and_permissions(tmp_path):
    instances, public = _runtime.create_instances(2, role='sealed_holdout',
        private_dir=tmp_path/'private', commitment_dir=tmp_path/'public')
    private = tmp_path/'private'/(public['batch']+'.json')
    loaded, record = _runtime.load_sealed_instances(private, public['commitment_sha256'], n=2)
    assert loaded == instances and record['role'] == 'sealed_holdout'
    with pytest.raises(ValueError, match='commitment'):
        _runtime.load_sealed_instances(private, '0'*64, n=2)
    with pytest.raises(ValueError, match='count'):
        _runtime.load_sealed_instances(private, public['commitment_sha256'], n=1)
    private.chmod(0o644)
    with pytest.raises(ValueError, match='0600'):
        _runtime.load_sealed_instances(private, public['commitment_sha256'], n=2)


@pytest.mark.parametrize('n',[0, -1, True, 1.0, '5', 101])
def test_invalid_trial_count_creates_no_files(tmp_path, n):
    with pytest.raises(ValueError):
        _runtime.create_instances(n, private_dir=tmp_path/'private', commitment_dir=tmp_path/'public')
    assert list(tmp_path.iterdir()) == []


def test_container_arguments_do_not_grant_candidate_host_access():
    args = _runtime.docker_argv('sha256:'+'a'*64, 'sciel-oracle-test')
    assert '--network=none' in args and '--read-only' in args
    assert '--security-opt=no-new-privileges' in args
    assert '--cap-drop=ALL' in args and '--pids-limit=64' in args
    assert not any(a.startswith(('--mount', '--volume', '--pid=', '--privileged', '--env')) for a in args)
    with pytest.raises(ValueError):
        _runtime.docker_argv('python:latest', 'sciel-oracle-test')


def _evaluator():
    spec=importlib.util.spec_from_file_location('oracle_evaluator_unit', _path.parent.parent/'evaluator.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_confirmation_has_no_positive_combined_score(monkeypatch):
    evaluator=_evaluator()
    monkeypatch.setattr(evaluator,'_evaluate',lambda *args,**kwargs: {
        'ok':True,'score':.4,'cases':{'reliability_score':.5,'combined_score':.4},'artifacts':{}})
    result=evaluator.confirm('unused',2)
    assert result['ok'] is False and result['score']==result['cases']['combined_score']==0.
    assert result['artifacts']['raw_trial_mean']==.4


@pytest.mark.parametrize('n',[0,-1,True,1.0,'2',101])
def test_confirm_rejects_invalid_count_before_evaluation(n):
    with pytest.raises(ValueError,match='trial count'):
        _evaluator().confirm('not-read',n)


def test_oversized_source_fails_before_creating_private_instances(monkeypatch,tmp_path):
    evaluator=_evaluator()
    monkeypatch.setattr(evaluator.runtime,'verify_runtime_image',lambda image: {})
    def unexpected(*args,**kwargs):
        raise AssertionError('private batch should not be created')
    monkeypatch.setattr(evaluator.runtime,'create_instances',unexpected)
    source=tmp_path/'large.py'
    source.write_bytes(b' '*(_runtime.MAX_SOURCE_BYTES+1))
    with pytest.raises(ValueError,match='candidate size'):
        evaluator.smoke(source)
