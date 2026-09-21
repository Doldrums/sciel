"""Typed list syntax reaches source/Engine contracts; all evaluations are inert."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.engine import checkpoint
from sciel.engine.circle_generation import _integration_identity
from sciel.hypothesis import circle_body_list_policy as policy
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis import circle_traced_body as traced
from sciel.hypothesis import circle_optimizer_traced_body as optimizer
from sciel.hypothesis.controlled_changes import validate_controlled_change
from test_circle_generation_engine import build, response, read_checkpoint

BODY = ('indices = []\nfor i in range(4):\n    indices.append(i)\n'
        'value = ordinary_params.copy()\nvalue[0] += 0.001 * len(indices)\nreturn value\n')


@pytest.mark.parametrize('builder', [body, traced, optimizer])
def test_all_typed_families_preserve_body_and_bind_policy(builder):
    parent = builder.source_for()
    made = builder.apply(parent, BODY)
    assert builder.body_of(made['code']) == BODY
    assert body._HEADER + body._indented(BODY) in made['code']
    proof = made['proof']
    if builder is body:
        assert proof['list_policy'] == policy.identity()
    elif builder is optimizer:
        assert proof['dependencies']['list_policy'] == policy.identity()
    else:
        # Historical traced builder bytes stay frozen; their dependency binds
        # the typed builder containing the checked literal policy digest.
        assert proof['dependencies']['typed_body_builder_sha256'] == hashlib.sha256(Path(body.__file__).read_bytes()).hexdigest()
    assert validate_controlled_change(proof, parent_source_sha256=proof['parent_source_sha256'],
        proposed_sha256=proof['child_source_sha256'], source_sha256=proof['child_source_sha256'],
        source_repaired=False, parent_code=parent, proposed_code=made['code']) == proof
    interface = builder.interface_description()['code_contract']['construction']
    assert interface['local_list_policy']['identity'] == policy.identity()
    with pytest.raises(ValueError, match='array attribute'):
        body.proposals.validate_helper(body._HEADER + body._indented(BODY))


@pytest.mark.parametrize('builder', [body, optimizer])
@pytest.mark.parametrize('change', ['remove', 'source', 'version'])
def test_controlled_edit_cannot_drop_or_replace_its_list_policy(builder, change):
    made = builder.apply(builder.source_for(), BODY)
    proof = copy.deepcopy(made['proof'])
    holder = proof if builder is body else proof['dependencies']
    if change == 'remove':
        del holder['list_policy']
    else:
        holder['list_policy']['source_sha256' if change == 'source' else 'version'] = 'changed'
    with pytest.raises(ValueError, match='differs from the exact builder proof'):
        validate_controlled_change(proof, parent_source_sha256=proof['parent_source_sha256'],
            proposed_sha256=proof['child_source_sha256'], source_sha256=proof['child_source_sha256'],
            source_repaired=False)


def test_policy_byte_drift_is_pinned_transitively_and_in_generation_identity(monkeypatch):
    before = _integration_identity()
    name = 'hypothesis/circle_body_list_policy.py'
    assert before[name] == body.PINNED_LIST_POLICY_SHA256
    read = Path.read_bytes
    target = Path(policy.__file__).resolve()
    def changed(path):
        raw = read(path)
        return raw + b'\n# inert drift' if path.resolve() == target else raw
    monkeypatch.setattr(Path, 'read_bytes', changed)
    assert _integration_identity()[name] != before[name]
    for builder in (body, traced, optimizer):
        with pytest.raises(ValueError, match='local-list policy differs'):
            builder.source_for(BODY)
        with pytest.raises(ValueError, match='local-list policy differs'):
            builder.identity()


def test_local_list_errors_keep_exact_submitted_line_without_repair():
    code = '# α\nvalues = best_params\nvalues.append(1)\nreturn None\n'
    decoded = body.decode_response(response(code))
    assert decoded['proposal'] is None
    assert 'body line 3, column 1' in decoded['body_error']
    assert 'values.append' in decoded['body_error']
    assert 'once-declared' in decoded['body_error']
    assert json.loads(response(code))['body'] == code


def test_list_body_crosses_actual_engine_binding_and_checkpoint_with_fake_evaluation(tmp_path):
    directory = tmp_path/'run'
    engine = build(directory, [response(BODY)], scores={BODY: .6})
    result = engine.run()
    assert result['B_evals'] == 1 and len(engine.llm.calls) == 1
    record, = engine.research.knowledge.experiments
    proof, = engine.circle_generation.seen_proofs
    assert record['status'] == 'evaluated' and record['source_repaired'] is False
    assert record['source_sha256'] == proof['child_source_sha256']
    assert proof['target_body'] == BODY and proof['list_policy'] == policy.identity()
    restored = build(directory, [], scores={BODY: .6})
    try:
        checkpoint.load(restored, str(directory/'checkpoint.json'))
        assert restored.research.state() == engine.research.state()
        assert not restored.llm.calls and not restored.fixture_evaluated
    finally:
        restored.log.close()


def test_changed_policy_checkpoint_cannot_be_forced_before_any_call(tmp_path):
    directory = tmp_path/'run'
    engine = build(directory, [response(BODY)], scores={BODY: .6})
    engine.run()
    saved = read_checkpoint(directory)
    del saved['identity']['circle_generation']['identity']['builder']['list_policy']
    tampered = directory/'missing_policy.json'
    tampered.write_text(json.dumps(saved))
    restored = build(directory, [], scores={BODY: .6})
    before = copy.deepcopy(restored.research.state())
    try:
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(restored, str(tampered), allow_identity_mismatch=True)
        assert restored.research.state() == before
        assert not restored.llm.calls and not restored.fixture_evaluated
    finally:
        restored.log.close()


def test_pair_codec_uses_same_local_list_policy_without_rewriting_either_body():
    from sciel.hypothesis import circle_proposal_pair
    entries = [json.loads(response(BODY)), json.loads(response('return None\n'))]
    decoded = circle_proposal_pair.decode_response(json.dumps({'candidates': entries}))
    assert decoded['error'] is None
    assert [row['proposal'].body for row in decoded['entries']] == [BODY, 'return None\n']


def test_optimizer_trace_control_source_is_byte_identical():
    assert hashlib.sha256(optimizer.source_for().encode()).hexdigest() == '93bc6ed5658a2a299c660de06d5dc4bf0e623f1d8a258aa2504b2a8a5ad98cae'
