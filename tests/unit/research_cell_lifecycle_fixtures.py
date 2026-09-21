"""Tool episodes and saved receipts shared by research cell lifecycle tests."""
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


IMAGE = 'sha256:' + 'a' * 64


CODE = 'def run(payload):\n    return {"finding": payload.get("probe", 1)}\n'


def snapshot(island=0, iteration=12, parent='p_original'):
    return bind_snapshot(SimpleNamespace(island=island, iteration_no=iteration, parent_id=parent,
        parent_code='def solve(): return 1\n',parent_score=.5,task_prompt='Fixture'))


def signals(*, active=True, anchor=4, **extra):
    return {'schema': 'sciel-stagnation-signals/1',
        'population_stall': {'threshold_reached': active, 'last_record_B_evals': anchor}, **extra}


@pytest.fixture
def lab(tmp_path):
    cfg = validate_config({'enabled': True, 'image_id': IMAGE, 'interval': 1,
                           'max_active_cells': 2})
    events = []
    lifecycle = CellLifecycle(tmp_path, cfg, lambda **event: events.append(event))
    library = ToolLibrary(tmp_path, {'image_id': IMAGE}, cfg['limits'], executor_factory=lambda _: None)
    return SimpleNamespace(lifecycle=lifecycle, library=library, events=events, config=cfg, run=tmp_path)


def package(lab, *, author=0, name='diagnostic', parents=()):
    return lab.library.register(name, 'A generic executable research check.', CODE, author, parents=parents)


def prepare(lab, island=0, *, active=True, anchor=4, parent='p_original'):
    plan = lab.lifecycle.prepare(snapshot(island, parent=parent), signals(active=active, anchor=anchor))
    return dict(plan, island_id=island, parent_program_id=parent) if plan is not None else None


def episode(lab, plan, bundle=None, *, status='ok', dispatch='invoked', matched=None, ordinal=1, cards=(), question=None, observed_output=None):
    rows = []
    if bundle:
        action_id = content_hash({'cell': plan['cell_id'], 'stage': plan['stage'], 'ordinal': ordinal})
        invocation_id = content_hash({'action_id': action_id, 'tool_id': bundle['id']})
        inputs = tool_inputs(snapshot(), {'task_hash':'fixture'}, {'probe':1})
        bound = binding(snapshot())
        output = tool_result({'finding': 1} if observed_output is None else observed_output) if status == 'ok' else None
        receipt = {'schema': 'authenticated-upstream-fixture/1', 'method_id': bundle['id'],
                   'invocation_id': invocation_id, 'plan_id': action_id, 'reservation_id': invocation_id,
                   'input_sha256': content_hash(inputs),
                   'output': output, 'output_sha256': content_hash(output) if output else None,
                   'runtime_sha256': content_hash({'image_id': IMAGE, 'schema': 'fixture'}),
                   'limits': lab.config['limits'], 'dependency_ids': [],
                   'resources': {'cpu_seconds': .01, 'wall_seconds': .02},
                   'source_sha256': content_hash(bundle['files']),
                   'status': status, 'dispatch_status': dispatch}
        receipt['receipt_sha256'] = content_hash(receipt)
        artifact = {'schema': 'sciel-research-tool-artifact/1', 'kind': 'invocation',
            'payload': {'action_id': action_id, 'island_id': plan['island_id'],
                'parent_program_id': plan['parent_program_id'], 'iteration': ordinal,
                'tool_id': bundle['id'], 'inputs': inputs, 'receipt': receipt,
                'cell_id': plan['cell_id'], 'cell_revision_id': plan['cell_revision_id'],
                'stage': plan['stage'], 'question': (question or plan['question'])[:600],
                'input_program_id': 'p_original', 'target_binding':bound, 'result': assess_receipt(receipt),
                'replay_policy': bundle['manifest']['protocol']['replay_policy']}}
        artifact_id = content_hash(artifact)
        directory = lab.library.path / 'artifacts'
        directory.mkdir(exist_ok=True)
        (directory / (artifact_id + '.json')).write_bytes(canonical_bytes(artifact))
        rows = [{'tool_id': bundle['id'], 'invocation_id': invocation_id, 'receipt_id': artifact_id,
                 'status': status, 'dispatch_status': dispatch, 'result': assess_receipt(receipt),
                 'replay_policy': bundle['manifest']['protocol']['replay_policy'],
                 'input_sha256': receipt['input_sha256'], 'output_sha256': receipt['output_sha256'],
                 'resources': receipt['resources'], 'target_binding':bound,
                 'author_island_id': bundle['manifest']['protocol']['author_island']}]
    comparison = None
    if bundle and plan['stage'] == 'independent_replay':
        original = json.loads((directory / (plan['reference_receipt_id'] + '.json')).read_text())['payload']['receipt']
        comparison = replay_evidence(original, receipt, bundle['manifest']['protocol']['replay_policy'])
        if matched is not None:  # Some tests deliberately forge upstream claims; resume must reject them.
            comparison['matched'] = matched
            comparison['measurement_reproduced'] = matched
    return {'plan': deepcopy(plan), 'invocations': rows, 'replay_matched': matched,
            'replay_evidence': comparison, 'cards': list(cards)}


def local_success(lab, *, author=0):
    bundle = package(lab, author=author)
    plan = prepare(lab, author)
    record = episode(lab, plan, bundle)
    assert lab.lifecycle.commit(record) == []
    return bundle, plan, record


def released(lab):
    bundle, origin, local = local_success(lab)
    plan = prepare(lab, 1)
    assert plan['stage'] == 'independent_replay'
    replay = episode(lab, plan, bundle, matched=True)
    assert lab.lifecycle.commit(replay) == [bundle['id']]
    return bundle, origin, plan, local, replay
