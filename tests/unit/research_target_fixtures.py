"""Source-bound input fixtures; no provider, evaluator or Docker activity."""
from types import SimpleNamespace
from sciel.engine.research_targets import freeze_targets, input_binding, tool_inputs


def bind_snapshot(snapshot, *, task_hash='fixture'):
    parent = SimpleNamespace(id=snapshot.parent_id, code=snapshot.parent_code,
                             score=snapshot.parent_score)
    engine = SimpleNamespace(log=SimpleNamespace(run_id='fixture-run'),
        task=SimpleNamespace(task_hash=task_hash), iteration_no=snapshot.iteration_no, B_evals=0,
        db=SimpleNamespace(pop_best_id=parent.id, get=lambda _: parent), observed_changes=None)
    snapshot.research_targets = freeze_targets(engine, parent, parent.code)
    return snapshot


def binding(snapshot):
    return input_binding(tool_inputs(snapshot, {'task_hash':snapshot.research_targets['task_hash']}, {}))
