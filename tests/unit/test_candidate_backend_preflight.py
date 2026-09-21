"""Independent policy/routing witnesses; candidate/provider/native leaves poisoned."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from sciel.config import apply_defaults
from sciel.engine import factory
from sciel.engine.loop import Engine
from sciel.eval.candidate_config import validate_task
from sciel.eval.registry import load_task

MVP=Path(__file__).resolve().parents[2]
POLICY={'backend':'candidate_only','image_id':'sha256:'+'1'*64,
        'cpu_seconds':2.,'memory_bytes':128*1024*1024,'pids':16}


def config(tmp_path,**updates):
    from test_parallel_engine import config as fixture
    return fixture('oe/circle_packing',tmp_path/'capacity',workers=1,num_islands=1,
                   evaluation=deepcopy(POLICY),**updates)


@pytest.fixture
def poison(monkeypatch):
    from sciel.eval import cascade,sandbox,registry
    import sciel.engine.loop as loop
    sentinel=Mock(side_effect=AssertionError('forbidden native or provider path'))
    monkeypatch.setattr(factory,'build_llm',sentinel)
    monkeypatch.setattr(loop,'cascade_evaluate',sentinel)
    monkeypatch.setattr(cascade,'run_stage',sentinel)
    monkeypatch.setattr(sandbox,'run_stage',sentinel)
    monkeypatch.setattr(registry,'determinism_check',sentinel)
    yield sentinel
    sentinel.assert_not_called()


@pytest.mark.parametrize('key',['repeat_population','repeat_promotion','circle_generation','circle_intervention'])
def test_unsupported_helper_layout_rejected_before_provider(tmp_path,poison,key):
    cfg=config(tmp_path,**{key:{'enabled':True}})
    with pytest.raises(ValueError):factory.build_engine(cfg,str(MVP/'tasks'),str(tmp_path/'run'))
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('key,value',[
    ('kb_free',True),('c2_event_bets',True),('container',{'selection':'centres'}),
    ('parallel_search',{'enabled':False}),('candidate_novelty',{'enabled':False}),
])
def test_unsupported_wave_shape_rejected_before_provider(tmp_path,poison,key,value):
    cfg=config(tmp_path,**{key:value})
    with pytest.raises(ValueError):factory.build_engine(cfg,str(MVP/'tasks'),str(tmp_path/'run'))
    assert not (tmp_path/'run').exists()


def test_unregistered_task_cannot_fall_back_to_native(tmp_path,poison):
    cfg=config(tmp_path);cfg['task']='oracle/editable_kb'
    # Use a maintained unregistered local task; this path is supplied below by
    # the test inventory rather than importing or executing its evaluator.
    from sciel.eval.registry import discover
    known={'oe/circle_packing','oe/heilbronn_triangle','math/beat_the_average'}
    task=next(t for t in discover(str(MVP/'tasks')) if t.qualified_name not in known)
    cfg['task']=task.qualified_name
    with pytest.raises(ValueError,match='no exact registered'):
        factory.build_engine(cfg,str(MVP/'tasks'),str(tmp_path/'run'))
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('field,value',[('confirm_n',2),('per_run_timeout',999.),('stochastic',False)])
def test_direct_effective_task_mutation_fails_before_engine_side_effect(tmp_path,poison,field,value):
    task=load_task(str(MVP/'tasks/oe/circle_packing'))
    if getattr(task,field)==value:value=True if field=='stochastic' else value+1
    setattr(task,field,value)
    with pytest.raises(ValueError,match='effective TaskSpec differs'):
        Engine(config(tmp_path),task,str(tmp_path/'run'),Mock())
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('task_name',['oe/circle_packing','oe/heilbronn_triangle','math/beat_the_average'])
def test_exact_registered_task_preflight_is_source_only(tmp_path,poison,task_name):
    cfg=config(tmp_path);cfg['task']=task_name
    cfg=apply_defaults(cfg);task=load_task(str(MVP/'tasks'/task_name))
    assert validate_task(cfg,task) in {'circle','heil','beat'}


def test_modified_evaluator_cannot_fall_back_to_native(tmp_path,poison):
    import shutil
    tasks=tmp_path/'tasks';target=tasks/'oe/circle_packing';target.parent.mkdir(parents=True)
    shutil.copytree(MVP/'tasks/oe/circle_packing',target)
    path=target/'evaluator.py';path.write_text(path.read_text()+'\n# altered inert identity witness\n')
    with pytest.raises(ValueError,match='no exact registered'):
        factory.build_engine(config(tmp_path),str(tasks),str(tmp_path/'run'))
    assert not (tmp_path/'run').exists()
