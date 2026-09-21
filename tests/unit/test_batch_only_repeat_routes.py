"""Batch-only policy on both Circle repeat routes, with inert I/O only."""
import sys

from sciel.engine.loop import Engine
from test_batch_only_lessons import admission, decode
from test_repeat_reflection_lifecycle import ReflectingModel
from test_repeat_engine_integration import config, FakeDraws, task, search_observations
from test_repeat_population_engine import settings, numeric, draws
from test_repeat_population_reflection_lifecycle import ResearchProvider


def test_promotion_reflection_uses_suppressed_review_and_terminal_group(task, tmp_path, monkeypatch, search_observations):
    measured = FakeDraws(monkeypatch)
    cfg = config(budget=1, max_iterations=1)
    cfg['research'].update(batch_only_lessons=True, reflection_interval=1, max_aux_calls=2)
    model = ReflectingModel(measured)
    engine = Engine(cfg, task, str(tmp_path/'promotion'), model, python=sys.executable)
    engine.run()
    call, = [c for c in model.calls if c['kind'] == 'reflection']
    assert call['draw_count'] == 9
    review, = decode(call['prompt'], 'REFLECTION_REVIEW_CONTEXT_JSON:')['cards']
    assert review['lesson_admission']['policy_suppression'] == admission()
    assert review['lesson_admission']['accepted_lesson_ids'] == []
    lesson, = engine.research.knowledge.lessons
    group, = engine.research.knowledge.comparisons
    assert lesson['source_support_ids'] == [group['id']]
    assert lesson['id'] == 'l000001'


def test_population_reflection_keeps_zero_proposal_marker_and_terminal_group(task, tmp_path, numeric, draws):
    cfg = settings(budget=1, max_iterations=1, research=dict(enabled=True,
        batch_only_lessons=True, reflection_interval=1, max_aux_calls=2))
    model = ResearchProvider(draws)
    engine = Engine(cfg, task, str(tmp_path/'population'), model, python=sys.executable)
    engine.run()
    call, = [c for c in model.auxiliary if c['kind'] == 'reflection']
    assert call['draw_count'] == 9
    review, = decode(call['prompt'], 'REFLECTION_REVIEW_CONTEXT_JSON:')['cards']
    assert review['lesson_admission']['policy_suppression'] == admission(0)
    group, = engine.research.knowledge.comparisons
    assert group['id'] in call['evidence']['observation_catalog']
    assert engine.research.knowledge.lessons == []
    assert engine.research.knowledge.state()['next_lesson_id'] == 1
