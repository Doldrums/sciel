"""Resume preserves imported research provenance without replaying old calls."""

import json
from pathlib import Path
import sys

from sciel.engine import checkpoint
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.catalogs import canonical_sha

from test_circle_intervention_engine import task_copy
from test_external_history_v2_engine import (
    BODIES, RULES, Model, capsule_for, make_source, research_config,
)
from test_optimizer_traced_engine import OutcomeDraws
from test_restart_prediction_engine import config as engine_config


def test_nonempty_history_real_engine_checkpoint_does_not_replay_source(tmp_path, monkeypatch):
    source = make_source(tmp_path / 'source')
    capsule = capsule_for(source)
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path / 'task', initial=builder.source_for())
    cfg = engine_config()
    cfg['research'] = research_config()
    cfg['research'].update(external_history_sha256=canonical_sha(capsule),
                           max_aux_calls=0, reflection_interval=0)
    evaluated = []

    def evaluate(_task, path, **kwargs):
        code = Path(path).read_text()
        body = builder.body_of(code)
        assert body in [builder.DEFAULT_BODY, *BODIES[2:]], 'historical source was evaluated again'
        evaluated.append(code)
        score = .5 if body == builder.DEFAULT_BODY else .55
        native = dict(sum_radii=score * 2.635983, target_ratio=score,
                      validity=1., combined_score=score)
        return CascadeOutcome(EvalResult(True, score, native, timing=.01), score, 'full', False)

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    folder = tmp_path / 'engine'
    first_model = Model(generation_start=2)
    first = Engine(cfg, task, str(folder), first_model, python=sys.executable)
    seed = first.seed

    def seed_and_import():
        seed()
        first.circle_generation.import_external_history(capsule,
            expected_sha256=canonical_sha(capsule), parent=first.db.get('p1'))
        assert first.research.attempts == first.research.outcomes == first.research.aux_calls == 0

    first.seed = seed_and_import
    first._stop_at = 1
    first.run()
    assert len(first_model.calls) == 1 and len(draws.calls) == 9
    assert first.research.knowledge.experiments[-1]['id'] == 'r000003a1'
    assert first.research.knowledge.lessons[0]['exposure_ids'] == ['r000003a1']
    raw, _ = json.JSONDecoder().raw_decode(first_model.calls[0]['prompt'].split('RESEARCH_CONTEXT_JSON:', 1)[1].lstrip())
    delivered = decode_evidence(raw)
    assert delivered['comparison_reviews']['presented_count'] > 0
    assert RULES[1] in first_model.calls[0]['prompt']

    second_model = Model(generation_start=3)
    second = Engine(cfg, task, str(folder), second_model, python=sys.executable)
    checkpoint.load(second, str(folder / 'checkpoint.json'))
    second.run()
    assert len(second_model.calls) == 1 and len(draws.calls) == 15
    assert evaluated == [builder.source_for(), builder.source_for(BODIES[2]), builder.source_for(BODIES[3])]
    assert second.research.outcomes == second.circle_generation.evaluations == 2
    assert second.research.aux_calls == second.research.aux_cost == 0
    assert second.cost_total == .002
    assert second.research.knowledge.experiments[-1]['id'] == 'r000004a1'
    assert second.research.knowledge.lessons[0]['refute_ids'] == ['r000002a1']
    # A final ordinary-window checkpoint also authenticates old foreign writers,
    # local exposures and any explicit omission due to the unchanged card cap.
    third = Engine(cfg, task, str(folder), Model(generation_start=2), python=sys.executable)
    checkpoint.load(third, str(folder / 'checkpoint.json'))
    assert third.research.state() == second.research.state()
    assert third.llm.calls == []
    third.repeat_promotion.close()
    third.log.close()
