"""One full Engine resume with saved evidence and an explicitly synthetic R.

The evaluator, model and paired draw adapters are inert in-process fixtures.
No candidate is executed, no provider is called and no solver is run.
"""
import json
from pathlib import Path
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.generated_paired_review_context import PREFIX
from test_circle_intervention_engine import task_copy
from test_generated_paired_integration import InertModel, pilot_artifacts, component, pilot_for_task
from test_generated_paired_review_delivery import review_sidecar, policy_for
from test_optimizer_traced_engine import OutcomeDraws
from test_restart_prediction_engine import config as engine_config


def test_actual_component_review_survives_engine_checkpoint_eviction_and_next_generation(
        component, pilot_artifacts, review_sidecar, tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Engine fixture attempted an actual provider or subprocess.')
    monkeypatch.setattr('sciel.engine.llm.client.EnsembleLLM.generate', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)
    # Process telemetry is not part of this scientific checkpoint fixture.
    monkeypatch.setattr('sciel.engine.runtime.process_identity', lambda pid: None)
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path / 'task', initial=builder.source_for())
    pilot_artifacts, component = pilot_for_task(task)
    if review_sidecar['schema'] == 'sciel-generated-paired-review-sidecar/3':
        from test_scoped_paired_reflection import saved
        review_sidecar = saved.__wrapped__(component).sidecar
    else:
        from test_generated_paired_review_delivery import review_sidecar as author_review
        review_sidecar = author_review.__wrapped__(component)
    assert metric_context(task) == component['metric']
    cfg = engine_config()
    cfg['research'].update(external_component_sha256=canonical_sha(component),
        external_component_review_sha256=canonical_sha(review_sidecar),
        structured_auxiliary=True, comparison_reviews=True,
        comparison_review_version=3, comparison_review_text=True,
        comparison_review_text_policy=policy_for(review_sidecar, 'accepted'), batch_only_lessons=True,
        max_aux_calls=0, reflection_interval=0, max_experiments=1)
    bodies = ['return best_params.copy()\n', 'return initial_params.copy()\n']
    evaluated = []

    def evaluate(_task, path, **_kwargs):
        source = Path(path).read_text()
        body = builder.body_of(source)
        assert body in [builder.DEFAULT_BODY, *bodies], 'Previously saved candidate submitted to evaluator.'
        evaluated.append(source)
        score = .5 if body == builder.DEFAULT_BODY else .4
        cases = dict(sum_radii=score * component['metric']['semantics']['denominator'],
                     target_ratio=score, validity=1., combined_score=score)
        return CascadeOutcome(EvalResult(True, score, cases, timing=.01), score, 'full', False)

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)

    class Model(InertModel):
        weights = [1.]

        def __init__(self, index):
            reply = json.loads(pilot_artifacts.gj('off/response.json')['text'])
            reply['body'] = bodies[index]
            super().__init__(json.dumps(reply))

        def choose_model(self, _rng):
            return self.models[0]

    original_card, = review_sidecar['review_cards'].values()

    def assert_delivered(prompt):
        assert prompt.count(PREFIX) == 1
        view, _ = json.JSONDecoder().raw_decode(prompt.split(PREFIX, 1)[1])
        card, = view['cards']
        assert view['branch'] == component['root_parent']['id'] == 'p1'
        assert view['source_component_sha256'] == canonical_sha(component)
        assert card['branch'] == original_card['branch'] != view['branch']
        assert card['stored_card_sha256'] == canonical_sha(original_card)
        assert card['text_eligibility']['status'] == 'accepted'
        assert card['unverified_interpretation'] == original_card['unverified_interpretation']
        assert view['rendered_chars'] <= view['reserved_chars'] <= 8000
        return view

    folder = tmp_path / 'engine'
    first = Engine(cfg, task, str(folder), Model(0), python=sys.executable)
    original_seed = first.seed
    blank_checkpoint = folder / 'before_import_checkpoint.json'

    def seed_and_import():
        original_seed()
        checkpoint.save(first, 'before_external_component_review')
        blank_checkpoint.write_bytes((folder / 'checkpoint.json').read_bytes())
        assert first.research.knowledge.generated_paired_review is None
        first.circle_generation.import_external_component(component,
            expected_sha256=canonical_sha(component), parent=first.db.get('p1'))
        first.circle_generation.import_external_component_review(review_sidecar,
            expected_sha256=canonical_sha(review_sidecar))
        assert first.research.attempts == first.research.outcomes == first.research.aux_calls == 0
        assert first.research.reflections == first.research.reflected_through == 0
        assert first.research.knowledge.experiments == first.research.knowledge.lessons == []

    first.seed = seed_and_import
    first._stop_at = 1
    first.run()
    assert len(first.llm.calls) == 1 and len(draws.calls) == 9
    first_view = assert_delivered(first.llm.calls[0]['prompt'])
    assert first.research.knowledge.generated_paired_review == review_sidecar

    second = Engine(cfg, task, str(folder), Model(1), python=sys.executable)
    checkpoint.load(second, str(folder / 'checkpoint.json'))
    assert second.llm.calls == [] and len(evaluated) == 2
    assert second.research.knowledge.generated_paired_review == review_sidecar
    second.run()
    assert len(second.llm.calls) == 1 and len(draws.calls) == 15
    assert assert_delivered(second.llm.calls[0]['prompt']) == first_view
    assert evaluated == [builder.source_for(), *(builder.source_for(body) for body in bodies)]
    assert second.research.attempts == second.research.outcomes == second.circle_generation.evaluations == 2
    assert second.research.aux_calls == second.research.aux_cost == second.research.reflections == 0
    assert second.cost_total == .002  # Only two synthetic G costs, never the foreign R cost.
    assert [x['id'] for x in second.research.knowledge.experiments] == ['r000002a1']
    assert second.research.knowledge.lessons == []
    assert second.research.knowledge.generated_paired_history == component
    assert second.research.knowledge.generated_paired_review == review_sidecar
    saved = json.loads((folder / 'checkpoint.json').read_text())
    third = Engine(cfg, task, str(folder), Model(0), python=sys.executable)
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match='uncheckpointed research'):
            checkpoint.load(third, str(blank_checkpoint))
        assert third.llm.calls == [] and third.research.knowledge.generated_paired_history is None
        assert third.research.knowledge.generated_paired_review is None
        checkpoint.load(third, str(folder / 'checkpoint.json'))
        third.circle_generation.assert_journal(third.log.path, saved['provenance']['journal_anchor'])
        assert third.research.state() == second.research.state()
        assert third.research.knowledge.generated_paired_review == review_sidecar
        assert third.llm.calls == []
        events = [json.loads(line) for line in Path(third.log.path).read_text().splitlines()]
        assert sum(row.get('phase') == 'external_component_review_import' for row in events) == 1
    finally:
        third.repeat_promotion.close()
        third.log.close()
