"""G-only paired memory crosses public controller boundaries without execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from sciel.engine.circle_generation import CONTEXT_SCOPE
from sciel.engine.types import NoChild
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha

from generated_paired_integration_fixture import (
    InertModel, component, construct, decoded_component, decoded_context,
    forbid_runtime_execution, import_component, pilot_artifacts, pilot_for_task, prepare, rows, snapshot,
)


def test_authored_streams_keep_two_authors_and_partial_negative_measurements(pilot_artifacts):
    p = pilot_artifacts
    # Same local ID in separate old arms is not the same authored candidate.
    assert p.bundle['arms']['off']['research_id'] == p.bundle['arms']['on']['research_id'] == 'r000003a1'
    assert p.gj('off/response.json')['text'] != p.gj('on/response.json')['text']
    summary = p.nj('summary.json')
    assert summary['primary_on_minus_off']['verdict'] == 'INCONCLUSIVE'
    assert summary['secondary_on_minus_control']['verdict'] == 'REFUTED'
    assert summary['primary_on_minus_off']['pairs'][0]['raw_delta'] == 0
    for group in p.groups.values():
        assert group['origin_experiment_id'] is None
        assert all(pair['comparator'] is pair['challenger'] is None for pair in group['pairs'][1:])
    assert not any(json.loads(line).get('phase') == 'outcome'
                   for arm in ('off', 'on') for line in p.generation[arm + '/preparation.jsonl'].splitlines())


def test_public_import_is_immutable_annex_with_zero_ordinary_activity(component, pilot_artifacts, tmp_path):
    f = construct(pilot_artifacts, component, tmp_path)
    before = canonical_json(component)
    try:
        assert import_component(f, component) is True
        assert f.c.knowledge.generated_paired_history == component
        assert f.c.knowledge.experiments == f.c.knowledge.comparisons == []
        assert f.c.knowledge.lessons == f.c.knowledge.fragments == []
        assert f.c.knowledge.state()['seen_experiments'] == {}
        assert all(getattr(f.c, key) == 0 for key in ('attempts', 'outcomes', 'valid_hypotheses',
            'aux_calls', 'aux_cost', 'reflections', 'reflection_changes', 'reflected_through'))
        assert f.worker.calls_started == f.worker.calls_returned == f.worker.evaluations == 0
        assert f.worker.seen_proofs == [] and f.model.calls == f.charges == []
        events = rows(f.log)
        assert len(events) == 2 and events[-1]['phase'] == 'external_component_import'
        assert import_component(f, copy.deepcopy(component)) is False
        assert rows(f.log) == events and canonical_json(component) == before
        returned = f.c.knowledge.generated_paired_history
        returned['proposals'].clear()
        assert f.c.knowledge.generated_paired_history == component
        assert len({p['id'] for p in component['proposals']}) == 2
        assert len({p['provenance']['research_id'] for p in component['proposals']}) == 1
        f.worker.assert_journal(f.log.path, f.log.anchor())
        from sciel.engine.generated_paired_import import assert_import_journal
        forged_rows = copy.deepcopy(rows(f.log))
        forged_rows[-1]['new_cost'] = False
        with pytest.raises(ValueError, match='journal import binding differs'):
            assert_import_journal(f.c, forged_rows)
    finally:
        f.log.close()


def test_actual_facts_and_unscored_forecasts_reach_public_generation_prompt(component, pilot_artifacts, tmp_path):
    f = construct(pilot_artifacts, component, tmp_path)
    try:
        import_component(f, component)
        ctx = f.c.context(f.db, f.parent, 0, 1)
        block = ctx['generated_paired_history']
        from sciel.knowledge_base.generated_paired_context import expand_context
        card, = expand_context(block)['cards']
        assert card['capsule_sha256'] == canonical_sha(component)
        assert {p['id'] for p in card['authored_proposals']} == {p['id'] for p in component['proposals']}
        qualities = {x['branch']: x['strict_quality'] for x in card['comparisons']}
        assert qualities['on_off']['verdict'] == 'INCONCLUSIVE'
        assert qualities['on_control']['verdict'] == 'REFUTED'
        assert qualities['on_control']['reason'] == 'known_paired_loss'
        for quality in qualities.values():
            assert all(p['raw_delta'] is p['certified_delta'] is p['no_loss'] is None
                       for p in quality['pairs'][1:])
        assert card['draw_coverage']['planned_count'] == 9
        assert card['draw_coverage']['not_dispatched_count'] == 6
        for proposal in card['authored_proposals']:
            assert proposal['native_observation']['score'] is None
            assert proposal['native_observation']['actual_delta'] is None
            assert proposal['native_observation']['forecast_scored'] is False
        _, prompt = prepare(f, ctx)
        assert decoded_component(prompt) == block
        assert f.c.outcomes == f.c.aux_calls == f.worker.evaluations == 0
        assert f.model.calls == []
    finally:
        f.log.close()


def test_forged_delivery_is_rejected_before_attempt_or_journal(component, pilot_artifacts, tmp_path):
    f = construct(pilot_artifacts, component, tmp_path)
    try:
        import_component(f, component)
        ctx = f.c.context(f.db, f.parent, 0, 1)
        shown = ctx['generated_paired_history']
        quality_index = shown['columns']['comparisons'].index('strict_quality')
        shown['cards'][0]['comparisons'][0][quality_index]['verdict'] = 'CONFIRMED'
        # Self-reported wire digest is not authority for altered facts.
        shown['context_sha256'] = canonical_sha({k: v for k, v in shown.items() if k != 'context_sha256'})
        before, events = f.c.state(), rows(f.log)
        with pytest.raises(ValueError):
            prepare(f, ctx)
        assert f.c.state() == before and rows(f.log) == events
        assert f.model.calls == []
    finally:
        f.log.close()


def test_forged_branch_cannot_silently_remove_declared_component(component, pilot_artifacts, tmp_path):
    f = construct(pilot_artifacts, component, tmp_path)
    try:
        import_component(f, component)
        ctx = f.c.context(f.db, f.parent, 0, 1)
        assert ctx['branch'] == component['root_parent']['id']
        ctx['branch'] = 'unrelated-forged-root'
        ctx.pop('generated_paired_history')
        before, events = f.c.state(), rows(f.log)
        with pytest.raises(ValueError, match='root|branch'):
            prepare(f, ctx)
        assert f.c.state() == before and rows(f.log) == events
        assert f.c.attempts == f.c.outcomes == f.worker.calls_started == 0
        assert f.model.calls == []
    finally:
        f.log.close()


@pytest.mark.parametrize('damage', ['changed', 'removed'])
def test_component_config_change_cannot_drain_queued_conclusion(component, pilot_artifacts, tmp_path, damage):
    reply = json.loads(pilot_artifacts.gj('off/response.json')['text'])
    reply['body'] = 'return initial_params.copy()\n'
    f = construct(pilot_artifacts, component, tmp_path,
        llm=InertModel(json.dumps(reply)), max_aux_calls=1)
    key = 'external_component_sha256'
    original_digest = f.c.config[key]
    try:
        import_component(f, component)
        def inert_evaluation(source):
            assert source == builder.source_for(reply['body'])
            # An explicitly invented unit-test receipt, never a packing run.
            score = .5
            return CascadeOutcome(EvalResult(True, score, dict(sum_radii=1.3175,
                target_ratio=score, validity=1., combined_score=score), timing=.01),
                score, 'full', False)
        result = f.worker.run(snapshot(f), f.model, inert_evaluation, base_temp=.2)
        assert result.ok and len(f.c.queue) == 1 and len(f.model.calls) == 1
        before = f.c.state()
        events, calls, charges = rows(f.log), copy.deepcopy(f.model.calls), list(f.charges)
        if damage == 'changed':
            f.c.config[key] = '0' * 64
        else:
            del f.c.config[key]
        try:
            with pytest.raises(ValueError, match='configured digest changed'):
                f.c.complete_step(reflect=False)
        finally:
            f.c.config[key] = original_digest
        assert f.c.state() == before
        assert rows(f.log) == events and f.model.calls == calls and f.charges == charges
        assert len(f.c.queue) == 1 and f.c.aux_calls == f.c.aux_cost == 0
    finally:
        f.c.config[key] = original_digest
        f.log.close()


def test_declared_component_must_be_imported_before_context_or_begin(component, pilot_artifacts, tmp_path):
    f = construct(pilot_artifacts, component, tmp_path)
    try:
        before, events = f.c.state(), rows(f.log)
        with pytest.raises(ValueError, match='not been imported'):
            f.c.context(f.db, f.parent, 0, 1)
        # Caller-supplied empty context must not bypass configured admission.
        with pytest.raises(ValueError, match='not been imported'):
            snap = snapshot(f, {'branch': f.parent.id, 'lesson_texts': [], 'injected_lesson_ids': []})
            f.c.begin(snap, 0, 'circle_generate', 'Inert prepared prompt.', context_scope=CONTEXT_SCOPE)
        assert f.c.state() == before and rows(f.log) == events
        assert f.model.calls == []
    finally:
        f.log.close()


@pytest.mark.parametrize('arm', ['off', 'on'])
def test_omitted_context_does_not_allow_reusing_saved_candidate(component, pilot_artifacts, tmp_path, arm):
    raw = pilot_artifacts.gj(arm + '/response.json')['text']
    f = construct(pilot_artifacts, component, tmp_path, llm=InertModel(raw), inject_lessons=False)
    try:
        import_component(f, component)
        snap = snapshot(f)
        shown = snap.research_context.get('generated_paired_history')
        assert shown is None or not shown['cards']
        def forbidden(_source):
            pytest.fail('A previously saved source was evaluated again.')
        result = f.worker.run(snap, f.model, forbidden, base_temp=.2)
        assert result.reason == 'generation_repeated'
        assert len(f.model.calls) == 1 and f.worker.evaluations == 0
        assert f.c.knowledge.generated_paired_history == component
        local, = f.c.knowledge.experiments
        assert local['score'] is local['actual_delta'] is None
        assert local['forecast_verdict'] == 'INCONCLUSIVE'
        f.worker.assert_journal(f.log.path, f.log.anchor())
    finally:
        f.log.close()


def test_controller_worker_state_and_durable_journal_survive_next_inert_g(component, pilot_artifacts, tmp_path):
    reply = json.loads(pilot_artifacts.gj('off/response.json')['text'])
    reply['body'] = 'return initial_params.copy()\n'
    f = construct(pilot_artifacts, component, tmp_path / 'first', llm=InertModel(json.dumps(reply)))
    second = None
    try:
        import_component(f, component)
        evaluated = []
        def evaluate(source):
            assert source == builder.source_for(reply['body'])
            evaluated.append(source)
            score = f.parent.score + .0001
            cases = dict(sum_radii=score * component['metric']['semantics']['denominator'],
                         target_ratio=score, validity=1., combined_score=score)
            return CascadeOutcome(EvalResult(True, score, cases, timing=.01), score, 'full', False)
        result = f.worker.run(snapshot(f), f.model, evaluate, base_temp=.2)
        assert result.ok and len(evaluated) == len(f.model.calls) == 1
        f.c.complete_step(reflect=False)
        assert f.c.attempts == f.c.outcomes == f.worker.evaluations == 1
        assert f.c.aux_calls == 0 and f.c.knowledge.generated_paired_history == component
        before = f.c.state()
        anchor = f.log.anchor()
        f.worker.assert_journal(f.log.path, anchor)
        second = construct(pilot_artifacts, component, tmp_path / 'second')
        second.c.load_state(before)
        second.worker.load_state(f.worker.state())
        second.worker.assert_journal(f.log.path, anchor)
        assert second.c.state() == before
        assert second.c.context(second.db, second.parent, 0, 2)['generated_paired_history'] == (
            f.c.context(f.db, f.parent, 0, 2)['generated_paired_history'])
        from sciel.engine.reflection_context import prepare_reflection_context
        from sciel.knowledge_base.generated_paired_context import block as component_block
        reflected = prepare_reflection_context(second.c.knowledge, second.c.conclusions,
            config=second.c.config, interval=1, through=1)
        shown = reflected['context_event']['generated_paired_history']
        assert shown['presented_count'] == 1
        from sciel.knowledge_base.generated_paired_context import expand_context
        assert expand_context(shown)['cards'][0]['capsule_sha256'] == canonical_sha(component)
        assert component_block(shown) in reflected['prompt']
        assert reflected['context_event']['experiment_ids'] == ['r000001a1']
        assert second.model.calls == second.charges == []
    finally:
        f.log.close()
        if second is not None:
            second.log.close()


@pytest.mark.parametrize('damage', ['response', 'source', 'root', 'policy'])
def test_invalid_component_cannot_partly_mutate_controller(component, pilot_artifacts, tmp_path, damage):
    bad = copy.deepcopy(component)
    if damage == 'response':
        bad['proposals'][0]['response']['text'] += ' '
    elif damage == 'source':
        bad['proposals'][0]['source'] += '\n# forged\n'
    elif damage == 'root':
        bad['root_parent']['id'] = 'another-branch'
    else:
        bad['policy']['schema'] = 'forged-policy'
    f = construct(pilot_artifacts, bad, tmp_path)
    try:
        before, events = f.c.state(), rows(f.log)
        with pytest.raises(ValueError):
            import_component(f, bad)
        assert f.c.state() == before and rows(f.log) == events
        assert f.worker.calls_started == f.worker.evaluations == 0
    finally:
        f.log.close()


@pytest.mark.parametrize('sink', ['returns_false', 'anchor_failure', 'post_event_anchor_failure'])
def test_import_needs_successful_durable_journal_before_state_publish(component, pilot_artifacts, tmp_path, monkeypatch, sink):
    f = construct(pilot_artifacts, component, tmp_path)
    try:
        before = f.c.state()
        if sink == 'returns_false':
            monkeypatch.setattr(f.log, 'research', lambda **_fields: False)
        else:
            original_anchor, calls = f.log.anchor, []
            def fail():
                calls.append(True)
                if sink == 'post_event_anchor_failure' and len(calls) == 1:
                    return original_anchor()
                raise OSError('Inert fsync failure; never an authorized import.')
            monkeypatch.setattr(f.log, 'anchor', fail)
        with pytest.raises((OSError, ValueError)):
            import_component(f, component)
        assert f.c.state() == before
        assert f.c.knowledge.generated_paired_history is None
        assert f.worker.calls_started == f.worker.evaluations == 0
        if sink == 'post_event_anchor_failure':
            monkeypatch.setattr(f.log, 'anchor', original_anchor)
            original_anchor()
            assert rows(f.log)[-1]['phase'] == 'external_component_import'
            with pytest.raises(ValueError, match='refusing replay'):
                import_component(f, component)
            assert f.c.state() == before
    finally:
        f.log.close()


def test_forged_saved_annex_and_missing_import_event_cannot_resume(component, pilot_artifacts, tmp_path):
    f = construct(pilot_artifacts, component, tmp_path / 'original')
    restored = construct(pilot_artifacts, component, tmp_path / 'restore')
    try:
        import_component(f, component)
        original = f.c.state()
        bad = copy.deepcopy(original)
        bad['knowledge']['generated_paired_history']['proposals'][0]['provenance']['request_key'] = 'forged-arm'
        before = restored.c.state()
        with pytest.raises(ValueError):
            restored.c.load_state(bad)
        assert restored.c.state() == before
        restored.c.load_state(original)
        restored.worker.load_state(f.worker.state())
        # This separate real RunLog has only run_start: its valid anchor cannot
        # authenticate an import that happened in a different journal.
        with pytest.raises(ValueError):
            restored.worker.assert_journal(restored.log.path, restored.log.anchor())
        assert restored.model.calls == restored.charges == []
    finally:
        f.log.close()
        restored.log.close()


def test_actual_ordinary_history_and_generated_annex_coexist_without_replaying_old_R(
        component, pilot_artifacts, tmp_path):
    from sciel.engine.external_research_v2 import policy_identity
    from sciel.knowledge_base.generated_paired_context import expand_context
    ordinary = pilot_artifacts.gj('frozen/capsule.json')
    original_journals = canonical_json(ordinary['journals'])
    assert ordinary['policy'] == policy_identity()  # Authored under the current policy.
    config = pilot_artifacts.gj('frozen/config.json')
    config = copy.deepcopy(config.get('research', config))
    config.update(external_history_sha256=canonical_sha(ordinary),
                  external_component_sha256=canonical_sha(component),
                  max_aux_calls=0, reflection_interval=0)
    model = InertModel()
    model.models = [config['model']]
    f = construct(pilot_artifacts, component, tmp_path, llm=model, **config)
    try:
        assert f.worker.import_external_history(ordinary,
            expected_sha256=canonical_sha(ordinary), parent=f.parent) is True
        old_experiments = copy.deepcopy(f.c.knowledge.experiments)
        old_conclusions = copy.deepcopy(f.c.conclusions)
        old_reviews = copy.deepcopy(f.c._comparison_reviews)
        assert len(old_experiments) == 2 and old_reviews == {}
        assert import_component(f, component) is True
        assert f.c.knowledge.experiments == old_experiments
        assert f.c.conclusions == old_conclusions and f.c._comparison_reviews == old_reviews
        assert f.c.knowledge.generated_paired_history == component
        assert canonical_json(ordinary['journals']) == original_journals
        assert f.c.attempts == f.c.outcomes == f.c.aux_calls == f.worker.evaluations == 0
        shown = f.c.context(f.db, f.parent, 0, 1)['generated_paired_history']
        assert expand_context(shown)['cards'][0]['capsule_sha256'] == canonical_sha(component)
        f.worker.assert_journal(f.log.path, f.log.anchor())
        # Three authored ordinary comparison groups coexist with the new whole
        # study. begin must reconstruct exactly the same reduced budget/cards.
        assert len(f.c.knowledge.comparisons) == 3
        _, prompt = prepare(f)
        assert decoded_component(prompt) == shown
        assert f.c.outcomes == f.c.aux_calls == f.worker.evaluations == 0
        assert f.model.calls == f.charges == []
    finally:
        f.log.close()




def test_actual_annex_survives_engine_checkpoint_eviction_and_next_segment(
        component, pilot_artifacts, tmp_path, monkeypatch):
    from sciel.engine import checkpoint
    from sciel.engine.loop import Engine
    from sciel.hypothesis.measurement import metric_context
    from test_circle_intervention_engine import task_copy
    from test_optimizer_traced_engine import OutcomeDraws
    from test_restart_prediction_engine import config as engine_config

    draws = OutcomeDraws(monkeypatch)  # In-process literal receipts; no adapter/solver.
    task = task_copy(tmp_path / 'task', initial=builder.source_for())
    pilot_artifacts, component = pilot_for_task(task)
    assert metric_context(task) == component['metric']
    cfg = engine_config()
    cfg['research'].update(external_component_sha256=canonical_sha(component),
                           max_aux_calls=0, reflection_interval=0, max_experiments=1)
    bodies = ['return best_params.copy()\n', 'return initial_params.copy()\n']
    evaluated = []

    def evaluate(_task, path, **_kwargs):
        source = Path(path).read_text()
        body = builder.body_of(source)
        assert body in [builder.DEFAULT_BODY, *bodies], 'A previously saved candidate was evaluated again.'
        evaluated.append(source)
        # Keep the receiving root as the chosen parent in both inert segments.
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

    folder = tmp_path / 'engine'
    first = Engine(cfg, task, str(folder), Model(0), python=sys.executable)
    original_seed = first.seed
    blank_checkpoint = folder / 'before_import_checkpoint.json'

    def seed_and_import():
        original_seed()
        checkpoint.save(first, 'before_external_component')
        blank_checkpoint.write_bytes((folder / 'checkpoint.json').read_bytes())
        assert first.research.knowledge.generated_paired_history is None
        first.circle_generation.import_external_component(component,
            expected_sha256=canonical_sha(component), parent=first.db.get('p1'))
        assert first.research.attempts == first.research.outcomes == first.research.aux_calls == 0
        assert first.research.knowledge.experiments == []

    first.seed = seed_and_import
    first._stop_at = 1
    first.run()
    assert first.research.knowledge.generated_paired_history == component
    assert len(first.llm.calls) == 1
    assert decoded_component(first.llm.calls[0]['prompt'])['presented_count'] == 1
    assert len(draws.calls) == 9  # Three mock baseline and six mock paired draws.

    second = Engine(cfg, task, str(folder), Model(1), python=sys.executable)
    checkpoint.load(second, str(folder / 'checkpoint.json'))
    assert second.llm.calls == [] and len(evaluated) == 2
    assert second.research.knowledge.generated_paired_history == component
    second.run()
    assert len(second.llm.calls) == 1 and len(draws.calls) == 15
    assert evaluated == [builder.source_for(), *(builder.source_for(body) for body in bodies)]
    assert second.research.attempts == second.research.outcomes == second.circle_generation.evaluations == 2
    assert second.research.aux_calls == second.research.aux_cost == 0
    assert second.cost_total == .002
    assert [x['id'] for x in second.research.knowledge.experiments] == ['r000002a1']
    assert second.research.knowledge.lessons == []
    assert second.research.knowledge.generated_paired_history == component
    saved = json.loads((folder / 'checkpoint.json').read_text())
    third = Engine(cfg, task, str(folder), Model(0), python=sys.executable)
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match='uncheckpointed research'):
            checkpoint.load(third, str(blank_checkpoint))
        assert third.llm.calls == [] and third.research.knowledge.generated_paired_history is None
        checkpoint.load(third, str(folder / 'checkpoint.json'))
        third.circle_generation.assert_journal(third.log.path, saved['provenance']['journal_anchor'])
        assert third.research.state() == second.research.state()
        assert third.research.knowledge.generated_paired_history == component
        assert third.llm.calls == []
    finally:
        third.repeat_promotion.close()
        third.log.close()
