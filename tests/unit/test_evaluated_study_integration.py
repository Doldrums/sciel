"""Portable detached-study import crosses real lifecycle boundaries without execution.

HOST evidence and generation/evaluation replies are authored unit fixtures;
no provider, candidate, Docker or optimizer runs.
"""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine.circle_generation import CircleGeneration, CONTEXT_SCOPE
from sciel.engine.research import ResearchController
from sciel.engine.types import Program, Snapshot
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.run_log import RunLog
from test_generated_paired_integration import InertModel, rows
from test_restart_prediction_engine import config as engine_config, spec
from test_circle_generation_engine import response



@pytest.fixture(autouse=True)
def forbid_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Detached study integration cannot dispatch a runtime or numeric draw.')
    monkeypatch.setattr('sciel.eval.repeat_draw.draw', forbidden)
    monkeypatch.setattr('sciel.eval.repeat_draw.adapter_identity', forbidden)


@pytest.fixture(scope='module')
def artifact():
    from evaluated_study_fixture import build
    return build()


def settings(artifact, **changes):
    cfg = engine_config()
    cfg['research'].update(external_evaluated_study_sha256=artifact['capsule_sha256'],
        external_evaluated_study_authority_sha256=artifact['author_authority_sha256'],
        model=InertModel.models[0], max_aux_calls=0, reflection_interval=0)
    cfg['research'].update(changes)
    return cfg


def construct(artifact, path, *, llm=None, **changes):
    cfg = apply_defaults(settings(artifact, **changes))['research']
    model = llm or InertModel()
    log = RunLog(str(path / 'research.jsonl'))
    log.run_start(scope='inert evaluated detached study import')
    charges = []
    capsule = artifact['capsule']
    c = ResearchController(cfg, model, log, charges.append, metric=capsule['metric'])
    worker = CircleGeneration(c, execution_profile=builder.EXECUTION_PROFILE)
    parent = Program(capsule['receiving_root']['id'], builder.source_for(), .5,
        dict(sum_radii=.5*capsule['metric']['semantics']['denominator'], target_ratio=.5, validity=1., combined_score=.5), 0, 'seed')
    return NS(c=c, worker=worker, model=model, log=log, charges=charges, parent=parent,
              db=NS(_all={parent.id: parent}))


def import_study(f, artifact):
    return f.worker.import_external_evaluated_study(artifact['capsule'],
        artifact['author_dependencies'], artifact['author_authority'],
        expected_sha256=artifact['capsule_sha256'], parent=f.parent)


def bundle(artifact):
    return {key: copy.deepcopy(artifact[key]) for key in ('capsule', 'author_dependencies', 'author_authority')}


def snapshot(f, context=None):
    ctx = f.c.context(f.db, f.parent, 0, f.c.attempts + 1) if context is None else context
    return Snapshot(island=0, op='circle_generate', model=f.model.models[0], iteration_no=f.c.attempts,
        parent_id=f.parent.id, parent_code=f.parent.code, parent_score=f.parent.score,
        task_prompt='Pack 26 circles in the unit square.', research_context=ctx,
        kb_lessons=ctx['lesson_texts'])


def prepare(f, context=None):
    snap = snapshot(f, context)
    return f.c.begin(snap, 0, 'circle_generate', f.worker.prepare_prompt(snap), context_scope=CONTEXT_SCOPE)


def test_authored_import_zero_native_activity_detached_idempotent_and_journal(artifact, tmp_path):
    f = construct(artifact, tmp_path)
    try:
        assert import_study(f, artifact) is True
        kb = f.c.knowledge
        assert kb.evaluated_detached_study == bundle(artifact)
        assert kb.experiments == kb.comparisons == kb.lessons == kb.fragments == []
        assert kb.state()['seen_experiments'] == {}
        assert all(getattr(f.c, key) == 0 for key in ('attempts', 'outcomes', 'aux_calls',
            'aux_cost', 'reflections', 'reflection_changes', 'reflected_through'))
        assert f.worker.calls_started == f.worker.evaluations == 0
        assert f.model.calls == f.charges == []
        before = rows(f.log)
        assert before[-1]['phase'] == 'external_evaluated_study_import'
        assert before[-1]['new_lessons'] == before[-1]['new_exposures'] == 0
        assert import_study(f, artifact) is False and rows(f.log) == before
        detached = kb.evaluated_detached_study
        detached['author_authority']['issuer'] = 'checkpoint is not an authority'
        assert kb.evaluated_detached_study == bundle(artifact)
        f.worker.assert_journal(f.log.path, f.log.anchor())
        from sciel.engine.evaluated_study_import import assert_import_journal
        forged = copy.deepcopy(before)
        forged[-1]['new_cost'] = False
        with pytest.raises(ValueError, match='journal import binding'):
            assert_import_journal(f.c, forged)
    finally:
        f.log.close()


def test_authored_study_reaches_next_G_and_R_with_missing_unknown_and_negative_quality(artifact, tmp_path):
    from sciel.knowledge_base.evaluated_study_context import PREFIX, block
    from sciel.engine.reflection_context import prepare_reflection_context
    f = construct(artifact, tmp_path)
    try:
        import_study(f, artifact)
        shown = f.c.context(f.db, f.parent, 0, 1)['evaluated_detached_study']
        assert shown['presented_count'] == 1
        card, = shown['cards']
        assert card['author']['kind'] == 'HOST'
        assert card['strict_quality']['verdict'] == 'REFUTED'
        assert len(artifact['capsule']['comparison']['pairs']) == 3
        missing = artifact['capsule']['comparison']['pairs'][-1]
        assert missing['comparator'] is missing['challenger'] is None
        assert len(block(shown)) <= 16000
        _, prompt = prepare(f)
        assert block(shown) in prompt
        literal, _ = json.JSONDecoder().raw_decode(prompt.split(PREFIX, 1)[1])
        assert literal == shown
        # Pure R preparation is not a new scheduler event or native outcome.
        prepared = prepare_reflection_context(f.c.knowledge, {}, config=f.c.config, interval=1, through=1)
        assert prepared['context_event']['evaluated_detached_study'] == shown
        assert block(shown) in prepared['prompt']
        assert prepared['records'] == prepared['lessons'] == []
        assert f.c.outcomes == f.c.aux_calls == f.worker.evaluations == 0
        assert f.model.calls == []
    finally:
        f.log.close()


@pytest.mark.parametrize('damage', ['drop', 'branch', 'card'])
def test_forged_context_rejected_before_attempt_or_journal(artifact, tmp_path, damage):
    f = construct(artifact, tmp_path)
    try:
        import_study(f, artifact)
        ctx = f.c.context(f.db, f.parent, 0, 1)
        if damage == 'drop':
            del ctx['evaluated_detached_study']
        elif damage == 'branch':
            ctx['branch'] = 'foreign-root'
            ctx.pop('evaluated_detached_study')
        else:
            shown = ctx['evaluated_detached_study']
            shown['cards'][0]['author']['kind'] = 'component_target_G'
            shown['context_sha256'] = canonical_sha({k:v for k,v in shown.items() if k != 'context_sha256'})
        before, events = f.c.state(), rows(f.log)
        with pytest.raises(ValueError):
            prepare(f, ctx)
        assert f.c.state() == before and rows(f.log) == events and f.model.calls == []
    finally:
        f.log.close()


def test_disabled_delivery_keeps_archived_source_dedup_and_no_forecast_credit(artifact, tmp_path):
    capsule = artifact['capsule']
    source = capsule['source_registry'][capsule['author']['candidate_source_sha256']]
    raw = response(builder.body_of(source), restart_predictions=spec())
    f = construct(artifact, tmp_path, llm=InertModel(raw), inject_lessons=False)
    try:
        import_study(f, artifact)
        snap = snapshot(f)
        assert 'evaluated_detached_study' not in snap.research_context
        def forbidden(source):
            pytest.fail('Archived candidate must not be evaluated again.')
        result = f.worker.run(snap, f.model, forbidden, base_temp=.2)
        assert result.reason == 'generation_repeated'
        assert len(f.model.calls) == 1 and f.worker.evaluations == 0
        assert f.c.knowledge.evaluated_detached_study == bundle(artifact)
        record, = f.c.knowledge.experiments
        assert record['score'] is record['actual_delta'] is None
        assert record['forecast_verdict'] == 'INCONCLUSIVE'
        events = rows(f.log)
        assert import_study(f, artifact) is False  # Repeated admission never replays paid/local work.
        assert rows(f.log) == events
        f.worker.assert_journal(f.log.path, f.log.anchor())
    finally:
        f.log.close()


@pytest.mark.parametrize('key', ['external_evaluated_study_sha256', 'external_evaluated_study_authority_sha256'])
def test_declared_config_mutation_and_missing_import_fail_before_work(artifact, tmp_path, key):
    f = construct(artifact, tmp_path)
    try:
        before, events = f.c.state(), rows(f.log)
        with pytest.raises(ValueError, match='not been imported'):
            snapshot(f)
        assert f.c.state() == before and rows(f.log) == events
        import_study(f, artifact)
        original = f.c.config[key]
        f.c.config[key] = '0' * 64
        for operation in (f.c.state, lambda: snapshot(f), lambda: f.c.complete_step(reflect=False)):
            with pytest.raises(ValueError, match='configured digests changed'):
                operation()
        f.c.config[key] = original
        assert f.c.attempts == f.c.aux_calls == 0 and f.model.calls == []
    finally:
        f.log.close()


@pytest.mark.parametrize('damage', ['authority', 'dependencies', 'missing'])
def test_checkpoint_cannot_self_authorize_replaced_evidence(artifact, tmp_path, damage):
    first, restored = construct(artifact, tmp_path/'first'), construct(artifact, tmp_path/'restored')
    try:
        import_study(first, artifact)
        state = first.c.state()
        if damage == 'authority':
            state['knowledge']['evaluated_detached_study']['author_authority']['issuer'] = 'forged-host'
        elif damage == 'dependencies':
            state['knowledge']['evaluated_detached_study']['author_dependencies']['design_text'] += ' '
        else:
            del state['knowledge']['evaluated_detached_study']
            # Missing at a truly blank checkpoint is allowed, but a state that
            # already has this declared study cannot discard it during restore.
            import_study(restored, artifact)
        before = restored.c.state()
        with pytest.raises(ValueError):
            restored.c.load_state(state)
        assert restored.c.state() == before
        assert restored.model.calls == []
    finally:
        first.log.close(); restored.log.close()


@pytest.mark.parametrize('sink', ['false', 'after_emit'])
def test_durable_import_failure_does_not_publish_or_replay(artifact, tmp_path, monkeypatch, sink):
    f = construct(artifact, tmp_path)
    try:
        before = f.c.state()
        original = f.log.anchor
        if sink == 'false':
            monkeypatch.setattr(f.log, 'research', lambda **kw: False)
        else:
            calls = []
            def anchor():
                calls.append(True)
                if len(calls) == 1:
                    return original()
                raise OSError('inert post-event persistence failure')
            monkeypatch.setattr(f.log, 'anchor', anchor)
        with pytest.raises((ValueError, OSError)):
            import_study(f, artifact)
        assert f.c.state() == before and f.c.knowledge.evaluated_detached_study is None
        if sink == 'after_emit':
            monkeypatch.setattr(f.log, 'anchor', original)
            assert rows(f.log)[-1]['phase'] == 'external_evaluated_study_import'
            with pytest.raises(ValueError, match='refusing replay'):
                import_study(f, artifact)
        assert f.model.calls == []
    finally:
        f.log.close()


@pytest.mark.parametrize('mutation', ['without_authority', 'without_study', 'null', 'bool'])
def test_config_requires_two_explicit_digest_pins(artifact, mutation):
    cfg = settings(artifact)
    if mutation == 'without_authority':
        del cfg['research']['external_evaluated_study_authority_sha256']
    elif mutation == 'without_study':
        del cfg['research']['external_evaluated_study_sha256']
    else:
        cfg['research']['external_evaluated_study_authority_sha256'] = None if mutation == 'null' else True
    with pytest.raises(ValueError):
        apply_defaults(cfg)




def test_real_engine_checkpoint_resume_keeps_study_and_new_G_delivery(tmp_path, monkeypatch):
    from sciel.engine import checkpoint
    from sciel.engine.loop import Engine
    from test_circle_intervention_engine import task_copy
    from test_optimizer_traced_engine import OutcomeDraws
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=builder.source_for())
    from evaluated_study_fixture import build
    from sciel.hypothesis.measurement import metric_context
    artifact = build(task_metric=metric_context(task))
    cfg = settings(artifact, max_experiments=1)
    bodies = ['return best_params.copy()\n', 'return initial_params.copy()\n']
    evaluated = []
    def evaluate(_task, path, **kwargs):
        source = Path(path).read_text()
        body = builder.body_of(source)
        assert body in [builder.DEFAULT_BODY, *bodies]
        evaluated.append(source)
        score = .5 if body == builder.DEFAULT_BODY else .4
        return CascadeOutcome(EvalResult(True, score, dict(sum_radii=score*artifact['capsule']['metric']['semantics']['denominator'],
            target_ratio=score, validity=1., combined_score=score), timing=.01), score, 'full', False)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    class Model(InertModel):
        weights = [1.]
        def choose_model(self, rng):
            return self.models[0]
    folder = tmp_path/'engine'
    first = Engine(cfg, task, str(folder), Model(response(bodies[0], restart_predictions=spec())), python=sys.executable)
    original_seed = first.seed
    blank = folder/'blank.json'
    def seed():
        original_seed()
        checkpoint.save(first, 'before_evaluated_study')
        blank.write_bytes((folder/'checkpoint.json').read_bytes())
        first.circle_generation.import_external_evaluated_study(artifact['capsule'],
            artifact['author_dependencies'], artifact['author_authority'],
            expected_sha256=artifact['capsule_sha256'], parent=first.db.get('p1'))
    first.seed = seed
    first._stop_at = 1
    first.run()
    assert first.research.knowledge.evaluated_detached_study == bundle(artifact)
    assert len(first.llm.calls) == 1 and len(draws.calls) == 9
    second = Engine(cfg, task, str(folder), Model(response(bodies[1], restart_predictions=spec())), python=sys.executable)
    checkpoint.load(second, str(folder/'checkpoint.json'))
    assert second.llm.calls == []
    second.run()
    assert second.research.knowledge.evaluated_detached_study == bundle(artifact)
    assert len(second.llm.calls) == 1 and len(draws.calls) == 15
    assert 'EVALUATED_DETACHED_STUDY_JSON:' in second.llm.calls[0]['prompt']
    assert second.research.attempts == second.research.outcomes == second.circle_generation.evaluations == 2
    assert second.research.aux_calls == second.research.aux_cost == 0
    assert second.research.knowledge.lessons == []
    assert evaluated == [builder.source_for(), *(builder.source_for(body) for body in bodies)]
    third = Engine(cfg, task, str(folder), Model(None), python=sys.executable)
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match='uncheckpointed research'):
            checkpoint.load(third, str(blank))
        checkpoint.load(third, str(folder/'checkpoint.json'))
        assert third.research.knowledge.evaluated_detached_study == bundle(artifact)
        assert third.llm.calls == []
    finally:
        third.repeat_promotion.close()
        third.log.close()

# These fixtures reconstruct the earlier generated study and synthetic sidecar
# from tracked bytes; the actual detached HOST study remains independently pinned.
from test_generated_paired_review_delivery import pilot_artifacts, component, review_sidecar


def test_mixed_annexes_share_budget_without_spending_hidden_review_reserve(
        artifact, component, pilot_artifacts, review_sidecar, tmp_path, monkeypatch):
    from test_generated_paired_review_delivery import policy_for
    from test_generated_paired_integration import construct as old_construct, import_component
    from sciel.knowledge_base.generated_paired_context import block as component_block
    from sciel.knowledge_base.evaluated_study_context import block as study_block
    from sciel.engine import research as research_module
    # Set an explicit admission budget: the authored fixtures are smaller than
    # the retired archive, but the complete HOST card must still be omitted.
    MAX_MEMORY_CHARS = 30000
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', MAX_MEMORY_CHARS)
    contexts = []
    for text_enabled in (True, False):
        f = old_construct(pilot_artifacts, component, tmp_path/str(text_enabled),
            external_component_review_sha256=canonical_sha(review_sidecar),
            external_evaluated_study_sha256=artifact['capsule_sha256'],
            external_evaluated_study_authority_sha256=artifact['author_authority_sha256'],
            comparison_reviews=True, comparison_review_version=3,
            comparison_review_text=text_enabled, comparison_review_text_policy=policy_for(review_sidecar, 'accepted'),
            batch_only_lessons=True)
        try:
            import_component(f, component)
            f.worker.import_external_component_review(review_sidecar, expected_sha256=canonical_sha(review_sidecar))
            import_study(f, artifact)
            ctx = f.c.context(f.db, f.parent, 0, 1)
            old_chars = len(component_block(ctx['generated_paired_history']))
            reserve = ctx['generated_paired_reviews']['reserved_chars']
            shown = ctx.get('evaluated_detached_study')
            assert len(study_block(shown)) + old_chars + reserve <= MAX_MEMORY_CHARS - 2048
            assert shown is not None and shown['presented_count'] == 0
            assert shown['omitted_count'] == 1 and shown['omissions'] and shown['cards'] == []
            assert shown['max_chars'] == min(16000, MAX_MEMORY_CHARS - old_chars - reserve - 2048)
            # Validation must rebuild the same allowance, not its old fixed16k projection.
            prepare(f, ctx)
            contexts.append(ctx)
            assert f.model.calls == []
        finally:
            f.log.close()
    assert contexts[0].get('evaluated_detached_study') == contexts[1].get('evaluated_detached_study')
    assert contexts[0]['generated_paired_reviews']['reserved_chars'] == contexts[1]['generated_paired_reviews']['reserved_chars']


def test_receiving_study_config_does_not_rewrite_historical_R_wire(artifact, pilot_artifacts, tmp_path):
    from sciel.engine.external_research_v2 import policy_identity
    ordinary = pilot_artifacts.gj('frozen/capsule.json')
    old_journals = canonical_json(ordinary['journals'])
    ordinary['policy'] = policy_identity()
    config = pilot_artifacts.gj('frozen/config.json')
    config = copy.deepcopy(config.get('research', config))
    config.pop('external_component_sha256', None)
    config.pop('external_component_review_sha256', None)
    config.update(external_history_sha256=canonical_sha(ordinary), max_aux_calls=0, reflection_interval=0)
    model = InertModel()
    model.models = [config['model']]
    f = construct(artifact, tmp_path, llm=model, **config)
    try:
        assert f.worker.import_external_history(ordinary, expected_sha256=canonical_sha(ordinary), parent=f.parent)
        old_records, old_conclusions = f.c.knowledge.experiments, copy.deepcopy(f.c.conclusions)
        assert old_records and old_conclusions
        assert import_study(f, artifact)
        assert f.c.knowledge.experiments == old_records and f.c.conclusions == old_conclusions
        assert canonical_json(ordinary['journals']) == old_journals
        assert f.c.attempts == f.c.outcomes == f.c.aux_calls == f.worker.evaluations == 0
        f.worker.assert_journal(f.log.path, f.log.anchor())
        assert 'evaluated_detached_study' in f.c.context(f.db, f.parent, 0, 1)
        assert f.model.calls == f.charges == []
    finally:
        f.log.close()


@pytest.mark.parametrize('damage', ['root_id', 'root_source'])
def test_import_requires_actual_matching_default_root(artifact, tmp_path, damage):
    f = construct(artifact, tmp_path)
    try:
        if damage == 'root_id':
            f.parent.id = 'unrelated-default-root'
        else:
            f.parent.code = builder.source_for('return best_params.copy()\n')
        before, events = f.c.state(), rows(f.log)
        with pytest.raises(ValueError, match='root'):
            import_study(f, artifact)
        assert f.c.state() == before and rows(f.log) == events
        assert f.model.calls == []
    finally:
        f.log.close()


def test_root_source_drift_is_rejected_before_context_and_begin(artifact, tmp_path):
    f = construct(artifact, tmp_path)
    try:
        import_study(f, artifact)
        ctx = f.c.context(f.db, f.parent, 0, 1)
        original_source = f.parent.code
        foreign_source = builder.source_for('return best_params.copy()\n')
        before, events = f.c.state(), rows(f.log)
        f.parent.code = foreign_source
        with pytest.raises(ValueError, match='root source'):
            snapshot(f)
        # A forged root snapshot also cannot bypass the authoritative ancestor
        # check by presenting its own internally consistent source/hash pair.
        from sciel.engine.research import digest
        ctx.update(parent_code=foreign_source, parent_source_sha256=digest(foreign_source))
        with pytest.raises(ValueError, match='root source'):
            prepare(f, ctx)
        f.parent.code = original_source
        assert f.c.state() == before and rows(f.log) == events and f.model.calls == []
    finally:
        f.log.close()
