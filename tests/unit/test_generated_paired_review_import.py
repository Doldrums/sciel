"""Real public import lifecycle with inert, explicitly synthetic R bytes."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine.circle_generation import CircleGeneration
from sciel.engine.generated_paired_reflection import GeneratedPairedReflectionJob
from sciel.engine import generated_paired_review_import as admission
from sciel.engine.research import ResearchController
from sciel.engine.types import Program
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.run_log import RunLog

import test_generated_paired_history as synthetic
from test_generated_paired_review import proposal


class NoCalls:
    models = ['fixture/model']
    max_retries = 1

    def generate(self, *_args, **_kwargs):
        pytest.fail('Import lifecycle cannot call a provider.')


@pytest.fixture(scope='module')
def saved():
    component = synthetic.capsule.__wrapped__()
    config = dict(external_component_sha256=canonical_sha(component),
        restart_predictions=True, structured_auxiliary=True, comparison_reviews=True,
        comparison_review_version=3, inject_lessons=True, max_aux_calls=1,
        model=NoCalls.models[0], batch_only_lessons=True, comparison_review_text=True,
        comparison_review_text_policy={'schema': 'sciel-review-text-policy/1', 'decisions': []})
    job = GeneratedPairedReflectionJob(component, expected_sha256=canonical_sha(component), config=config)
    plan = job.prepare(run_id='inert-new-review', segment_id='inert-new-review-segment')
    response = dict(text=json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=[proposal(component)])),
        model=NoCalls.models[0], cost=.002, prompt_tokens=100, completion_tokens=100,
        finish_reason='stop', refusal=None)
    sidecar = job.admit(plan, response, manifest_sha256='a' * 64,
                       journal_prefix=plan['journal_prefix'])['extended_capsule']
    return NS(component=component, sidecar=sidecar, config=config)


def construct(saved, path):
    from test_restart_prediction_engine import config as base_config
    config = base_config()
    config['research'] = dict(saved.config, enabled=True,
        external_component_review_sha256=canonical_sha(saved.sidecar))
    cfg = apply_defaults(config)['research']
    log = RunLog(str(path / 'events.jsonl'))
    log.run_start(scope='inert paired review import test')
    charges = []
    c = ResearchController(cfg, NoCalls(), log, charges.append, metric=saved.component['metric'])
    parent = Program(id='p1', code=synthetic.history.builder.source_for(), score=.5,
                     cases={}, island=0, origin='fixture')
    return NS(c=c, worker=CircleGeneration(c, execution_profile='paired_restart_trace/2'),
              log=log, charges=charges, parent=parent)


def import_base(f, saved):
    return f.worker.import_external_component(saved.component,
        expected_sha256=canonical_sha(saved.component), parent=f.parent)


def import_review(f, saved):
    return f.worker.import_external_component_review(saved.sidecar,
        expected_sha256=canonical_sha(saved.sidecar))


def rows(f):
    return [json.loads(line) for line in Path(f.log.path).read_text().splitlines()]


def test_public_import_is_durable_idempotent_and_never_counts_historical_r_as_local(saved, tmp_path):
    f = construct(saved, tmp_path)
    before = canonical_json(f.c.state())
    with pytest.raises(ValueError, match='source component first'):
        import_review(f, saved)
    assert canonical_json(f.c.state()) == before
    assert import_base(f, saved)
    assert import_review(f, saved)
    state, journal = f.c.state(), Path(f.log.path).read_bytes()
    assert not import_review(f, saved)
    assert f.c.state() == state and Path(f.log.path).read_bytes() == journal
    assert f.c.knowledge.generated_paired_review == saved.sidecar
    assert f.c.knowledge._evidence() == {} and f.c.knowledge.lessons == []
    assert all(getattr(f.c, key) == 0 for key in
        ('attempts', 'outcomes', 'aux_calls', 'aux_cost', 'reflections', 'reflected_through'))
    assert f.charges == [] and f.worker.known_cost == 0
    events = [row for row in rows(f) if row.get('phase') == admission.EVENT]
    assert len(events) == 1
    assert events[0]['sidecar_sha256'] == canonical_sha(saved.sidecar)
    assert events[0]['review_cards_sha256'] == canonical_sha(saved.sidecar['review_cards'])
    assert type(events[0]['new_outcomes']) is int and events[0]['new_outcomes'] == 0
    admission.assert_import_journal(f.c, rows(f))


def test_declared_missing_review_blocks_auxiliary_before_any_mutation(saved, tmp_path):
    f = construct(saved, tmp_path)
    import_base(f, saved)
    before, journal = f.c.state(), Path(f.log.path).read_bytes()
    with pytest.raises(ValueError, match='review has not been imported'):
        f.c.complete_step(reflect=False)
    assert f.c.state() == before and Path(f.log.path).read_bytes() == journal
    assert f.charges == []


@pytest.mark.parametrize('damage', ['no_write', 'raise'])
def test_failed_durable_event_never_publishes_detached_review(saved, tmp_path, monkeypatch, damage):
    f = construct(saved, tmp_path)
    import_base(f, saved)
    before, journal = f.c.knowledge.state(), Path(f.log.path).read_bytes()
    def emit(*_args, **_kwargs):
        if damage == 'raise':
            raise OSError('inert durable write failure')
    monkeypatch.setattr(f.c, 'emit', emit)
    with pytest.raises((ValueError, OSError)):
        import_review(f, saved)
    assert f.c.knowledge.state() == before and Path(f.log.path).read_bytes() == journal
    assert f.c.knowledge.generated_paired_review is None and f.charges == []


def test_preexisting_local_work_cannot_be_hidden_by_late_review_import(saved, tmp_path):
    f = construct(saved, tmp_path)
    import_base(f, saved)
    f.c.attempts = 1
    before, journal = f.c.knowledge.state(), Path(f.log.path).read_bytes()
    with pytest.raises(ValueError, match='precede local research'):
        import_review(f, saved)
    assert f.c.knowledge.state() == before and Path(f.log.path).read_bytes() == journal


@pytest.mark.parametrize('damage', ['duplicate', 'bool_counter', 'wrong_order', 'raw_sha', 'local_before'])
def test_reanchoring_does_not_make_forged_import_provenance_valid(saved, tmp_path, damage):
    f = construct(saved, tmp_path)
    import_base(f, saved); import_review(f, saved)
    events = rows(f)
    index = next(i for i, row in enumerate(events) if row.get('phase') == admission.EVENT)
    if damage == 'duplicate': events.append(copy.deepcopy(events[index]))
    elif damage == 'bool_counter': events[index]['new_outcomes'] = False
    elif damage == 'raw_sha': events[index]['raw_response_sha256'] = '0' * 64
    elif damage == 'wrong_order':
        events.insert(1, events.pop(index))
    else: events.insert(index, {'event': 'research', 'phase': 'aux_start'})
    with pytest.raises(ValueError):
        admission.assert_import_journal(f.c, events)


def test_config_and_host_text_arm_are_frozen_before_import(saved, tmp_path):
    f = construct(saved, tmp_path)
    import_base(f, saved)
    before, journal = f.c.knowledge.state(), Path(f.log.path).read_bytes()
    f.c.config['comparison_review_text'] = False
    with pytest.raises(ValueError, match='ablation mode changed'):
        import_review(f, saved)
    assert f.c.knowledge.state() == before and Path(f.log.path).read_bytes() == journal


def test_checkpoint_reconstructs_sidecar_and_refuses_erasure_after_import(saved, tmp_path):
    f = construct(saved, tmp_path / 'original')
    import_base(f, saved); import_review(f, saved)
    state = f.c.state()
    receiver = construct(saved, tmp_path / 'receiver')
    receiver.c.load_state(copy.deepcopy(state))
    assert receiver.c.knowledge.generated_paired_review == saved.sidecar
    erased = copy.deepcopy(state)
    del erased['knowledge']['generated_paired_review']
    before = receiver.c.knowledge.state()
    with pytest.raises(ValueError, match='removed declared review'):
        receiver.c.load_state(erased)
    assert receiver.c.knowledge.state() == before
    # Exercise the new helper's independent local-work boundary directly,
    # without inventing an inconsistent complete ResearchController snapshot.
    from sciel.knowledge_base.base import KnowledgeBase
    detached = KnowledgeBase(**erased['knowledge']['limits'])
    detached.load_state(erased['knowledge'])
    fresh = construct(saved, tmp_path / 'fresh')
    with pytest.raises(ValueError, match='removed declared review'):
        admission.validate_loaded(fresh.c, detached, dict(erased, attempts=1))
