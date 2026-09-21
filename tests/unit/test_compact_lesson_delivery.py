"""The actual context -> Snapshot -> prompt -> attempt path, without a model."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.engine.research import ResearchController
from sciel.engine.types import Program, Snapshot
from sciel.engine.prompt.build import build
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.decision import DECISION_CARD_SCHEMA, resolve_decision_evidence
from test_compact_lesson_decision import fixture


def delivery_fixture():
    original = fixture()
    kb = KnowledgeBase(max_experiments=500)
    for index, record in enumerate(original.experiments):
        # This fixture isolates lesson delivery from the separately retained
        # successful-source fragment channel, and uses controller metric identity.
        record.update(metric=metric_context(), parent_cases=None, evaluation=None, code_fragment='')
        kb.record(record)
        if index == 0:
            old = original.lessons[0]
            kb.add_lessons([{key: old[key] for key in ('rule', 'polarity', 'scope', 'evidence_ids')}],
                           allowed_evidence_ids=old['evidence_ids'])
    return kb


def controller(kb, *, compact, guarded=False, enabled=True):
    events = []
    config = dict(model=None, max_lessons=kb.max_lessons, max_experiments=kb.max_experiments,
        max_fragments=kb.max_fragments, retrieve_k=3, max_aux_calls=0, reflection_interval=0,
        inject_lessons=enabled, **({'compact_lesson_memory': True} if compact else {}),
        **({'lesson_selection': {'mode': 'source_diverse_guarded'}} if guarded else {}))
    llm = SimpleNamespace(models=['fixture'], max_retries=1,
                          generate=lambda *a, **k: pytest.fail('no model calls'))
    instance = ResearchController(config, llm, SimpleNamespace(research=lambda **e: events.append(e)),
                                  lambda cost: pytest.fail('no model cost'))
    instance.knowledge = kb
    instance.attempts = instance.outcomes = len(kb.experiments)
    parent = Program('p1', 'def solve():\n    return 1\n', .5, {}, 0, 'seed')
    return instance, SimpleNamespace(_all={'p1': parent}), parent, events


@pytest.mark.parametrize('guarded', [False, True])
def test_large_retained_lesson_reaches_actual_prompt_and_manifest_stays_audit_only(monkeypatch, guarded):
    monkeypatch.setattr('sciel.engine.research.MAX_MEMORY_CHARS', 16000)
    kb = delivery_fixture()
    before = canonical_json(kb.state())
    old, db, parent, _ = controller(kb, compact=False, guarded=guarded)
    historical = old.context(db, parent, 0, 10)
    assert historical['injected_ids'] == []
    assert historical['lesson_presentation']['omitted'][0]['reason'] == 'card_char_limit'
    assert 'projection' not in historical['lesson_presentation']
    current, db, parent, events = controller(kb, compact=True, guarded=guarded)
    ctx = current.context(db, parent, 0, 10)
    ident = kb.lessons[0]['id']
    assert ctx['injected_ids'] == [ident]
    presentation = ctx['lesson_presentation']
    assert presentation['selected_count'] == presentation['stored_active_count'] == 1
    assert presentation['projection'] == DECISION_CARD_SCHEMA
    assert presentation['omitted'] == []
    assert presentation['rendered_chars'] == sum(map(len, ctx['lesson_texts'])) < 16000
    card = json.loads(ctx['lesson_texts'][0].split('RESEARCH_LESSON_DECISION_JSON:\n', 1)[1])
    manifest, = presentation['evidence_manifests']
    assert canonical_sha(manifest) == card['evidence_ref']['manifest_sha256']
    assert resolve_decision_evidence(card, kb.lessons[0])['manifest'] == manifest
    assert len(manifest['entries']) == 5
    assert canonical_json(kb.state()) == before
    if guarded:
        assert presentation['selection']['decision_presentation'] == DECISION_CARD_SCHEMA
        assert presentation['selection']['source_fact_scope'] == 'shown_decision_source_rows_only'
    snapshot = Snapshot(0, 'diff', 'fixture', 9, parent.id, parent.code, parent.score,
        research_context=ctx, kb_lessons=ctx['lesson_texts'])
    base_prompt = build('diff', snapshot)
    experiment_id, prompt = current.begin(snapshot, 0, 'diff', base_prompt)
    assert ctx['lesson_texts'][0] in prompt
    assert card['unverified_interpretation'] in prompt
    assert 'FULL_DIFF_MARKER' not in prompt
    assert canonical_json(manifest) not in prompt
    assert '"entries":' not in prompt
    attempt = next(event for event in events if event['phase'] == 'attempt')
    assert attempt['experiment_id'] == experiment_id
    assert attempt['lesson_presentation'] == presentation
    assert attempt['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
    assert current.pending[experiment_id]['injected_ids'] == [ident]
    assert canonical_json(kb.state()) == before


@pytest.mark.parametrize('guarded', [False, True])
def test_compact_state_roundtrip_and_disabled_injection_do_not_change_evidence(guarded):
    kb = delivery_fixture()
    current, db, parent, _ = controller(kb, compact=True, guarded=guarded)
    before = current.state()
    restored, _, _, _ = controller(kb, compact=True, guarded=guarded)
    restored.load_state(deepcopy(before))
    assert restored.state() == before
    assert restored.context(db, parent, 0, 10) == current.context(db, parent, 0, 10)
    damaged = deepcopy(before)
    del damaged['presentation']['compact_lesson_memory']
    with pytest.raises(ValueError, match='presentation mismatch'):
        restored.load_state(damaged)
    ablated, db, parent, _ = controller(kb, compact=True, guarded=guarded, enabled=False)
    context = ablated.context(db, parent, 0, 10)
    assert context['injected_ids'] == [] and context['lesson_texts'] == []
    assert context['lesson_presentation']['evidence_manifests'] == []
    assert context['lesson_presentation']['selected_count'] == 0
    assert {row['reason'] for row in context['lesson_presentation']['omitted']} == {'ablation'}
    assert kb.state() == before['knowledge']
