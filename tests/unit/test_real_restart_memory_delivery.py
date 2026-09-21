"""Full Research prompts from authored full-prefix traces, without numerical replay.

Incomplete receipts cannot authorize lessons. Separate inert observations supply
lessons, while state restore and prompt assembly retain all receipt provenance.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.prompt.build import build
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Program, Snapshot
from sciel.hypothesis.prompts import restart_block
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.memory_references import RULES as SHOWN_RULES, SCHEMA as SHOWN_SCHEMA
from sciel.knowledge_base.paired_comparisons import build_comparison
from sciel.knowledge_base.restart_context import PREFIX, expand_context

from restart_memory_fixtures import BRANCH, FIRST_SEED, legacy_dataset

RULE = 'INERT CONTROLLER FIXTURE ONLY: this synthetic ordinary source increased the fixture score.'
CLAIM = 'INERT CONTROLLER FIXTURE ONLY: this synthetic proposal forecasts an increase.'
PARENT_CODE = '# Inert controller fixture; never executed.\ndef run_packing():\n    return None\n'


def _parse(prompt, marker):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0]


@pytest.fixture(scope='module')
def historical():
    return legacy_dataset()


class ReflectionSink:
    models = ['fake']
    max_retries = 1

    def __init__(self):
        self.prompts = []
        self.conclusion_prompts = []

    def generate(self, prompt, model, **kwargs):
        # Only production complete_step auxiliary dispatches reach this fake.
        if 'Review existing experimental evidence and lessons.' in prompt:
            self.prompts.append(prompt)
            text = 'REFLECTION_JSON: {"lessons":[],"retire_ids":[]}'
        else:
            assert CLAIM in prompt and 'mechanism_review' in prompt
            self.conclusion_prompts.append(prompt)
            text = 'CONCLUSION_JSON: ' + json.dumps(dict(verdict='REFUTED',
                explanation='Inert fixture score fell below its stated forecast; no causal claim.',
                lessons=[], mechanism_review=[dict(claim_quote=CLAIM, source_quotes=[],
                    assessment='unknown', direct_effect='This fixture does not assess source semantics.',
                    untested_downstream='No real optimizer or candidate was run by this fixture.')]))
        return SimpleNamespace(text=text,
            model='fake', cost=0., prompt_tokens=1, completion_tokens=1)


def _controller(metric, *, inject=True):
    config = apply_defaults({'research': {'enabled': True, 'max_experiments': 8,
        'retrieve_k': 1, 'reflection_interval': 2, 'max_aux_calls': 2,
        'inject_lessons': inject}})['research']
    sink, events, charges = ReflectionSink(), [], []
    controller = ResearchController(config, sink,
        SimpleNamespace(research=lambda **event: events.append(copy.deepcopy(event))),
        charges.append, metric=metric)
    parent = Program(BRANCH, PARENT_CODE, .5, {}, 0, 'inert-test-parent')
    return controller, parent, SimpleNamespace(_all={parent.id: parent}), sink, events, charges


def _begin(controller, parent, db):
    context = controller.context(db, parent, 0, controller.attempts + 1)
    snapshot = Snapshot(0, 'diff', 'fake', controller.attempts, parent.id, parent.code,
        parent.score, research_context=context, kb_lessons=context['lesson_texts'],
        task_prompt='Inert prompt-assembly test; no candidate or numerical evaluation will run.')
    ident, prompt = controller.begin(snapshot, 0, 'diff', build('diff', snapshot))
    return ident, prompt, context


def _history(controller, parent, db):
    records = []
    for score in (.6, .4):
        ident, _, _ = _begin(controller, parent, db)
        source = f'# INERT ordinary fixture score={score}; never executed.\ndef run_packing():\n    return None\n'
        text = ('Inert fixture without a hypothesis.' if score > parent.score else
            'HYPOTHESIS_JSON: ' + json.dumps(dict(hypothesis=CLAIM, predicted_low=0., predicted_high=.2,
                reasoning='Explicit inert fixture; no real model or numerical execution.', confidence=.5)))
        controller.proposal(ident, SimpleNamespace(text=text,
            cost=0., prompt_tokens=0, completion_tokens=0), source, SimpleNamespace(code=source))
        records.append(controller.resolve(ident, status='evaluated', source=source, score=score,
            evaluation=dict(ok=True, score=score,
                cases={'sum_radii': score * 2.635, 'validity': 1., 'fixture_only': 1.},
                reached='full', confirmed=False, raw_first_score=score)))
    assert len(controller.queue) == 1
    controller.complete_step(reflect=False)
    assert not controller.queue and controller.reflected_through == 0
    assert controller.conclusions[records[1]['id']]['status'] == 'parsed'
    lesson, = controller.knowledge.add_lessons([dict(rule=RULE, polarity='positive',
        scope='branch', evidence_ids=[records[0]['id']])], allowed_evidence_ids=[records[0]['id']])
    return records, lesson


def _import_group(controller, original):
    # Import under a distinct task-bound ID without replacing receipt provenance.
    keys = ('batch_id', 'branch', 'origin_experiment_id', 'comparator_source_sha256',
        'challenger_source_sha256', 'adapter_identity', 'limits', 'pairs', 'source_pair')
    imported = build_comparison(**{k: copy.deepcopy(original[k]) for k in keys},
        namespace='test_saved_restart_memory:' + canonical_sha(original),
        task_metric=controller.metric, provenance={**original['provenance'],
            'test_context_only': True, 'original_group_sha256': canonical_sha(original),
            'metric_binding': 'authored task binding; no candidate execution'})
    for key in keys:
        assert canonical_json(imported[key]) == canonical_json(original[key])
    before = (controller.attempts, controller.outcomes, controller.aux_calls, controller.aux_cost)
    assert controller.import_comparison(imported) is True
    assert (controller.attempts, controller.outcomes, controller.aux_calls, controller.aux_cost) == before
    assert imported['derived'] == original['derived']
    return imported


def _assert_whole_pair(prompt, group, historical, name):
    from sciel.hypothesis import restart_observations as observations
    from sciel.hypothesis.restart_motion_observations import base_observation
    wire = _parse(prompt, PREFIX)
    expanded = expand_context(wire)
    assert wire['rendered_chars'] == len(canonical_json(wire)) <= 12000
    assert expanded['selected_count'] == 6
    assert expanded['groups'][0]['group_sha256'] == canonical_sha(group)
    for role in ('comparator', 'challenger'):
        card, = [row for row in expanded['cards'] if row['seed'] == FIRST_SEED and row['role'] == role]
        original = historical['views'][(name, FIRST_SEED, role)]
        restored_base = base_observation(card['observation_view']['observation'])
        assert canonical_json(restored_base) == canonical_json(original['observation'])
        assert canonical_json(observations.observation_view(restored_base)) == canonical_json(original)
        assert len(original['observation']['cases']) == 16
        assert card['source_sha256'] == group[role + '_source_sha256']
    coverage = expanded['cards'] + expanded['omissions']
    assert {(row['seed'], row['role']) for row in coverage} == {
        (pair['seed'], role) for pair in group['pairs'] for role in ('comparator', 'challenger')}
    assert next(row for row in coverage if row['seed'] == 1657835957 and row['role'] == 'challenger')['receipt_status'] == 'unknown'
    return wire


@pytest.mark.parametrize('name', ['pair_vs_control', 'pair_vs_single'])
def test_real_full_prefix_reaches_complete_prompt_and_survives_research_restore(historical, name):
    controller, parent, db, sink, events, charges = _controller(historical['metric'])
    records, lesson_id = _history(controller, parent, db)
    original = historical['groups'][name]
    original_bytes = canonical_json(original)
    group = _import_group(controller, original)
    with pytest.raises(ValueError):
        controller.knowledge.add_lessons([dict(rule='Must not mint a lesson from incomplete repeats.',
            polarity='negative', scope='branch', evidence_ids=[group['id']])],
            allowed_evidence_ids=[group['id']])
    assert len(controller.knowledge.lessons) == 1
    controller.complete_step()
    reflection, = sink.prompts
    _assert_whole_pair(reflection, group, historical, name)
    assert RULE in reflection and records[1]['id'] in reflection
    assert charges == [0., 0.] and controller.reflected_through == 2

    state = json.loads(canonical_json(controller.state()))
    restored, other_parent, other_db, other_sink, other_events, other_charges = _controller(historical['metric'])
    restored.load_state(state)
    assert canonical_json(restored.state()) == canonical_json(state)
    restored.complete_step()  # Restoring a completed reflection must not replay it.
    assert other_sink.prompts == other_sink.conclusion_prompts == other_charges == []
    first_id, first_prompt, context = _begin(controller, parent, db)
    second_id, second_prompt, restored_context = _begin(restored, other_parent, other_db)
    assert first_id == second_id and first_prompt == second_prompt
    assert canonical_json(context) == canonical_json(restored_context)
    wire = _assert_whole_pair(first_prompt, group, historical, name)
    assert context['injected_ids'] == [lesson_id]
    assert context['lesson_texts'][0] in first_prompt and RULE in first_prompt
    delivered = decode_evidence(_parse(first_prompt, 'RESEARCH_CONTEXT_JSON:\n'))
    assert records[1]['source_sha256'] in canonical_json(delivered['failed_hypotheses'])
    shown_rules = len(SHOWN_RULES) if context['paired_comparisons']['schema'] == SHOWN_SCHEMA else 0
    assert (context['lesson_presentation']['rendered_chars']
            + len(canonical_json(context['paired_comparisons'])) + shown_rules
            + (len(canonical_json(context['comparison_facts'])) if 'comparison_facts' in context else 0)
            + len(restart_block(wire))
            + (len(canonical_json(context['conclusion_feedback']))
               if 'conclusion_feedback' in context else 0)) <= 32000
    for journal in (events, other_events):
        shown = [row for row in journal if row['phase'] == 'restart_observation_presentation'
                 and row['role'] == 'generation'][-1]
        assert shown['prompt_sha256'] == digest(first_prompt)
        assert shown['context'] == wire and shown['delivery'] == 'prompt_prepared_not_model_use'
    for current in (controller, restored):
        assert canonical_json(current.knowledge.comparisons[0]) == canonical_json(group)
        lesson, = current.knowledge.lessons
        assert lesson['source_support_ids'] == [records[0]['id']]
        assert lesson['exposure_ids'] == []  # begin is only prepared intent, no dispatch.
        assert current.aux_calls == 2 and current.aux_cost == 0.
    assert canonical_json(original) == original_bytes


def test_real_history_ablation_keeps_receipts_without_trace_or_lesson_delivery(historical):
    controller, parent, db, sink, _, _ = _controller(historical['metric'], inject=False)
    _, lesson_id = _history(controller, parent, db)
    group = _import_group(controller, historical['groups']['pair_vs_control'])
    state = json.loads(canonical_json(controller.state()))
    restored, p2, db2, _, _, _ = _controller(historical['metric'], inject=False)
    restored.load_state(state)
    _, prompt, context = _begin(restored, p2, db2)
    assert PREFIX not in prompt and RULE not in prompt
    assert context['injected_ids'] == [] and context['lesson_texts'] == []
    assert context['lesson_presentation']['omitted'][0]['id'] == lesson_id
    assert context['lesson_presentation']['omitted'][0]['reason'] == 'ablation'
    assert context['failed_hypotheses']
    assert canonical_json(restored.knowledge.comparisons[0]) == canonical_json(group)
    assert sink.prompts == [] and len(sink.conclusion_prompts) == 1 and restored.aux_calls == 1
