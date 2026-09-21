"""Generation receives derived signs without confusing scopes or losing budgets."""
import copy

import pytest

from sciel.engine import research as research_module
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.comparison_facts import bounded_comparison_facts
from test_research_measurement_state import begin_fixture, circle_metric, controller, observe
from test_research_paired_comparisons import decoded_context, group, proposal


def prepared():
    c, parent, db, events = controller(metric=circle_metric())
    record, _ = observe(c, parent, db, .6)
    paired = group(record=record, radii=(.02, .005, .02))
    c.import_comparison(paired)
    return c, parent, db, events, paired


def test_next_generation_and_journal_receive_rederived_distinct_scopes():
    c, parent, db, events, paired = prepared()
    before = copy.deepcopy(c.knowledge.state())
    ctx = c.context(db, parent, 0, 2)
    ident, prompt = begin_fixture(c, ctx)
    sent = decoded_context(prompt)
    facts = sent['comparison_facts']
    assert facts == ctx['comparison_facts']
    assert facts['ordinary']['r000001a1']['controller_score_direction'] == 'positive'
    assert facts['paired'][paired['id']]['quality_outcome'] == 'mixed'
    assert [p['raw_physical_sum_direction'] for p in facts['paired'][paired['id']]['pairs']] == [
        'positive', 'negative', 'positive']
    attempt = next(e for e in events if e['phase'] == 'attempt' and e['experiment_id'] == ident)
    assert attempt['comparison_facts'] == facts
    assert c.knowledge.state() == before


@pytest.mark.parametrize('change', ['direction', 'scope', 'foreign_group', 'same_id_card', 'cap', 'boolean_cap', 'ablation'])
def test_forged_generation_facts_fail_before_attempt_or_event(change):
    c, parent, db, events, paired = prepared()
    ctx = c.context(db, parent, 0, 2)
    facts = ctx['comparison_facts']
    if change == 'direction':
        facts['paired'][paired['id']]['pairs'][1]['raw_physical_sum_direction'] = 'positive'
    elif change == 'scope':
        facts['ordinary']['r000001a1']['scope'] = 'paired_shared_seed_group'
    elif change == 'foreign_group':
        ctx['paired_comparisons']['cards'][0]['id'] = 'invented'
    elif change == 'same_id_card':
        block = ctx['paired_comparisons']
        card = block['cards'][0]
        card['derived']['pairs'][1]['raw_delta'] = 0.1
        card['view_sha256'] = canonical_sha({k: v for k, v in card.items() if k != 'view_sha256'})
        while block['rendered_chars'] != len(canonical_json(block)):
            block['rendered_chars'] = len(canonical_json(block))
    elif change == 'cap':
        facts['max_chars'] = 6001
    elif change == 'boolean_cap':
        facts['max_chars'] = True
    else:
        c.config['inject_lessons'] = False
    before, old_events = copy.deepcopy(c.state()), copy.deepcopy(events)
    with pytest.raises(ValueError, match='comparison facts'):
        begin_fixture(c, ctx)
    assert c.state() == before and events == old_events


def test_sign_block_consumes_memory_before_lessons_and_ablation_omits_it(monkeypatch):
    c, parent, db, _, paired = prepared()
    record = c.knowledge.experiments[0]
    lesson_id, = c.knowledge.add_lessons([proposal(record['id'])], allowed_evidence_ids=[record['id']])
    first = c.context(db, parent, 0, 2)
    pair_chars = len(canonical_json(first['paired_comparisons']))
    lesson_chars = len(c.lesson_text(c.knowledge.lessons[0]))
    # Enough for the lesson alone, but not for lesson plus explicit measurements.
    cap = pair_chars + lesson_chars
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', cap)
    ctx = c.context(db, parent, 0, 2)
    assert ctx['comparison_facts'] == bounded_comparison_facts(
        c._comparison_fact_records(ctx['branch'], ctx['paired_comparisons']),
        max_chars=min(6000, lesson_chars))
    assert lesson_id not in ctx['injected_ids']
    total = ctx['lesson_presentation']['rendered_chars'] + sum(
        len(canonical_json(ctx[key])) for key in ('paired_comparisons', 'comparison_facts',
            'proposal_groups', 'conclusion_feedback') if key in ctx)
    assert total <= cap
    c.config['inject_lessons'] = False
    assert 'comparison_facts' not in c.context(db, parent, 0, 2)
