"""Actual journal sources, not generation-parent guesses, reach paired memory."""
import copy
import json
from pathlib import Path

import pytest
from sciel.knowledge_base import paired_comparisons as paired
from sciel.knowledge_base.presentation import comparison_card
from sciel.hypothesis.rendering import decode_evidence
from test_repeat_engine_integration import (construct, draws, fixture_code,
    search_observations, task)
from test_repeat_comparison_integration import generation_prompts


def test_actual_paired_sources_and_changed_text_reach_next_generation(task,tmp_path,search_observations,draws):
    engine,llm=construct(task,tmp_path/'source-context');engine.run()
    groups=engine.research.knowledge.comparisons
    assert groups[0]['source_pair']=={'comparator':Path(task.initial_program_path).read_text(),
                                    'challenger':fixture_code(1)}
    assert groups[1]['source_pair']=={'comparator':fixture_code(1),'challenger':fixture_code(2)}
    card=comparison_card(groups[0]); assert card['source_context']['available'] is True
    raw,_=json.JSONDecoder().raw_decode(generation_prompts(llm)[1].split('RESEARCH_CONTEXT_JSON:\n',1)[1])
    shown=decode_evidence(raw)['paired_comparisons']['cards'][0]
    assert shown==card and shown['source_context']['changes']
    state=engine.research.state()
    assert engine._sync_repeat_comparisons()==0 and engine.research.state()==state


def test_large_source_copy_omission_does_not_discard_measurements(task,tmp_path,search_observations,draws,monkeypatch):
    monkeypatch.setattr(paired,'MAX_SOURCE_BYTES',1)
    engine,_=construct(task,tmp_path/'source-cap');engine.run()
    groups=engine.research.knowledge.comparisons
    assert len(groups)==2 and all('source_pair' not in g for g in groups)
    assert [g['derived']['outcome'] for g in groups]==['improved','regressed']
    assert all(g['provenance']['source_pair_omission']=='source_byte_limit' for g in groups)
    assert engine.research.attempts==engine.research.outcomes==2


def test_long_source_diff_keeps_numeric_evidence_when_generation_budget_is_small():
    from sciel.knowledge_base import KnowledgeBase
    from test_paired_source_context import build

    group = build('x = 1\n', 'x = 1\n# ' + 'a' * 8000 + '\n')
    kb = KnowledgeBase(); kb.record_comparison(group)
    saved = kb.state()
    whole = comparison_card(group)
    assert whole['source_context']['available']
    large = kb.comparison_context(group['branch'], max_chars=16000)
    assert large['cards'] == [whole]
    small = kb.comparison_context(group['branch'], max_chars=6000)
    shown = small['cards'][0]
    assert shown['source_context']['omission_reason'] == 'presentation_budget'
    assert shown['source_context']['changes'] == []
    assert {k:v for k,v in shown.items() if k not in ('source_context', 'view_sha256')} == {
        k:v for k,v in whole.items() if k not in ('source_context', 'view_sha256')}
    assert small['omitted_count'] == 0
    assert kb.state() == saved
    restored = KnowledgeBase(); restored.load_state(saved)
    assert restored.comparisons[0]['source_pair'] == group['source_pair']
