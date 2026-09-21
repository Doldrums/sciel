"""A stale/tampered feedback snapshot must never enter a new model request."""
from copy import deepcopy

import pytest

from sciel.engine.db import Database
from sciel.engine.types import Snapshot
from sciel.hypothesis import circle_restart_body as body
from test_structured_auxiliary_flow import construct
from test_typed_mechanism_evidence import forecast


@pytest.mark.parametrize('damage', ['verdict', 'sign', 'interpretation', 'removed_card', 'budget', 'extra'])
def test_begin_rederives_feedback_before_any_attempt_or_dispatch(tmp_path, damage):
    f = construct(tmp_path, mechanism_entry=forecast(assessment='directly_supported'))
    f.research.complete_step()
    db = Database(1); source = body.source_for(); parent = db.add(source, .8, {}, 0, 'seed')
    snap = Snapshot(0, 'circle_generate', 'fake', 0, parent.id, source, .8)
    snap.research_context = f.research.context(db, parent, 0, 2)
    feedback = snap.research_context['conclusion_feedback']
    card = feedback['cards'][0]
    facts = card['mechanism_grounding']['forecasts'][0]
    if damage == 'verdict': facts['computed_verdict'] = 'REFUTED'
    elif damage == 'sign': facts['observed_comparison']['measurement']['direction'] = 'worsened'
    elif damage == 'interpretation': card['unverified_interpretation']['explanation'] = 'Invented explanation.'
    elif damage == 'removed_card': feedback['cards'] = []
    elif damage == 'budget': feedback['max_chars'] = 32001
    else: feedback['invented'] = True
    before = deepcopy(f.research.state()); events = deepcopy(f.events); calls = deepcopy(f.calls)
    with pytest.raises(ValueError):
        f.research.begin(snap, 0, 'circle_generate', 'Inert next request.',
            response_instruction=body.response_instruction(), context_scope=body.OPERATION)
    assert f.research.state() == before and f.events == events and f.calls == calls
