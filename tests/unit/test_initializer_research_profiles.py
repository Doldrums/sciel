"""Named live witness profiles with a fake transport, never helper execution."""
import copy
from types import SimpleNamespace

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.config import apply_defaults
from sciel.engine.research import ResearchController
from sciel.hypothesis import circle_initializer_witness as witness
from sciel.knowledge_base.catalogs import canonical_json

from test_circle_generation_engine import build, config, response, rows
from test_initializer_research_flow import controller


def inert_transport(monkeypatch):
    requests = []

    def unavailable(request, *_args):
        requests.append(copy.deepcopy(request))
        raise OSError('inert test transport; no helper process started')

    monkeypatch.setattr(witness, '_launch', unavailable)
    return requests


def fresh_like(controller, profile):
    cfg = copy.deepcopy(controller.config)
    cfg['initializer_witness_profile'] = profile
    return ResearchController(cfg, controller.llm,
        SimpleNamespace(research=lambda **_event: None), lambda _cost: None,
        metric=controller.metric)


def test_profile_config_is_explicit_typed_and_does_not_enable_diagnostics():
    cfg = config()
    assert 'initializer_witness_profile' not in apply_defaults(cfg)['research']
    for profile in ('legacy_v1', 'unequal_radii_v1'):
        selected = copy.deepcopy(cfg)
        selected['research']['initializer_witness_profile'] = profile
        normalized = apply_defaults(selected)['research']
        assert normalized['initializer_witness_profile'] == profile
        assert not normalized.get('initializer_witness', False)
    for invalid in (None, True, 1, {}, '', 'unequal_radii_v2'):
        bad = copy.deepcopy(cfg)
        bad['research']['initializer_witness_profile'] = invalid
        with pytest.raises(ValueError, match='initializer_witness_profile'):
            apply_defaults(bad)
    c, _, _ = controller(enabled=False)
    with pytest.raises(ValueError, match='initializer_witness_profile'):
        fresh_like(c, True)  # Direct construction must validate too.


@pytest.mark.parametrize('profile', [None, 'unequal_radii_v1'])
def test_real_engine_uses_one_bound_profile_and_restores_without_replay(monkeypatch, tmp_path, profile):
    requests = inert_transport(monkeypatch)
    cfg = config()
    cfg['research']['initializer_witness'] = True
    if profile is not None:
        cfg['research']['initializer_witness_profile'] = profile
    engine = build(tmp_path/'run', [response()], cfg=cfg)
    result = engine.run()
    expected_name = profile or 'legacy_v1'
    fixtures = witness.make_fixture_bundle(profile=expected_name)
    assert len(requests) == 2  # Two planned sources; neither process was started.
    assert all(canonical_json(r['plan']['fixtures']) == canonical_json(fixtures) for r in requests)
    assert sum(len(r['plan']['fixtures']['cases']) for r in requests) == 32
    events = rows(engine)
    intent = next(e for e in events if e.get('phase') == 'initializer_witness_intent')
    receipt = next(e['receipt'] for e in events if e.get('phase') == 'initializer_witness_result')
    assert intent['helper_invocation_limit'] == 32
    assert canonical_json(receipt['plan']) == canonical_json(intent['plan'])
    assert receipt['status'] == 'incomplete'
    assert result['B_evals'] == 1 and len(engine.fixture_evaluated) == 2
    assert len(engine.llm.calls) == 1 and engine.research.aux_calls == 0
    record = engine.research.knowledge.experiments[0]
    assert record['status'] == 'evaluated' and record['score'] == .6
    saved = engine.research.state()
    identity = saved['initializer_witness']['identity']
    assert identity['fixture_profile'] == expected_name
    assert identity['fixture_sha256'] == fixtures['sha256']
    binding = engine.research.knowledge.initializer_evidence_for(record['id'])
    assert binding['evidence']['fixture_sha256'] == fixtures['sha256']
    assert all(s['status'] == 'incomplete' for s in binding['evidence']['sources'].values())
    # Engine.run closes its journal. Restore into a fresh controller/log, as a
    # supported continuation does, rather than writing through the closed run.
    restored_events = []
    restored = ResearchController(copy.deepcopy(engine.research.config), engine.llm,
        SimpleNamespace(research=lambda **event: restored_events.append(event)),
        lambda _cost: None, metric=engine.research.metric)
    restored.load_state(saved)
    assert restored.state() == saved
    restored.queue = copy.deepcopy(restored.knowledge.experiments)
    restored.complete_step()
    assert len(requests) == 2 and restored.knowledge.experiments == saved['knowledge']['experiments']
    assert not any(e['phase'] == 'initializer_witness_intent' for e in restored_events)


def test_switching_profile_cannot_resume_even_with_restamped_outer_config(monkeypatch):
    requests = inert_transport(monkeypatch)
    c, _, _ = controller(auxiliary=0, reflection=0)
    c.complete_step()
    saved = c.state()
    fresh = fresh_like(c, 'unequal_radii_v1')
    before = fresh.state()
    for restamp in ('none', 'config', 'config_and_identity'):
        changed = copy.deepcopy(saved)
        if restamp != 'none':
            changed['config'] = copy.deepcopy(fresh.config)
        if restamp == 'config_and_identity':
            changed['initializer_witness']['identity'] = copy.deepcopy(fresh._initializer_identity)
        with pytest.raises(ValueError, match='configuration|initializer'):
            fresh.load_state(changed)
        assert fresh.state() == before
    assert len(requests) == 2


def test_profile_mutation_after_construction_stops_before_transport_preserving_score(monkeypatch):
    requests = inert_transport(monkeypatch)
    c, model, _ = controller(auxiliary=0, reflection=0)
    before = c.knowledge.state()
    c.config['initializer_witness_profile'] = 'unequal_radii_v1'
    with pytest.raises(CampaignStopped, match='Initializer diagnostics stopped'):
        c.complete_step()
    assert requests == [] and model.prompts == []
    assert c.knowledge.state() == before
    assert c._initializer_attempts['r000001a1']['status'] == 'started'


def test_unequal_profile_rejected_generation_has_no_diagnostic_slot(monkeypatch, tmp_path):
    requests = inert_transport(monkeypatch)
    cfg = config()
    cfg['research'].update(initializer_witness=True, initializer_witness_profile='unequal_radii_v1')
    engine = build(tmp_path/'run', [response(target='return forbidden_name\n')], cfg=cfg)
    engine.run()
    saved = engine.research.state()
    assert requests == [] and len(engine.fixture_evaluated) == 1
    assert saved['outcomes'] == 1 and saved['initializer_witness']['attempts'] == {}
    engine.research.load_state(saved)
    assert engine.research.state() == saved
