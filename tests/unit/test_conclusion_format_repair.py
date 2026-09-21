"""Known-complete C corrections across real research accounting; no provider calls."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.config import apply_defaults, load_config
from sciel.engine.llm.client import LLMError
from sciel.engine.research import ResearchController
from sciel.hypothesis import conclusion_refs
from test_mechanism_review import controller


def setup(items, **options):
    old, _, _, _ = controller()
    config = apply_defaults({'research': {'enabled': True, 'structured_auxiliary': True,
        'conclusion_source_refs': True, 'conclusion_repair_attempts': 1,
        'reflection_interval': 0, 'max_aux_calls': 10, **options}})['research']
    events, charges = [], []
    class Model:
        models, max_retries = ['fixture'], 1
        def __init__(self): self.calls = []
        def generate(self, prompt, model, **kwargs):
            self.calls.append((prompt, deepcopy(kwargs)))
            value = items[len(self.calls) - 1]
            if isinstance(value, Exception):
                raise value
            return NS(text=value, model=model, cost=.03, prompt_tokens=10,
                completion_tokens=10, finish_reason='stop', refusal=None)
    model = Model()
    c = ResearchController(config, model, NS(research=lambda **row: events.append(deepcopy(row))), charges.append)
    for record in old.knowledge.experiments:
        c.knowledge.record(deepcopy(record))
    c.attempts = c.outcomes = c.valid_hypotheses = 1
    c.queue = deepcopy(c.knowledge.experiments)
    return c, model, events, charges


def response(*, too_long=False, bad_ref=False, lessons=False):
    old, _, _, _ = controller()
    row = old.knowledge.experiments[0]
    catalog = conclusion_refs.catalog(row)
    entry = dict(claim_ref='invented' if bad_ref else catalog['claims'][0]['id'], source_refs=[],
        assessment='unknown', direct_effect='Not established.',
        untested_downstream='x' * 513 if too_long else 'A single measurement does not isolate a cause.',
        evidence_kind='unavailable')
    return json.dumps(dict(verdict=row['forecast_verdict'], explanation='Measured regression.',
        lessons=([dict(rule='This change regressed in the measured case.', polarity='negative',
            scope='branch', branch='p1', evidence_ids=[row['id']])] if lessons else []),
        mechanism_review=[entry]))


@pytest.mark.parametrize('invalid', [response(too_long=True), response(bad_ref=True), '{bad json'])
def test_invalid_complete_review_is_corrected_once_with_exact_diagnostic_and_both_charges(invalid):
    c, model, events, charges = setup([invalid, response(lessons=True)])
    before = deepcopy(c.knowledge.experiments)
    c.complete_step()
    assert c.conclusions['r000001a1']['status'] == 'parsed'
    assert len(c.knowledge.lessons) == 1
    assert len(model.calls) == c.aux_calls == 2 and charges == [.03, .03]
    assert c.aux_cost == pytest.approx(.06) and c.knowledge.experiments == before
    assert [r['phase'] for r in events].count('conclusion_rejected') == 1
    assert [r['phase'] for r in events].count('conclusion_repair') == 1
    assert [r['phase'] for r in events].count('conclusion') == 1
    correction = json.loads(model.calls[1][0].split('REJECTED_RESPONSE_JSON:\n')[1])
    assert correction['previous_response'] == invalid and correction['diagnostic']
    assert model.calls[0][1]['response_format'] == conclusion_refs.evidence_response_format(before[0], model='fixture')
    assert model.calls[1][1]['response_format'] == conclusion_refs.repair_response_format(before[0], model='fixture')
    resolution = next(r for r in events if r['phase'] == 'conclusion_reference_resolution')
    catalog = next(r for r in events if r['phase'] == 'conclusion_reference_catalog')
    assert resolution['catalog_sha256'] == catalog['catalog_sha256']
    restored, _, _, _ = setup([])
    restored.queue = []  # A newly constructed engine has no queued local fixture.
    restored.load_state(c.state())
    assert restored.state() == c.state()


def test_exhausted_repair_retains_measurement_without_lessons_or_extra_calls():
    c, model, events, charges = setup([response(too_long=True), response(bad_ref=True)])
    c.complete_step()
    assert c.conclusions['r000001a1']['status'] == 'invalid'
    assert not c.knowledge.lessons and c.knowledge.experiments[0]['score'] == .75
    assert len(model.calls) == 2 and charges == [.03, .03] and c.queue == []


def test_aux_limit_does_not_erase_first_rejected_response_or_dispatch_repair():
    c, model, events, charges = setup([response(too_long=True)], max_aux_calls=1)
    c.complete_step()
    assert c.conclusions['r000001a1']['status'] == 'invalid'
    assert len(model.calls) == 1 and charges == [.03]
    assert any(r['phase'] == 'aux_skipped' and r['reason'] == 'aux_call_limit' for r in events)


@pytest.mark.parametrize('failure', [RequestNotDispatched('budget'), LLMError(cause=ConnectionError())])
def test_failed_correction_preserves_charge_and_seals_queue_without_replay(failure):
    c, model, events, charges = setup([response(too_long=True), failure])
    with pytest.raises(CampaignStopped): c.complete_step()
    assert len(model.calls) == 2 and charges == [.03]
    assert c.queue == [] and c.conclusions == {} and not c.knowledge.lessons
    stop = next(r for r in events if r['phase'] == 'aux_stopped')
    assert stop['call_index'] == 2
    assert stop['current_call_cost_unknown'] is (not isinstance(failure, RequestNotDispatched))
    restored, fresh, _, _ = setup([])
    restored.queue = []
    restored.load_state(c.state())
    restored.complete_step()
    assert fresh.calls == []


def test_correction_disabled_is_legacy_single_invalid_terminal():
    c, model, events, charges = setup([response(too_long=True)], conclusion_repair_attempts=0)
    c.complete_step()
    assert len(model.calls) == 1 and charges == [.03]
    assert not any(r['phase'] == 'conclusion_repair' for r in events)


@pytest.mark.parametrize('option,value', [
    ('compact_lesson_memory', 1), ('conclusion_source_refs', 1),
    ('conclusion_repair_attempts', True), ('conclusion_repair_attempts', 2),
    ('conclusion_repair_attempts', -1)])
def test_bad_interface_options_rejected_in_factory_and_direct_controller(option, value):
    with pytest.raises(ValueError): apply_defaults({'research': {'enabled': True, option: value}})
    cfg = apply_defaults({'research': {'enabled': True}})['research']
    cfg[option] = value
    with pytest.raises(ValueError): ResearchController(cfg, NS(models=['fixture']), NS(), lambda _: None)


def test_live_profiles_enable_reliability_and_novelty_without_changing_old_checkpoints():
    for suffix in ('', '_rl_full', '_rl_presets_vq'):
        cfg = apply_defaults(load_config('configs/circle_live_gemini' + suffix + '.yaml'))
        assert cfg['candidate_novelty']['enabled']
        assert cfg['research']['compact_lesson_memory']
        assert cfg['research']['conclusion_source_refs']
        assert cfg['research']['conclusion_repair_attempts'] == 1
        if cfg.get('idea_two_stage', {}).get('enabled'):
            assert cfg['idea_two_stage']['card_repair_attempts'] == 1
