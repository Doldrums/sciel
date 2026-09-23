"""Final delivery, oracle resume refusal and trusted upstream no-cap goldens."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.parallel_calls import _encode, _decode, bind_visibility
from sciel.engine.prompt.build import build
from sciel.engine.research import ResearchController
from sciel.engine.llm.local import freeze_snapshot_inference
from sciel.adaptation.visibility import freeze_context, validate_capsule
from sciel.campaign.budget import RequestNotDispatched
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.knowledge_base import KnowledgeBase
from sciel.hypothesis.conclusion_feedback import reflection_review_context
from test_reflection_anchor_allocation import add, observe
from test_conclusion_feedback import inputs
from test_upstream_generation_budget import setup, configured
from test_upstream_local_port import server, ROOT
from upstream_native_fixture import provider


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def no_cap_inputs(provider, monkeypatch):
    llm, sdk, _, snap, _ = setup(provider, monkeypatch)
    snap.inspirations = [dict(program_id='foreign', code='LABEL="Δ世界"\n', score=.2)]
    cfg = configured(); cfg.pop('generation_context_tokens')
    events = []
    controller = ResearchController(cfg, llm, SimpleNamespace(
        research=lambda **row: events.append(deepcopy(row))), lambda _: None)
    return controller, snap, events


def selection_inputs():
    kb = KnowledgeBase()
    first = observe(kb, 1)
    lessons = [add(kb, 'Unverified Δ interpretation '+str(i), [first]) for i in range(2)]
    observe(kb, 2, delta=-.1, injected=lessons)
    for i in range(3, 8): observe(kb, i)
    return kb


def test_no_cap_generation_and_selection_match_trusted_3a8_golden(provider, monkeypatch):
    golden = json.loads((Path(__file__).parents[1]/'fixtures/upstream_native_nocap.json').read_bytes())
    controller, snap, events = no_cap_inputs(provider, monkeypatch)
    result = controller.begin(snap, 0, 'full_rewrite', build('full_rewrite', snap))
    assert digest({'result':result, 'state':controller.state(), 'events':events}) == golden['generation_sha256']
    kb = selection_inputs()
    before = deepcopy(kb.state())
    selected = [kb.reflection_context(4, selection_epoch=i) for i in range(3)]
    assert digest(selected) == golden['selection_sha256'] and kb.state() == before
    reviews, evidence = inputs(count=3)
    assert digest(reflection_review_context(reviews, evidence, list(evidence))) == golden['reviews_sha256']


def test_exact_final_research_prompt_is_bound_and_changed_wal_refuses(provider, monkeypatch):
    controller, snap, _ = no_cap_inputs(provider, monkeypatch)
    llm = provider[0]
    freeze_snapshot_inference(llm, snap)
    parent = SimpleNamespace(code=snap.parent_code, id=snap.parent_id, score=snap.parent_score, origin='seed')
    engine = SimpleNamespace(llm=llm, cfg={}, observed_changes=SimpleNamespace(block=lambda _:''),
        log=SimpleNamespace(run_id='public-fixture'), task=SimpleNamespace(task_hash='a'*64))
    freeze_context(engine, snap, parent)
    _, prompt = controller.begin(snap, 0, 'full_rewrite', build('full_rewrite', snap))
    from sciel.engine.llm.local import generation_options
    options = generation_options(llm, snap, prompt=prompt, op='full_rewrite')
    bound = bind_visibility(prompt, options)
    capsule = validate_capsule(bound['adaptation_visibility'], prompt=prompt, inference_spec=snap.inference_spec)
    assert capsule['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
    assert snap.adaptation_delivery['prompt_sha256'] == capsule['prompt_sha256']
    assert any(ref['kind']=='research_context' for ref in capsule['delivered_refs'])
    with pytest.raises(RequestNotDispatched): bind_visibility(prompt+'changed', bound)
    provider[1].chat.completions.create.assert_not_called()


def test_oracle_host_evidence_roundtrips_without_entering_candidate_metrics():
    evidence = {'host':{'retained':'opaque fixture'}}
    outcome = CascadeOutcome(EvalResult(True, .5, {'ordinary':'metric'}), .5, 'confirm',
        confirmed=True, oracle_evidence=evidence)
    raw = _encode('evaluation', outcome)
    assert 'oracle_evidence' not in raw['result']
    assert _decode('evaluation', raw).oracle_evidence == evidence
    assert _decode('evaluation', raw).result.to_json() == outcome.result.to_json()


@pytest.mark.parametrize('change', ['add', 'drop', 'rebind', 'stripped-identity'])
def test_force_cannot_add_drop_or_rebind_oracle_identity(server, tmp_path, change):
    engine = build_engine(deepcopy(server.config), str(ROOT/'tasks'), str(tmp_path))
    try:
        # The object stands for already validated constructor authority; this
        # test exercises checkpoint identity/force guards, not oracle validity.
        authority = {'schema':'host-launch-fixture/1', 'panel':'train-0'}
        if change != 'add':
            engine.oracle_evaluation = deepcopy(authority)
            engine.cfg['oracle_evaluation'] = {'schema':'host-config-fixture/1'}
        path = Path(checkpoint.save(engine, 'oracle-authority'))
        if change == 'add': engine.oracle_evaluation = deepcopy(authority)
        elif change == 'drop': engine.oracle_evaluation = None
        elif change == 'rebind': engine.oracle_evaluation = {**authority, 'panel':'train-1'}
        else:
            saved = json.loads(path.read_bytes())
            del saved['identity']['oracle_evaluation']
            path.write_text(json.dumps(saved))
            engine.oracle_evaluation = None
            del engine.cfg['oracle_evaluation']
        with pytest.raises(checkpoint.IdentityMismatch, match='oracle'):
            checkpoint.load(engine, str(path), allow_identity_mismatch=True)
        assert not server.calls
    finally: engine.close()
