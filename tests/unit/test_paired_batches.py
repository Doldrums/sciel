"""Explicit comparator durability with cheap saved geometry; never a solver."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from sciel.engine.paired_batches import CONTRACT, PairedBatches
from sciel.engine.repeat_journal import digest
from sciel.engine.repeat_promotion import RepeatPromotionError
from sciel.hypothesis.measurement import metric_context
from tests.unit.test_paired_comparisons import receipt as geometry_receipt
from tests.unit.test_repeat_promotion import sha

INITIAL, FIRST, SECOND = "# bootstrap\n", "# first\n", "# second\n"
ADAPTER = {"schema": "sciel-repeat-adapter/1", "fixture": "saved geometry only"}


def config(**changes):
    cfg = dict(enabled=True, seed_master=31, max_comparisons=2, max_draw_attempts=15,
               max_wall_seconds=600, timeout_seconds=30, certification_seconds=10,
               cpu_seconds=35, mem_bytes=2**31)
    return dict(cfg, **changes)


def identity():
    task = SimpleNamespace(qualified_name="oe/circle_packing", task_hash="a"*64,
                           primary_metric="combined_score", research_metric=None)
    return dict(contract=CONTRACT, task=task.qualified_name, task_hash=task.task_hash,
                canonical_initial_source_sha256=sha(INITIAL), task_evaluator_sha256="b"*64,
                adapter=deepcopy(ADAPTER), task_metric=metric_context(task))


def context(child=FIRST, comparator=INITIAL, *, parent=INITIAL, parent_id="p0", comparator_id="p0"):
    active = list(dict.fromkeys([parent_id, comparator_id]))
    snapshot = dict(schema="sciel-repeat-population-selection/1", selection_id="s000001", island=0,
        parent_id=parent_id, parent_source_sha256=sha(parent), comparator_program_id=comparator_id,
        comparator_source_sha256=sha(comparator), victim_id=None, active_ids=active,
        evidence_heads={i: "d"*64 for i in active}, distribution=[dict(program_id=i,
            source_sha256=sha(parent if i == parent_id else comparator), evidence_id="d"*64,
            rank=1., probability=1/len(active)) for i in active], draw=.2, parent_propensity=1/len(active))
    return dict(branch="root-ancestry", origin_experiment_id="r000001a1", parent_id=parent_id,
        parent_source_sha256=sha(parent), comparator_program_id=comparator_id,
        comparator_source_sha256=sha(comparator), source_sha256=sha(child), prospective_snapshot=snapshot)


class Draws:
    def __init__(self, effects=None, radii=None):
        self.calls = []
        self.effects = effects or {}
        self.radii = radii or {INITIAL: .1, FIRST: .2, SECOND: .25}

    def __call__(self, source, *, seed, **limits):
        self.calls.append((source, seed))
        effect = self.effects.get(len(self.calls), {})
        if isinstance(effect, BaseException):
            raise effect
        radius = effect.get("radius", self.radii[source])
        result = geometry_receipt(sha(source), seed, radius, x=effect.get("x", .5),
                                  status=effect.get("status", "valid"))
        result["adapter_identity"] = deepcopy(ADAPTER)
        result["limits"] = dict(candidate_wall_seconds=limits["timeout"],
            certification_wall_seconds=limits["certification_seconds"], candidate_cpu_seconds=limits["cpu_seconds"],
            certification_cpu_seconds=limits["cpu_seconds"], candidate_mem_bytes=limits["mem_bytes"],
            certification_mem_bytes=limits["mem_bytes"])
        if effect.get("tamper_geometry"):
            result["capture"]["centers"][0] = [-10., -10.]
        return result


@pytest.fixture
def make(tmp_path):
    instances = []
    def factory(draw=None, cfg=None, ident=None):
        item = PairedBatches(tmp_path, cfg or config(), ident or identity(), draw or Draws())
        instances.append(item)
        return item
    yield factory
    for item in instances:
        item.close()


def ack(item):
    state = item.state()
    item.checkpoint_ack(state["ledger_anchor"])
    return state


def bootstrap(item):
    item.start("fixture-run")
    assert item.plan_bootstrap(INITIAL)
    saved = ack(item)
    item.run_pending()
    return saved


def comparison(item, child=FIRST, comparator=INITIAL, **ctx):
    assert item.plan_comparison(child, comparator, "child-"+sha(child)[:8],
                                context=context(child, comparator, **ctx))
    return ack(item)


def test_explicit_comparator_can_differ_from_parent_and_global_publication(make):
    draw = Draws()
    item = make(draw)
    bootstrap(item)
    published = item.export()["confirmed_output_incumbent"]
    comparison(item)
    item.run_pending()
    assert item.terminal_batches()[1]["decision"]["result"]["status"] == "promote"
    assert item.export()["confirmed_output_incumbent"] == published
    comparison(item, SECOND, FIRST, parent=INITIAL, parent_id="p0", comparator_id="p1")
    item.run_pending()
    packet = item.terminal_batches()[-1]
    assert packet["source_pair"] == {"comparator": FIRST, "challenger": SECOND}
    assert packet["plan"]["context"]["parent_source_sha256"] == sha(INITIAL)
    assert packet["plan"]["incumbent"] == sha(FIRST)
    assert packet["decision"]["result"]["status"] == "promote"
    assert item.export()["confirmed_output_incumbent"] == published
    assert len(draw.calls) == 15 and not item.can_compare()
    assert packet["task_metric"] == identity()["task_metric"]
    # Every pair is freshly executed, with prescribed alternating roles.
    rows = packet["receipts"]
    assert [r["role"] for r in rows] == ["incumbent", "challenger", "challenger", "incumbent", "incumbent", "challenger"]
    assert all(rows[i]["seed"] == rows[i+1]["seed"] for i in (0, 2, 4))
    assert set(r["seed"] for r in rows).isdisjoint(s for _, s in draw.calls[:9])
    export = item.terminal_comparisons()[-1]
    assert export["research_context"]["branch"] == "root-ancestry"
    assert export["source_pair"] == packet["source_pair"]
    assert export["provenance"]["plan"] == packet["plan"]


def test_no_ordinary_score_screen_and_only_one_copy_of_each_source(make):
    item = make(); bootstrap(item)
    # The service never accepts an ordinary numeric score as a quality gate.
    assert item.plan_comparison(FIRST, INITIAL, "low-native-score", context=context())
    ack(item); item.run_pending()
    before = item.state()
    assert not item.plan_comparison(FIRST, INITIAL, "different-id", context=context())
    assert not item.plan_comparison(INITIAL, FIRST, "bootstrap-again", context=context(INITIAL, FIRST, parent=FIRST))
    assert item.state() == before
    with pytest.raises(RepeatPromotionError, match="explicit comparator"):
        item.plan_candidate(SECOND, {"validity": 1, "sum_radii": 99}, "p3")


@pytest.mark.parametrize("field,value", [("max_comparisons",0), ("max_draw_attempts",8), ("max_wall_seconds",359)])
def test_complete_reserve_preflight_before_generation_and_late_budget_failclosed(make, field, value):
    changes = {field:value}
    if field == "max_comparisons": changes["max_draw_attempts"] = 3
    draw=Draws();item=make(draw, config(**changes));bootstrap(item)
    before=item.state()
    assert not item.can_compare() and item.state()==before and len(draw.calls)==3
    assert not item.plan_comparison(FIRST, INITIAL, "p1", context=context())
    assert item.stopped_reason=="comparison_budget_unavailable"
    assert item.state()["data"]["budget_unavailable"][0]["context"]==context()
    assert item.export()["confirmed_output_incumbent"]["source_sha256"]==sha(INITIAL)


@pytest.mark.parametrize("role,stopped", [("incumbent","comparator_failed"), ("challenger",None)])
def test_candidate_failures_preserve_full_six_slots_but_comparator_failure_stops(make, role, stopped):
    # In sequence1 the challenger runs first, comparator second.
    draw=Draws({5 if role=="incumbent" else 4:{"status":"candidate_failure"}})
    item=make(draw);bootstrap(item);comparison(item);item.run_pending()
    packet=item.terminal_batches()[-1]
    assert len(draw.calls)==9 and len(packet["receipts"])==6
    assert item.stopped_reason==stopped and packet["decision"]["result"]["status"]=="incomplete"
    assert sum(r["receipt"]["status"]=="candidate_failure" for r in packet["receipts"])==1
    assert item.export()["confirmed_output_incumbent"]["source_sha256"]==sha(INITIAL)
    assert item.can_compare() is (stopped is None)


@pytest.mark.parametrize("effect", [{"status":"infrastructure"}, {"tamper_geometry":True}, RuntimeError("fixture")])
def test_infrastructure_or_invalid_geometry_stops_with_null_unrun_slots(make, effect):
    draw=Draws({4:effect});item=make(draw);bootstrap(item);comparison(item);item.run_pending()
    packet=item.terminal_batches()[-1]
    assert len(draw.calls)==4 and item.stopped_reason=="infrastructure"
    assert packet["receipts"][0]["receipt"]["status"]=="infrastructure"
    assert all(r["receipt"] is None for r in packet["receipts"][1:])
    assert packet["decision"]["result"]["raw_deltas"] is None


def test_no_mean_rescue_after_one_pair_loss(make):
    draw=Draws({4:{"radius":.09}});item=make(draw);bootstrap(item);comparison(item);item.run_pending()
    decision=item.terminal_batches()[-1]["decision"]
    assert len(draw.calls)==9 and decision["result"]["status"]=="reject"
    assert decision["result"]["reason"]=="observed_loss" and item.stopped_reason is None


def test_raw_gain_exact_loss_never_admits(make):
    # Raw radius .10000005 at x=.1 is tolerance valid but cert shrinks to .1;
    # comparator .10000002 in the interior has a larger exact radius.
    draw=Draws({4:{"radius":.10000005,"x":.1},5:{"radius":.10000002}})
    item=make(draw);bootstrap(item);comparison(item);item.run_pending()
    decision=item.terminal_batches()[-1]["decision"]["result"]
    assert decision["status"]=="reject" and decision["raw_deltas"][0]>0
    assert decision["certified_deltas"][0]["numerator"]<0


def test_ack_required_and_saved_terminal_resume_never_replays(make):
    draw=Draws();item=make(draw);bootstrap(item)
    assert item.plan_comparison(FIRST,INITIAL,"p1",context=context())
    with pytest.raises(RepeatPromotionError,match="checkpoint"):
        item.run_pending()
    saved=ack(item);item.run_pending();terminal=item.state();packets=item.terminal_batches();item.close()
    resumed_draw=Draws();resumed=make(resumed_draw);resumed.load_state(saved)
    assert resumed.state()==terminal and resumed.terminal_batches()==packets
    resumed.run_pending();assert not resumed_draw.calls


def test_unknown_intent_resumes_as_unknown_once_without_replay(make):
    draw=Draws({4:KeyboardInterrupt()});item=make(draw);bootstrap(item);saved=comparison(item)
    with pytest.raises(KeyboardInterrupt):item.run_pending()
    item.close();next_draw=Draws();restored=make(next_draw);restored.load_state(saved)
    assert restored.stopped_reason=="unknown_dispatch" and not restored.pending
    packet=restored.terminal_batches()[-1]
    assert packet["receipts"][0]["receipt"]["status"]=="unknown"
    assert all(r["receipt"] is None for r in packet["receipts"][1:])
    state=restored.state();restored.close();again=make(Draws());again.load_state(state)
    assert again.state()==state and not next_draw.calls


def test_partial_terminal_tail_continues_only_remaining_draws(make, monkeypatch):
    draw=Draws();item=make(draw);bootstrap(item);saved=comparison(item)
    append=item._append
    def stop_after_receipt(event):
        append(event)
        if event["type"]=="terminal":raise KeyboardInterrupt
    monkeypatch.setattr(item,"_append",stop_after_receipt)
    with pytest.raises(KeyboardInterrupt):item.run_pending()
    item.close();next_draw=Draws();restored=make(next_draw);restored.load_state(saved);restored.run_pending()
    assert len(draw.calls)==4 and len(next_draw.calls)==5
    assert restored.terminal_batches()[-1]["decision"]["result"]["status"]=="promote"


@pytest.mark.parametrize("tamper", ["context", "metric", "source", "receipt", "contract"])
def test_restore_tampering_fails_before_dispatch_and_releases_lock(make, tamper):
    item=make();bootstrap(item);comparison(item);item.run_pending();saved=item.state();item.close()
    bad=deepcopy(saved);ident=identity();restore=None
    if tamper=="context":bad["data"]["batches"][1]["plan"]["context"]["parent_id"]="invented"
    elif tamper=="metric":
        ident["task_metric"]["primary_metric"]="other";ident["task_metric"]["sha256"]=digest({k:v for k,v in ident["task_metric"].items() if k!="sha256"})
    elif tamper=="contract":bad["contract"]="sciel-repeat-promotion/1"
    else:
        path=(item.directory/"sources"/(sha(FIRST)+".py") if tamper=="source" else next((item.directory/"receipts").glob('*.json')))
        restore=(path,path.read_bytes());path.write_bytes(b"tampered")
    draw=Draws();failed=make(draw,ident=ident)
    with pytest.raises((ValueError,KeyError)):failed.load_state(bad)
    assert not draw.calls and failed._lock is None
    if restore:restore[0].write_bytes(restore[1])
    good=make(Draws());good.load_state(saved);assert good.state()==saved


def test_uncheckpointed_plan_cannot_resume_from_previous_idle_state(make):
    item=make();bootstrap(item);saved=item.state();comparison(item);item.close()
    with pytest.raises(RepeatPromotionError,match="uncheckpointed"):
        make(Draws()).load_state(saved)


@pytest.mark.parametrize("field", ["parent_source_sha256","comparator_program_id","source_sha256","prospective_snapshot"])
def test_context_binding_rejects_before_journal_mutation(make, field):
    item=make();bootstrap(item);before=item.state();bad=context()
    bad[field] = {} if field=="prospective_snapshot" else "bogus"
    with pytest.raises(RepeatPromotionError):item.plan_comparison(FIRST,INITIAL,"p1",context=bad)
    assert item.state()==before


def test_unmeasured_comparator_rejected_and_packet_is_detached(make):
    item=make();bootstrap(item)
    with pytest.raises(RepeatPromotionError,match="complete measured"):
        item.plan_comparison(SECOND,FIRST,"p2",context=context(SECOND,FIRST,comparator_id="p1"))
    packet=item.terminal_batches()[0]
    packet["receipts"][0]["receipt"]["raw_sum"] = 999
    assert item.terminal_batches()[0]["receipts"][0]["receipt"]["raw_sum"]==.1


@pytest.mark.parametrize("status", ["candidate_failure", "infrastructure"])
def test_bootstrap_failure_never_opens_generation_or_publication(make,status):
    draw=Draws({1:{"status":status}});item=make(draw);bootstrap(item)
    assert not item.can_compare() and item.export()["confirmed_output_incumbent"] is None
    packet=item.terminal_batches()[0]
    assert packet["kind"]=="bootstrap" and packet["decision"]["result"]["status"]=="incomplete"
    assert len(packet["receipts"])==3 and item.terminal_comparisons()==[]
    assert len(draw.calls)==(1 if status=="infrastructure" else 3)


def test_valid_source_receipt_recomputes_geometry_before_admission(make):
    draw=Draws();item=make(draw);bootstrap(item);comparison(item);item.run_pending()
    packet=item.terminal_batches()[-1]
    # Mutating only process-memory context must not masquerade as a journal-bound plan.
    item._data["batches"][-1]["plan"]["context"]["branch"]="changed-root"
    with pytest.raises(RepeatPromotionError,match="bound decision"):
        item.terminal_batches()
    item._data["batches"][-1]["plan"]["context"]["branch"]=packet["plan"]["context"]["branch"]
    assert item.terminal_batches()[-1]==packet


@pytest.mark.parametrize("field", ["task_metric", "adapter"])
def test_new_identity_rejects_mismatched_self_hash_before_io(make,tmp_path,field):
    ident=identity()
    if field=="task_metric":ident[field]["task_hash"]="f"*64
    else:ident[field]["identity_sha256"]="f"*64
    with pytest.raises(ValueError):make(ident=ident)
    assert not (tmp_path/"paired_batches").exists()
