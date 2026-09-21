"""Durable explicit-comparator batches for repeat-controlled populations.

The global publication pointer is bootstrap-only. A terminal ``promote`` is a
paired admission verdict against the named comparator, never publication or a
mutation of ordinary Program scores. Candidate execution is the injected adapter.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .repeat_journal import digest, encoded, immutable_file
from .repeat_promotion import (RepeatPromotion, RepeatPromotionError,
                               _aggregate, _finite, _source_sha)
from ..knowledge_base.paired_comparisons import _adapter, _receipt
from ..hypothesis.measurement import validate_metric_context

CONTRACT = "sciel-paired-batches/1"


def _require(condition, message):
    if not condition:
        raise RepeatPromotionError(message)


def _identifier(value, *, nullable=False):
    return ((nullable and value is None) or
            (type(value) is str and bool(value.strip()) and len(value) <= 128))


def _hash(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class PairedBatches(RepeatPromotion):
    CONTRACT = CONTRACT
    DIRECTORY = "paired_batches"
    IDENTITY_REQUIRED_EXTRA = frozenset({"task_metric"})

    def __init__(self, run_dir, config, identity, draw_fn):
        super().__init__(run_dir, config, identity, draw_fn)
        _adapter(self.identity["adapter"])
        validate_metric_context(self.identity["task_metric"])
        _require(self.identity["task_metric"]["task_hash"] == self.identity["task_hash"]
                 and self.identity["task_metric"]["task_name"] == self.identity["task"],
                 "paired task metric hash differs from task identity")

    def _validate_receipt(self, receipt, draw):
        result = super()._validate_receipt(receipt, draw)
        # Unlike a numeric metric claim, this recomputes tolerance geometry and
        # the exact certificate from the saved arrays. No candidate is imported.
        _receipt(result, source=draw["source_sha256"], seed=draw["seed"],
                 adapter=self.identity["adapter"], limits=self._receipt_limits())
        return result

    def can_compare(self):
        """Read-only full-six-draw preflight, before the next paid generation.

        Single inflight Engine ownership is required. Actual reservation is the
        durable comparison plan after a source exists; unused phase allowance is
        never recycled. A pending/stopped or unbootstrapped service returns False.
        """
        self._check_live()
        if self._data is None or self.pending or self.stopped_reason or not self._data["confirmed"]:
            return False
        counters = self.summary()["counters"]
        wall = 6 * (self.config["timeout_seconds"] + self.config["certification_seconds"])
        return (counters["comparisons"] < self.config["max_comparisons"]
                and counters["reserved_draw_attempts"] + 6 <= self.config["max_draw_attempts"]
                and counters["planned_wall_seconds"] + wall <= self.config["max_wall_seconds"])

    def plan_candidate(self, *args, **kwargs):
        raise RepeatPromotionError("population batches require an explicit comparator and prospective context")

    def note_unexamined_candidate(self, *args, **kwargs):
        raise RepeatPromotionError("population interruption must retain its prospective pending context")

    @staticmethod
    def _context(value, challenger, comparator):
        keys = {"branch", "origin_experiment_id", "parent_id", "parent_source_sha256",
                "comparator_program_id", "comparator_source_sha256", "source_sha256",
                "prospective_snapshot"}
        _require(type(value) is dict and set(value) == keys, "invalid paired population context fields")
        _require(all(_identifier(value[k]) for k in ("branch", "parent_id", "comparator_program_id"))
                 and _identifier(value["origin_experiment_id"], nullable=True), "invalid paired population IDs")
        _require(_hash(value["parent_source_sha256"]) and value["source_sha256"] == challenger
                 and value["comparator_source_sha256"] == comparator,
                 "paired population context source mismatch")
        snapshot = value["prospective_snapshot"]
        keys = {"schema", "selection_id", "island", "parent_id", "parent_source_sha256",
                "comparator_program_id", "comparator_source_sha256", "victim_id", "active_ids",
                "evidence_heads", "distribution", "draw", "parent_propensity"}
        _require(type(snapshot) is dict and set(snapshot) == keys
                 and snapshot["schema"] == "sciel-repeat-population-selection/1"
                 and _identifier(snapshot["selection_id"])
                 and type(snapshot["island"]) is int and snapshot["island"] >= 0,
                 "prospective population snapshot is required")
        _require(all(snapshot[k] == value[k] for k in ("parent_id", "parent_source_sha256",
                 "comparator_program_id", "comparator_source_sha256")),
                 "prospective parent/comparator binding differs")
        active = snapshot["active_ids"]
        _require(type(active) is list and all(_identifier(i) for i in active)
                 and len(active) == len(set(active)) and value["parent_id"] in active
                 and value["comparator_program_id"] in active
                 and snapshot["victim_id"] in (None, value["comparator_program_id"])
                 and type(snapshot["evidence_heads"]) is dict and set(snapshot["evidence_heads"]) == set(active)
                 and all(_hash(h) for h in snapshot["evidence_heads"].values()),
                 "prospective membership/evidence binding differs")
        rows = snapshot["distribution"]
        row_keys = {"program_id", "source_sha256", "evidence_id", "rank", "probability"}
        _require(type(rows) is list and len(rows) == len(active)
                 and all(type(r) is dict and set(r) == row_keys and _identifier(r["program_id"]) for r in rows),
                 "invalid prospective selection distribution")
        _require({r["program_id"] for r in rows} == set(active)
                 and all(_hash(r["source_sha256"]) and r["evidence_id"] == snapshot["evidence_heads"][r["program_id"]]
                         and _finite(r["rank"]) and 1 <= r["rank"] <= len(active)
                         and _finite(r["probability"]) and 0 < r["probability"] <= 1 for r in rows)
                 and abs(math.fsum(r["probability"] for r in rows) - 1) <= 1e-12
                 and _finite(snapshot["draw"]) and 0 <= snapshot["draw"] < 1,
                 "invalid prospective rank/probability evidence")
        parent = next(r for r in rows if r["program_id"] == value["parent_id"])
        comparator_row = next(r for r in rows if r["program_id"] == value["comparator_program_id"])
        _require(parent["source_sha256"] == value["parent_source_sha256"]
                 and comparator_row["source_sha256"] == comparator
                 and type(snapshot["parent_propensity"]) is not bool
                 and parent["probability"] == snapshot["parent_propensity"],
                 "prospective selected source/propensity differs")
        # The population component independently validates membership/ranking and
        # binds its exact snapshot. Preserve it, do not infer a victim after draws.
        _require(len(encoded(value)) <= 131072, "paired population context exceeds limit")
        return deepcopy(value)

    def _complete_source(self, sha):
        """A comparator must already have a complete measured three-draw block."""
        for batch in reversed(self._data["batches"]):
            if batch["decision"] is None:
                continue
            receipts = [batch["receipts"].get(d["draw_id"], {})
                        for d in sorted(batch["plan"]["draws"], key=lambda d: d["draw_index"])
                        if d["source_sha256"] == sha]
            if _aggregate(receipts) is not None:
                return True
        return False

    def plan_comparison(self, code, comparator_code, program_id, *, context):
        if not self._can_plan():
            return False
        _require(self._data["confirmed"] is not None, "comparison requires confirmed bootstrap")
        challenger, comparator = _source_sha(code), _source_sha(comparator_code)
        _require(_identifier(program_id), "candidate program ID required")
        context = self._context(context, challenger, comparator)
        _require(program_id not in context["prospective_snapshot"]["active_ids"],
                 "candidate program ID is already active")
        if challenger == comparator or challenger in self._data["claimed_sources"]:
            return False
        _require(self._complete_source(comparator), "comparator lacks a complete measured source block")
        _require(self._complete_source(context["parent_source_sha256"]),
                 "parent lacks a complete measured source block")
        _require(self._source(comparator) == comparator_code, "comparator retained source differs")
        immutable_file(self.directory / "sources" / (challenger + ".py"), code.encode("utf-8"))
        if not self.can_compare():
            self._append({"type": "population_budget_unavailable", "source_sha256": challenger,
                          "comparator_source_sha256": comparator, "program_id": program_id,
                          "context": context})
            return False
        sequence = len(self._data["batches"])
        plan = {"batch_id": f"b{sequence:06d}", "sequence": sequence, "kind": "comparison",
                "challenger": challenger, "incumbent": comparator, "program_id": program_id,
                "screening": None, "context": context,
                "research_context": {k: context[k] for k in ("branch", "origin_experiment_id", "source_sha256")},
                "draws": self._planned_draws(sequence, "comparison", challenger, comparator),
                "reserved_wall_seconds": 6 * (self.config["timeout_seconds"] + self.config["certification_seconds"])}
        self._append({"type": "plan", "plan": plan})
        self._ack = None
        return True

    def _decision(self, batch):
        result = super()._decision(batch)
        if batch["plan"]["kind"] == "comparison" and result["stopped_reason"] is None:
            comparator = [batch["receipts"].get(d["draw_id"], {}) for d in batch["plan"]["draws"]
                          if d["role"] == "incumbent"]
            if _aggregate(comparator) is None:
                result["stopped_reason"] = "comparator_failed"
        return result

    def _apply(self, event):
        kind = event["type"]
        if kind == "population_budget_unavailable":
            _require(self._data is not None and not self.pending and not self.stopped_reason,
                     "population budget event inside pending/stopped batch")
            _require(set(event) == {"type", "source_sha256", "comparator_source_sha256", "program_id", "context"}
                     and _identifier(event["program_id"]), "invalid population budget event")
            self._context(event["context"], event["source_sha256"], event["comparator_source_sha256"])
            _require(not self.can_compare(), "population budget refusal despite full available reserve")
            self._data["sources"][event["source_sha256"]] = self._source(event["source_sha256"])
            self._data["budget_unavailable"].append(deepcopy(event))
            self._data["stopped_reason"] = "comparison_budget_unavailable"
            return
        if kind == "plan" and event["plan"]["kind"] == "comparison":
            plan = event["plan"]
            _require(self._data is not None and self._data["confirmed"] is not None
                     and not self.pending and not self.stopped_reason, "comparison before bootstrap or while pending/stopped")
            _require(set(plan) == {"batch_id", "sequence", "kind", "challenger", "incumbent", "program_id",
                     "screening", "context", "research_context", "draws", "reserved_wall_seconds"},
                     "invalid explicit comparison plan fields")
            seq = len(self._data["batches"])
            context = self._context(plan["context"], plan["challenger"], plan["incumbent"])
            _require(type(plan["sequence"]) is int and plan["sequence"] == seq and plan["batch_id"] == f"b{seq:06d}"
                     and _identifier(plan["program_id"]) and plan["screening"] is None
                     and plan["program_id"] not in context["prospective_snapshot"]["active_ids"]
                     and plan["challenger"] != plan["incumbent"]
                     and plan["challenger"] not in self._data["claimed_sources"]
                     and self._complete_source(plan["incumbent"])
                     and self._complete_source(context["parent_source_sha256"])
                     and plan["research_context"] == {k: context[k] for k in ("branch", "origin_experiment_id", "source_sha256")}
                     and plan["draws"] == self._planned_draws(seq, "comparison", plan["challenger"], plan["incumbent"]),
                     "explicit comparison source/context/seed/order mismatch")
            _require(self.can_compare() and plan["reserved_wall_seconds"] ==
                     6 * (self.config["timeout_seconds"] + self.config["certification_seconds"]),
                     "explicit comparison budget mismatch")
            for sha in (plan["challenger"], plan["incumbent"]):
                self._data["sources"][sha] = self._source(sha)
            self._data["claimed_sources"].append(plan["challenger"])
            self._data["batches"].append({"plan": deepcopy(plan), "intents": [], "receipts": {}, "decision": None})
            return
        if kind == "decision" and self.pending and self._batch()["plan"]["kind"] == "comparison":
            batch = self._batch()
            decision = event["decision"]
            _require(decision == self._decision(batch), "population decision differs from receipts")
            _require(len(batch["receipts"]) == len(batch["plan"]["draws"])
                     or decision["stopped_reason"] in ("unknown_dispatch", "infrastructure"),
                     "premature population decision")
            batch["decision"] = deepcopy(decision)
            self._data["stopped_reason"] = decision["stopped_reason"]
            # Deliberately never update confirmed: victim dominance is admission,
            # not global publication. Bootstrap owns the only published source.
            return
        return super()._apply(event)

    def terminal_batches(self):
        """Verified immutable snapshots of bootstrap and explicit comparisons.

        Roles retain the coordinator's ``incumbent`` spelling for comparator;
        all planned positions survive, including None/unstarted and unknown.
        """
        if self._data is None:
            return []
        rows = self.journal.read()
        _require(rows == self.journal.rows, "paired batch journal changed before export")
        decisions = {r["event"]["decision"]["batch_id"]: r for r in rows if r["event"]["type"] == "decision"}
        plans = {r["event"]["plan"]["batch_id"]: r["event"]["plan"] for r in rows if r["event"]["type"] == "plan"}
        terminals = {r["event"]["draw_id"]: r["event"] for r in rows if r["event"]["type"] == "terminal"}
        packets = []
        for batch in self._data["batches"]:
            plan, decision = batch["plan"], batch["decision"]
            if decision is None:
                continue
            row = decisions.get(plan["batch_id"])
            _require(row is not None and row["event"]["decision"] == decision
                     and plans.get(plan["batch_id"]) == plan
                     and decision == self._decision(batch), "terminal batch lacks bound decision")
            receipts = []
            for draw in plan["draws"]:
                key = draw["draw_id"]
                receipt = batch["receipts"].get(key)
                if key in terminals:
                    _require(self._read_receipt(terminals[key], draw) == receipt, "terminal receipt artifact differs")
                receipts.append({**deepcopy(draw), "receipt": deepcopy(receipt)})
            packets.append({"namespace": "h15:" + digest([self._data["run_id"], self.contract_hash]),
                "batch_id": plan["batch_id"], "kind": plan["kind"], "plan": deepcopy(plan),
                "decision": deepcopy(decision), "receipts": receipts,
                "source_pair": {"comparator": self._source(plan["incumbent"]) if plan["incumbent"] else None,
                                "challenger": self._source(plan["challenger"])},
                "adapter_identity": deepcopy(self.identity["adapter"]), "limits": self._receipt_limits(),
                "task_metric": deepcopy(self.identity["task_metric"]),
                "task_identity": {k: deepcopy(self.identity[k]) for k in
                    ("task", "task_hash", "canonical_initial_source_sha256", "task_evaluator_sha256")},
                "decision_anchor": {"entries": row["sequence"] + 1, "sha256": row["sha256"]}})
        return packets
