"""Repeat journal reduction, restored state and exported comparisons."""
from __future__ import annotations

from copy import deepcopy
import math
from .repeat_journal import digest, encoded
from .repeat_policy import (RepeatPromotionError, _source_sha)


class RepeatState:

    def _apply(self, event):
        kind = event["type"]
        if kind == "header":
            if self._data is not None or event != {"type": "header", "run_id": event["run_id"],
                    "contract_hash": self.contract_hash, "config": self.config, "identity": self.identity}:
                raise RepeatPromotionError("repeat header identity mismatch")
            self._data = {"run_id": event["run_id"], "batches": [], "sources": {}, "claimed_sources": [],
                          "confirmed": None, "budget_unavailable": [], "unexamined_candidates": [],
                          "stopped_reason": None}
            return
        if self._data is None:
            raise RepeatPromotionError("repeat header missing")
        if kind == "unexamined_candidate":
            if (set(event) != {"type", "source_sha256", "program_id", "screening", "reason"}
                    or self.pending or self.stopped_reason
                    or not isinstance(event["reason"], str) or not event["reason"].strip()
                    or len(event["reason"]) > 200):
                raise RepeatPromotionError("invalid unexamined candidate event")
            code = self._source(event["source_sha256"])
            screening = self._screen_candidate(code, event["screening"], event["program_id"])
            if screening is None or encoded(screening) != encoded(event["screening"]):
                raise RepeatPromotionError("unexamined candidate was not eligible")
            self._data["sources"][event["source_sha256"]] = code
            self._data["unexamined_candidates"].append(deepcopy(event))
            self._data["stopped_reason"] = "search_step_incomplete"
            return
        if kind == "budget_unavailable":
            if self.pending:
                raise RepeatPromotionError("budget event inside pending batch")
            self._data["budget_unavailable"].append(deepcopy(event))
            if event["kind"] == "bootstrap":
                self._data["stopped_reason"] = "bootstrap_budget_unavailable"
            return
        if kind == "plan":
            if self.pending or self.stopped_reason:
                raise RepeatPromotionError("new plan while pending/stopped")
            plan = event["plan"]
            if "research_context" in plan:
                if plan["kind"] != "comparison" or plan["research_context"] is None:
                    raise RepeatPromotionError("research context requires a comparison plan")
                self._research_context(plan["research_context"], plan["challenger"])
            seq = len(self._data["batches"])
            expected_incumbent = self._data["confirmed"]["source_sha256"] if self._data["confirmed"] else None
            if (plan["sequence"] != seq or plan["batch_id"] != f"b{seq:06d}"
                    or plan["kind"] != ("bootstrap" if seq == 0 else "comparison")
                    or plan["incumbent"] != expected_incumbent
                    or (seq == 0 and plan["challenger"] != self.identity["canonical_initial_source_sha256"])
                    or plan["challenger"] in self._data["claimed_sources"]
                    or plan["draws"] != self._planned_draws(seq, plan["kind"], plan["challenger"], plan["incumbent"])):
                raise RepeatPromotionError("repeat plan source/seed/order mismatch")
            n = len(plan["draws"])
            counters = self.summary()["counters"]
            wall = n * (self.config["timeout_seconds"] + self.config["certification_seconds"])
            if (plan["reserved_wall_seconds"] != wall
                    or counters["reserved_draw_attempts"] + n > self.config["max_draw_attempts"]
                    or counters["planned_wall_seconds"] + wall > self.config["max_wall_seconds"]
                    or (seq and counters["comparisons"] >= self.config["max_comparisons"])):
                raise RepeatPromotionError("repeat plan budget mismatch")
            for sha in {d["source_sha256"] for d in plan["draws"]}:
                self._data["sources"][sha] = self._source(sha)
            self._data["claimed_sources"].append(plan["challenger"])
            self._data["batches"].append({"plan": deepcopy(plan), "intents": [], "receipts": {}, "decision": None})
            return
        if not self.pending:
            raise RepeatPromotionError("repeat event outside pending batch")
        batch = self._batch()
        if kind == "decision":
            decision = event["decision"]
            if decision != self._decision(batch):
                raise RepeatPromotionError("repeat decision differs from receipts")
            if len(batch["receipts"]) < len(batch["plan"]["draws"]) and decision["stopped_reason"] not in ("unknown_dispatch", "infrastructure"):
                raise RepeatPromotionError("premature repeat decision")
            batch["decision"] = deepcopy(decision)
            self._data["stopped_reason"] = decision["stopped_reason"]
            if decision["result"]["status"] in ("confirmed", "promote"):
                metrics = decision["result"]["aggregates"]
                if batch["plan"]["kind"] == "comparison":
                    metrics = metrics["challenger"]
                self._data["confirmed"] = {"source_sha256": batch["plan"]["challenger"],
                    "decision_id": decision["decision_id"], "batch_id": batch["plan"]["batch_id"],
                    "metrics": deepcopy(metrics), "program_id": batch["plan"]["program_id"]}
            return
        if event.get("batch_id") != batch["plan"]["batch_id"]:
            raise RepeatPromotionError("repeat event batch mismatch")
        key = event.get("draw_id")
        draw = next((d for d in batch["plan"]["draws"] if d["draw_id"] == key), None)
        if draw is None or key in batch["receipts"]:
            raise RepeatPromotionError("unknown or repeated repeat draw key")
        if kind == "intent":
            remaining = [d["draw_id"] for d in batch["plan"]["draws"] if d["draw_id"] not in batch["receipts"]]
            if (key in batch["intents"] or key != remaining[0]
                    or any(r["status"] in ("infrastructure", "unknown") for r in batch["receipts"].values())
                    or any(i not in batch["receipts"] for i in batch["intents"])):
                raise RepeatPromotionError("repeat dispatch order/replay violation")
            batch["intents"].append(key)
        elif kind in ("terminal", "unknown"):
            if key not in batch["intents"]:
                raise RepeatPromotionError("repeat terminal lacks dispatch intent")
            batch["receipts"][key] = (self._read_receipt(event, draw) if kind == "terminal" else
                {"status": "unknown", "raw_sum": None, "certified_sum": None,
                 "reason": "dispatch_intent_without_terminal_receipt"})
        else:
            raise RepeatPromotionError("unknown repeat event type")

    def state(self):
        if self._data is None:
            raise RepeatPromotionError("repeat coordinator not started")
        return {"contract": self.CONTRACT, "contract_hash": self.contract_hash,
                "run_id": self._data["run_id"], "ledger_anchor": deepcopy(self.journal.anchor),
                "data": deepcopy(self._data)}

    def load_state(self, saved):
        if self._data is not None or self._lock is not None:
            raise RepeatPromotionError("repeat state already initialized")
        if not isinstance(saved, dict) or set(saved) != {"contract", "contract_hash", "run_id", "ledger_anchor", "data"}:
            raise RepeatPromotionError("missing/unbound repeat checkpoint")
        if saved["contract"] != self.CONTRACT or saved["contract_hash"] != self.contract_hash:
            raise RepeatPromotionError("repeat checkpoint identity mismatch")
        self._acquire()
        try:
            rows = self.journal.read()
            anchor = saved["ledger_anchor"]
            size = anchor.get("entries")
            if (type(size) is not int or not 1 <= size <= len(rows)
                    or anchor != {"entries": size, "sha256": rows[size - 1]["sha256"]}):
                raise RepeatPromotionError("repeat checkpoint ledger prefix mismatch")
            for row in rows[:size]:
                self._apply(row["event"])
            if self._data != saved["data"] or self._data["run_id"] != saved["run_id"]:
                raise RepeatPromotionError("repeat checkpoint state differs from journal prefix")
            pending_id = self._batch()["plan"]["batch_id"] if self.pending else None
            for row in rows[size:]:
                event = row["event"]
                batch_id = event.get("batch_id", event.get("decision", {}).get("batch_id"))
                if event["type"] not in ("intent", "terminal", "unknown", "decision") or batch_id != pending_id:
                    raise RepeatPromotionError("uncheckpointed repeat plan or unrelated journal tail")
                self._apply(event)
            self.journal.rows = rows
            self._ack = deepcopy(anchor)
            if self.pending:
                batch = self._batch()
                unresolved = [key for key in batch["intents"] if key not in batch["receipts"]]
                if unresolved:
                    self._append({"type": "unknown", "batch_id": batch["plan"]["batch_id"], "draw_id": unresolved[0]})
                    self._finish()
                elif len(batch["receipts"]) == len(batch["plan"]["draws"]) or any(r["status"] in ("infrastructure", "unknown") for r in batch["receipts"].values()):
                    self._finish()
        except BaseException:
            self.close()
            raise

    def terminal_comparisons(self):
        """Export durable paired batches, preserving absent versus unknown draws.

        The decision-prefix anchor remains stable when later batches are added.
        Bootstrap is not a comparison. This method neither creates measurements
        nor changes promotion/search state, and works after the lock is closed.
        """
        if self._data is None:
            return []
        rows = self.journal.read()
        if rows != self.journal.rows:
            raise RepeatPromotionError("repeat journal changed before comparison export")
        decisions = {r["event"]["decision"]["batch_id"]: r for r in rows
                     if r["event"]["type"] == "decision"}
        terminals = {r["event"]["draw_id"]: r["event"] for r in rows
                     if r["event"]["type"] == "terminal"}
        result = []
        for batch in self._data["batches"]:
            plan, decision = batch["plan"], batch["decision"]
            if plan["kind"] != "comparison" or decision is None:
                continue
            row = decisions.get(plan["batch_id"])
            if row is None or row["event"]["decision"] != decision:
                raise RepeatPromotionError("comparison lacks its durable decision")
            pairs = [{"seed": self._seed(plan["sequence"], i),
                      "comparator": None, "challenger": None} for i in range(3)]
            ordered = []
            for draw in plan["draws"]:
                key = draw["draw_id"]
                receipt = batch["receipts"].get(key)
                event = terminals.get(key)
                if event is not None and self._read_receipt(event, draw) != receipt:
                    raise RepeatPromotionError("comparison receipt differs from durable artifact")
                pairs[draw["draw_index"]]["comparator" if draw["role"] == "incumbent"
                                          else "challenger"] = deepcopy(receipt)
                ordered.append({**deepcopy(draw), "receipt_sha256":
                                event["artifact_sha256"] if event else None})
            result.append({"namespace": "h15:" + digest([self._data["run_id"], self.contract_hash]),
                "batch_id": plan["batch_id"], "program_id": plan["program_id"],
                "research_context": deepcopy(plan.get("research_context")),
                "comparator_source_sha256": plan["incumbent"],
                "challenger_source_sha256": plan["challenger"],
                "source_pair": {"comparator": self._source(plan["incumbent"]),
                                "challenger": self._source(plan["challenger"])},
                "adapter_identity": deepcopy(self.identity["adapter"]),
                "limits": self._receipt_limits(), "pairs": pairs,
                "provenance": {"kind": "h15_terminal_batch/1", "run_id": self._data["run_id"],
                    "contract_hash": self.contract_hash,
                    "decision_anchor": {"entries": row["sequence"] + 1, "sha256": row["sha256"]},
                    "plan": deepcopy(plan), "decision": deepcopy(decision), "ordered_draws": ordered}})
        return result

    def summary(self):
        batches = self._data["batches"] if self._data else []
        receipts = [r for b in batches for r in b["receipts"].values()]
        cap = self.config["timeout_seconds"] + self.config["certification_seconds"]
        reserved = sum(len(b["plan"]["draws"]) for b in batches)
        terminal = [r for r in receipts if r["status"] != "unknown"]
        actual = math.fsum(r["elapsed_seconds"] for r in terminal)
        held = (reserved - len(terminal)) * cap
        released = math.fsum(max(0.0, cap - r["elapsed_seconds"]) for r in terminal)
        confirmed = self._data["confirmed"] if self._data else None
        return {"contract": self.CONTRACT, "status": self.stopped_reason or
                ("pending" if self.pending else "confirmed" if confirmed else "uninitialized"),
                "stopped_reason": self.stopped_reason, "confirmed_source_sha256": confirmed["source_sha256"] if confirmed else None,
                "pending": self.pending, "counters": {
                    "reserved_draw_attempts": reserved, "draw_attempts": sum(len(b["intents"]) for b in batches),
                    "comparisons": sum(b["plan"]["kind"] == "comparison" for b in batches),
                    "completed": len(terminal), "failed": sum(r["status"] in ("candidate_failure", "infrastructure") for r in receipts),
                    "unknown": sum(r["status"] == "unknown" for r in receipts),
                    "certified_draws": sum(r["status"] == "valid" for r in receipts),
                    "planned_wall_seconds": reserved * cap, "actual_wall_seconds": actual,
                    "held_wall_seconds": held, "released_wall_seconds": released},
                "budget_unavailable": deepcopy(self._data["budget_unavailable"] if self._data else []),
                "unexamined_candidates": deepcopy(self._data["unexamined_candidates"] if self._data else []),
                "coverage_complete": bool(confirmed and not self.pending and not self.stopped_reason
                                          and not self._data["budget_unavailable"]),
                "decisions": [deepcopy(b["decision"]) for b in batches if b["decision"] is not None],
                "wall_budget_policy": "lifetime_planned_phase_allowances_no_reallocation; actual includes adapter elapsed and cleanup"}

    def export(self, observed_program=None):
        observed = None
        if observed_program is not None:
            def value(name):
                return observed_program.get(name) if isinstance(observed_program, dict) else getattr(observed_program, name, None)
            observed = {"id": value("id"), "source_sha256": _source_sha(value("code")), "score": value("score")}
        confirmed = deepcopy(self._data["confirmed"]) if self._data else None
        if confirmed:
            confirmed["code"] = self._source(confirmed["source_sha256"])
        return {"status": self.summary()["status"], "observed_search_best": observed,
                "confirmed_output_incumbent": confirmed, "repeat_summary": self.summary()}
