"""Repeat physical receipts, limits and terminal draw decisions."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import math
import time
from .repeat_journal import digest, encoded, immutable_file
from .repeat_policy import (
    RepeatPromotionError,
    _finite,
    _fraction,
    _numeric_capture,
    _valid,
    bootstrap_decision,
    paired_dominance,
)


class RepeatExecution:

    def _limits(self):
        return {"timeout": self.config["timeout_seconds"], "cpu_seconds": self.config["cpu_seconds"],
                "mem_bytes": self.config["mem_bytes"], "certification_seconds": self.config["certification_seconds"]}

    def _receipt_limits(self):
        return {"candidate_wall_seconds": self.config["timeout_seconds"],
                "certification_wall_seconds": self.config["certification_seconds"],
                "candidate_cpu_seconds": self.config["cpu_seconds"], "candidate_mem_bytes": self.config["mem_bytes"],
                "certification_cpu_seconds": self.config["cpu_seconds"], "certification_mem_bytes": self.config["mem_bytes"]}

    def _validate_receipt(self, receipt, draw):
        # The adapter owns independent geometry checks; the coordinator binds
        # their complete result to exactly one reserved source/seed/limit tuple.
        if not isinstance(receipt, dict):
            raise RepeatPromotionError("adapter receipt must be an object")
        encoded(receipt)
        if (receipt.get("source_sha256") != draw["source_sha256"]
                or receipt.get("seed") != draw["seed"] or type(receipt.get("seed")) is not int
                or encoded(receipt.get("adapter_identity")) != encoded(self.identity["adapter"])
                or encoded(receipt.get("limits")) != encoded(self._receipt_limits())):
            raise RepeatPromotionError("adapter receipt identity/seed/limits mismatch")
        if receipt.get("status") not in ("valid", "candidate_failure", "infrastructure"):
            raise RepeatPromotionError("invalid adapter status")
        if receipt.get("schema") not in ("sciel-repeat-draw/1", "sciel-repeat-coordinator-failure/1"):
            raise RepeatPromotionError("adapter receipt schema mismatch")
        try:
            start = datetime.fromisoformat(receipt["started_at"])
            end = datetime.fromisoformat(receipt["completed_at"])
            if start.utcoffset() is None or end.utcoffset() is None or end < start:
                raise ValueError
        except (KeyError, ValueError, TypeError):
            raise RepeatPromotionError("invalid adapter timestamps") from None
        for key in ("elapsed_seconds",):
            if not _finite(receipt.get(key)) or receipt[key] < 0:
                raise RepeatPromotionError("invalid adapter timestamps/elapsed")
        for key in ("candidate_elapsed_seconds", "certification_elapsed_seconds", "identity_elapsed_seconds"):
            if receipt.get(key) is not None and (not _finite(receipt[key]) or receipt[key] < 0):
                raise RepeatPromotionError("invalid stage elapsed")
        if receipt["status"] == "valid":
            expected_seed = {"python_random": draw["seed"], "numpy_legacy": draw["seed"],
                "python_hash_seed": str(draw["seed"]), "before_candidate_import": True, "determinism_proven": False}
            if not _valid(receipt) or encoded(receipt.get("seed_applied")) != encoded(expected_seed):
                raise RepeatPromotionError("valid receipt lacks bound metrics/seed receipt")
            capture, certificate = receipt.get("capture"), receipt.get("certificate")
            if (not isinstance(capture, dict) or capture.get("valid") is not True
                    or capture.get("sum_radii") != receipt["raw_sum"]
                    or not isinstance(certificate, dict) or certificate.get("status") != "certified"
                    or certificate.get("sum_exact") != receipt["certified_sum"]):
                raise RepeatPromotionError("receipt metrics differ from numeric capture/certificate")
            raw_geometry, certified_geometry = _numeric_capture(capture), _numeric_capture(certificate)
            if (certificate.get("schema") != "sciel-circle-packing-certificate/1"
                    or certificate.get("input_numeric_sha256") != digest(raw_geometry)
                    or certificate.get("certified_numeric_sha256") != digest(certified_geometry)
                    or encoded(raw_geometry["centers"]) != encoded(certified_geometry["centers"])
                    or math.fsum(raw_geometry["radii"]) != receipt["raw_sum"]
                    or sum((Fraction(r) for r in certified_geometry["radii"]), Fraction()) != _fraction(receipt["certified_sum"])):
                raise RepeatPromotionError("numeric geometry hashes/sums differ from receipt")
        elif receipt.get("raw_sum") is not None or receipt.get("certified_sum") is not None:
            raise RepeatPromotionError("failed receipt cannot carry physical metrics")
        return deepcopy(receipt)

    def _read_receipt(self, event, draw):
        expected = draw["draw_id"] + ".json"
        if event["artifact"] != expected:
            raise RepeatPromotionError("receipt artifact path/key mismatch")
        raw = (self.directory / "receipts" / expected).read_bytes()
        if hashlib.sha256(raw).hexdigest() != event["artifact_sha256"]:
            raise RepeatPromotionError("receipt artifact hash mismatch")
        import json
        receipt = json.loads(raw)
        if encoded(receipt) != raw:
            raise RepeatPromotionError("receipt canonical bytes mismatch")
        if receipt.get("schema") == "sciel-repeat-coordinator-failure/1":
            if receipt.get("status") != "infrastructure" or receipt.get("raw_sum") is not None or receipt.get("certified_sum") is not None:
                raise RepeatPromotionError("invalid coordinator failure receipt")
            rejected = receipt.get("rejected_return")
            if rejected is not None:
                if (not isinstance(rejected, dict) or set(rejected) != {"artifact", "sha256"}
                        or rejected["artifact"] != draw["draw_id"] + ".rejected.json"
                        or hashlib.sha256((self.directory / "receipts" / rejected["artifact"]).read_bytes()).hexdigest() != rejected["sha256"]):
                    raise RepeatPromotionError("rejected adapter artifact mismatch")
        return self._validate_receipt(receipt, draw)

    def run_pending(self):
        self._check_live()
        if not self.pending or self.stopped_reason:
            return self.summary()
        if self._ack is None:
            raise RepeatPromotionError("repeat plan needs a durable checkpoint acknowledgement")
        batch = self._batch()
        for draw in batch["plan"]["draws"]:
            key = draw["draw_id"]
            if key in batch["receipts"]:
                continue
            if key in batch["intents"]:
                raise RepeatPromotionError("unresolved dispatch intent must recover as unknown")
            self._append({"type": "intent", "batch_id": batch["plan"]["batch_id"], "draw_id": key})
            started, clock = datetime.now(timezone.utc).isoformat(), time.monotonic()
            returned = None
            try:
                returned = self.draw_fn(self._source(draw["source_sha256"]), seed=draw["seed"], **self._limits())
                receipt = self._validate_receipt(returned, draw)
            except Exception as exc:
                rejected = None
                if returned is not None:
                    try:
                        rejected_raw = encoded(returned)
                    except (ValueError, TypeError, OverflowError):
                        pass
                    else:
                        rejected_name = key + ".rejected.json"
                        immutable_file(self.directory / "receipts" / rejected_name, rejected_raw)
                        rejected = {"artifact": rejected_name, "sha256": hashlib.sha256(rejected_raw).hexdigest()}
                receipt = {"schema": "sciel-repeat-coordinator-failure/1", "status": "infrastructure",
                           "source_sha256": draw["source_sha256"], "seed": draw["seed"],
                           "adapter_identity": self.identity["adapter"], "limits": self._receipt_limits(),
                           "seed_applied": None, "reason": "adapter_exception_or_invalid_receipt",
                           "error_type": type(exc).__name__, "rejected_return": rejected, "capture": None, "certificate": None,
                           "raw_sum": None, "certified_sum": None, "started_at": started,
                           "completed_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - clock,
                           "candidate_elapsed_seconds": None, "certification_elapsed_seconds": None}
            artifact = key + ".json"
            raw = encoded(receipt)
            immutable_file(self.directory / "receipts" / artifact, raw)
            self._append({"type": "terminal", "batch_id": batch["plan"]["batch_id"], "draw_id": key,
                          "artifact": artifact, "artifact_sha256": hashlib.sha256(raw).hexdigest()})
            if receipt["status"] == "infrastructure":
                break
        self._finish()
        return self.summary()

    def _decision(self, batch):
        plan = batch["plan"]
        receipts = {role: [] for role in ("bootstrap", "incumbent", "challenger")}
        statuses = []
        for draw in sorted(plan["draws"], key=lambda d: d["draw_index"]):
            receipt = batch["receipts"].get(draw["draw_id"], {"status": "not_dispatched", "raw_sum": None, "certified_sum": None})
            receipts[draw["role"]].append(receipt)
            statuses.append({"draw_id": draw["draw_id"], "status": receipt["status"]})
        result = (bootstrap_decision(receipts["bootstrap"]) if plan["kind"] == "bootstrap"
                  else paired_dominance(receipts["incumbent"], receipts["challenger"]))
        stop = None
        if any(v["status"] == "unknown" for v in statuses):
            stop = "unknown_dispatch"
        elif any(v["status"] == "infrastructure" for v in statuses):
            stop = "infrastructure"
        elif plan["kind"] == "bootstrap" and result["status"] != "confirmed":
            stop = "bootstrap_failed"
        return {"decision_id": "d" + plan["batch_id"][1:], "batch_id": plan["batch_id"],
                "result": result, "draw_statuses": statuses, "stopped_reason": stop}

    def _finish(self):
        batch = self._batch()
        self._append({"type": "decision", "decision": self._decision(batch)})
