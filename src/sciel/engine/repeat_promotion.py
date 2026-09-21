"""Opt-in, output-only repeat promotion with durable, non-replayed draw identities.

The caller must durably checkpoint every accepted plan, then acknowledge that
checkpoint before dispatch. This component never changes search fitness or RNG.
"""

from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from fractions import Fraction
import hashlib
import math
from pathlib import Path
import time
from .repeat_journal import Journal, digest, encoded, fsync_directory, immutable_file
from .repeat_policy import (
    CONTRACT,
    RepeatPromotionError,
    SEED_CONTRACT,
    _FIXED,
    _LIMITS,
    _REQUIRED,
    _aggregate,
    _finite,
    _fraction,
    _numeric_capture,
    _rational,
    _source_sha,
    _valid,
    bootstrap_decision,
    paired_dominance,
    validate_config,
)

from .repeat_planning import RepeatPlanning
from .repeat_execution import RepeatExecution
from .repeat_state import RepeatState


class RepeatPromotion(
    RepeatPlanning,
    RepeatExecution,
    RepeatState,
):
    CONTRACT = CONTRACT

    DIRECTORY = "repeat_promotion"

    IDENTITY_REQUIRED_EXTRA = frozenset()


    def __init__(self, run_dir, config, identity, draw_fn):
        self.config = validate_config(config)
        if not self.config["enabled"]:
            raise ValueError("do not construct RepeatPromotion for disabled configuration")
        required = {"contract", "task", "task_hash", "canonical_initial_source_sha256",
                    "task_evaluator_sha256", "adapter"} | self.IDENTITY_REQUIRED_EXTRA
        if (not isinstance(identity, dict) or required - set(identity)
                or set(identity) - required - {"integration_sha256", "execution_profile", "restart_trace"}
                or identity["contract"] != self.CONTRACT):
            raise ValueError("repeat identity fields/contract differ")
        if 'execution_profile' in identity or 'restart_trace' in identity:
            from .repeat_support import traced_execution_policy, circle_initial_source_policy
            if identity.get('execution_profile') not in ('paired_restart_trace/1', 'paired_restart_trace/2'):
                raise ValueError('repeat traced execution policy differs')
            policy = circle_initial_source_policy({'circle_generation': {
                'enabled': True, 'execution_profile': identity.get('execution_profile')}})
            expected = traced_execution_policy(policy)
            if digest({key: identity.get(key) for key in expected}) != digest(expected):
                raise ValueError('repeat traced execution policy differs')
        for key in ("task_hash", "canonical_initial_source_sha256", "task_evaluator_sha256"):
            if not isinstance(identity[key], str) or len(identity[key]) != 64 or any(c not in "0123456789abcdef" for c in identity[key]):
                raise ValueError(f"invalid repeat identity {key}")
        if not isinstance(identity["task"], str) or not identity["task"] or not isinstance(identity["adapter"], dict):
            raise ValueError("invalid task/adapter identity")
        if "integration_sha256" in identity:
            pins = identity["integration_sha256"]
            if (not isinstance(pins, dict) or not pins or any(not isinstance(k, str) or not k
                    or not isinstance(v, str) or len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
                    for k, v in pins.items())):
                raise ValueError("invalid integration source hashes")
        self.identity = deepcopy(identity)
        self.contract_hash = digest({"contract": self.CONTRACT, "seed_contract": SEED_CONTRACT,
                                     "config": self.config, "identity": self.identity})
        self.directory = Path(run_dir) / self.DIRECTORY
        self.journal = Journal(self.directory / "journal.jsonl")
        self.draw_fn = draw_fn
        self._lock = None
        self._ack = None
        self._data = None

    def _acquire(self):
        self._lock = (self.directory / "coordinator.lock").open("a+b")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock.close()
            self._lock = None
            raise RepeatPromotionError("repeat coordinator already active") from None

    def close(self):
        if self._lock is not None:
            self._lock.close()
            self._lock = None

    def start(self, run_id):
        if self._data is not None:
            if run_id != self._data["run_id"]:
                raise RepeatPromotionError("repeat run ID changed")
            return
        if not isinstance(run_id, str) or not run_id or len(run_id) > 200:
            raise ValueError("invalid repeat run ID")
        self.directory.mkdir()  # exclusive; an existing run must use load_state
        fsync_directory(self.directory.parent)
        self._acquire()
        (self.directory / "sources").mkdir()
        (self.directory / "receipts").mkdir()
        fsync_directory(self.directory)
        self._append({"type": "header", "run_id": run_id, "contract_hash": self.contract_hash,
                      "config": self.config, "identity": self.identity})

    @property
    def pending(self):
        return bool(self._data and self._data["batches"] and self._data["batches"][-1]["decision"] is None)

    @property
    def stopped_reason(self):
        return self._data["stopped_reason"] if self._data else None

    def _batch(self):
        return self._data["batches"][-1]

    def _append(self, event):
        self._check_live()
        self.journal.append(deepcopy(event))
        self._apply(event)

    def _check_live(self):
        if self._lock is None:
            raise RepeatPromotionError("repeat coordinator is not active")
        if digest({"contract": self.CONTRACT, "seed_contract": SEED_CONTRACT,
                   "config": self.config, "identity": self.identity}) != self.contract_hash:
            raise RepeatPromotionError("repeat configuration/identity changed in memory")

    def _source(self, sha):
        code = (self.directory / "sources" / (sha + ".py")).read_bytes().decode("utf-8")
        if _source_sha(code) != sha:
            raise RepeatPromotionError("retained source hash mismatch")
        return code

    def checkpoint_ack(self, anchor):
        self._check_live()
        if anchor != self.journal.anchor:
            raise RepeatPromotionError("checkpoint does not bind the current repeat ledger")
        self._ack = deepcopy(anchor)
