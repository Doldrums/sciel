"""Replayable trial quotas independent of resource balances."""
from __future__ import annotations

import hashlib
from copy import deepcopy

from .resource_values import _canonical, _text, resolution


class FundingPolicy:
    """Pure, replayable trial policy; its owner durably stores state() changes.

    Quotas are *next-block step caps*, never resource balances or new money.
    The caller supplies independently assessed information/evidence; this class
    neither judges scientific utility nor grants permission to spend resources.
    Replication blocks retain evidence and do not alter a search penalty streak.
    """

    def __init__(self, initial_quota=8, no_information_blocks=2):
        if type(initial_quota) is not int or not 1 <= initial_quota <= 10**9:
            raise ValueError("initial trial quota must be a bounded positive integer")
        if type(no_information_blocks) is not int or not 1 <= no_information_blocks <= 1000:
            raise ValueError("invalid no-information block threshold")
        self.initial_quota = initial_quota
        self.no_information_blocks = no_information_blocks
        self._events, self._lineages, self._versions, self._blocks = [], {}, {}, {}

    def create_lineage(self, lineage_id):
        lineage_id = _text(lineage_id, "lineage ID")
        if lineage_id not in self._lineages:
            self._lineages[lineage_id] = dict(lineage_id=lineage_id, next_quota=self.initial_quota,
                                             no_information_streak=0, status="trial")
            self._events.append(dict(op="create_lineage", lineage_id=lineage_id))
        return deepcopy(self._lineages[lineage_id])

    def bind_version(self, version_id, lineage_id, *, parent_versions=()):
        version_id, lineage_id = _text(version_id, "version ID"), _text(lineage_id, "lineage ID")
        if lineage_id not in self._lineages:
            raise ValueError("allocate the lineage explicitly before binding a version")
        if type(parent_versions) not in (list, tuple) or len(parent_versions) > 64:
            raise ValueError("invalid parent versions")
        parents = [_text(p, "parent version") for p in parent_versions]
        if len(set(parents)) != len(parents) or any(p not in self._versions for p in parents):
            raise ValueError("parent versions must be distinct known versions")
        if parents and lineage_id not in {self._versions[p]["lineage_id"] for p in parents}:
            raise ValueError("a revision must inherit an existing parent funding lineage")
        event = dict(op="bind_version", version_id=version_id, lineage_id=lineage_id, parent_versions=parents)
        if version_id in self._versions:
            if self._versions[version_id] != event:
                raise ValueError("version funding lineage cannot change")
        else:
            self._versions[version_id] = deepcopy(event)
            self._events.append(event)
        return deepcopy(self._lineages[lineage_id])

    def record_block(self, lineage_id, block_id, *, informative, evidence_refs, replication=False):
        _text(block_id, "block ID")
        if lineage_id not in self._lineages or type(informative) is not bool or type(replication) is not bool:
            raise ValueError("known lineage and explicit information/replication flags required")
        audit = resolution("assessed block", evidence_refs)
        event = dict(op="record_block", lineage_id=lineage_id, block_id=block_id,
                     informative=informative, evidence_refs=audit["evidence_refs"], replication=replication)
        if block_id in self._blocks:
            if self._blocks[block_id] != event:
                raise ValueError("block ID already binds different evidence")
            return deepcopy(self._lineages[lineage_id])
        row = self._lineages[lineage_id]
        if not replication:
            if row["status"] == "dormant":
                raise ValueError("dormant lineage requires explicit new trial authorization")
            row["no_information_streak"] = 0 if informative else row["no_information_streak"] + 1
            if row["no_information_streak"] >= self.no_information_blocks:
                row["next_quota"] //= 2
                row["no_information_streak"] = 0
            row["status"] = "active" if row["next_quota"] else "dormant"
        self._blocks[block_id] = deepcopy(event)
        self._events.append(event)
        return deepcopy(row)

    def authorize_trial(self, lineage_id, quota, *, reason, evidence_refs):
        if lineage_id not in self._lineages or type(quota) is not int or not 1 <= quota <= self.initial_quota:
            raise ValueError("rescue trial must be explicit and bounded by the initial quota")
        audit = resolution(reason, evidence_refs)
        row = self._lineages[lineage_id]
        if row["status"] != "dormant":
            raise ValueError("new trial authorization only resumes a dormant lineage")
        row.update(next_quota=quota, no_information_streak=0, status="trial")
        self._events.append(dict(op="authorize_trial", lineage_id=lineage_id, quota=quota, **audit))
        return deepcopy(row)

    def snapshot(self, lineage_id=None):
        return deepcopy(self._lineages if lineage_id is None else self._lineages[lineage_id])

    def state(self):
        body = dict(schema="sciel-method-funding/1", initial_quota=self.initial_quota,
                    no_information_blocks=self.no_information_blocks, events=deepcopy(self._events))
        return dict(body, sha256=hashlib.sha256(_canonical(body)).hexdigest())

    @classmethod
    def from_state(cls, state):
        if type(state) is not dict or set(state) != {"schema", "initial_quota", "no_information_blocks", "events", "sha256"}:
            raise ValueError("invalid funding policy state")
        body = {k: v for k, v in state.items() if k != "sha256"}
        if (body["schema"] != "sciel-method-funding/1" or type(body["events"]) is not list
                or hashlib.sha256(_canonical(body)).hexdigest() != state["sha256"]):
            raise ValueError("funding policy identity mismatch")
        policy = cls(body["initial_quota"], body["no_information_blocks"])
        for event in body["events"]:
            if type(event) is not dict or event.get("op") not in (
                    "create_lineage", "bind_version", "record_block", "authorize_trial"):
                raise ValueError("invalid funding policy event")
            before = len(policy._events)
            try:
                getattr(policy, event["op"])(**{k: deepcopy(v) for k, v in event.items() if k != "op"})
            except (TypeError, KeyError) as exc:
                raise ValueError("malformed funding policy event") from exc
            if len(policy._events) != before + 1 or policy._events[-1] != event:
                raise ValueError("noncanonical funding policy event")
        return policy
