"""Replay committed native KB authorship without inventing training episodes.

The auxiliary graph authenticates delivery and ancestry, not the truth of a
lesson's explanation. Rejected and no-child generation calls remain calls.
"""
from __future__ import annotations

from copy import deepcopy
import json

from .types import content_hash, sha256_bytes


def _need(condition, message):
    # Imported lazily: NativeAncestry delegates its auxiliary work here.
    from .ancestry import ResolutionUnknown

    if not condition:
        raise ResolutionUnknown(message)


def _sha(text):
    return sha256_bytes(text.encode("utf-8"))


def _one(values, message):
    _need(len(values) == 1, message)
    return values[0]


def validate_node(node):
    """Validate the typed commitment; KBAncestry authenticates actual history."""
    from ..knowledge_base import authoring
    from ..knowledge_base.records import _LESSON_FIELDS
    from .types import require_hash, require_identifier

    identity, proof = node["identity"], node["proof"]
    _need(node["kind"] == "lesson_revision" and set(identity) == {"lesson_id", "snapshot_sha256"}
          and set(proof) == {"snapshot", "observed_at"}, "invalid lesson revision binding")
    snapshot = proof["snapshot"]
    _need(type(snapshot) is dict and set(snapshot) == _LESSON_FIELDS | {"authoring"},
          "lesson revision lacks exact native authoring/snapshot fields")
    require_identifier(identity["lesson_id"], "lesson identity")
    require_hash(identity["snapshot_sha256"], "lesson snapshot")
    _need(snapshot["id"] == identity["lesson_id"] and content_hash(snapshot) == identity["snapshot_sha256"],
          "lesson revision snapshot hash differs")
    authoring.validate(snapshot["authoring"], rule=snapshot["rule"], lesson_id=snapshot["id"],
        evidence={row["id"]: row for row in snapshot["evidence_snapshots"]}, polarity=snapshot["polarity"])
    observed = proof["observed_at"]
    _need(type(observed) is dict and set(observed) == {"segment_id", "event_seq", "event_sha256"},
          "invalid lesson revision observation")
    require_identifier(observed["segment_id"], "lesson observation segment")
    require_hash(observed["event_sha256"], "lesson observation event")
    _need(type(observed["event_seq"]) is int and observed["event_seq"] > 0,
          "invalid lesson observation sequence")
    _need(node["owner"] == "base" and node["owner_scope"] == "shared",
          "lesson author does not have the actual shared C/R scope")
    return deepcopy(node)
