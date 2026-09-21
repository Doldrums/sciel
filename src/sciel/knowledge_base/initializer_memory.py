"""Retain source-bound initializer witnesses and select compatible comparisons."""
from __future__ import annotations

import copy
from .records import (
    INITIALIZER_BINDING_SCHEMA,
    _digest,
    _is_comparison,
    _require,
    _snapshot,
    _text,
)

def _initializer_observations(knowledge):
    """Only retained ordinary observations can carry initializer bindings."""
    records = {e["id"]: _snapshot(e) for e in knowledge._experiments}
    for lesson in knowledge._lessons:
        snapshots = lesson["evidence_snapshots"] + lesson["exposure_snapshots"]
        snapshots += [item["experiment"] for item in lesson["inherited_exposures"]]
        records.update((e["id"], e) for e in snapshots if not _is_comparison(e))
    return records


def _age_initializer_evidence(knowledge):
    retained = knowledge._initializer_observations()
    knowledge._initializer_evidence = {ident: binding for ident, binding in knowledge._initializer_evidence.items()
        if ident in retained and (binding["prior_observation_id"] is None
                                  or binding["prior_observation_id"] in retained)}
    # A prior may age out before its owner. Keep the original digest while
    # that owner remains, preventing a changed replacement or resurrection.
    knowledge._initializer_seen = {ident: digest for ident, digest in knowledge._initializer_seen.items()
                              if ident in retained}


def _initializer_binding(knowledge, value):
    from ..hypothesis.initializer_evidence import validate_evidence
    fields = {"schema", "observation_id", "observation_sha256", "prior_observation_id",
              "evidence", "binding_sha256"}
    _require(type(value) is dict and set(value) == fields
             and value["schema"] == INITIALIZER_BINDING_SCHEMA, "invalid initializer binding schema")
    _require(value["binding_sha256"] == _digest({k: v for k, v in value.items() if k != "binding_sha256"}),
             "initializer binding digest mismatch")
    records = knowledge._initializer_observations()
    ident = value["observation_id"]
    _text(ident, "initializer observation ID", limit=128)
    record = records.get(ident)
    _require(record is not None and record["source_sha256"] is not None
             and record["parent_source_sha256"] is not None, "initializer observation source unavailable")
    _require(value["observation_sha256"] == knowledge._seen.get(ident) == _digest(record),
             "initializer observation digest mismatch")
    evidence = validate_evidence(value["evidence"], source_sha256=record["source_sha256"],
                                 parent_source_sha256=record["parent_source_sha256"])
    prior = value["prior_observation_id"]
    role = evidence["roles"].get("prior")
    _require((prior is None) == (role is None), "initializer prior role/observation mismatch")
    if prior is not None:
        _text(prior, "initializer prior ID", limit=128)
        _require(prior != ident and prior in records, "initializer prior observation unavailable")
        _require(records[prior]["source_sha256"] is not None
                 and records[prior]["source_sha256"] == evidence["sources"][role]["source_sha256"],
                 "initializer prior source mismatch")
        _require(knowledge._seen.get(prior) == _digest(records[prior]), "initializer prior observation digest mismatch")
    return copy.deepcopy(value)


def register_initializer_evidence(knowledge, observation_id, evidence, prior_observation_id=None):
    """Attach finite initializer evidence without recording another outcome."""
    _text(observation_id, "initializer observation ID", limit=128)
    value = dict(schema=INITIALIZER_BINDING_SCHEMA, observation_id=observation_id,
                 observation_sha256=knowledge._seen.get(observation_id),
                 prior_observation_id=prior_observation_id, evidence=copy.deepcopy(evidence))
    value["binding_sha256"] = _digest(value)
    binding = knowledge._initializer_binding(value)
    if observation_id in knowledge._initializer_seen:
        _require(knowledge._initializer_seen[observation_id] == binding["binding_sha256"],
                 "conflicting initializer evidence binding")
        _require(observation_id in knowledge._initializer_evidence, "initializer evidence has aged out")
        return False
    if len(knowledge._initializer_evidence) == knowledge.max_experiments:
        # Bounded full-record FIFO, with tombstones until ordinary owners
        # age out. This never evicts or mutates their measured observations.
        del knowledge._initializer_evidence[next(iter(knowledge._initializer_evidence))]
    knowledge._initializer_evidence[observation_id] = binding
    knowledge._initializer_seen[observation_id] = binding["binding_sha256"]
    return True


def initializer_evidence_for(knowledge, observation_id):
    _text(observation_id, "initializer observation ID", limit=128)
    return copy.deepcopy(knowledge._initializer_evidence.get(observation_id))


def initializer_comparisons_for(knowledge, observation_id):
    from ..hypothesis.initializer_evidence import compare_evidence
    from .presentation import INITIALIZER_COMPARISON_SCOPE
    _text(observation_id, "initializer observation ID", limit=128)
    binding = knowledge._initializer_evidence.get(observation_id)
    if binding is None:
        return []
    observations = knowledge._initializer_observations()
    comparisons = [dict(observation_id=observation_id, other_observation_id=ident,
        branch=observations[observation_id]["branch"], other_branch=observations[ident]["branch"],
        scope=INITIALIZER_COMPARISON_SCOPE,
        observation_sha256=binding["observation_sha256"], other_observation_sha256=other["observation_sha256"],
        binding_sha256=binding["binding_sha256"], other_binding_sha256=other["binding_sha256"],
        comparison=compare_evidence(binding["evidence"], other["evidence"]))
        for ident, other in reversed(list(knowledge._initializer_evidence.items())) if ident != observation_id]
    # Python's stable sort preserves newest-registration order within status.
    comparisons.sort(key=lambda item: item["comparison"]["status"] != "same_on_witnesses")
    return comparisons


def initializer_context(knowledge, observation_ids, *, max_items=3, max_chars=12000):
    from .presentation import initializer_context
    return initializer_context(observation_ids, knowledge._initializer_evidence,
                               seen=knowledge._initializer_seen, comparisons_for=knowledge.initializer_comparisons_for,
                               branches={ident:record["branch"] for ident,record in knowledge._initializer_observations().items()},
                               max_items=max_items, max_chars=max_chars)
