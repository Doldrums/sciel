"""Present paired measurements and lossless lesson evidence references."""
from __future__ import annotations

import copy
import re
from ..hypothesis.observed_facts import observed_comparison
from .paired_comparisons import SCHEMA as GROUP_SCHEMA, validate_comparison, source_pair_view, validate_source_pair_view
from .presentation_contracts import (
    CARD_SCHEMA,
    COMPARISON_CARD_SCHEMA,
    LEGACY_COMPARISON_CARD_SCHEMA,
    PAIRED_LESSON_SCHEMA,
    _PAIRED_LESSON_FIELDS,
    _PAIRED_LESSON_VIEWS,
    _reflection_digest,
)


def comparison_card(group):
    """Whole three-pair projection, bound to full receipt evidence by hashes.

    This view is not an importable receipt and never supplies an artificial
    cascade score/delta. Source and adapter hashes bind data, not authenticity.
    """
    group = validate_comparison(group)
    fields = ("id", "namespace", "batch_id", "branch", "origin_experiment_id",
              "comparator_source_sha256", "challenger_source_sha256", "task_metric",
              "metric", "limits", "derived", "forecast", "scope")
    result = {key: copy.deepcopy(group[key]) for key in fields}
    result.update(schema=COMPARISON_CARD_SCHEMA if "source_pair" in group else LEGACY_COMPARISON_CARD_SCHEMA,
                  evidence_sha256=_reflection_digest(group),
                  adapter_sha256=_reflection_digest(group["adapter_identity"]),
                  receipt_sha256=[{"seed": p["seed"], **{
                      role: _reflection_digest(p[role]) if p[role] is not None else None
                      for role in ("comparator", "challenger")}} for p in group["pairs"]])
    if "source_pair" in group:
        result["source_context"] = source_pair_view(group)
    result["view_sha256"] = _reflection_digest(result)
    return result

def _comparison_view(record):
    if record.get("schema") == GROUP_SCHEMA:
        return comparison_card(record)
    if record.get("schema") in (COMPARISON_CARD_SCHEMA, LEGACY_COMPARISON_CARD_SCHEMA):
        # Full evidence is validated at import/state load. The prompt projection
        # has its own integrity binding; it is never accepted as KB evidence.
        fields = {"schema", "id", "namespace", "batch_id", "branch", "origin_experiment_id",
                  "comparator_source_sha256", "challenger_source_sha256", "task_metric",
                  "metric", "limits", "derived", "forecast", "scope", "evidence_sha256",
                  "adapter_sha256", "receipt_sha256", "view_sha256"}
        if record['schema'] == COMPARISON_CARD_SCHEMA:
            fields.add('source_context')
            if set(record) != fields:
                raise ValueError('paired comparison card fields differ')
            validate_source_pair_view(record.get('source_context'),
                record['comparator_source_sha256'], record['challenger_source_sha256'])
            if record['source_context']['omission_reason'] == 'source_pair_absent':
                raise ValueError('source-aware paired card cannot claim missing source pair')
        if (set(record) != fields or record["view_sha256"] != _reflection_digest(
                {k: v for k, v in record.items() if k != "view_sha256"})):
            raise ValueError("paired comparison card integrity mismatch")
        return copy.deepcopy(record)
    return None

def omit_source_context(card):
    """Remove the entire optional source diff, preserving every measured fact.

    This is a capacity projection only. Full source evidence remains in the KB;
    callers still admit or omit the resulting whole card, never partial hunks.
    """
    if not isinstance(card, dict) or card.get("schema") != COMPARISON_CARD_SCHEMA:
        raise ValueError("source omission requires a source-aware paired card")
    result = _comparison_view(card)
    result["source_context"].update(available=False, changes=[],
                                    omission_reason="presentation_budget")
    result["view_sha256"] = _reflection_digest(
        {key: value for key, value in result.items() if key != "view_sha256"})
    return result

def _comparison(record):
    view = _comparison_view(record)
    if view is None:
        return observed_comparison(record)
    return {"schema": view["schema"], "id": view["id"],
            "evidence_sha256": view["evidence_sha256"], "derived": view["derived"],
            "metric": view["metric"], "forecast": view["forecast"]}

def _paired_lesson_parts(card):
    """Validate only the explicit reference domain of an eligible /7 card."""
    sources, ids, catalog = (card.get(key) for key in
                             ("source_observations", "source_ids", "comparison_catalog"))
    def ident(value):
        return type(value) is str and bool(value)
    if (type(sources) is not list or type(ids) is not list or type(catalog) is not dict
            or not all(ident(i) for i in ids) or len(set(ids)) != len(ids)
            or any(type(row) is not dict or not ident(row.get('id')) for row in sources)
            or [row['id'] for row in sources] != ids):
        raise ValueError('paired lesson source membership mismatch')
    projections = {}
    for row in sources:
        if row.get('schema') not in _PAIRED_LESSON_VIEWS:
            continue
        try:
            view = _comparison_view(row)
            projections[row['id']] = {key: copy.deepcopy(view[key]) for key in _PAIRED_LESSON_FIELDS}
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('invalid paired lesson source view') from exc
    return catalog, projections

def compact_paired_lesson_card(card):
    """Factor exact paired-source comparison projections, never opaque data.

    The caller supplies the already projected /7 prompt card. No eligible paired
    source means detached passthrough, including legacy/minimal ordinary cards.
    An eligible but inconsistent projection raises instead of being repaired.
    This codec changes neither source support nor the decoded card schema.
    """
    from .catalogs import canonical_json, canonical_sha
    canonical_json(card)
    if type(card) is not dict or card.get('schema') != CARD_SCHEMA:
        raise ValueError('paired lesson compaction requires an observation card')
    result = copy.deepcopy(card)
    sources = result.get('source_observations')
    if (type(sources) is not list or not any(type(row) is dict and
            row.get('schema') in _PAIRED_LESSON_VIEWS for row in sources)):
        return result
    if 'expanded_sha256' in result:
        raise ValueError('paired lesson original contains reserved transport field')
    catalog, projections = _paired_lesson_parts(result)
    for ident, projection in projections.items():
        if ident not in catalog or canonical_json(catalog[ident]) != canonical_json(projection):
            raise ValueError('paired lesson comparison projection mismatch')
    # A token is legal only when this codec creates it at an authorized source
    # slot. Nested dictionaries/text and exposure-only comparisons stay literal.
    if any(type(item) is dict and 'paired_ref' in item for item in catalog.values()):
        raise ValueError('paired lesson original contains a reference token')
    result['schema'] = PAIRED_LESSON_SCHEMA
    result['expanded_sha256'] = canonical_sha(card)
    for ident in projections:
        catalog[ident] = {'paired_ref': True}
    return result

def expand_paired_lesson_card(wire):
    """Restore one exact /7 card; hashes check integrity, not authenticity.

    References exist only at top-level comparison_catalog slots, with unique
    same-ID supported paired views in source_observations/source_ids. Neither
    arbitrary nested lookalikes nor exposure observations are reference targets.
    """
    from .catalogs import canonical_json, canonical_sha
    canonical_json(wire)
    if type(wire) is not dict or wire.get('schema') not in (CARD_SCHEMA, PAIRED_LESSON_SCHEMA):
        raise ValueError('invalid paired lesson transport schema')
    result = copy.deepcopy(wire)
    if result['schema'] == CARD_SCHEMA:
        return result  # Keep the historical plain-card decoder behavior.
    digest = result.pop('expanded_sha256', None)
    if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError('invalid paired lesson expanded hash')
    result['schema'] = CARD_SCHEMA
    catalog, projections = _paired_lesson_parts(result)
    refs = []
    for ident, item in catalog.items():
        if type(item) is dict and 'paired_ref' in item:
            if set(item) != {'paired_ref'} or item['paired_ref'] is not True or ident not in projections:
                raise ValueError('invalid paired lesson reference target or token')
            refs.append(ident)
        elif ident in projections and canonical_json(item) != canonical_json(projections[ident]):
            raise ValueError('paired lesson comparison projection mismatch')
    if not refs or any(ident not in catalog for ident in projections):
        raise ValueError('paired lesson transport missing comparison reference')
    for ident in refs:
        catalog[ident] = copy.deepcopy(projections[ident])
    if canonical_sha(result) != digest:
        raise ValueError('paired lesson expanded hash mismatch')
    return result
