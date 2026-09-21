"""Project and validate whole initializer-witness cards within context budgets."""
from __future__ import annotations

import copy
import re
from .presentation_contracts import (
    INITIALIZER_CARD_SCHEMA,
    INITIALIZER_COMPARISON_SCOPE,
    INITIALIZER_CONTEXT_SCHEMA,
    LEGACY_INITIALIZER_CONTEXT_SCHEMA,
    ORIGIN_INITIALIZER_CONTEXT_SCHEMA,
    _reflection_digest,
)


def initializer_context(observation_ids, bindings, *, seen, comparisons_for, branches, max_items=3, max_chars=12000):
    """Separate, whole finite-witness cards; never expand ordinary observations."""
    from ..hypothesis.initializer_evidence import evidence_view
    from .catalogs import canonical_json
    if (type(observation_ids) is not list or len(observation_ids)>64
            or any(type(i) is not str or not i or len(i)>128 for i in observation_ids)
            or len(set(observation_ids)) != len(observation_ids)):
        raise ValueError("invalid initializer context observation IDs")
    if type(max_items) is not int or max_items < 0 or type(max_chars) is not int or max_chars < 0:
        raise ValueError("invalid initializer context budget")
    if not observation_ids or max_chars==0:
        return None
    cards, comparisons, omitted, compared, compared_owners = [], [], [], [], []

    def block():
        ids = [card["observation_id"] for card in cards]
        keys = [(c["observation_id"], c["other_observation_id"]) for c in comparisons]
        missing = [(c["observation_id"], c["other_observation_id"]) for c in compared
                   if (c["observation_id"], c["other_observation_id"]) not in keys]
        skipped_owners=[ident for ident in observation_ids if ident in bindings and ident not in compared_owners]
        value = dict(schema=INITIALIZER_CONTEXT_SCHEMA, cards=cards, comparisons=comparisons,
            selected_ids=observation_ids, selected_count=len(observation_ids),
            selected_ids_truncated=False, presented_ids=ids, presented_count=len(ids),
            omitted=omitted[:16], omitted_count=len(omitted), omitted_ids_truncated=len(omitted)>16,
            available_count=sum(i in bindings for i in observation_ids),
            registry_count=len(bindings),
            comparison_owner_ids=compared_owners, comparison_owner_omitted_ids=skipped_owners[:16],
            comparison_owner_omitted_count=len(skipped_owners), comparison_owner_omitted_ids_truncated=len(skipped_owners)>16,
            not_computed_comparison_count=len(skipped_owners)*max(0,len(bindings)-1),
            compared_count=len(compared), compatible_count=sum(c["comparison"]["status"]!="incompatible" for c in compared),
            presented_comparison_count=len(comparisons), omitted_comparison_count=len(missing),
            omitted_comparisons=[list(pair) for pair in missing[:16]], omitted_comparison_ids_truncated=len(missing)>16,
            rendered_chars=0, context_sha256="0"*64)
        while value["rendered_chars"] != len(canonical_json(value)):
            value["rendered_chars"] = len(canonical_json(value))
        value["context_sha256"] = _reflection_digest({k:v for k,v in value.items() if k!="context_sha256"})
        return value

    # Account for every omission before considering complete cards. Failed fits
    # never leave an unreported selected item or a partially shown witness.
    for ident in observation_ids:
        omitted.append(dict(observation_id=ident, reason="evidence_aged_out" if ident in seen else "evidence_unavailable"))
    for ident in observation_ids:
        if ident not in bindings:
            continue
        omission = next(item for item in omitted if item["observation_id"] == ident)
        if len(cards) >= max_items:
            omission["reason"] = "item_limit"
            continue
        binding = bindings[ident]
        view = evidence_view(binding["evidence"], max_chars=8192)
        if view["status"] == "omitted":
            omission["reason"] = "evidence_view_budget"
            continue
        card = {key: copy.deepcopy(binding[key]) for key in
                ("observation_id", "observation_sha256", "prior_observation_id", "binding_sha256")}
        role_sources = {role: binding["evidence"]["sources"][label]["source_sha256"]
                        for role, label in binding["evidence"]["roles"].items()}
        card.update(schema=INITIALIZER_CARD_SCHEMA, branch=branches[ident], role_sources=role_sources, evidence=view)
        cards.append(card); omitted.remove(omission)
        if block()["rendered_chars"] > max_chars:
            cards.pop(); omission["reason"] = "presentation_budget"; omitted.append(omission)
    presented = {card["observation_id"] for card in cards}
    for card in cards:
        ident=card["observation_id"]
        compared.extend(comparisons_for(ident));compared_owners.append(ident)
    registration_order={ident:index for index,ident in enumerate(bindings)}
    compared.sort(key=lambda item: (item["comparison"]["status"]!="same_on_witnesses",
                                   -registration_order[item["other_observation_id"]]))
    for comparison in compared:
        if comparison["observation_id"] not in presented:
            continue
        comparisons.append(comparison)
        if block()["rendered_chars"] > max_chars:
            comparisons.pop()
    value = block()
    if value["rendered_chars"] > max_chars:
        return None  # Caller journals omitted selected IDs when even metadata cannot fit.
    return validate_initializer_context(value)

def validate_initializer_context(value):
    """Check a detached view's integrity, never import it as execution evidence."""
    try:
        return _validate_initializer_context(value)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("initializer context: malformed value") from exc

def _validate_initializer_context(value):
    from ..hypothesis.initializer_evidence import (validate_evidence_view, validate_evidence_comparison,
                                                  VIEW_SCHEMA, ORIGIN_VIEW_SCHEMA, LEGACY_VIEW_SCHEMA)
    from .catalogs import canonical_json
    def require(ok, reason):
        if not ok:
            raise ValueError("initializer context: " + reason)
    def ident(item):
        return type(item) is str and 0 < len(item) <= 128
    def sha(item):
        return type(item) is str and re.fullmatch('[0-9a-f]{64}', item) is not None
    fields = {"schema", "cards", "comparisons", "selected_ids", "selected_count", "selected_ids_truncated",
        "presented_ids", "presented_count", "omitted", "omitted_count", "omitted_ids_truncated", "available_count", "registry_count",
        "comparison_owner_ids", "comparison_owner_omitted_ids", "comparison_owner_omitted_count",
        "comparison_owner_omitted_ids_truncated", "not_computed_comparison_count",
        "compared_count", "compatible_count", "presented_comparison_count", "omitted_comparison_count",
        "omitted_comparisons", "omitted_comparison_ids_truncated", "rendered_chars", "context_sha256"}
    require(type(value) is dict and set(value)==fields and value["schema"] in
            (INITIALIZER_CONTEXT_SCHEMA, ORIGIN_INITIALIZER_CONTEXT_SCHEMA,
             LEGACY_INITIALIZER_CONTEXT_SCHEMA), "schema")
    expected_view = {INITIALIZER_CONTEXT_SCHEMA: VIEW_SCHEMA,
                     ORIGIN_INITIALIZER_CONTEXT_SCHEMA: ORIGIN_VIEW_SCHEMA,
                     LEGACY_INITIALIZER_CONTEXT_SCHEMA: LEGACY_VIEW_SCHEMA}[value["schema"]]
    require(sha(value["context_sha256"]) and value["context_sha256"]==_reflection_digest(
        {k:v for k,v in value.items() if k!="context_sha256"}), "digest")
    for key in ("selected_count", "presented_count", "omitted_count", "available_count", "registry_count", "compared_count",
                "compatible_count", "presented_comparison_count", "omitted_comparison_count", "rendered_chars"):
        require(type(value[key]) is int and value[key]>=0, "invalid count")
    for key in ("comparison_owner_omitted_count","not_computed_comparison_count"):
        require(type(value[key]) is int and value[key]>=0, "invalid comparison owner count")
    require(value["rendered_chars"]==len(canonical_json(value)), "rendered length")
    ids=value["selected_ids"]
    require(type(ids) is list and len(ids)<=64 and all(ident(i) for i in ids)
            and len(set(ids))==len(ids)==value["selected_count"] and value["selected_ids_truncated"] is False, "selection")
    cards=value["cards"]; require(type(cards) is list, "cards")
    by_id={}
    for card in cards:
        require(type(card) is dict and set(card)=={"schema", "observation_id", "observation_sha256",
            "prior_observation_id", "binding_sha256", "role_sources", "evidence", "branch"}
            and card["schema"]==INITIALIZER_CARD_SCHEMA, "card schema")
        owner=card["observation_id"]
        require(owner in ids and owner not in by_id and sha(card["observation_sha256"])
                and sha(card["binding_sha256"]) and ident(card["branch"]) and bool(card["branch"].strip()), "observation binding")
        view=validate_evidence_view(card["evidence"])
        require(view["schema"]==expected_view and view["status"]=="presented"
                and card["role_sources"]==view["role_sources"], "source/view binding")
        prior=card["prior_observation_id"]
        require((prior is None)==("prior" not in card["role_sources"])
                and (prior is None or (ident(prior) and prior!=owner)), "prior binding")
        by_id[owner]=card
    require(value["presented_ids"]==list(by_id) and value["presented_count"]==len(cards)
            and len(cards)<=value["available_count"]<=len(ids)
            and value["omitted_count"]==len(ids)-len(cards), "presentation counts")
    require(value["comparison_owner_ids"]==list(by_id)
            and value["comparison_owner_omitted_count"]==value["available_count"]-len(cards)
            and value["registry_count"]>=value["available_count"]
            and value["compared_count"]==len(cards)*max(0,value["registry_count"]-1)
            and value["not_computed_comparison_count"]==value["comparison_owner_omitted_count"]*max(0,value["registry_count"]-1),
            "comparison owner scope")
    uncomputed=value["comparison_owner_omitted_ids"]
    require(type(uncomputed) is list and len(uncomputed)==min(16,value["comparison_owner_omitted_count"])
            and all(ident(i) and i in ids and i not in by_id for i in uncomputed)
            and len(set(uncomputed))==len(uncomputed)
            and value["comparison_owner_omitted_ids_truncated"] is (value["comparison_owner_omitted_count"]>16),
            "comparison owner omissions")
    omitted=value["omitted"]
    require(type(omitted) is list and len(omitted)==min(16,value["omitted_count"])
            and value["omitted_ids_truncated"] is (value["omitted_count"]>16), "omission counts")
    missing=[]
    for item in omitted:
        require(type(item) is dict and set(item)=={"observation_id","reason"}
            and item["observation_id"] in ids and item["observation_id"] not in by_id
            and item["observation_id"] not in missing and item["reason"] in
            ("evidence_aged_out","evidence_unavailable","item_limit","evidence_view_budget","presentation_budget"), "omission")
        missing.append(item["observation_id"])
    comparisons=value["comparisons"];require(type(comparisons) is list, "comparisons")
    pairs=[]
    for item in comparisons:
        require(type(item) is dict and set(item)=={"observation_id","other_observation_id","observation_sha256",
            "other_observation_sha256","binding_sha256","other_binding_sha256","comparison","branch","other_branch","scope"}, "comparison fields")
        owner,other=item["observation_id"],item["other_observation_id"]
        require(owner in by_id and ident(other) and other!=owner and (owner,other) not in pairs, "comparison IDs")
        require(all(ident(item[key]) and bool(item[key].strip()) for key in ("branch","other_branch"))
                and item["scope"]==INITIALIZER_COMPARISON_SCOPE, "comparison branch/scope")
        for key in ("observation_sha256","other_observation_sha256","binding_sha256","other_binding_sha256"):
            require(sha(item[key]), "comparison binding digest")
        comparison=validate_evidence_comparison(item["comparison"])
        for side,oid,prefix in (("left",owner,""),("right",other,"other_")):
            if oid in by_id:
                card=by_id[oid]
                require(item[prefix+"observation_sha256"]==card["observation_sha256"]
                    and item[prefix+"binding_sha256"]==card["binding_sha256"]
                    and item[prefix+"branch"]==card["branch"], "comparison observation mismatch")
                require(comparison[side]=={key:card["evidence"][key]
                    for key in ("evidence_sha256","receipt_sha256","role_sources")}, "comparison source mismatch")
        pairs.append((owner,other))
    require(value["presented_comparison_count"]==len(comparisons)
            and len(comparisons)<=value["compared_count"] and 0<=value["compatible_count"]<=value["compared_count"]
            and value["omitted_comparison_count"]==value["compared_count"]-len(comparisons), "comparison counts")
    missing=value["omitted_comparisons"]
    require(type(missing) is list and len(missing)==min(16,value["omitted_comparison_count"])
            and value["omitted_comparison_ids_truncated"] is (value["omitted_comparison_count"]>16), "comparison omissions")
    seen=[]
    for pair in missing:
        require(type(pair) is list and len(pair)==2 and pair[0] in ids and ident(pair[1]) and pair[0]!=pair[1]
                and tuple(pair) not in pairs and pair not in seen, "omitted comparison IDs")
        seen.append(pair)
    return copy.deepcopy(value)
