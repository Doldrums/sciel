"""Select complete evidence for bounded current and legacy reflection contexts."""
from __future__ import annotations

import copy
from ..hypothesis.measurement import observation_view
from ..hypothesis.circle_proposal_group import group_card
from .eligibility import output_repeat
from .catalogs import canonical_json
from .records import (
    _is_comparison,
    _require,
    _text,
)

def proposal_group_context(knowledge, branch, *, max_items=3, max_chars=12000):
    """Whole bounded proposal cards, never extra experiments or exposures."""
    from .catalogs import canonical_json
    _text(branch, "branch", limit=128)
    knowledge._limit(max_items)
    _require(type(max_chars) is int and max_chars > 0, "invalid proposal context budget")
    eligible = [e for e in knowledge._experiments if e["branch"] == branch and "proposal_group" in e]
    chosen = []

    def block(cards):
        ids = [c["request_id"] for c in cards]
        omitted = [e["id"] for e in reversed(eligible) if e["id"] not in ids]
        result = {"schema": "sciel-circle-proposal-context/1", "cards": cards,
                  "presented_ids": ids, "omitted_ids": omitted[:16],
                  "omitted_count": len(omitted), "omitted_ids_truncated": len(omitted) > 16,
                  "retained_count": len(eligible), "rendered_chars": 0}
        while result["rendered_chars"] != len(canonical_json(result)):
            result["rendered_chars"] = len(canonical_json(result))
        return result

    for experiment in reversed(eligible):
        if len(chosen) >= max_items:
            break
        trial = chosen + [group_card(experiment["proposal_group"])]
        if block(trial)["rendered_chars"] <= max_chars:
            chosen = trial
    result = block(chosen)
    _require(result["rendered_chars"] <= max_chars, "proposal omission metadata exceeds context budget")
    return result


def reflection_context(knowledge, recent_limit, *, max_experiments=64,
                       max_lessons=8, max_chars=90000, selection_epoch=None):
    """Select whole observations and evidence-complete lesson summaries.

        Older retained source measurements travel with a lesson, so consolidation
        is reachable after the recent window has advanced. Exposure is association,
        not source support: show the latest observation of each outcome class and
        disclose omissions/counts. All counterevidence stays in the KB regardless
        of what fits in a prompt. Selection never mutates or creates observations.
        Reserve one evidence-complete anchor before optional recent history.
        Pair and singleton turns alternate when structural pairs are available;
        singleton turns cover all lessons, including members of oversized pairs.
        Each feasible unit gets its own turn for fixed inventory/budgets and
        consecutive selection epochs; changing inventories or skipped epochs do
        not imply a finite wait bound. Live scheduling supplies its persisted
        reflection-attempt ordinal. Standalone callers without selection_epoch
        retain the lifetime-count/recent-window fallback (no cadence guarantee). Structural pairs share source IDs,
        polarity and scope, not necessarily rule meaning. Pair focus is atomic.
        """
    from ..hypothesis.parsing import MAX_EVIDENCE_IDS
    from ..hypothesis.rendering import serialize_reflection

    for value in (recent_limit, max_experiments, max_lessons, max_chars):
        _require(type(value) is int and value > 0, "invalid reflection context limit")

    _require(selection_epoch is None or (type(selection_epoch) is int and selection_epoch >= 0),
             "invalid reflection selection epoch")

    def project(record):
        if _is_comparison(record):
            from .presentation import comparison_card
            return comparison_card(record)
        value = {k: copy.deepcopy(v) for k, v in record.items()
                 if k not in ("code_fragment", "code_fragment_sha256", "proposal_group")}
        if "proposal_group" in record:
            value["proposal_group_card"] = group_card(record["proposal_group"])
        return observation_view(value)

    def fits(records, lessons):
        return (len(records) <= max_experiments and len(lessons) <= max_lessons and
                len(serialize_reflection(list(records.values()), lessons)) <= max_chars)

    unit_cache = {}
    def unit(lesson, *, focus=False):
        cached = unit_cache.get(lesson["id"])
        if cached is not None:
            summary, extra = cached
            return dict(summary, focus_for_consolidation=focus), extra
        summary = {k: copy.deepcopy(lesson[k]) for k in
                   ("id", "rule", "polarity", "scope", "branch", "evidence_ids")}
        if "authoring" in lesson:
            summary["authoring"] = copy.deepcopy(lesson["authoring"])
        summary["retirement_eligible"] = len(lesson["evidence_ids"]) <= MAX_EVIDENCE_IDS
        summary["focus_for_consolidation"] = focus
        # Dedupe real observations that may have multiple inherited origins.
        exposures = {e["id"]: e for e in lesson["exposure_snapshots"]}
        origins = {e["id"]: [{"kind": "direct", "from_lesson_id": lesson["id"],
                              "from_rule": lesson["rule"]}]
                   for e in lesson["exposure_snapshots"]}
        for item in lesson["inherited_exposures"]:
            ident = item["experiment"]["id"]
            exposures[ident] = item["experiment"]
            origins.setdefault(ident, []).append({"kind": "inherited",
                "from_lesson_id": item["from_lesson_id"], "from_rule": item["from_rule"]})
        counts = dict(positive=0, negative=0, zero=0, unmeasured=0)
        representatives = {}
        for e in sorted(exposures.values(), key=lambda e: (e.get("iteration") or 0, e["id"])):
            category = ("recorded_output_repeat" if output_repeat(e) else
                        "unmeasured" if e["status"] != "evaluated" else
                        "positive" if e["actual_delta"] > 0 else
                        "negative" if e["actual_delta"] < 0 else "zero")
            counts[category] = counts.get(category, 0) + 1
            for kind in dict.fromkeys(origin["kind"] for origin in origins[e["id"]]):
                representatives[(kind, category)] = e
        shown = {e["id"]: e for e in representatives.values()}
        summary["observed_exposures"] = {
            **counts, "total": len(exposures), "shown_ids": list(shown),
            "omitted_count": len(exposures) - len(shown),
            "representatives": [{"experiment_id": ident, "attribution": copy.deepcopy(origins[ident])}
                                for ident in shown],
            "interpretation": "Association after advice exposure, not causal support for this rule"}
        extra = {e["id"]: project(e) for e in lesson["evidence_snapshots"] + list(shown.values())}
        unit_cache[lesson["id"]] = summary, extra
        return dict(summary, focus_for_consolidation=focus), extra

    epoch = len(knowledge._seen) // recent_limit if selection_epoch is None else selection_epoch
    groups = {}
    for lesson in knowledge._lessons:
        if len(lesson["evidence_ids"]) <= MAX_EVIDENCE_IDS:
            key = (tuple(sorted(lesson["evidence_ids"])), lesson["polarity"],
                   lesson["scope"], lesson["branch"])
            groups.setdefault(key, []).append(lesson)
    pair_groups = [group for group in groups.values() if len(group) > 1]
    pair_count = sum(len(group) * (len(group) - 1) // 2 for group in pair_groups)
    alternate = bool(pair_count and max_lessons >= 2)
    lane_epoch = epoch // 2 if alternate else epoch

    def pair_at(index):
        # Index the cyclic pair schedule without allocating O(lessons**2)
        # pairs or serializing every possible pair at each reflection.
        for group in pair_groups:
            size = len(group) * (len(group) - 1) // 2
            if index >= size:
                index -= size
                continue
            for left in range(len(group) - 1):
                row_size = len(group) - left - 1
                if index < row_size:
                    return group[left], group[left + 1 + index]
                index -= row_size
        raise AssertionError("reflection pair index outside schedule")

    records, selected, selected_ids = {}, [], set()
    if alternate and epoch % 2 == 0:
        pair = pair_at(lane_epoch % pair_count)
        trial = {}
        summaries = []
        for lesson in pair:
            summary, extra = unit(lesson, focus=True)
            summaries.append(summary)
            for ident, record in extra.items():
                trial.setdefault(ident, record)
        if fits(trial, summaries):
            records, selected = trial, summaries
            selected_ids.update(lesson["id"] for lesson in pair)

    if not selected and knowledge._lessons:
        offset = lane_epoch % len(knowledge._lessons)
        for lesson in knowledge._lessons[offset:] + knowledge._lessons[:offset]:
            summary, extra = unit(lesson)
            if fits(extra, [summary]):
                records, selected = dict(extra), [summary]
                selected_ids.add(lesson["id"])
                break

    # The anchor's sources and counterevidence cannot be displaced. Whole
    # recent paired groups keep priority over other optional observations.
    for group in reversed(knowledge._comparisons[-min(3, recent_limit, max_experiments):]):
        trial = {**records, group["id"]: project(group)}
        if fits(trial, selected):
            records = trial
    recent = knowledge._experiments[-min(recent_limit, max_experiments):]
    admitted_recent = {}
    for observation in reversed(recent):
        if observation["id"] in records:
            continue
        trial = {**records, **admitted_recent, observation["id"]: project(observation)}
        if fits(trial, selected):
            admitted_recent[observation["id"]] = trial[observation["id"]]
    # Preserve chronological order for the selected recent observations.
    records.update((e["id"], admitted_recent[e["id"]]) for e in recent
                   if e["id"] in admitted_recent)
    if not records and recent:
        raise ValueError("single research observation exceeds reflection context")

    offset = epoch % len(knowledge._lessons) if knowledge._lessons else 0
    ordered = knowledge._lessons[offset:] + knowledge._lessons[:offset]
    for lesson in ordered:
        if len(selected) >= max_lessons:
            break
        if lesson["id"] in selected_ids:
            continue
        summary, extra = unit(lesson)
        trial = dict(records)
        for ident, record in extra.items():
            trial.setdefault(ident, record)
        # A lesson never appears with missing source evidence or a silently
        # omitted negative representative. Skip the entire unit if it cannot fit.
        if fits(trial, selected + [summary]):
            records, selected = trial, selected + [summary]
    if not fits(records, selected):
        raise ValueError("reflection context cannot fit serialization overhead")
    return copy.deepcopy(list(records.values())), selected


def legacy_reflection_context(knowledge, recent_limit, *, max_experiments=64,
                       max_lessons=8, max_chars=90000):
    """Reproduce unversioned archived R contexts; never use for live scheduling."""
    from ..hypothesis.parsing import MAX_EVIDENCE_IDS
    from ..hypothesis.rendering import serialize_reflection
    from .presentation import COMPARISON_CARD_SCHEMA, LEGACY_COMPARISON_CARD_SCHEMA

    for value in (recent_limit, max_experiments, max_lessons, max_chars):
        _require(type(value) is int and value > 0, "invalid reflection context limit")

    def project(record):
        if _is_comparison(record):
            from .presentation import comparison_card
            return comparison_card(record)
        value = {k: copy.deepcopy(v) for k, v in record.items()
                 if k not in ("code_fragment", "code_fragment_sha256", "proposal_group")}
        if "proposal_group" in record:
            value["proposal_group_card"] = group_card(record["proposal_group"])
        return observation_view(value)

    def fits(records, lessons):
        return (len(records) <= max_experiments and len(lessons) <= max_lessons and
                len(serialize_reflection(list(records.values()), lessons)) <= max_chars)

    records = {e["id"]: project(e) for e in knowledge._experiments[-min(recent_limit, max_experiments):]}
    while records and not fits(records, []):
        if len(records) == 1:
            raise ValueError("single research observation exceeds reflection context")
        records.pop(next(iter(records)))
    # Recent repeat groups get explicit slots; do not silently reinterpret
    # them as six more attempts. When the whole card cannot fit, retain the
    # original observations and leave the group in durable KB state.
    for group in reversed(knowledge._comparisons[-min(3, recent_limit, max_experiments):]):
        trial = {**records, group["id"]: project(group)}
        while not fits(trial, []):
            ordinary = next((ident for ident, item in trial.items()
                             if item.get("schema") not in (COMPARISON_CARD_SCHEMA, LEGACY_COMPARISON_CARD_SCHEMA)), None)
            if ordinary is None:
                break
            trial.pop(ordinary)
        if fits(trial, []):
            records = trial
    def unit(lesson, *, focus=False):
        summary = {k: copy.deepcopy(lesson[k]) for k in
                   ("id", "rule", "polarity", "scope", "branch", "evidence_ids")}
        if "authoring" in lesson:
            summary["authoring"] = copy.deepcopy(lesson["authoring"])
        summary["retirement_eligible"] = len(lesson["evidence_ids"]) <= MAX_EVIDENCE_IDS
        summary["focus_for_consolidation"] = focus
        # Dedupe real observations that may have multiple inherited origins.
        exposures = {e["id"]: e for e in lesson["exposure_snapshots"]}
        origins = {e["id"]: [{"kind": "direct", "from_lesson_id": lesson["id"],
                              "from_rule": lesson["rule"]}]
                   for e in lesson["exposure_snapshots"]}
        for item in lesson["inherited_exposures"]:
            ident = item["experiment"]["id"]
            exposures[ident] = item["experiment"]
            origins.setdefault(ident, []).append({"kind": "inherited",
                "from_lesson_id": item["from_lesson_id"], "from_rule": item["from_rule"]})
        counts = dict(positive=0, negative=0, zero=0, unmeasured=0)
        representatives = {}
        for e in sorted(exposures.values(), key=lambda e: (e.get("iteration") or 0, e["id"])):
            category = ("recorded_output_repeat" if output_repeat(e) else
                        "unmeasured" if e["status"] != "evaluated" else
                        "positive" if e["actual_delta"] > 0 else
                        "negative" if e["actual_delta"] < 0 else "zero")
            counts[category] = counts.get(category, 0) + 1
            for kind in dict.fromkeys(origin["kind"] for origin in origins[e["id"]]):
                representatives[(kind, category)] = e
        shown = {e["id"]: e for e in representatives.values()}
        summary["observed_exposures"] = {
            **counts, "total": len(exposures), "shown_ids": list(shown),
            "omitted_count": len(exposures) - len(shown),
            "representatives": [{"experiment_id": ident, "attribution": copy.deepcopy(origins[ident])}
                                for ident in shown],
            "interpretation": "Association after advice exposure, not causal support for this rule"}
        extra = {e["id"]: project(e) for e in lesson["evidence_snapshots"] + list(shown.values())}
        return summary, extra

    epoch = len(knowledge._seen) // recent_limit
    groups = {}
    for lesson in knowledge._lessons:
        if len(lesson["evidence_ids"]) <= MAX_EVIDENCE_IDS:
            key = (tuple(sorted(lesson["evidence_ids"])), lesson["polarity"],
                   lesson["scope"], lesson["branch"])
            groups.setdefault(key, []).append(lesson)
    pairs = [(left, right) for group in groups.values()
             for i, left in enumerate(group) for right in group[i + 1:]]
    selected, selected_ids = [], set()
    if pairs and max_lessons >= 2:
        pair = pairs[epoch % len(pairs)]
        trial = dict(records)
        summaries = []
        for lesson in pair:
            summary, extra = unit(lesson, focus=True)
            summaries.append(summary)
            for ident, record in extra.items():
                trial.setdefault(ident, record)
        if fits(trial, summaries):
            records, selected = trial, summaries
            selected_ids.update(lesson["id"] for lesson in pair)

    offset = epoch % len(knowledge._lessons) if knowledge._lessons else 0
    ordered = knowledge._lessons[offset:] + knowledge._lessons[:offset]
    for lesson in ordered:
        if len(selected) >= max_lessons:
            break
        if lesson["id"] in selected_ids:
            continue
        summary, extra = unit(lesson)
        trial = dict(records)
        for ident, record in extra.items():
            trial.setdefault(ident, record)
        # A lesson never appears with missing source evidence or a silently
        # omitted negative representative. Skip the entire unit if it cannot fit.
        if fits(trial, selected + [summary]):
            records, selected = trial, selected + [summary]
    return copy.deepcopy(list(records.values())), selected
