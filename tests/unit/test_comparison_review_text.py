"""Pure text-delivery ablation: identical facts, card selection and allocation.

No model, candidate, evaluator, helper or historical Engine is executed.
"""
import copy
import hashlib

import pytest

from sciel.config import apply_defaults
from sciel.hypothesis import comparison_review as review
from sciel.knowledge_base.catalogs import canonical_json
from tests.unit.test_comparison_review import WRITER, cards_for, group, proposal


def pair(cards, groups, *, branch="p1", max_chars=10000, **options):
    return tuple(review.context(cards, groups, branch, max_chars,
                 text_enabled=flag, **options) for flag in (True, False))


def without_text(view):
    result = copy.deepcopy(view)
    result.pop("text_enabled")
    result.pop("rendered_chars")
    for card in result["cards"]:
        card.pop("unverified_interpretation")
        card.pop("interpretation_omission")
    return result


def assert_pair(on, off, cards, groups, *, branch="p1", enabled=True):
    assert (on is None) == (off is None)
    if on is None:
        return
    assert without_text(on) == without_text(off)
    assert on["schema"] == off["schema"] == review.TEXT_CONTEXT_SCHEMA
    assert on["text_enabled"] is True and off["text_enabled"] is False
    assert on["reserved_chars"] == off["reserved_chars"] == on["rendered_chars"] + 1
    for flag, view in ((True, on), (False, off)):
        assert view["rendered_chars"] == len(canonical_json(view))
        assert view["rendered_chars"] <= view["reserved_chars"] <= view["max_chars"]
        assert review.validate_context(view, cards, groups, branch,
                   enabled=enabled, text_enabled=flag) == view
        for card in view["cards"]:
            stored = cards[card["comparison_id"]]
            assert card["schema"] == review.TEXT_CARD_SCHEMA
            assert card["host_facts"] == stored["host_facts"]
            assert card["writer"] == stored["writer"]
            assert card["comparison_sha256"] == stored["comparison_sha256"]
            if flag:
                assert card["unverified_interpretation"] == stored["unverified_interpretation"]
                assert card["interpretation_omission"] is None
            else:
                assert card["unverified_interpretation"] is None
                assert card["interpretation_omission"] == "comparison_review_text_disabled"


def test_omitted_setting_keeps_legacy_wire_without_ablation_fields():
    groups = [group()]
    cards = cards_for(groups)
    default = review.context(cards, groups, "p1", 10000)
    explicit_none = review.context(cards, groups, "p1", 10000, text_enabled=None)
    assert canonical_json(default) == canonical_json(explicit_none)
    assert default["schema"] == "sciel-comparison-review-context/1"
    assert "reserved_chars" not in default and "text_enabled" not in default
    assert default["cards"] == list(cards.values())
    assert "interpretation_omission" not in default["cards"][0]
    # Pinned before the optional /2 presentation was added.
    assert hashlib.sha256(canonical_json(default).encode()).hexdigest() == "b12e4f834a662918b05db5149456b087b82abad5f3a4aa4cf1f9ad2f07dcf431"


def test_text_switch_preserves_every_host_field_and_detaches_both_views():
    groups = [group(1), group(2)]
    cards = cards_for(groups)
    before = copy.deepcopy((cards, groups))
    on, off = pair(cards, groups)
    assert_pair(on, off, cards, groups)
    assert on["presented_count"] == 2
    assert [c["comparison_id"] for c in on["cards"]] == [g["id"] for g in reversed(groups)]
    assert off["rendered_chars"] < on["rendered_chars"]
    off["cards"][0]["host_facts"]["derived"]["all_no_loss"] = True
    on["cards"][0]["unverified_interpretation"]["interpretation"] = "changed"
    assert (cards, groups) == before
    assert all(c["schema"] == review.CARD_SCHEMA for c in cards.values())
    assert review.validate_cards(cards, groups) == cards


def test_off_does_not_use_saved_text_space_to_admit_an_extra_card():
    groups = [group(1), group(2)]
    cards = cards_for(groups)
    full_on, full_off = pair(cards, groups)
    assert full_on["presented_count"] == 2
    cap = full_on["reserved_chars"] - 10
    assert full_off["rendered_chars"] < cap
    on, off = pair(cards, groups, max_chars=cap)
    assert_pair(on, off, cards, groups)
    assert on["presented_count"] == off["presented_count"] == 1
    assert on["omitted"] == off["omitted"]
    assert on["omitted"][0]["reason"] == "memory_char_limit"
    # Omission commitments describe the same complete ON card in both arms.
    omitted = on["omitted"][0]
    whole = next(c for c in full_on["cards"] if c["comparison_id"] == omitted["comparison_id"])
    assert omitted["card_chars"] == len(canonical_json(whole))
    assert omitted["card_sha256"] == hashlib.sha256(canonical_json(whole).encode()).hexdigest()


@pytest.mark.parametrize("options", [{"max_items": 0}, {"max_items": 1}, {"enabled": False}])
def test_branch_item_and_disabled_limits_are_shared(options):
    groups = [group(1), group(2, branch="elsewhere"), group(3)]
    cards = cards_for(groups)
    on, off = pair(cards, groups, **options)
    assert_pair(on, off, cards, groups, enabled=options.get("enabled", True))
    assert on["retained_count"] == 2
    assert all(c["branch"] == "p1" for c in on["cards"])
    assert all(x["comparison_id"] != groups[1]["id"] for x in on["omitted"])
    assert on["presented_count"] == (0 if options.get("enabled") is False
        else options["max_items"])


@pytest.mark.parametrize("has_retained_card", [False, True])
def test_empty_and_fully_omitted_channels_cover_the_longer_false_flag(has_retained_card):
    groups = [group()] if has_retained_card else []
    cards = cards_for(groups)
    on, off = pair(cards, groups, max_items=0)
    assert_pair(on, off, cards, groups)
    assert on["presented_count"] == off["presented_count"] == 0
    assert off["rendered_chars"] == on["rendered_chars"] + 1 == off["reserved_chars"]
    # max_chars changes from five digits to four, saving one wire character.
    cap = on["reserved_chars"] - 2
    on, off = pair(cards, groups, max_items=0, max_chars=cap)
    assert_pair(on, off, cards, groups)
    if not has_retained_card:
        assert on is off is None


@pytest.mark.parametrize("cap", [0, 1, 1000, 1600, 2400, 4000, 10000])
def test_whole_channel_admission_and_omission_details_have_identical_limits(cap):
    groups = [group(i) for i in range(4)]
    cards = cards_for(groups)
    on, off = pair(cards, groups, max_chars=cap)
    assert_pair(on, off, cards, groups)
    if on is not None:
        assert on["presented_count"] + on["omitted_count"] == 4
        assert len(on["omitted"]) + on["omission_details_omitted_count"] == on["omitted_count"]


def test_unicode_is_charged_as_escaped_wire_text_and_tiny_text_still_fits_reserve():
    groups = [group()]
    for text in ("x", "\u2603" * 100):
        p = proposal(groups[0], hypothesis=False)
        p["interpretation"] = p["limitations"] = text
        cards = review.build_cards([p], groups, WRITER)
        on, off = pair(cards, groups)
        assert_pair(on, off, cards, groups)
        assert on["presented_count"] == 1
        if text != "x":
            assert "\\u2603" in canonical_json(on)
            assert on["rendered_chars"] - off["rendered_chars"] > 1000


@pytest.mark.parametrize("tamper", [
    lambda v: v["cards"][0]["host_facts"]["derived"].update(all_no_loss=True),
    lambda v: v["cards"][0].update(unverified_interpretation={"interpretation": "leaked"}),
    lambda v: v["cards"][0].pop("interpretation_omission"),
    lambda v: v["cards"][0].update(interpretation_omission=None),
    lambda v: v["cards"][0]["writer"].update(reference=999),
    lambda v: v.update(reserved_chars=v["rendered_chars"]),
    lambda v: v.update(rendered_chars=v["rendered_chars"] - 1),
    lambda v: v.update(text_enabled=True),
])
def test_disabled_context_cannot_change_facts_leak_text_or_refund_reservation(tamper):
    groups = [group()]
    cards = cards_for(groups)
    _, off = pair(cards, groups)
    tamper(off)
    with pytest.raises(ValueError, match="authoritative cards"):
        review.validate_context(off, cards, groups, "p1", text_enabled=False)


def test_mode_cannot_be_inferred_from_an_untrusted_context_or_stored_as_a_card():
    groups = [group()]
    cards = cards_for(groups)
    on, off = pair(cards, groups)
    for view, setting in ((on, False), (off, True), (on, None), (off, None)):
        with pytest.raises(ValueError, match="authoritative cards"):
            review.validate_context(view, cards, groups, "p1", text_enabled=setting)
    for view in (on, off):
        with pytest.raises(ValueError):
            review.validate_cards({view["cards"][0]["comparison_id"]: view["cards"][0]}, groups)


@pytest.mark.parametrize("value", [0, 1, "false", [], {}, None])
def test_config_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="comparison_review_text must be boolean"):
        apply_defaults({"research": {"comparison_reviews": True, "structured_auxiliary": True,
                                     "comparison_review_text": value}})


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("review_setting", [{}, {"comparison_reviews": False}])
def test_text_switch_requires_retained_review_channel(value, review_setting):
    with pytest.raises(ValueError, match="comparison_review_text requires comparison_reviews"):
        apply_defaults({"research": {**review_setting, "comparison_review_text": value}})


def test_config_omission_and_both_explicit_modes_remain_distinct_and_idempotent():
    raw = {"research": {"comparison_reviews": True, "structured_auxiliary": True}}
    before = copy.deepcopy(raw)
    legacy = apply_defaults(raw)
    assert raw == before
    assert "comparison_review_text" not in legacy["research"]
    for value in (True, False):
        cfg = apply_defaults({"research": {**raw["research"], "comparison_review_text": value}})
        assert cfg["research"]["comparison_review_text"] is value
        assert apply_defaults(cfg) == cfg
        cfg["research"].pop("comparison_review_text")
        assert cfg == legacy


@pytest.mark.parametrize("value", [0, 1, "false", [], {}])
def test_direct_context_rejects_nonboolean_text_delivery(value):
    with pytest.raises(ValueError, match="text enabled flag"):
        review.context({}, [], "p1", 10000, text_enabled=value)
