"""Lexical geometry invariants, not semantic-family equivalence tests."""
from copy import deepcopy
import json
import math

import pytest

from sciel.components.selection.card_space import card_novelty, card_vector, cosine_distance


BASE = {"family": "grid", "strategy": "place circles", "mechanism": "adjust spacing"}
OTHER = {"family": "anneal", "strategy": "minimize energy", "mechanism": "temperature schedule"}


def test_json_layout_key_order_and_outer_space_do_not_change_geometry():
    alt = {key: " " + BASE[key] + " " for key in reversed(BASE)}
    one, first = card_vector(BASE)
    two, second = card_vector("IDEA_CARD_JSON: " + json.dumps(alt, indent=2))
    assert one == two and cosine_distance(one, two) == 0
    assert first["canonical_sha256"] == second["canonical_sha256"]
    assert first["sha256"] != second["sha256"]  # Raw inputs remain separately identified.
    assert math.fsum(v*v for v in one.values()) == pytest.approx(1)


def test_field_channels_and_equal_field_weight_are_explicit():
    one, _ = card_vector(BASE)
    shifted, _ = card_vector({**BASE, "family": BASE["strategy"], "strategy": BASE["family"]})
    assert cosine_distance(one, shifted) > 0
    for field in BASE:
        assert math.fsum(value*value for (name, _), value in one.items() if name == field) == pytest.approx(1/3)
    repeated, _ = card_vector({key: " ".join([value] * 10) for key, value in BASE.items()})
    assert cosine_distance(one, repeated) == 0  # Repetition is not novelty.


@pytest.mark.parametrize("left,right", [("x < 0", "x > 0"), ("x <= 0", "x < 0"),
    ("x + y", "x - y"), ("X", "x"), ("match 'a b'", "match 'a  b'"),
    ('match "a b"', 'match "a  b"'), ("scale 0.99", "scale 1.01")])
def test_operators_case_numbers_and_quoted_literals_are_not_erased(left, right):
    one, _ = card_vector({**BASE, "mechanism": left})
    two, _ = card_vector({**BASE, "mechanism": right})
    assert 0 < cosine_distance(one, two) <= 1


def test_order_outside_literals_is_a_declared_bag_of_words_limitation():
    one, _ = card_vector({**BASE, "mechanism": "move inner before outer"})
    two, _ = card_vector({**BASE, "mechanism": "move outer before inner"})
    assert cosine_distance(one, two) == 0
    # This collision is why lexical similarity cannot establish semantic equality.


@pytest.mark.parametrize("value,kind", [(None,"missing"),("", "missing"),("  ","missing"),
    ("old prose strategy", "legacy"), ("{" ,"malformed"), ([1],"malformed"),
    ({**BASE,"extra":"oops"},"malformed"), ({**BASE,"family":1},"malformed"),
    ('{"family":"one","family":"two","strategy":"a","mechanism":"b"}',"malformed"),
    ("x"*8193,"oversized"), ({**BASE,"family":"x"*129},"malformed")])
def test_legacy_missing_malformed_and_oversized_have_no_invented_vector(value, kind):
    vector, metadata = card_vector(value)
    assert not vector and metadata["kind"] == kind
    with pytest.raises(ValueError, match="no measured distance"):
        cosine_distance(vector, {})


def test_identical_cluster_and_rare_card_nearest_geometry_with_neutral_legacy():
    cards = {"a":BASE, "b":BASE, "c":BASE, "d":OTHER, "legacy":"untyped plan", "empty":None}
    before = deepcopy(cards)
    result = card_novelty(cards, k=2)
    assert result["available"] and cards == before
    assert result["novelty"] == {"a":0., "b":0., "c":0., "d":1., "legacy":.25, "empty":.25}
    assert result["typed_ids"] == ["a","b","c","d"] and result["neighbors"] == 2
    assert result == card_novelty(dict(reversed(list(cards.items()))), k=2)


def test_population_removal_recomputes_geometry_without_archive_or_cache():
    first = card_novelty({"a":BASE,"b":BASE,"c":OTHER}, k=1)
    assert first["novelty"] == {"a":0.,"b":0.,"c":1.}
    second = card_novelty({"a":BASE,"c":OTHER}, k=1)
    assert second["novelty"] == {"a":1.,"c":1.} and second["available"]
    assert card_novelty({"a":BASE,"b":BASE}, k=1)["reason"] == "zero_lexical_diversity"


@pytest.mark.parametrize("cards", [{},{"a":BASE},{"a":BASE,"legacy":"old text"}])
def test_insufficient_structured_support_disables_tilt(cards):
    result = card_novelty(cards)
    assert not result["available"] and result["reason"] == "too_few_structured_cards"
    assert set(result["novelty"]) == set(cards)
    assert all(v == 0 for v in result["novelty"].values())


@pytest.mark.parametrize("k", [0,-1,True,1.5,None])
def test_invalid_neighbor_count_rejected(k):
    with pytest.raises(ValueError): card_novelty({"a":BASE},k=k)
