"""Pure two-proposal transport tests: no providers, candidate execution or search."""
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest

from sciel.hypothesis import circle_proposal_pair as pair
from sciel.hypothesis import circle_restart_body as builder
from sciel.hypothesis.parsing import parse_hypothesis_value


FIRST = {"hypothesis": "Change the center perturbation", "predicted_low": -0.002,
         "predicted_high": 0.003, "reasoning": "A prospective first alternative.", "confidence": 0.4}
SECOND = {"hypothesis": "Change the radius perturbation", "predicted_low": -0.01,
          "predicted_high": 0.02, "reasoning": "An independently attributed alternative.", "confidence": 0.8}


def entry(hypothesis=FIRST, body="return ordinary_params\n"):
    return {"hypothesis": copy.deepcopy(hypothesis), "body": body}


def wire(first=None, second=None):
    return json.dumps({"candidates": [entry() if first is None else first,
                                     entry(SECOND, "return best_params.copy()") if second is None else second]},
                      ensure_ascii=False, allow_nan=False)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def test_two_complete_entries_keep_order_and_independent_immutable_forecasts():
    decoded = pair.decode_response(wire())
    assert decoded["error"] is None
    assert [item["index"] for item in decoded["entries"]] == [0, 1]
    assert [item["hypothesis"] for item in decoded["entries"]] == [
        parse_hypothesis_value(FIRST), parse_hypothesis_value(SECOND)]
    for item in decoded["entries"]:
        assert item["error"] is item["body_error"] is None
        assert isinstance(item["proposal"], builder.BodyProposal)
        assert item["proposal"].hypothesis is item["hypothesis"]
        with pytest.raises(FrozenInstanceError):
            item["proposal"].body = "return None"


@pytest.mark.parametrize("source", [
    "return ordinary_params", "return ordinary_params\n",
    "x = ordinary_params.copy()\r\nx[:52] += uniform_noise[:52] / 2\r\nreturn x\r\n",
    '# α and literal two  spaces\nx = "a\u2028b"\nreturn ordinary_params\n',
])
def test_body_bytes_are_retained_without_synthetic_wire_or_source_repair(source):
    raw = wire(entry(body=source), entry(SECOND, source))
    original = raw.encode()
    decoded = pair.decode_response(raw)
    assert decoded["error"] is None
    assert all(item["proposal"].body.encode() == source.encode() for item in decoded["entries"])
    assert raw.encode() == original


@pytest.mark.parametrize("raw", [
    None, b"{}", False, 1, "", "null", "[]", "{}", '{"candidates":null}',
    '{"candidates":{}}', '{"candidates":[]}', '{"candidates":[{}]}',
    '{"candidates":[{},{},{}]}', '{"candidates":[{},{}],"extra":1}',
    "```json\n" + wire() + "\n```", "HYPOTHESIS_JSON:" + wire(),
    "before " + wire(), wire() + " after", wire() + wire(), wire()[:-1],
    "\ufeff" + wire(),
    pytest.param('{"candidates":[' + "[" * 1200 + "0", id="deep_unclosed_array"),
])
def test_invalid_transport_returns_no_partial_entries(raw):
    assert pair.decode_response(raw) == {"entries": [], "error": "invalid_json_envelope"}


@pytest.mark.parametrize("raw", [
    '{"candidates":[{},{}],"candidates":[{},{}]}',
    '{"candidates":[{"body":"return None","body":"return best_params","hypothesis":{}},{}]}',
    '{"candidates":[{}, {"hypothesis":{"confidence":0.1,"confidence":0.8},"body":"return None"}]}',
    '{"candidates":[{"unknown":{"x":1,"x":2}},{}]}',
])
def test_duplicate_key_anywhere_invalidates_entire_transport(raw):
    assert pair.decode_response(raw) == {"entries": [], "error": "invalid_json_envelope"}


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e309", "-1e309"])
@pytest.mark.parametrize("position", [0, 1])
def test_nonfinite_number_in_either_slot_invalidates_whole_response(number, position):
    values = [entry(), entry(SECOND)]
    values[position]["hypothesis"]["confidence"] = "NUMBER_SENTINEL"
    raw = json.dumps({"candidates": values}).replace('"NUMBER_SENTINEL"', number)
    assert pair.decode_response(raw) == {"entries": [], "error": "invalid_json_envelope"}


@pytest.mark.parametrize("invalid", [None, False, 4, "text", [], {},
    {"hypothesis": FIRST}, {"body": "return None"},
    {"hypothesis": FIRST, "body": "return None", "extra": False}])
@pytest.mark.parametrize("position", [0, 1])
def test_invalid_entry_shape_does_not_poison_other_slot(invalid, position):
    values = [entry(), entry(SECOND)]
    values[position] = invalid
    decoded = pair.decode_response(json.dumps({"candidates": values}))
    assert decoded["error"] is None
    bad, good = decoded["entries"][position], decoded["entries"][1-position]
    assert bad == {"index": position, "proposal": None, "hypothesis": None,
                   "error": "invalid_entry", "body_error": None}
    assert good["proposal"] is not None and good["error"] is None
    assert good["hypothesis"] == parse_hypothesis_value(values[1-position]["hypothesis"])


@pytest.mark.parametrize("invalid_body", [None, 3, False, "", "    return None\n",
    "return np.random.choice(26)\n", "return None\r", "return None\n#\ud800"])
def test_body_failure_preserves_its_forecast_and_other_entry(invalid_body):
    raw = json.dumps({"candidates": [entry(body=invalid_body), entry(SECOND)]})
    decoded = pair.decode_response(raw)
    first, second = decoded["entries"]
    assert decoded["error"] is None
    assert first["error"] == "invalid_body" and first["body_error"]
    assert first["hypothesis"] == parse_hypothesis_value(FIRST) and first["proposal"] is None
    assert second["proposal"] is not None and second["hypothesis"] == parse_hypothesis_value(SECOND)


@pytest.mark.parametrize("hypothesis", [None, {}, {**FIRST, "confidence": True},
    {**FIRST, "predicted_low": 1, "predicted_high": -1},
    {**FIRST, "reasoning": ""}, {**FIRST, "predicted_low": 10**400}])
def test_hypothesis_failure_is_slot_local_even_with_valid_body(hypothesis):
    decoded = pair.decode_response(wire(entry(hypothesis), entry(SECOND)))
    first, second = decoded["entries"]
    assert decoded["error"] is None
    assert first["hypothesis"] is first["proposal"] is first["body_error"] is None
    assert first["error"] == "invalid_hypothesis" and second["proposal"] is not None


def test_invalid_hypothesis_and_body_errors_remain_independent():
    decoded = pair.decode_response(wire(entry({}, "return np.random.choice(26)")))
    first, second = decoded["entries"]
    assert first["error"] == "invalid_hypothesis"
    assert first["hypothesis"] is first["proposal"] is None
    assert "np.random.choice" in first["body_error"]
    assert second["proposal"] is not None


def test_duplicate_bodies_are_not_selected_or_discarded_by_codec():
    raw = wire(entry({}, "return ordinary_params"), entry(SECOND, "return ordinary_params"))
    decoded = pair.decode_response(raw)
    assert decoded["entries"][0]["proposal"] is None
    assert decoded["entries"][1]["proposal"].body == "return ordinary_params"
    assert set(decoded) == {"entries", "error"}
    assert all(set(item) == {"index", "proposal", "hypothesis", "error", "body_error"}
               for item in decoded["entries"])
    both = pair.decode_response(wire(entry(body="return None"), entry(SECOND, "return None")))
    assert all(item["proposal"] is not None for item in both["entries"])


def test_utf8_transport_cap_is_not_implicitly_doubled_and_has_exact_boundary(monkeypatch):
    assert pair.MAX_RESPONSE_BYTES == builder.MAX_RESPONSE_BYTES == 131072
    raw = wire()
    padded = raw + " " * (pair.MAX_RESPONSE_BYTES - len(raw.encode()))
    assert pair.decode_response(padded)["error"] is None
    calls = []
    monkeypatch.setattr(builder, "validate_body", lambda value: calls.append(value))
    assert pair.decode_response(padded + " ") == {"entries": [], "error": "invalid_json_envelope"}
    assert calls == []
    nonascii = wire(entry(body="#" + "α" * 4000 + "\nreturn None\n"))
    oversized = nonascii + " " * (pair.MAX_RESPONSE_BYTES - len(nonascii))
    assert len(oversized) == pair.MAX_RESPONSE_BYTES < len(oversized.encode())
    assert pair.decode_response(oversized)["error"] == "invalid_json_envelope"


def test_decode_does_not_assemble_sources_or_execute_a_proposed_body(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("wire validation must not assemble candidate source")
    monkeypatch.setattr(builder, "source_for", forbidden)
    raw = wire(entry(body="x = 1 / 0\nreturn ordinary_params\n"))
    assert all(item["proposal"] is not None for item in pair.decode_response(raw)["entries"])


def test_unexpected_internal_validator_error_is_not_laundered_as_bad_slot(monkeypatch):
    def broken(_value):
        raise RuntimeError("unexpected validator infrastructure failure")
    monkeypatch.setattr(builder, "validate_body", broken)
    with pytest.raises(RuntimeError, match="unexpected validator"):
        pair.decode_response(wire())


def test_schema_exactly_embeds_detached_single_candidate_schema(monkeypatch):
    single = builder.response_format()
    original = copy.deepcopy(single)
    monkeypatch.setattr(builder, "response_format", lambda: single)
    result = pair.response_format()
    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "circle_proposal_pair_v1"
    assert result["json_schema"]["strict"] is True
    schema = result["json_schema"]["schema"]
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert schema["required"] == ["candidates"] and set(schema["properties"]) == {"candidates"}
    candidates = schema["properties"]["candidates"]
    assert candidates["type"] == "array" and candidates["minItems"] == candidates["maxItems"] == 2
    assert candidates["items"] == single["json_schema"]["schema"]
    candidates["items"]["properties"].clear()
    assert single == original
    assert pair.response_format()["json_schema"]["schema"]["properties"]["candidates"]["items"] == original["json_schema"]["schema"]


def test_identity_binds_current_codec_builder_schema_and_limits_and_is_detached():
    expected = pair.identity()
    assert expected["contract"] == pair.CONTRACT == "circle_proposal_pair/1"
    assert expected["codec_source_sha256"] == hashlib.sha256(Path(pair.__file__).read_bytes()).hexdigest()
    assert expected["builder"] == builder.identity()
    assert expected["response_format_sha256"] == hashlib.sha256(canonical(pair.response_format())).hexdigest()
    assert expected["max_response_bytes"] == 131072 and expected["entry_count"] == 2
    changed = pair.identity()
    changed["builder"].clear()
    changed["codec_source_sha256"] = "0" * 64
    assert pair.identity() == expected


def test_instruction_declares_same_parent_order_one_child_and_unmeasured_alternatives():
    instruction = pair.response_instruction()
    assert "exactly two candidates" in instruction
    assert "SAME separately declared measured parent" in instruction
    assert "first locally eligible entry in this order" in instruction
    assert "at most one child" in instruction
    assert "rejected alternatives remain unmeasured" in instruction
    assert "confidence is not a selection weight" in instruction
    assert "checks both alternatives against the same source history" in instruction
    assert "failure of the chosen candidate never triggers the other" in instruction
    assert json.loads(instruction.split("\n", 1)[1]) == builder.interface_description()
