"""Boundary and evidence-preservation checks for the untrusted C2 wire format."""
import hashlib
import json
from pathlib import Path
import pytest
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.presentation import expand_reflection_payload
from sciel.hypothesis import (
    CONTRACT, Conclusion, Hypothesis, Reflection, conclusion_prompt,
    hypothesis_instruction, parse_conclusion, parse_hypothesis, parse_reflection,
    reflection_prompt,
)
from sciel.hypothesis.parsing import (
    MAX_BLOCK_CHARS, MAX_EVIDENCE_IDS, MAX_ID_CHARS, MAX_LESSONS,
    MAX_REASONING_CHARS, MAX_RESPONSE_CHARS, MAX_RETIRE_IDS, MAX_TEXT_CHARS, AUX_TRANSPORT,
)
from sciel.hypothesis.prompts import MAX_CONTEXT_CHARS, MAX_EXPERIMENTS
from hypothesis_protocol_fixtures import (
    hypothesis, lesson, wire, OBSERVED_COMPARISON_TREATMENT,
)


def test_hypothesis_before_python_code_roundtrips_field_names_and_numbers():
    response = ("Here is the planned experiment.\n" +
                wire("HYPOTHESIS_JSON", hypothesis()) +
                '\n```python\nprint("HYPOTHESIS_JSON: not a protocol block")\n```\n')
    result = parse_hypothesis(response)
    assert isinstance(result, Hypothesis)
    assert result.to_dict() == {
        "text": hypothesis()["hypothesis"], "predicted_low": -0.02,
        "predicted_high": 0.04, "reasoning": hypothesis()["reasoning"], "confidence": 0.65,
    }
    assert CONTRACT == "sciel-research/12"


def test_hypothesis_next_line_braces_and_escaped_quotes():
    value = hypothesis(hypothesis='Change mapping {"key": "}"}; keep quoted strings.',
                       predicted_low=-1, predicted_high=-1, confidence=0)
    result = parse_hypothesis("  HYPOTHESIS_JSON:\n" + json.dumps(value) + "\n")
    assert result.text == value["hypothesis"]
    assert result.predicted_low == result.predicted_high == -1.0
    assert result.confidence == 0.0


@pytest.mark.parametrize("leading", ["", "  ", "\n\t\r\n"])
def test_unmarked_hypothesis_at_response_start_before_python(leading):
    value = hypothesis()
    response = leading + json.dumps(value) + "\n\n```python\nx = 1\n```"
    marked = parse_hypothesis(wire("HYPOTHESIS_JSON", value))
    assert parse_hypothesis(response) == marked


def test_actual_v2_response_prefix_transport_regression():
    # Exact prefix from v2 r000001a1, response SHA-256
    # e7c24ff76866397f16c4a001e507719782d63e9f83b0ac35eddf0d169c8a9147.
    # Candidate body is unnecessary for this transport test and is not executed.
    prefix = (
        '{"hypothesis": "Implement a gradient-based optimization approach to refine circle '
        'positions and radii iteratively.","predicted_low":-0.01,"predicted_high":0.05,'
        '"reasoning":"Gradient descent can find local optima more efficiently than random '
        'search or simple heuristics, potentially leading to higher packing densities.",'
        '"confidence":0.7}\n'
    )
    result = parse_hypothesis(prefix + "```python\n# Body omitted from transport fixture.\n```")
    assert result is not None
    assert result.predicted_low == -0.01 and result.predicted_high == 0.05
    assert result.confidence == 0.7


def test_local_v2_raw_responses_under_new_parser_without_rewriting_historical_outcomes(tmp_path):
    # Author legacy journal rows locally; parser acceptance never rewrites them.
    path = tmp_path / "sciel_events.jsonl"
    records = []
    for index in range(4):
        raw = json.dumps(hypothesis(confidence=index / 4)) + "\n```python\nx=1\n```"
        records.append(dict(event="research", phase="hypothesis", contract="sciel-research/1",
            evidence={"hypothesis": None}, response=raw, response_truncated=False,
            response_sha256=hashlib.sha256(raw.encode()).hexdigest()))
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    before = path.read_bytes()
    records = [row for row in map(json.loads, before.splitlines())
               if row.get("event") == "research" and row.get("phase") == "hypothesis"]
    assert len(records) == 4
    for row in records:
        assert row["contract"] == "sciel-research/1" and row["evidence"]["hypothesis"] is None
        assert not row["response_truncated"]
        assert hashlib.sha256(row["response"].encode()).hexdigest() == row["response_sha256"]
        assert parse_hypothesis(row["response"]) is not None
    assert path.read_bytes() == before


@pytest.mark.parametrize("response", [
    json.dumps(hypothesis()),
    "Prose before raw JSON.\n" + json.dumps(hypothesis()) + "\n```python\nx=1\n```",
    "OTHER_JSON: " + json.dumps(hypothesis()) + "\n```python\nx=1\n```",
    json.dumps(hypothesis()) + "\n```json\n{}\n```\n```python\nx=1\n```",
    "```json\n" + json.dumps(hypothesis()) + "\n```\n```python\nx=1\n```",
    json.dumps(hypothesis()) + "\n```\nx=1\n```",
    json.dumps(hypothesis()) + "\n```javascript\nx=1\n```",
    json.dumps(hypothesis()) + "\n" + json.dumps(hypothesis()) + "\n```python\nx=1\n```",
    json.dumps(hypothesis()) + "\n" + wire("HYPOTHESIS_JSON", hypothesis()) + "\n```python\nx=1\n```",
    wire("HYPOTHESIS_JSON", hypothesis()) + "\n" + json.dumps(hypothesis()) + "\n```python\nx=1\n```",
    json.dumps(hypothesis()) + "\nExtra prose.\n```python\nx=1\n```",
    json.dumps(hypothesis(extra="unknown")) + "\n```python\nx=1\n```",
    json.dumps(hypothesis(predicted_low=True)) + "\n```python\nx=1\n```",
    json.dumps(hypothesis(predicted_high=float("inf"))) + "\n```python\nx=1\n```",
])
def test_unmarked_hypothesis_keeps_strict_transport_and_schema_failures(response):
    assert parse_hypothesis(response) is None


def test_standalone_aux_json_uses_the_same_schema_as_marked_transport():
    conclusion = {"verdict": "INCONCLUSIVE", "explanation": "Unknown.", "lessons": []}
    reflection = {"lessons": [], "retire_ids": []}
    for parser, marker, data in [(parse_conclusion, "CONCLUSION_JSON", conclusion),
                                  (parse_reflection, "REFLECTION_JSON", reflection)]:
        raw = json.dumps(data)
        assert parser(" \n" + raw + "\n") == parser(marker + ": " + raw)
        assert parser("```json\n" + raw + "\n```") == parser(marker + ": " + raw)
        for invalid in ("Here is the result: " + raw, raw + " commentary", raw + raw,
                        raw.replace('"lessons": []', '"lessons": [], "lessons": []')):
            assert parser(invalid) is None


@pytest.mark.parametrize("parser,marker,data", [
    (parse_hypothesis, "HYPOTHESIS_JSON", hypothesis()),
    (parse_conclusion, "CONCLUSION_JSON", {"verdict": "INCONCLUSIVE", "explanation": "Unknown.", "lessons": []}),
    (parse_reflection, "REFLECTION_JSON", {"lessons": [], "retire_ids": []}),
])
@pytest.mark.parametrize("other", ["HYPOTHESIS_JSON", "CONCLUSION_JSON", "REFLECTION_JSON", "OTHER_JSON"])
def test_mixed_marked_protocol_blocks_are_ambiguous(parser, marker, data, other):
    assert parser(other + ': {}\n' + wire(marker, data)) is None
    assert parser(wire(marker, data) + '\n' + other + ': {}') is None


def test_protocol_marker_text_inside_json_or_candidate_is_not_a_second_block():
    value = {"verdict": "INCONCLUSIVE", "explanation": "Quoted text:\nREFLECTION_JSON: {}", "lessons": []}
    assert parse_conclusion(wire("CONCLUSION_JSON", value)) is not None
    response = wire("HYPOTHESIS_JSON", hypothesis(reasoning="Quoted:\nOTHER_JSON: {}"))
    assert parse_hypothesis(response + '\n```python\n# REFLECTION_JSON: {}\n```') is not None


@pytest.mark.parametrize("response", [
    None, b"HYPOTHESIS_JSON: {}", "", "HYPOTHESIS: use more restarts\nPREDICTION: 0.1",
    "HYPOTHESIS_JSON: {}", "HYPOTHESIS_JSON: []", "HYPOTHESIS_JSON: null",
    "HYPOTHESIS_JSON: {", "HYPOTHESIS_JSON: ```json\n{}\n```",
    "```json\n" + wire("HYPOTHESIS_JSON", hypothesis()) + "\n```",
    "```python\npass\n```\n" + wire("HYPOTHESIS_JSON", hypothesis()),
    "~~~python\npass\n~~~\n" + wire("HYPOTHESIS_JSON", hypothesis()),
    "inline " + wire("HYPOTHESIS_JSON", hypothesis()),
    wire("HYPOTHESIS_JSON", hypothesis()) + " trailing explanation",
    wire("HYPOTHESIS_JSON", hypothesis()) + " {}",
    wire("HYPOTHESIS_JSON", hypothesis()) + "\n" + wire("HYPOTHESIS_JSON", hypothesis()),
])
def test_hypothesis_rejects_missing_ambiguous_or_unsupported_formats(response):
    assert parse_hypothesis(response) is None


@pytest.mark.parametrize("key,value", [
    ("predicted_low", True), ("predicted_high", False), ("confidence", True),
    ("predicted_low", "0.1"), ("predicted_high", None), ("confidence", "0.5"),
    ("predicted_low", float("nan")), ("predicted_high", float("inf")),
    ("confidence", float("-inf")), ("predicted_low", 0.05),
    ("confidence", -0.01), ("confidence", 1.01),
    ("hypothesis", " "), ("reasoning", ""), ("hypothesis", ["change"]),
    ("reasoning", {}), ("hypothesis", "x" * (MAX_TEXT_CHARS + 1)),
    ("reasoning", "x" * (MAX_REASONING_CHARS + 1)),
])
def test_hypothesis_rejects_invalid_fields_without_defaults(key, value):
    assert parse_hypothesis(wire("HYPOTHESIS_JSON", hypothesis(**{key: value}))) is None


def test_hypothesis_rejects_overflow_duplicate_and_unknown_keys():
    raw = wire("HYPOTHESIS_JSON", hypothesis())
    assert parse_hypothesis(raw.replace('-0.02', '1e999')) is None
    assert parse_hypothesis(raw.replace('-0.02', '9' * 400)) is None
    assert parse_hypothesis(raw[:-1] + ', "confidence": 0.1}') is None
    assert parse_hypothesis(wire("HYPOTHESIS_JSON", hypothesis(predicted_delta=0.0))) is None
    for key in hypothesis():
        value = hypothesis()
        del value[key]
        assert parse_hypothesis(wire("HYPOTHESIS_JSON", value)) is None


def test_hypothesis_bounds_response_block_and_nested_json():
    assert parse_hypothesis("x" * MAX_RESPONSE_CHARS +
                            wire("HYPOTHESIS_JSON", hypothesis())) is None
    assert parse_hypothesis('HYPOTHESIS_JSON: {"x":"' +
                            "x" * MAX_BLOCK_CHARS + '"}') is None
    assert parse_hypothesis('HYPOTHESIS_JSON: {"x":' +
                            "[" * 2000 + "0" + "]" * 2000 + "}") is None


def test_exact_hypothesis_text_limits_and_confidence_one_are_accepted():
    assert parse_hypothesis(wire("HYPOTHESIS_JSON", hypothesis(
        hypothesis="x" * MAX_TEXT_CHARS, reasoning="x" * MAX_REASONING_CHARS,
        predicted_low=1e100, predicted_high=1e100, confidence=1))) is not None


@pytest.mark.parametrize("verdict", ["CONFIRMED", "REFUTED", "INCONCLUSIVE"])
def test_conclusion_roundtrip_and_serialization_does_not_alias_lists(verdict):
    result = parse_conclusion(wire("CONCLUSION_JSON", {
        "verdict": verdict, "explanation": "One observation only.", "lessons": [lesson()],
    }))
    assert isinstance(result, Conclusion)
    data = result.to_dict()
    assert data["verdict"] == verdict
    data["lessons"][0]["evidence_ids"].append("fabricated")
    assert result.lessons[0]["evidence_ids"] == ["r000001a1"]


@pytest.mark.parametrize("scope,branch", [("branch", "p1"), ("branch", "foreign-branch"),
                                         ("global", None)])
def test_optional_branch_is_preserved_for_kb_evidence_validation(scope, branch):
    proposed = lesson(scope=scope, branch=branch)
    conclusion = parse_conclusion(wire("CONCLUSION_JSON", {
        "verdict": "CONFIRMED", "explanation": "One observation.", "lessons": [proposed],
    }))
    reflection = parse_reflection(wire("REFLECTION_JSON", {"lessons": [proposed], "retire_ids": []}))
    assert conclusion.lessons == reflection.lessons == [proposed]
    assert conclusion.to_dict()["lessons"][0]["branch"] == branch
    assert reflection.to_dict()["lessons"][0]["branch"] == branch


@pytest.mark.parametrize("scope,branch", [
    ("branch", None), ("branch", ""), ("branch", " p1"), ("branch", "p 1"),
    ("branch", "p1\n"), ("branch", "p\x001"), ("branch", "p" * (MAX_ID_CHARS + 1)),
    ("branch", True), ("branch", 1), ("branch", []), ("branch", {}),
    ("global", "p1"), ("global", ""), ("global", False), ("global", 0),
    ("global", []), ("global", {}),
])
def test_optional_branch_has_strict_type_and_scope_constraints(scope, branch):
    proposed = lesson(scope=scope, branch=branch)
    assert parse_conclusion(wire("CONCLUSION_JSON", {
        "verdict": "CONFIRMED", "explanation": "Observed.", "lessons": [proposed],
    })) is None
    assert parse_reflection(wire("REFLECTION_JSON", {"lessons": [proposed], "retire_ids": []})) is None


def test_optional_branch_does_not_allow_missing_required_or_other_unknown_keys():
    proposed = lesson(branch="p1")
    for key in ("rule", "polarity", "scope", "evidence_ids"):
        missing = dict(proposed)
        del missing[key]
        assert parse_reflection(wire("REFLECTION_JSON", {"lessons": [missing], "retire_ids": []})) is None
    assert parse_reflection(wire("REFLECTION_JSON", {
        "lessons": [lesson(branch="p1", confidence=1.0)], "retire_ids": [],
    })) is None
    raw = wire("REFLECTION_JSON", {"lessons": [proposed], "retire_ids": []})
    assert parse_reflection(raw.replace('"branch": "p1"', '"branch": "p1", "branch": "p2"')) is None


def test_actual_v4_reflection_transport_regression():
    # Exact archived response; interpretation uses /3 without rewriting its /2 outcome.
    raw = ('REFLECTION_JSON:\n{"lessons": [{"branch": "p1", "evidence_ids": ["r000001a1"], '
           '"polarity": "negative", "rule": "Adjusting the radius of the central circle '
           'resulted in a decrease in the score, refuting the hypothesis that it would lead '
           'to tighter packing.", "scope": "branch"}], "retire_ids": []}')
    assert hashlib.sha256(raw.encode()).hexdigest() == "2c49a792b3e07a65d87452d6dc21a56bab7aecf93eb7d6c4855e360faf8c53ff"
    result = parse_reflection(raw)
    assert result is not None and result.retire_ids == []
    assert result.lessons[0]["branch"] == "p1"
    assert result.lessons[0]["evidence_ids"] == ["r000001a1"]


def test_local_v4_reflection_under_new_parser_preserves_historical_outcome(tmp_path):
    path = tmp_path / "sciel_events.jsonl"
    raw = wire("REFLECTION_JSON", {"lessons": [lesson(branch="p1", polarity="negative")],
                                  "retire_ids": []})
    authored = [dict(event="research", phase="aux_result", kind="reflection", response=raw,
                    response_sha256=hashlib.sha256(raw.encode()).hexdigest()),
                dict(event="research", phase="reflection", contract="sciel-research/2",
                     status="invalid", applied=None)]
    path.write_text("".join(json.dumps(row) + "\n" for row in authored))
    before = path.read_bytes()
    rows = [row for row in map(json.loads, before.splitlines()) if row.get("event") == "research"]
    raw = [row for row in rows if row.get("phase") == "aux_result" and row.get("kind") == "reflection"]
    assert len(raw) == 1
    assert hashlib.sha256(raw[0]["response"].encode()).hexdigest() == raw[0]["response_sha256"]
    assert parse_reflection(raw[0]["response"]).lessons[0]["branch"] == "p1"
    historical = [row for row in rows if row.get("phase") == "reflection"]
    assert len(historical) == 1 and historical[0]["contract"] == "sciel-research/2"
    assert historical[0]["status"] == "invalid" and historical[0]["applied"] is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad_lesson", [
    {}, "rule", lesson(rule=""), lesson(rule="x" * (MAX_TEXT_CHARS + 1)),
    lesson(polarity="confirmed"), lesson(polarity={}), lesson(scope="island"),
    lesson(evidence_ids=[]), lesson(evidence_ids="r000001a1"),
    lesson(evidence_ids=[None]), lesson(evidence_ids=[{}]),
    lesson(evidence_ids=["r000001a1", "r000001a1"]),
    lesson(evidence_ids=["contains space"]), lesson(evidence_ids=["newline\n"]),
    lesson(evidence_ids=["id\x00"]), lesson(evidence_ids=["x" * (MAX_ID_CHARS + 1)]),
    lesson(evidence_ids=[str(i) for i in range(MAX_EVIDENCE_IDS + 1)]),
    lesson(confidence=1.0),
])
def test_invalid_lesson_rejects_whole_conclusion_and_reflection(bad_lesson):
    assert parse_conclusion(wire("CONCLUSION_JSON", {
        "verdict": "CONFIRMED", "explanation": "Observed.", "lessons": [lesson(), bad_lesson],
    })) is None
    assert parse_reflection(wire("REFLECTION_JSON", {
        "lessons": [lesson(), bad_lesson], "retire_ids": [],
    })) is None


@pytest.mark.parametrize("changes", [
    {"verdict": "confirmed"}, {"verdict": True}, {"explanation": None},
    {"explanation": "x" * (MAX_REASONING_CHARS + 1)}, {"lessons": None},
    {"lessons": [lesson()] * (MAX_LESSONS + 1)}, {"extra": "unrecognized"},
])
def test_conclusion_invalid_fields(changes):
    value = {"verdict": "CONFIRMED", "explanation": "Observed.", "lessons": [], **changes}
    assert parse_conclusion(wire("CONCLUSION_JSON", value)) is None


@pytest.mark.parametrize("parser,marker,value", [
    (parse_conclusion, "CONCLUSION_JSON", {"verdict": "INCONCLUSIVE", "explanation": "Unknown.", "lessons": []}),
    (parse_reflection, "REFLECTION_JSON", {"lessons": [], "retire_ids": []}),
])
def test_conclusion_reflection_reject_extra_objects_fences_duplicates_and_missing_fields(parser, marker, value):
    raw = wire(marker, value)
    assert parser(raw) is not None
    assert parser(raw + "\n" + raw) is None
    assert parser(raw + " {}") is None
    assert parser("```json\n" + raw + "\n```") is None
    assert parser(raw + "\n```python\npass\n```") is None
    assert parser(raw[:-1] + ', "lessons": []}') is None
    assert parser(raw.replace('"lessons": []', '"lessons": NaN')) is None
    for key in value:
        missing = dict(value)
        del missing[key]
        assert parser(wire(marker, missing)) is None


def test_reflection_retirement_ids_are_requests_not_membership_validated():
    value = {"lessons": [lesson(evidence_ids=["unknown-experiment"])],
             "retire_ids": ["unknown-lesson"]}
    result = parse_reflection(wire("REFLECTION_JSON", value))
    assert isinstance(result, Reflection)
    assert result.to_dict() == value  # Only the KB can check membership/evidence.
    result.to_dict()["retire_ids"].append("other")
    assert result.retire_ids == ["unknown-lesson"]


@pytest.mark.parametrize("ids", [None, "x", ["x", "x"], [True], [""], [" x"],
                                    [str(i) for i in range(MAX_RETIRE_IDS + 1)]])
def test_reflection_rejects_malformed_retirement_ids(ids):
    assert parse_reflection(wire("REFLECTION_JSON", {"lessons": [], "retire_ids": ids})) is None


@pytest.mark.parametrize("parser,marker,value", [
    (parse_conclusion, "CONCLUSION_JSON", {"verdict": "REFUTED", "explanation": "One observation.", "lessons": [lesson()]}),
    (parse_reflection, "REFLECTION_JSON", {"lessons": [lesson()], "retire_ids": ["l000001"]}),
])
@pytest.mark.parametrize("newline,outer", [("\n", ""), ("\r\n", " \t\n")])
def test_sole_aux_json_fence_changes_only_transport(parser, marker, value, newline, outer):
    raw = json.dumps(value, indent=2).replace("\n", newline)
    assert parser(outer + "```json" + newline + raw + newline + "```" + outer) == parser(wire(marker, value))
    assert AUX_TRANSPORT == "sciel-aux-json-transport/3"
    # This extension is intentionally absent from the generation protocol.
    assert parse_hypothesis("```json\n" + json.dumps(hypothesis()) + "\n```") is None


@pytest.mark.parametrize("parser,marker,value", [
    (parse_conclusion, "CONCLUSION_JSON", {"verdict": "REFUTED", "explanation": "One observation.", "lessons": []}),
    (parse_reflection, "REFLECTION_JSON", {"lessons": [], "retire_ids": []}),
])
@pytest.mark.parametrize("template", [
    "Prose\n```json\n{raw}\n```", "```json\n{raw}\n```\nProse",
    "```json\n{marker}: {raw}\n```",
    "```json\n{raw}\n```\n```json\n{raw}\n```", "```json\n{raw}\n```\n{raw}",
    "```json\n{raw}\n```\n```python\npass\n```", "```python\n{raw}\n```",
    "```JSON\n{raw}\n```", "```Json\n{raw}\n```", "~~~json\n{raw}\n~~~",
    "````json\n{raw}\n````", "```\n{raw}\n```", "```json \n{raw}\n```",
    "```json{raw}\n```", "```json\n{raw}```", "```json\n{raw}\n````",
    "```json\n[{raw}]\n```", "```json\n{raw}\n{raw}\n```",
])
def test_aux_fence_does_not_salvage_mixed_or_ambiguous_envelopes(parser, marker, value, template):
    assert parser(template.format(raw=json.dumps(value), marker=marker)) is None


@pytest.mark.parametrize("parser,value", [
    (parse_conclusion, {"verdict": "REFUTED", "explanation": "One observation.", "lessons": []}),
    (parse_reflection, {"lessons": [], "retire_ids": []}),
])
def test_aux_fence_retains_duplicate_schema_numeric_size_and_depth_guards(parser, value):
    raw = json.dumps(value)
    malformed = [raw[:-1] + ', "lessons": []}', raw[:-1] + ', "extra": 1}',
        raw.replace('"lessons": []', '"lessons": NaN'),
        raw.replace('"lessons": []', '"lessons": Infinity'),
        raw.replace('"lessons": []', '"lessons": true'),
        raw.replace('"lessons": []', '"lessons": [1e9999]'),
        raw.replace('"lessons": []', '"lessons": [' + '[' * 2000 + '0' + ']' * 2000 + ']'),
        '{"extra":"' + 'x' * MAX_BLOCK_CHARS + '"}',
        raw.replace('"lessons": []', '"lessons": [' + json.dumps(lesson(evidence_ids=[])) + ']')]
    for body in malformed:
        assert parser("```json\n" + body + "\n```") is None
    assert parser(" " * MAX_RESPONSE_CHARS + "```json\n" + raw + "\n```") is None
    for field in value:
        missing = dict(value); del missing[field]
        assert parser("```json\n" + json.dumps(missing) + "\n```") is None


def test_exact_saved_treatment_is_parser_valid_without_rewriting_its_claims():
    assert hashlib.sha256(OBSERVED_COMPARISON_TREATMENT.encode()).hexdigest() == "38957570e157365413012824e9f3689148656c855d608f0d0e8b11927d20679d"
    parsed = parse_conclusion(OBSERVED_COMPARISON_TREATMENT)
    assert parsed is not None and parsed.verdict == "REFUTED"
    assert parsed.lessons[0]["polarity"] == "positive"
    assert "significant improvement" in parsed.explanation  # Parsing does not certify prose.


def test_archived_treatment_applies_offline_to_exact_frozen_pre_record_memory(tmp_path):
    from sciel.knowledge_base import KnowledgeBase
    from hypothesis_protocol_fixtures import measured_record

    # Authored pre-record memory retains an earlier lesson and its exposure.
    prior = KnowledgeBase()
    prior.record(measured_record("r000001a1", .5, .6))
    assert prior.add_lessons([lesson()], allowed_evidence_ids=["r000001a1"]) == ["l000001"]
    path = tmp_path / "prior_knowledge.json"
    path.write_text(json.dumps(prior.state()))
    before = path.read_bytes()
    response = dict(text=OBSERVED_COMPARISON_TREATMENT, historical_parsed=False)
    record = measured_record("r000002a1", .6, .8, injected_ids=["l000001"])
    kb = KnowledgeBase()
    kb.load_state(json.loads(before))
    assert [e["id"] for e in kb.experiments] == ["r000001a1"]
    kb.record(record)
    conclusion = parse_conclusion(response["text"])
    assert conclusion.verdict == kb.experiments[-1]["forecast_verdict"] == "REFUTED"
    assert kb.add_lessons(conclusion.lessons, allowed_evidence_ids=[record["id"]]) == ["l000002"]
    new = kb.lessons[-1]
    assert new["source_support_ids"] == ["r000002a1"] and new["polarity"] == "positive"
    assert response["historical_parsed"] is False
    assert path.read_bytes() == before
