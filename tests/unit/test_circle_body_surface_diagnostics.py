"""Public body diagnostics, using source/AST only; proposed code is never run."""
import ast
import json
import re

import pytest

from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis.parsing import parse_hypothesis_value


HYP = {
    "hypothesis": "Change initializer values using the supplied arrays",
    "predicted_low": -0.002,
    "predicted_high": 0.003,
    "reasoning": "A prospective forecast, not an observed quality result.",
    "confidence": 0.6,
}


def wire(source, hypothesis=HYP):
    return json.dumps({"hypothesis": hypothesis, "body": source}, ensure_ascii=False)


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + "." + node.attr
    return "<expression>"


def submitted_position(source, access):
    """AST columns are UTF8 bytes; the public diagnostic promises characters."""
    matches = [node for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.Attribute) and dotted(node) == access]
    assert matches
    node = matches[0]
    # Split LF only: quoted Unicode separators are not physical source lines.
    line = source.split("\n")[node.lineno - 1]
    column = len(line.encode("utf-8")[:node.col_offset].decode("utf-8")) + 1
    return node.lineno, column


def rejected(source, prefix, access, *, hypothesis=HYP, original_wire=None):
    raw = wire(source, hypothesis) if original_wire is None else original_wire
    original = json.loads(raw)
    decoded = body.decode_response(raw)
    assert decoded["proposal"] is None
    assert decoded["hypothesis"] == parse_hypothesis_value(hypothesis)
    assert decoded["error"] == "invalid_body"
    message = decoded["body_error"]
    assert message.startswith(prefix)
    assert access in message
    line, column = submitted_position(source, access)
    assert f"body line {line}, column {column}" in message
    assert "untrusted" in message
    assert len(message) < 1024
    with pytest.raises(ValueError) as error:
        body.source_for(source)
    assert str(error.value) == message
    assert json.loads(raw) == original
    assert original["body"] == source
    assert set(decoded) == {"proposal", "hypothesis", "error", "body_error"}
    return message


@pytest.fixture(scope="module")
def refused_response():
    return wire("radii_indices = np.random.choice(ordinary_params)\nreturn ordinary_params\n",
                dict(HYP, predicted_low=0.0001, predicted_high=0.0005, confidence=0.7))


def test_refusal_preserves_forecast_and_identifies_random_choice(refused_response):
    value = json.loads(refused_response)
    assert value["hypothesis"]["predicted_low"] == 0.0001
    assert value["hypothesis"]["predicted_high"] == 0.0005
    assert value["hypothesis"]["confidence"] == 0.7
    message = rejected(value["body"], "array attribute is outside the explicit surface",
                       "np.random.choice", hypothesis=value["hypothesis"],
                       original_wire=refused_response)
    assert "np.random.uniform" not in message
    assert "radii_indices = np.random.choice" in message


@pytest.mark.parametrize("source,prefix,access", [
    ("return np.linalg.norm(best_params)\n", "array attribute is outside the explicit surface", "np.linalg.norm"),
    ("return np.random.uniform(size=78)\n", "array attribute is outside the explicit surface", "np.random.uniform"),
    ("return np.unavailable(best_params)\n", "module attribute is outside the explicit surface", "np.unavailable"),
    ("return best_params.resize(78)\n", "array attribute is outside the explicit surface", "best_params.resize"),
    ("return best_params.tolist()\n", "array attribute is outside the explicit surface", "best_params.tolist"),
    ("return np.pi()\n", "module value is not a permitted callable", "np.pi"),
    ("return best_params.shape()\n", "only explicit array methods can be called", "best_params.shape"),
    ("best_params.shape = (78,)\nreturn ordinary_params\n", "attribute mutation/dunder access is not allowed", "best_params.shape"),
    ("return best_params.__array_interface__\n", "attribute mutation/dunder access is not allowed", "best_params.__array_interface__"),
])
def test_surface_reasons_keep_original_prefix_and_exact_access(source, prefix, access):
    rejected(source, prefix, access)


@pytest.mark.parametrize("source", [
    "é = 1; x = np.linalg.norm(best_params)\nreturn ordinary_params\n",
    'label = "αβ🌱"; x = np.linalg.norm(best_params)\nreturn ordinary_params\n',
    'label = "a\u2028b"; x = np.linalg.norm(best_params)\nreturn ordinary_params\n',
    "# Unicode α and CRLF are literal submitted bytes.\r\né = 1; x = np.linalg.norm(best_params)\r\nreturn ordinary_params\r\n",
    "if restart_index > 0:\n    x = np.linalg.norm(best_params)\nreturn ordinary_params\n",
])
def test_reported_columns_count_submitted_unicode_characters(source):
    message = rejected(source, "array attribute is outside the explicit surface", "np.linalg.norm")
    line, column = submitted_position(source, "np.linalg.norm")
    assert re.search(r"body line (\d+), column (\d+)", message).groups() == (str(line), str(column))


@pytest.mark.parametrize("source", [
    "x = (\n    np.linalg.norm(\n        best_params\n    )\n)\nreturn ordinary_params\n",
    "x = (\r\n    np.linalg.norm(\r\n        best_params\r\n    )\r\n)\r\nreturn ordinary_params\r\n",
    "x = (np.\n     linalg.\n     norm)(best_params)\nreturn ordinary_params\n",
])
def test_multiline_access_uses_attribute_start_not_call_parentheses(source):
    rejected(source, "array attribute is outside the explicit surface", "np.linalg.norm")


def test_long_argument_is_bounded_and_explicitly_untrusted():
    payload = "UNTRUSTED_PAYLOAD_" * 150
    source = "best_params.tofile('" + payload + "')\nreturn ordinary_params\n"
    message = rejected(source, "array attribute is outside the explicit surface", "best_params.tofile")
    assert payload not in message
    assert "UNTRUSTED_PAYLOAD_" * 20 not in message
    assert "body line 1, column 1" in message


def test_long_attribute_is_bounded_without_guessing_or_repair():
    attribute = "forbidden_" + "x" * 2500
    source = f"return best_params.{attribute}()\n"
    decoded = body.decode_response(wire(source))
    message = decoded["body_error"]
    assert decoded["proposal"] is None and decoded["error"] == "invalid_body"
    assert decoded["hypothesis"] == parse_hypothesis_value(HYP)
    assert message.startswith("array attribute is outside the explicit surface")
    assert "best_params.forbidden_" in message
    assert "body line 1, column 8" in message
    assert "untrusted" in message and len(message) < 1024
    assert "x" * 161 not in message
    with pytest.raises(ValueError) as error:
        body.source_for(source)
    assert str(error.value) == message


def test_missing_trusted_ast_location_preserves_original_value_error(monkeypatch):
    original = ValueError("array attribute is outside the explicit surface")

    def unavailable_traceback(_source):
        # A validator boundary can retain the reason but lose its AST frame.
        # No diagnostic may guess a rejected node by rescanning source order.
        raise original.with_traceback(None)

    monkeypatch.setattr(body.proposals, "validate_helper", unavailable_traceback)
    source = "return np.random.choice(26)\n"
    decoded = body.decode_response(wire(source))
    assert decoded["body_error"] == str(original)
    assert decoded["proposal"] is None and decoded["error"] == "invalid_body"
    assert decoded["hypothesis"] == parse_hypothesis_value(HYP)
    with pytest.raises(ValueError) as error:
        body.source_for(source)
    assert error.value is original
    assert "body line" not in str(error.value)


def test_diagnostic_follows_actual_ast_rejection_not_first_textual_invalid_access():
    source = ("x = np.copy(np.random.choice(26))\n"
              "y = best_params.resize(78)\n"
              "return ordinary_params\n")
    calls = [dotted(node.func) for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)]
    assert calls.index("best_params.resize") < calls.index("np.random.choice")
    message = rejected(source, "array attribute is outside the explicit surface", "best_params.resize")
    assert "body line 2, column 5" in message
    assert "np.random.choice" not in message


def test_unicode_repr_expansion_is_bounded_after_escaping():
    # Nonprintable Unicode expands from one source character to six repr chars.
    payload = "\u200b" * 400
    source = "best_params.tofile('" + payload + "')\nreturn ordinary_params\n"
    message = rejected(source, "array attribute is outside the explicit surface", "best_params.tofile")
    excerpt = message.split("Submitted line excerpt", 1)[1]
    assert "\\u200b" in excerpt
    assert "\\u200b" * 35 not in excerpt
    assert len(excerpt) < 300


@pytest.mark.parametrize("source", [
    "return None\n",
    "return np.copy(best_params)\n",
    "x = best_params.copy()\nx[52:] = np.maximum(x[52:], 0.001)\nx[:52] = x[:52].clip(0.001, 0.999)\nreturn x\n",
    "x = ordinary_params.copy()\r\ny = x[:52].reshape(26, 2).ravel()\r\nx[:52] = y\r\nreturn x\r\n",
    "# allowed calls remain literal α\nx = np.where(uniform_noise > 0, best_params, ordinary_params)\nreturn x",
])
def test_permitted_calls_and_compiled_source_bytes_are_unchanged(source):
    decoded = body.decode_response(wire(source))
    assert decoded["error"] is decoded["body_error"] is None
    assert decoded["proposal"].body == source
    assert decoded["proposal"].hypothesis == parse_hypothesis_value(HYP)
    compiled = body.source_for(source)
    canonical = source if source.endswith("\n") else source + "\n"
    assert body.body_of(compiled) == canonical
    assert body.source_for(canonical) == compiled
    # Only the compiler's indentation/final LF may be added; no API repair.
    indented = "".join("    " + line + "\n" for line in canonical.split("\n")[:-1])
    assert indented in compiled


def test_public_source_construction_does_not_execute_accepted_body():
    source = "x = 1 / 0\nreturn ordinary_params\n"
    decoded = body.decode_response(wire(source))
    assert decoded["proposal"].body == source
    assert body.body_of(body.source_for(source)) == source


def test_writer_interface_exposes_the_existing_forbidden_surface():
    contract = body.interface_description()['code_contract']
    assert contract['forbidden'] == body.proposals.interface_description()['forbidden']
    assert 'random' in contract['forbidden'] and 'I/O' in contract['forbidden']


def test_existing_syntax_diagnostic_stays_distinct_from_api_surface_error():
    source = "    return np.copy(best_params)\n"
    decoded = body.decode_response(wire(source))
    assert decoded["proposal"] is None and decoded["hypothesis"] == parse_hypothesis_value(HYP)
    assert decoded["body_error"].startswith("invalid body syntax: unexpected indent")
    assert "body line 1, column" in decoded["body_error"]
    assert "untrusted" in decoded["body_error"]
