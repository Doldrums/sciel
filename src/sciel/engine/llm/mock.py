"""Deterministic mock LLM keyed by prompt hash, with controlled dups and failures."""

from __future__ import annotations

import hashlib
import json

from .client import LLMResponse

_DUP_PROGRAM = (
    "import numpy as np\n"
    "def run_search():\n"
    "    return (0.0, 0.0)\n"
)

_FAIL_PROGRAM = (
    "def run_search():\n"
    "    raise RuntimeError('mock injected failure')\n"
)


def _fill(schema: dict):
    kind = schema.get("type")
    kind = next((k for k in kind if k != "null"), None) if isinstance(kind, list) else kind
    if "enum" in schema:
        return "skip" if "skip" in schema["enum"] else schema["enum"][0]
    if kind == "object":
        return {key: _fill(value) for key, value in schema.get("properties", {}).items()}
    if kind == "array":
        return []
    if kind == "string":
        return "mock"
    return None


class MockLLM:
    def __init__(self, dup_modulo: int = 23, fail_modulo: int = 19,
                 bet_marker: str = "BET:"):
        self.dup_modulo = dup_modulo
        self.fail_modulo = fail_modulo
        self.bet_marker = bet_marker
        self.models = ["mock-model"]
        self.weights = [1.0]

    def choose_model(self, rng) -> str:
        return "mock-model"

    def generate(self, prompt: str, model: str,
                 temperature: float | None = None, response_format=None, **options) -> LLMResponse:
        if response_format is not None:
            action = _fill(response_format["json_schema"]["schema"])
            if "arguments_json" in action:
                action["arguments_json"] = "{}"
            text = json.dumps(action)
            return LLMResponse(text=text, model="mock-model", cost=0.0,
                               prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4)
        keyed = f"{prompt}\x00temp={temperature}"
        digest = hashlib.sha256(keyed.encode()).digest()
        n = int.from_bytes(digest[:8], "big")
        if n % self.fail_modulo == 0:
            code = _FAIL_PROGRAM
        elif n % self.dup_modulo == 0:
            code = _DUP_PROGRAM
        else:
            x = -5.0 + 10.0 * ((n % 100000) / 100000.0)
            y = -5.0 + 10.0 * (((n >> 20) % 100000) / 100000.0)
            code = ("import numpy as np\n"
                    "def run_search():\n"
                    f"    return ({x:.6f}, {y:.6f})\n")
        text = f"```python\n{code}```\n"
        if self.bet_marker in prompt:
            q = 0.1 + 0.8 * ((n >> 8) % 1000) / 1000.0
            text += f'\nBET: {{mechanism: "mock local move", q: {q:.3f}}}\n'
        return LLMResponse(text=text, model="mock-model", cost=0.0,
                           prompt_tokens=len(prompt) // 4,
                           completion_tokens=len(text) // 4)
