"""Bounded, dependency-free hypothesis, conclusion and reflection protocol."""

from .models import CONTRACT, Conclusion, Hypothesis, Reflection
from .parsing import parse_conclusion, parse_hypothesis, parse_reflection
from .prompts import conclusion_prompt, hypothesis_instruction, reflection_prompt

__all__ = [
    "CONTRACT", "Hypothesis", "Conclusion", "Reflection",
    "parse_hypothesis", "parse_conclusion", "parse_reflection",
    "hypothesis_instruction", "conclusion_prompt", "reflection_prompt",
]
