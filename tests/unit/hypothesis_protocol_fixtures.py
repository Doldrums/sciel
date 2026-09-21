"""Small protocol payloads shared by parser and prompt contract tests."""
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


def hypothesis(**updates):
    value = {"hypothesis": "Replace the scalar constraint loop with a vectorized one.",
             "predicted_low": -0.02, "predicted_high": 0.04,
             "reasoning": "Fewer Python calls may reduce runtime at fixed work.",
             "confidence": 0.65}
    return {**value, **updates}


def lesson(**updates):
    return {"rule": "Vectorizing constraints reduced runtime in this experiment.",
            "polarity": "positive", "scope": "branch",
            "evidence_ids": ["r000001a1"], **updates}


def wire(marker, value):
    return marker + ": " + json.dumps(value)


OBSERVED_COMPARISON_TREATMENT = '''```json
{
  "verdict": "REFUTED",
  "explanation": "The forecast predicted a narrow interval around zero, suggesting minimal change or a slight decrease in the controller score. However, the actual measured outcome showed a significant improvement in the controller score. The forecast's supplied verdict was REFUTED, aligning with the observed outcome where the prediction did not match the actual improvement.",
  "lessons": [
    {
      "rule": "The controller score improved, as indicated by a positive actual_delta, despite the forecast predicting a narrow interval around zero and being refuted.",
      "polarity": "positive",
      "scope": "branch",
      "evidence_ids": [
        "r000002a1"
      ]
    }
  ]
}
```'''


def measured_record(ident, parent_score, score, *, injected_ids=()):
    """Authored observed improvement whose forecast is deliberately too narrow."""
    source = hashlib.sha256((ident + "\n").encode()).hexdigest()
    forecast = parse_hypothesis(wire("HYPOTHESIS_JSON", hypothesis())).to_dict()
    return dict(id=ident, branch="p1", status="evaluated", parent_score=parent_score,
        score=score, actual_delta=score-parent_score, source_sha256=source,
        proposed_sha256=source, hypothesis=forecast, injected_ids=list(injected_ids))
