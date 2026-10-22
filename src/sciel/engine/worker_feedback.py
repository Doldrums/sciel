"""Bounded retry feedback, evaluation attempts and failure accounting."""
from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from .prompt.budget import tail
from .llm.client import llm_error_metadata
from .validation import source_sha256
from ..campaign.budget import CampaignStopped


def _bounded_failure(category, detail=None):
    """Task diagnostics only: bounded data, without traceback/provider bodies."""
    if not isinstance(category, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,63}", category):
        category = "EvaluationError"
    if not isinstance(detail, str) or not detail.strip():
        return category
    message = detail[:4096]
    # Do not move recognizable credentials from an evaluator message to the
    # next model prompt. No environment values are read to build this context.
    message = re.sub(r"(?i)\b(?:sk-[A-Za-z0-9_-]+|bearer\s+[^\s,;]+)", "[redacted]", message)
    message = re.sub(r"(?i)\b([A-Za-z0-9_]*(?:api_key|token|password|secret))\s*[:=]\s*[^\s,;]+",
                     r"\1=[redacted]", message)
    message = " ".join(message.split())
    if len(message) > 600:
        message = message[:597] + "..."
    return f"{category}: {message}"

def _evaluation_failure(artifacts):
    if not isinstance(artifacts, dict):
        return "EvaluationError"
    category = artifacts.get("error_type")
    if not category:
        # Older task evaluators report error+traceback but no error_type. Read
        # only the terminal exception class; never send the traceback itself.
        trace = artifacts.get("traceback")
        if isinstance(trace, str):
            match = re.search(r"(?:^|\n)([A-Za-z_][A-Za-z0-9_.]{0,63}):[^\n]*\s*$", trace[-2048:])
            if match:
                category = match.group(1)
    return _bounded_failure(category, artifacts.get("error"))

def _escalate(op: str, temp: float) -> tuple:
    if op == "diff":
        return "full_rewrite", temp
    return "full_rewrite", min(temp + 0.2, 1.5)

def unknown_request_stop(exc, spent):
    """Seal an unreturned request without erasing prior known charges."""
    stopped = CampaignStopped('generation outcome and cost are unknown')
    stopped.known_cost = spent
    stopped.current_call_cost_unknown = True
    stopped.generation_returned = False
    stopped.generation_dispatch = 'unknown'
    stopped.llm_error = llm_error_metadata(exc)
    return stopped

def candidate_retry_prompt(prompt, response, validation=None, *, reason=None, measured_repeat=None):
    """Return exact previous output and distinguish measured admission refusal."""
    if measured_repeat is not None:
        verdict = measured_repeat.get('output_novelty') or {}
        near = verdict.get('status') == 'non_improving_near_repeat'
        reason = ('its saved output is within the configured task output resolution of an earlier '
                  'compatible measurement and does not improve its canonical score' if near else
                  'its authenticated saved output exactly matched a prior compatible measurement')
        measured_repeat = deepcopy(measured_repeat)
        if near:
            evidence = verdict['near_output']
            delta = evidence['score'] - evidence['match']['score']
            measured_repeat['canonical_score_delta_from_match'] = delta if math.isfinite(delta) else None
            if not math.isfinite(delta):
                measured_repeat['score_delta_unavailable_reason'] = 'Difference exceeds finite float range; both original scores are retained.'
        return (prompt + '\n\nPOST-EVALUATION OUTPUT REPEAT FEEDBACK:\n'
            'The previous candidate was evaluated, but ' + reason +
            '. It was not admitted as a new mutation. This does not prove program '
            'equivalence or future behavior. Produce a distinct executable search change; changing only '
            'comments, wording or a parameter that leaves this output unchanged is insufficient. '
            'A strictly improved measured score remains eligible under the near-output policy, '
            'however small the improvement; shrinking or perturbing a known output without improvement is insufficient. '
            'Do not claim efficacy from reproducing this known output. A program that continues a construction '
            'from its population view and returns it when its search finds nothing better reproduces that '
            'construction. Do not fall back to it: return the construction your own search reached, even if it '
            'scores below the parent. A distinct output with a lower score is admitted to the population; '
            'only an exact repeat is refused.\n'
            'MEASURED_REPEAT_JSON:\n' + json.dumps(measured_repeat, ensure_ascii=True, allow_nan=False) + '\n'
            'PREVIOUS_RESPONSE_JSON (end of the previous generator response):\n'
            + json.dumps(tail(response), ensure_ascii=True) + '\n')
    novelty = validation.novelty if validation is not None else None
    violation = validation.violation if validation is not None else None
    feedback = {'reason': reason or (violation.detail or violation.kind if violation else 'invalid candidate'),
                'quality_measured': False, 'novelty': novelty}
    return (prompt + '\n\nLOCAL CANDIDATE VALIDATION FEEDBACK:\n'
            'The previous proposal was not admitted as a new mutation. This is a source-validation '
            'result, not an evaluation score. Produce a corrected, distinct executable candidate; '
            'do not merely reword its forecast, comments, or formatting. Embedding similarity alone '
            'is not proof of equivalent behavior.\n'
            'REFUSAL_JSON:\n' + json.dumps(feedback, ensure_ascii=True, allow_nan=False) + '\n'
            'PREVIOUS_RESPONSE_JSON (end of the previous generator response):\n'
            + json.dumps(tail(response), ensure_ascii=True) + '\n')

def evaluation_attempt(outcome, source):
    """Retain every returned measurement, including rejected mutations."""
    verdict = getattr(outcome, 'output_novelty', None)
    if verdict is not None:
        from .output_novelty import validate_verdict
        verdict = validate_verdict(verdict, source_sha256=source_sha256(source))
        if verdict['evaluation_id'] != getattr(outcome, 'evaluation_id', None):
            raise ValueError('output verdict differs from returned evaluation identity')
        if verdict['identity'] is not None:
            stage = verdict['identity']['scope']['stage']
            if (not outcome.result.ok or stage['reached'] != outcome.reached
                    or stage['confirmed'] != outcome.confirmed):
                raise ValueError('output identity differs from returned evaluation stage')
        near = verdict.get('near_output')
        if near is not None and near['score'] != outcome.result.score:
            raise ValueError('near-output evidence differs from returned evaluation score')
    result = outcome.result
    return dict(evaluation_id=getattr(outcome, 'evaluation_id', None), source_sha256=source_sha256(source),
        ok=result.ok, score=result.score, reached=outcome.reached, confirmed=outcome.confirmed,
        output_novelty=verdict)

def evaluation_metadata(evaluations):
    from .output_novelty import is_repeat
    return dict(evaluations=deepcopy(evaluations), evaluation_count=len(evaluations),
        evaluation_ids=[row['evaluation_id'] for row in evaluations if row['evaluation_id'] is not None],
        output_repeat_rejections=sum(row.get('admission') == 'output_repeat_rejected' for row in evaluations),
        output_repeats=[deepcopy(row['output_novelty']) for row in evaluations
                        if row['output_novelty'] is not None and is_repeat(row['output_novelty'])])
