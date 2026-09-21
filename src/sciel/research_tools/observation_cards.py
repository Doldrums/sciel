"""Pure bounded observation cards shared by fresh and retained evidence delivery."""
from copy import deepcopy
import json

from ..methods.models import canonical_bytes
from .result_projection import MIN_CONTEXT_CHARS, success_context
from .replay_feedback import shorten_feedback
from .comparison_assessment import CARD_SCHEMA, unavailable, validate as validate_comparison


def format_context(cards):
    return ('\n\nRESEARCH CHECK RESULTS\n'
        'Observations are auxiliary tool results, separate from canonical task evaluation. '
        'Interpretations and next checks are hypotheses, not established improvements.\n'
        'RESEARCH_CHECK_RESULTS_JSON:\n' + json.dumps(cards, ensure_ascii=True, sort_keys=True))


def _clip(text, limit):
    text = str(text)
    if len(text) <= limit:
        return text
    note = f'\n[Excerpt: {len(text) - limit} or more characters omitted.]'
    return text[:max(0, limit - len(note))] + note


def _head_tail(text, limit):
    """Keep traceback context and its terminal cause within a hard text bound."""
    text = str(text)
    if len(text) <= limit:
        return text
    note = '\n[Excerpt: middle omitted.]\n'
    if limit <= len(note):
        return note[:max(0, limit)]
    available = limit - len(note)
    head = available // 2
    tail = available - head
    terminal = text.rsplit('\n',1)[-1]
    if '\n' in text and len(terminal) <= available - min(16,head):
        # Preserve a complete short exception line, instead of dropping its
        # first characters merely to keep a perfectly balanced excerpt.
        tail = max(tail,len(terminal))
        head = available - tail
    return text[:head] + note + text[-tail:]


def _failure_feedback(item, limit):
    """Share space between failure causes instead of losing the final exception.

    This is only a presentation excerpt. Original stderr, declared errors and
    outputs remain unchanged in the invocation receipt and artifact.
    """
    result = item.get('result') or {}
    entries, seen = [], set()
    for label, value in (('Tool error', result.get('error')),
                         ('Reason', item.get('reason')),
                         ('Diagnostic', item.get('diagnostic')),
                         ('Stderr', item.get('stderr'))):
        if value and str(value) not in seen:
            entries.append((label, str(value)))
            seen.add(str(value))
    if not entries or limit <= 0:
        return ''
    remaining, parts = limit, []
    for index, (label, value) in enumerate(entries):
        budget = remaining // (len(entries) - index)
        prefix = label + ': '
        if budget <= len(prefix):
            break
        part = prefix + _head_tail(value, budget - len(prefix))
        parts.append(part)
        remaining -= len(part) + 1
    return '\n'.join(parts)


def _failure_context(item, limit):
    """Prioritize concrete failure evidence over potentially large partial output."""
    output = item.get('output')
    details = _failure_feedback(item, limit if output is None else limit * 2 // 3)
    if output is None:
        return details
    remaining = max(0, limit - len(details) - 9)
    if remaining >= 64:
        details += '\nOutput: ' + _clip(json.dumps(output, ensure_ascii=True, sort_keys=True), remaining)
    return details


def _clip_json(text, limit):
    """Bound escaped JSON size too, including non-ASCII research observations."""
    text = str(text)
    if len(canonical_bytes(text)) <= limit:
        return text
    suffix = ' [excerpt]'
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(canonical_bytes(text[:middle] + suffix)) <= limit:
            low = middle
        else:
            high = middle - 1
    return text[:low] + suffix


def measurement_card(config, plan, observation, replay_comparison=None, *, transform=None):
    output = observation.get('output')
    successful = observation['result']['status'] == 'success'
    findings = output['observations'] if successful else {}
    interpretation = findings.get('interpretation')
    next_check = findings.get('next_check')
    limitations = ['Exact pinned inputs/runtime only.',
        'Auxiliary evidence, not canonical task score or benefit.',
        'References available; actual use is unproven.',
        'Result projections retain exact values at original JSON paths; sparse tree branches are partial. Omitted fields are unavailable, not negative observations.']
    if plan['stage'] == 'independent_replay':
        limitations.append('Replay uses the original saved inputs.')
    comparison_assessment = deepcopy(observation.get('comparison_assessment', unavailable(observation['receipt_id'])))
    if not validate_comparison(comparison_assessment, observation['receipt_id']):
        raise ValueError('invalid host comparison assessment')
    card = {'schema':CARD_SCHEMA, 'comparison_assessment':comparison_assessment, 'cell_id':plan['cell_id'],
        'revision_id':plan['cell_revision_id'],
        'question':_clip(observation.get('question') or plan['question'], 600),
        'source':{'tool_id':observation['tool_id'], 'invocation_id':observation['invocation_id'],
            'receipt_id':observation['receipt_id'], **deepcopy(observation['target_binding'])},
        'observations':{'status':observation['result']['status'], 'output_excerpt':'',
            'resources':deepcopy(observation.get('resources', {}))},
        'interpretation':_clip(interpretation, 500) if type(interpretation) is str
            else ('Interpretation remains open; do not infer task improvement from tool execution.'
                  if successful else 'The research tool did not complete its declared check; no candidate finding is established.'),
        'next_check':_clip(next_check, 500) if type(next_check) is str
            else ('Use relevant findings to propose a change; test the resulting program with the ordinary evaluator.'
                  if successful else 'Repair the tool using its recorded error before drawing conclusions about the candidate.'),
        'limitations':limitations, 'replay_comparison':deepcopy(replay_comparison)}
    if replay_comparison is not None and not replay_comparison['matched']:
        card['interpretation'] = 'Execution and replay comparison are separate; this result did not satisfy the saved replay policy.'
        card['next_check'] = 'Inspect replay_comparison and its saved receipts; consider a child tool version without changing the existing policy.'
    if transform is not None:
        card = transform(card)
    maximum = min(config['max_context_chars'], 10000)
    result = {'output':output, 'result':observation['result'],
              'replay_policy':observation['replay_policy'], 'reason':observation.get('reason'),
              'diagnostic':observation.get('diagnostic')}
    if successful:
        # Measure the complete evidence in its actual enclosing format.
        # A smaller intermediate allowance can discard nested measurements
        # even when the whole immutable result fits the configured memory.
        card['observations']['output_excerpt'] = canonical_bytes(result).decode('ascii')
        if len(format_context([card])) <= maximum:
            return card
        card['observations']['output_excerpt'] = ''
    # Preserve identity/scope before projecting genuinely oversized results.
    # The exact escaped enclosing size is checked below; no value is sliced.
    budget = max(64, maximum - len(format_context([card])))
    card['observations']['output_excerpt'] = (success_context(result, budget)
        if successful else _failure_context(observation, budget))
    if len(format_context([card])) > maximum:
        card['interpretation'] = _clip_json(card['interpretation'], 90)
        card['next_check'] = _clip_json(card['next_check'], 90)
    # Fit escaped JSON exactly instead of halving until the terminal error
    # disappears. Identity, the saved question and full raw receipt stay intact.
    if len(format_context([card])) > maximum:
        original = card['observations']['output_excerpt']
        low = MIN_CONTEXT_CHARS if successful else 0
        high = budget if successful else len(original)
        fitted = success_context(result, MIN_CONTEXT_CHARS) if successful else ''
        while low <= high:
            middle = (low + high) // 2
            text = (success_context(result, middle) if successful
                    else _failure_context(observation, middle))
            card['observations']['output_excerpt'] = text
            if len(format_context([card])) <= maximum:
                fitted, low = text, middle + 1
            else:
                high = middle - 1
        card['observations']['output_excerpt'] = fitted
    if len(format_context([card])) > maximum and replay_comparison is not None:
        for count in range(len(replay_comparison['changed_paths']) - 1, -1, -1):
            card['replay_comparison'] = shorten_feedback(replay_comparison, count)
            if len(format_context([card])) <= maximum:
                break
    return card
