"""Reconstruct bounded prior observations from authenticated immutable receipts.

No new invocation or checkpoint evidence is created. Terminal interpretation and
its original source remain separate from every cited measurement's source.
"""
from copy import deepcopy
import json

from .decision_projection import project_decision
from .observation_cards import measurement_card, format_context
from .replay_feedback import build_feedback
from .result_projection import preserves_measurements as _preserves_measurements

MAX_PRIOR_CARDS = 4


def _size(value):
    return len(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')))


def _compact_decision(card):
    result = deepcopy(card)
    result['interpretation'] = 'Model decision below.'
    result['next_check'] = ''
    result['limitations'] = ['No new invocation; prior measurements remain separate.']
    return result


def _decision(card, proposal, budget):
    def size(value):
        return max(_size(value), len(json.dumps(value, ensure_ascii=True)))
    def restore_short_text(value):
        # An omission marker can cost more than the missing text. Restore a
        # whole field whenever it fits instead of keeping an unnecessary cut.
        for _ in range(2):
            for position, key in enumerate(('question', 'interpretation', 'next_decision')):
                shown = value['decision']
                if not shown['omitted_chars'][key]:
                    continue
                limits = [shown['original_chars'][name] - shown['omitted_chars'][name]
                          for name in ('question', 'interpretation', 'next_decision')]
                limits[position] = shown['original_chars'][key]
                trial = {**value, 'decision': project_decision(proposal,
                    card['decision']['proposal_id'], text_limits=tuple(limits),
                    max_receipts=len(shown['receipt_ids']))}
                if size(trial) <= budget:
                    value = trial
        return value
    result = deepcopy(card)
    # Delivery may receive a previously compacted card. Always reconstruct
    # from its authenticated proposal, not that earlier lossy projection.
    result['decision'] = project_decision(proposal, card['decision']['proposal_id'])
    if size(result) <= budget:
        return result
    # Avoid spending evidence space on repeated host boilerplate. The complete
    # model text and authenticated scope remain in decision; invocation status
    # remains explicit in observations and source.
    result = _compact_decision(result)
    if size(result) <= budget:
        return result
    # Whole text omissions disclose their exact original lengths and hashes.
    for chars in (600, 400, 200, 100, 0):
        for refs in (16, 4, 1, 0):
            result['decision'] = project_decision(proposal, card['decision']['proposal_id'],
                text_limits=(min(chars, 600), chars, min(chars, 600)), max_receipts=refs)
            if size(result) <= budget:
                return restore_short_text(result)
    return result  # A caller may explicitly omit an impossible minimum card.


def _measurement(lifecycle, library, index, ident, record, budget, *, compact=False):
    if ident in index.records:
        record = index.receipt(ident, record['tool_id'])
    receipt = record['receipt']
    comparison = None
    if record['stage'] == 'independent_replay':
        method = lifecycle._state['methods'].get(record['tool_id'])
        if method is not None:
            original_id = method['local_receipt_id']
            original = index.receipt(original_id, record['tool_id'])
            comparison = build_feedback(original, record, tool_id=record['tool_id'],
                reference_receipt_id=original_id, replay_receipt_id=ident)
    from .comparison_assessment import from_record
    assessment = from_record(record, ident, lambda proposal_id: index.proposal(proposal_id))
    observation = {'comparison_assessment':assessment, 'tool_id': record['tool_id'], 'invocation_id': receipt['invocation_id'],
        'receipt_id': ident, 'target_binding': record['target_binding'],
        'question': record['question'], 'result': record['result'],
        'replay_policy': record['replay_policy'],
        **{key: receipt.get(key) for key in ('output', 'reason', 'diagnostic', 'stderr', 'resources')}}
    plan = {'cell_id': record['cell_id'], 'stage': record['stage'], 'question': record['question'],
        'cell_revision_id': index.receipt_revisions.get(ident, record['cell_revision_id'])}
    def compact_host_text(card):
        # Repeated host guidance must not crowd out the evidence it describes.
        # Keep source/question/resources and any tool-authored prose unchanged.
        output = receipt.get('output')
        findings = output.get('observations') if type(output) is dict else None
        findings = findings if type(findings) is dict else {}
        successful = record['result']['status'] == 'success'
        if comparison is None or comparison['matched']:
            if not successful:
                card['interpretation'] = 'Tool failed; no candidate finding established.'
                card['next_check'] = 'Repair using the saved error before drawing conclusions.'
            else:
                if type(findings.get('interpretation')) is not str:
                    card['interpretation'] = 'Auxiliary evidence; no canonical task benefit established.'
                if type(findings.get('next_check')) is not str:
                    card['next_check'] = ''
        card['limitations'] = ['Auxiliary evidence, not canonical task score or benefit. '
            'Exact pinned inputs/runtime only; omissions are unknown. References do not prove use.']
        if record['stage'] == 'independent_replay':
            card['limitations'].append('Replay uses the original saved inputs.')
        return card

    return measurement_card({**lifecycle.config, 'max_context_chars': max(1, budget)},
                            plan, observation, comparison,
                            transform=compact_host_text if compact else None)


def retained_bundle(lifecycle, library, card, evidence, *, budget):
    """Measurement(s), then interpretation, with explicit reference/budget loss.

    Reserve interpretation and a whole measurement projection together. A small
    context may omit measurements, but cannot present an omitted value as zero.
    """
    records = list(reversed(evidence['records']))
    reasons = {}
    if len(records) > MAX_PRIOR_CARDS:
        reasons['prior_evidence_limit'] = len(records) - MAX_PRIOR_CARDS
        records = records[:MAX_PRIOR_CARDS]
    # First test complete payloads under the actual escaped enclosing format.
    # A per-card allowance must not discard a whole fitting comparison array.
    from ..engine.research_cells_context import SCHEMA, MAX_CHARS
    def available(cards, omissions):
        return min(min(MAX_CHARS, budget) - _size({'schema': SCHEMA, 'cards': cards,
            'omitted_count': sum(omissions.values()), 'omission_reasons': omissions}),
            budget - len(format_context(cards)))

    def fits(cards, omissions):
        return available(cards, omissions) >= 0
    complete = [_measurement(lifecycle, library, evidence['index'], ident, record, budget)
                for ident, record in records] + [_decision(card, evidence['proposal'], budget)]
    if fits(complete, reasons):
        return complete, reasons
    # Keep complete measured values when shortening only the later decision
    # suffices. Count the same escaped formatter used by ordinary generation.
    measured = complete[:-1]
    occupied = max(_size({'schema':SCHEMA, 'cards':measured + [{}],
        'omitted_count':sum(reasons.values()), 'omission_reasons':reasons}),
        len(format_context(measured + [{}]))) - 2
    decision = _decision(card, evidence['proposal'], max(0, budget - occupied))
    text = decision['decision']
    retains_decision_text = all(text['original_chars'][key] == 0
        or text['omitted_chars'][key] < text['original_chars'][key]
        for key in ('interpretation', 'next_decision'))
    if fits(measured + [decision], reasons) and retains_decision_text:
        return measured + [decision], reasons
    # Mandatory host assessments may make formerly fitting boilerplate too
    # large. Try whole measurements with only redundant host text compacted
    # before assigning per-card shares or projecting any observation values.
    compact_measured = [_measurement(lifecycle, library, evidence['index'], ident, record,
                                    budget, compact=True) for ident, record in records]
    occupied = max(_size({'schema':SCHEMA, 'cards':compact_measured + [{}],
        'omitted_count':sum(reasons.values()), 'omission_reasons':reasons}),
        len(format_context(compact_measured + [{}]))) - 2
    compact_terminal = _decision(card, evidence['proposal'], max(0, budget - occupied))
    text = compact_terminal['decision']
    retains_decision_text = all(text['original_chars'][key] == 0
        or text['omitted_chars'][key] < text['original_chars'][key]
        for key in ('interpretation', 'next_decision'))
    preserves = all(old['observations']['status'] != 'success' or _preserves_measurements(
        json.loads(old['observations']['output_excerpt']), json.loads(new['observations']['output_excerpt']))
        for old, new in zip(measured, compact_measured))
    if preserves and retains_decision_text and fits(compact_measured + [compact_terminal], reasons):
        return compact_measured + [compact_terminal], reasons
    maximum = max(0, budget - 350)
    minimum_decision = _compact_decision(card)
    minimum_decision['decision'] = project_decision(evidence['proposal'],
        card['decision']['proposal_id'], text_limits=(100, 160, 100), max_receipts=1)
    decision_floor = max(_size(minimum_decision), len(json.dumps(minimum_decision, ensure_ascii=True)))
    for count in range(len(records), -1, -1):
        if count:
            decision = _decision(card, evidence['proposal'],
                                 max(decision_floor, maximum // max(4, count + 1)))
            share = max(1, (maximum - _size(decision)) // count)
            measured = [_measurement(lifecycle, library, evidence['index'], ident, record, share,
                                     compact=True)
                        for ident, record in records[:count]]
        else:
            measured = []
            decision = _decision(card, evidence['proposal'], maximum)
        cards = measured + [decision]
        omissions = dict(reasons)
        omitted = len(records) - count
        if omitted:
            omissions['character_limit'] = omitted
        if fits(cards, omissions):
            # Shares are only a starting allocation. A short receipt must not
            # strand its allowance while another loses measurements. Grow
            # under both actual enclosing formats, newest evidence first, then
            # the terminal decision; never select by a reported outcome.
            for _ in range(2):
                before = cards
                for index, (ident, record) in enumerate(records[:count]):
                    allowance = available(cards, omissions)
                    if allowance <= 0:
                        break
                    bound = len(format_context([cards[index]])) + allowance
                    grown = _measurement(lifecycle, library, evidence['index'], ident, record,
                                         bound, compact=True)
                    trial = cards[:index] + [grown] + cards[index + 1:]
                    preserves = (cards[index]['observations']['status'] != 'success'
                        or _preserves_measurements(
                            json.loads(cards[index]['observations']['output_excerpt']),
                            json.loads(grown['observations']['output_excerpt'])))
                    if preserves and fits(trial, omissions):
                        cards = trial
                allowance = available(cards, omissions)
                bound = max(_size(cards[-1]), len(json.dumps(cards[-1], ensure_ascii=True))) + allowance
                grown = _decision(card, evidence['proposal'], bound)
                trial = cards[:-1] + [grown]
                if fits(trial, omissions):
                    cards = trial
                if cards == before:
                    break
            return cards, omissions
        if not count:
            return cards, omissions
