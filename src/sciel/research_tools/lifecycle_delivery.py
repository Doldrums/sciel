"""Select and authenticate bounded observations for research and generation."""

from itertools import zip_longest
from .config import library_limits
from .library import ToolLibrary
from .replay_feedback import build_feedback
from ..engine.research_cells_context import _card

def author_feedback(lifecycle, island_id, visible, library):
    """One latest saved attempt per visible authored method, no new calls.

        The original attempt iteration orders evidence; a receipt hash breaks
        same-wave ties deterministically, including after checkpoint restore.
        """
    selected = []
    for ident in visible:
        method = lifecycle._state['methods'].get(ident)
        if method is None or method['author_island_id'] != island_id or not method['replay_attempts']:
            continue
        reference_id = method['local_receipt_id']
        reference = lifecycle._receipt(reference_id, ident, library)
        attempts = [(receipt_id, lifecycle._receipt(receipt_id, ident, library))
                    for receipt_id in method['replay_attempts'].values()]
        receipt_id, replay = max(attempts, key=lambda item: (item[1]['iteration'], item[0]))
        selected.append(build_feedback(reference, replay, tool_id=ident,
            reference_receipt_id=reference_id, replay_receipt_id=receipt_id))
    return selected


def shared_observations(lifecycle, library, island_id, task_identity):
    """Frozen all-island knowledge; no publication or invocation permission."""
    from .shared_evidence import collect
    return collect(lifecycle, library, island_id, task_identity)


def context_view(lifecycle, snapshot):
    ids = list(reversed(lifecycle._state['affinity'].get(str(snapshot.island), [])))
    if lifecycle.config.get('shared_observations'):
        # Keep original source cell, island, candidate and receipt. A card
        # is knowledge delivery, never permission to execute a foreign tool.
        foreign = [ident for ident in reversed(lifecycle._state['cards']) if ident not in ids]
        ids = [ident for pair in zip_longest(ids, foreign) for ident in pair if ident is not None]
    selected = []
    for ident in ids:
        if ident in lifecycle._state['cards']:
            selected.append(lifecycle._state['cards'][ident])
    plan = getattr(snapshot, 'tool_episode_plan', None) or {}
    # Eligibility precedes the logical-cell limit: an unsupported newer
    # interpretation must not evict an older measured observation.
    return lifecycle.delivery_context(selected, recipient_cell_id=plan.get('cell_id'),
                                 max_cells=lifecycle.config['max_visible_tools'])


def _delivery_measurement(lifecycle, card, episode=None):
    """Authenticate host warnings before a measured card enters a prompt.

        Use the existing bounded immutable evidence cache. Current invocation
        membership comes only from the supplied episode; orphan artifacts never
        gain observation rights merely by naming a receipt hash.
        """
    from .shared_evidence import EvidenceIndex
    from .comparison_assessment import from_record, CARD_SCHEMA as ASSESSED_CARD_SCHEMA
    if lifecycle._shared_evidence_index is None:
        library = ToolLibrary(lifecycle.run_dir, {'image_id':lifecycle.config['image_id']}, library_limits(lifecycle.config))
        lifecycle._shared_evidence_index = EvidenceIndex(lifecycle, library)
    index = lifecycle._shared_evidence_index
    index.update()
    source = card['source']
    ident = source['receipt_id']
    if ident in index.records:
        record = index.receipt(ident, source['tool_id'])
    else:
        current = next((row for row in (episode or {}).get('invocations', [])
                        if row['receipt_id'] == ident and row['tool_id'] == source['tool_id']), None)
        if current is None:
            raise ValueError('measured memory lacks committed or current-episode provenance')
        record = lifecycle._receipt(ident, source['tool_id'], index.library)
        if (record['action_id'] != episode['plan']['action_id']
                or record['cell_id'] != episode['plan']['cell_id']):
            raise ValueError('measured memory belongs to another episode')
    if (record['cell_id'] != card['cell_id'] or source['tool_id'] != record['tool_id']
            or source['invocation_id'] != record['receipt']['invocation_id']
            or any(source[k] != v for k, v in record['target_binding'].items())
            or card['observations']['status'] != record['result']['status']
            or card['observations']['resources'] != record['receipt']['resources']):
        raise ValueError('measured memory differs from immutable receipt')
    expected = from_record(record, ident, index.proposal)
    if card['schema'] == ASSESSED_CARD_SCHEMA:
        if card['comparison_assessment'] != expected:
            raise ValueError('research card comparison assessment differs from immutable evidence')
        return card
    if 'experiment_proposal_id' in record:
        raise ValueError('research card comparison assessment was removed')
    # Legacy cards remain readable, but an absent declaration is not a
    # retroactive valid comparison. Upgrade only the detached delivery view.
    return {**card, 'schema':ASSESSED_CARD_SCHEMA, 'comparison_assessment':expected}


def delivery_context(lifecycle, context, *, recipient_cell_id=None, episode=None, max_cells=None):
    """No-evidence decisions belong to their cell, not experiment memory.

        Ordinary generation has no recipient cell. A planner's exact cell ID
        comes from its frozen plan, never from island affinity. Keep immutable
        cards/decisions intact; this is only a delivery projection.
        """
    from ..engine.research_cells_context import compact_context
    from .shared_evidence import interpretation_evidence
    from .retained_observations import retained_bundle
    raw = context.get('cards') if type(context) is dict else context
    if raw is None:
        raw = []
    value = compact_context({**context, 'cards':[]} if type(context) is dict else [])
    if type(raw) is not list:
        return compact_context(context)
    # Read retained payloads in priority order. Eligibility must precede
    # the cell limit, but a full selection need not inspect/copy the tail.
    own = lifecycle._state['cards'].get(recipient_cell_id)
    if own is not None:
        raw = [own] + raw
    from ..engine.research_cells_context import card_identity
    library = None
    eligible, allowed, limited, seen = [], set(), set(), set()
    for item in raw:
        raw_cell = item.get('cell_id') if type(item) is dict else None
        if (max_cells is not None and len(allowed) >= max_cells
                and type(raw_cell) is str and raw_cell not in allowed):
            # Selection limit only: this untouched cell is not classified
            # as authenticated evidence or an unsupported interpretation.
            limited.add(raw_cell)
            continue
        card = _card(item, detach=False)
        if card is None:
            value['omitted_count'] += 1
            reasons = value['omission_reasons']
            reasons['invalid_card'] = reasons.get('invalid_card', 0) + 1
            continue
        identity = card_identity(card)
        if identity in seen:
            continue
        seen.add(identity)
        evidence = None
        if card['source']['receipt_id'] is None:
            if library is None:
                library = ToolLibrary(lifecycle.run_dir, {'image_id':lifecycle.config['image_id']},
                                      library_limits(lifecycle.config))
            evidence = interpretation_evidence(lifecycle, library, card, episode=episode)
            if not evidence['supported'] and card['cell_id'] != recipient_cell_id:
                value['omitted_count'] += 1
                reasons = value['omission_reasons']
                reasons['unmeasured_interpretation'] = reasons.get('unmeasured_interpretation', 0) + 1
                continue
        if card['source']['receipt_id'] is not None:
            card = lifecycle._delivery_measurement(card, episode)
        allowed.add(card['cell_id'])
        eligible.append((card, evidence))
    if limited:
        value['omitted_count'] += len(limited)
        reasons = value['omission_reasons']
        reasons['cell_limit'] = reasons.get('cell_limit', 0) + len(limited)
    # Process decisions first, so their authenticated reconstruction wins
    # over stale frozen projections of the same receipt. Budget each bundle
    # in the remaining enclosing context; reserve its decision explicitly.
    from .retained_observations import _size
    maximum = min(lifecycle.config['max_context_chars'], 10000)
    selected, expanded = [], set()
    for card, evidence in eligible:
        if evidence is None:
            continue
        evidence = {**evidence, 'records':[(ident, record) for ident, record in evidence['records']
            if (record['cell_id'], ident) not in expanded]}
        empty_size = _size({'schema':value['schema'], 'cards':[], 'omitted_count':0, 'omission_reasons':{}})
        from .observation_cards import format_context
        # Ordinary G uses a spaced/escaped formatter, whereas the frozen
        # envelope is compact JSON. Charge whichever representation is
        # larger, plus the separator joining this bundle to prior cards.
        used = max(_size({**value, 'cards':selected}) - empty_size,
                   len(format_context(selected)) - len(format_context([])))
        if selected:
            used += 2
        bundle, omissions = retained_bundle(lifecycle, library, card, evidence,
                                             budget=max(0, maximum - used))
        for item in bundle:
            key = (item['cell_id'], item['source']['receipt_id'])
            if key not in expanded:
                selected.append(item)
                expanded.add(key)
        for reason, count in omissions.items():
            value['omitted_count'] += count
            value['omission_reasons'][reason] = value['omission_reasons'].get(reason, 0) + count
    for card, evidence in eligible:
        key = (card['cell_id'], card['source']['receipt_id'])
        if evidence is None and key not in expanded:
            selected.append(card)
            expanded.add(key)
    value['cards'] = selected
    return compact_context(value, max_chars=maximum)
