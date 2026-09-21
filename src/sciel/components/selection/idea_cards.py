"""Opt-in lexical card-distance target; no extra draws, learned state or calls."""
from __future__ import annotations

import hashlib
import json

from ..policy.tilt import Tilt
from .card_space import CONTRACT as REPRESENTATION, card_novelty

CONTRACT = "sciel-idea-card-selection/1"
EPSILON = 1e-6


def prepare_card_tilt(view, cards, visits, *, beta, neighbors):
    """Use only the supplied active IDs; legacy/missing cards stay neutral."""
    active = {pid: cards.get(pid) for pid in view.ids}
    space = card_novelty(active, k=neighbors)
    before = {pid: visits.get(pid, 0) for pid in view.ids}
    target = None
    if space['available']:
        weights = {pid: (EPSILON + space['novelty'][pid]) / (1 + before[pid])
                   for pid in view.ids}
        total = sum(weights.values())
        target = {pid: weights[pid] / total for pid in view.ids}
    inputs = {'ids': list(view.ids), 'scores': dict(view.scores), 'visits_before': before,
              'card_metadata': space['card_metadata']}
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    details = {'contract': CONTRACT, 'representation': REPRESENTATION,
               'beta': beta, 'neighbors': neighbors, 'epsilon': EPSILON,
               'input_sha256': digest, **inputs, 'card_space': space,
               'target_distribution': target}
    return (Tilt(beta, lambda ignored: dict(target)) if target is not None else None), details
