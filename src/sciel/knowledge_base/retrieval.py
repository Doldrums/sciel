"""Opt-in, deterministic source-diversity selection over an unchanged KB ranking.

Source IDs identify shared observations, not semantic equivalence or measured
benefit. Only whole cards actually admitted by the caller affect redundancy.
This selector never changes the knowledge base or its confidence/evidence.
"""
from __future__ import annotations

import copy
import math


CONTRACT = 'sciel-lesson-selection/1'
GUARDED_CONTRACT = 'sciel-lesson-selection-guarded/1'
GUARD_VERSION = 'whole-card-source-lexical/1'


def validate_selection(raw):
    """Validate an explicitly supplied selection object; absence is caller-owned."""
    if type(raw) is not dict or set(raw) - {'mode', 'diversity_weight'}:
        raise ValueError('research lesson_selection must be a selection object')
    mode = raw.get('mode', 'ranked')
    if type(mode) is not str or mode not in {'ranked', 'source_diverse', 'source_diverse_guarded'}:
        raise ValueError('research lesson_selection mode must be ranked, source_diverse or source_diverse_guarded')
    weight = raw.get('diversity_weight', 0.5)
    if (type(weight) not in (int, float) or not 0 <= weight <= 1
            or not math.isfinite(weight)):
        raise ValueError('research lesson_selection diversity_weight must be finite in [0, 1]')
    if mode == 'source_diverse_guarded' and weight != 0.5:
        raise ValueError('source_diverse_guarded fixes diversity_weight at 0.5')
    return {'mode': mode, 'diversity_weight': float(weight)}


def selection_identity(research_config):
    """Bind every opt-in selection, including zero weight and injection ablation."""
    if 'lesson_selection' not in research_config:
        return None
    selection = validate_selection(research_config['lesson_selection'])
    if selection['mode'] == 'ranked':
        return None
    if selection['mode'] == 'source_diverse_guarded':
        from ..hypothesis.rendering import presentation_identity
        return {'schema': GUARDED_CONTRACT, **selection, 'guard_version': GUARD_VERSION,
                'rendering': presentation_identity(
                    compact_lesson_memory=research_config.get('compact_lesson_memory', False))}
    return {'schema': CONTRACT, **selection}


class SourceDiverseSelection:
    """Sequential ranking whose caller explicitly acknowledges card admission.

    ``next`` returns the next lesson and its decision facts; ``record`` must follow
    after rendering and checking the whole-card limits. A rejected card is removed
    from consideration but contributes no redundancy penalty to future cards.
    """

    def __init__(self, ranked, weight):
        self.weight = validate_selection({'mode': 'source_diverse',
                                          'diversity_weight': weight})['diversity_weight']
        self._lessons = copy.deepcopy(list(ranked))
        self._base = []
        seen = set()
        for rank, lesson in enumerate(self._lessons):
            ident, sources = lesson['id'], lesson['source_support_ids']
            if (type(ident) is not str or not ident or ident in seen
                    or type(sources) is not list
                    or any(type(source) is not str or not source for source in sources)
                    or len(set(sources)) != len(sources)):
                raise ValueError('invalid source-diverse lesson candidates')
            seen.add(ident)
            self._base.append({'id': ident, 'base_rank': rank,
                'base_utility': 1.0 / (rank + 1), 'source_support_ids': sources[:]})
        self._remaining = list(range(len(self._base)))
        self._admitted = []
        self._decisions = []
        self._pending = None

    @property
    def has_remaining(self):
        return bool(self._remaining)

    def _decision(self, index):
        base = self._base[index]
        sources = set(base['source_support_ids'])
        redundancy = 0.0
        for other in self._admitted:
            observed = set(self._base[other]['source_support_ids'])
            union = sources | observed
            overlap = len(sources & observed) / len(union) if union else 0.0
            redundancy = max(redundancy, overlap)
        penalty = self.weight * redundancy
        return {'id': base['id'], 'base_rank': index,
                'base_utility': base['base_utility'],
                'against_ids': [self._base[i]['id'] for i in self._admitted],
                'redundancy': redundancy, 'penalty': penalty,
                'score': base['base_utility'] - penalty}

    def next(self):
        if self._pending is not None:
            raise ValueError('source-diverse selection requires an admission decision')
        if not self._remaining:
            raise ValueError('source-diverse selection is exhausted')
        decision = max((self._decision(i) for i in self._remaining),
                       key=lambda item: (item['score'], -item['base_rank']))
        self._pending = decision
        return copy.deepcopy(self._lessons[decision['base_rank']]), copy.deepcopy(decision)

    def record(self, decision, *, admitted, reason=None):
        if self._pending is None or decision != self._pending:
            raise ValueError('source-diverse admission does not match selected candidate')
        if (type(admitted) is not bool or (admitted and reason is not None)
                or (not admitted and (type(reason) is not str or not reason))):
            raise ValueError('invalid source-diverse admission result')
        index = self._pending['base_rank']
        self._remaining.remove(index)
        if admitted:
            self._admitted.append(index)
        self._decisions.append({**self._pending, 'admitted': admitted, 'reason': reason})
        self._pending = None

    def metadata(self):
        if self._pending is not None:
            raise ValueError('source-diverse selection has an unresolved admission')
        return copy.deepcopy({'schema': CONTRACT, 'mode': 'source_diverse',
            'diversity_weight': self.weight, 'base_candidates': self._base,
            'decisions': self._decisions,
            'remaining_ids': [self._base[i]['id'] for i in self._remaining],
            'unshown_ids': [b['id'] for i, b in enumerate(self._base)
                            if i not in self._admitted]})
