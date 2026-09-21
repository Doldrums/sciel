"""C3 knowledge base: lessons from event postmortems, credited by lift.

Lesson weight is empirical lift of P(event) against a matched base rate (same island
and phase), not LLM opinion. Alignment gate = coverage of mechanism tokens over the
diff's changed lines (the mechanism must be grounded in what the mutation changed; the
loop passes the changed region, not the whole child) plus a minimum mechanism-token
guard so a one-word mechanism cannot pass trivially. Dedup by Jaccard on lesson tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-zA-Z]{3,}")


def _tokens(text: str) -> set:
    return set(w.lower() for w in _WORD.findall(text or ""))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _coverage(mechanism: set, diff: set) -> float:
    """Share of the MECHANISM's tokens that appear in the diff.

    This used to divide by min(|mechanism|, |diff|), the overlap coefficient. `diff`
    mutations routinely change one or two lines, so the divisor was usually |diff|,
    and a single shared word such as `return` passed a 0.34 gate at 0.5 while the
    documented quantity was 0.17. The min_mechanism_tokens guard blocks a one-word
    mechanism; nothing blocked a one-token diff, which is the same trivial pass on
    the other axis."""
    if not mechanism or not diff:
        return 0.0
    return len(mechanism & diff) / len(mechanism)


@dataclass
class Lesson:
    id: int
    text: str
    mechanism: str
    created_iter: int
    exposures: int = 0
    exposure_events: int = 0
    baseline_sum: float = 0.0
    tokens: set = field(default_factory=set)

    def lift(self) -> float:
        if self.exposures == 0:
            return 0.0
        rate = self.exposure_events / self.exposures
        base = self.baseline_sum / self.exposures
        return rate - base

    def to_json(self) -> dict:
        return {"id": self.id, "text": self.text, "mechanism": self.mechanism,
                "created_iter": self.created_iter, "exposures": self.exposures,
                "exposure_events": self.exposure_events,
                "baseline_sum": self.baseline_sum, "tokens": sorted(self.tokens)}

    @classmethod
    def from_json(cls, d: dict) -> "Lesson":
        return cls(id=d["id"], text=d["text"], mechanism=d["mechanism"],
                   created_iter=d["created_iter"], exposures=d["exposures"],
                   exposure_events=d["exposure_events"],
                   baseline_sum=d["baseline_sum"], tokens=set(d["tokens"]))


class KnowledgeBase:
    def __init__(self, retrieve_k: int = 2, dedup_threshold: float = 0.8,
                 alignment_threshold: float = 0.34, cap: int = 50,
                 min_mechanism_tokens: int = 2):
        self.retrieve_k = retrieve_k
        self.dedup_threshold = dedup_threshold
        self.alignment_threshold = alignment_threshold
        self.cap = cap
        self.min_mechanism_tokens = min_mechanism_tokens
        self.lessons: list = []
        self._counter = 0

    def add_lesson(self, text: str, mechanism: str, diff_text: str,
                   iteration: int, align: bool = True) -> Lesson | None:
        mech_tokens = _tokens(mechanism)
        if len(mech_tokens) < self.min_mechanism_tokens:
            return None
        # align=False for free (card-sourced) lessons: a strategy card is the
        # LLM's own natural-language self-description, not code, so token overlap
        # with the diff is meaningless (prose vs identifiers).
        if align:
            diff_tokens = _tokens(diff_text)
            if _coverage(mech_tokens, diff_tokens) < self.alignment_threshold:
                return None
        toks = _tokens(text) | mech_tokens
        for ls in self.lessons:
            if _jaccard(toks, ls.tokens) >= self.dedup_threshold:
                return None
        self._counter += 1
        lesson = Lesson(id=self._counter, text=text.strip(), mechanism=mechanism.strip(),
                        created_iter=iteration, tokens=toks)
        self.lessons.append(lesson)
        self._prune()
        return lesson

    def retrieve(self, parent_score: float) -> list:
        if not self.lessons:
            return []
        ranked = sorted(self.lessons, key=lambda l: (l.lift(), l.created_iter),
                        reverse=True)
        out = ranked[:self.retrieve_k]
        # cold-start explore: a never-exposed lesson has lift 0 and would never
        # surface to earn exposures. When there is more than one slot, reserve the
        # last one for the newest unexposed lesson so credit can accumulate (with a
        # single slot, keep pure lift ordering).
        if self.retrieve_k >= 2:
            unexposed = [l for l in self.lessons if l.exposures == 0]
            if unexposed and all(l.exposures > 0 for l in out):
                out = out[:self.retrieve_k - 1] + [max(unexposed, key=lambda l: l.created_iter)]
        return out

    def note_exposure(self, lesson_id: int, outcome: bool, baseline: float) -> None:
        for ls in self.lessons:
            if ls.id == lesson_id:
                ls.exposures += 1
                ls.exposure_events += 1 if outcome else 0
                ls.baseline_sum += baseline
                return

    def _prune(self) -> None:
        if len(self.lessons) <= self.cap:
            return
        self.lessons.sort(key=lambda l: (l.lift(), l.created_iter), reverse=True)
        self.lessons = self.lessons[:self.cap]

    def state(self) -> dict:
        return {"retrieve_k": self.retrieve_k, "dedup_threshold": self.dedup_threshold,
                "alignment_threshold": self.alignment_threshold, "cap": self.cap,
                "counter": self._counter,
                "lessons": [l.to_json() for l in self.lessons]}

    def load_state(self, d: dict) -> None:
        self.retrieve_k = d["retrieve_k"]
        self.dedup_threshold = d["dedup_threshold"]
        self.alignment_threshold = d["alignment_threshold"]
        self.cap = d["cap"]
        self._counter = d["counter"]
        self.lessons = [Lesson.from_json(x) for x in d["lessons"]]
