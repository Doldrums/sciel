"""Theorist strategies: proposed search directives that enter as randomized arms and survive by measured effect."""

from __future__ import annotations

from dataclasses import asdict, dataclass

PREAMBLE = "You are the theorist of an automated program search."
PROPOSE_INTERVAL = 25
MAX_ACTIVE = 3
NONE_SHARE = 0.5
MAX_CHARS = 400


@dataclass
class Strategy:
    id: str
    text: str
    born: int
    status: str = "trial"


def _excerpt(code: str, limit: int = 1500) -> str:
    return code if len(code) <= limit else code[:limit] + "\n# ..."


class Strategies:
    def __init__(self):
        self.pool: list[Strategy] = []
        self.last_request = 0

    def live(self) -> list[Strategy]:
        return [s for s in self.pool if s.status != "retired"]

    def distribution(self) -> dict:
        live = self.live()
        if not live:
            return {"none": 1.0}
        return {"none": NONE_SHARE} | {s.id: (1 - NONE_SHARE) / len(live) for s in live}

    def text(self, strategy_id: str) -> str | None:
        return next((s.text for s in self.pool if s.id == strategy_id), None)

    def due(self, iteration: int) -> bool:
        trials = sum(s.status == "trial" for s in self.pool)
        return trials < MAX_ACTIVE and iteration - self.last_request >= PROPOSE_INTERVAL

    def request(self, iteration: int, *, task: str, programs: list, claims: list[str]) -> str:
        self.last_request = iteration
        best = "\n\n".join(f"Score {p.score:.4f}:\n```python\n{_excerpt(p.code)}\n```" for p in programs)
        tried = "\n".join(f"- [{s.status}] {s.text}" for s in self.pool) or "- none yet"
        known = "\n".join(f"- {c}" for c in claims) or "- none yet"
        return (f"{PREAMBLE}\n\nTask:\n{task}\n\nBest programs so far:\n{best}\n\n"
                f"Measured claims about the search:\n{known}\n\nStrategies already tried:\n{tried}\n\n"
                "Propose one new, concrete strategy for the next mutations: a specific algorithmic idea "
                "or construction that the best programs do not use yet. It must differ from the strategies "
                "already tried. Reply with the strategy only, in at most three sentences.")

    def adopt(self, text: str | None, iteration: int) -> Strategy | None:
        text = " ".join((text or "").split())[:MAX_CHARS]
        if not text or any(s.text == text for s in self.pool):
            return None
        strategy = Strategy(id=f"s{len(self.pool) + 1}", text=text, born=iteration)
        self.pool.append(strategy)
        return strategy

    def judge(self, theory) -> list[Strategy]:
        changed = []
        for claim in theory.claims:
            strategy = next((s for s in self.pool if s.id == claim.action), None)
            if claim.decision != "strategy" or strategy is None or not claim.decided:
                continue
            status = "culture" if claim.effect > 0 else "retired"
            if strategy.status != status:
                strategy.status = status
                changed.append(strategy)
        return changed

    def state(self) -> dict:
        return {"pool": [asdict(s) for s in self.pool], "last_request": self.last_request}

    def load_state(self, data: dict) -> None:
        self.pool = [Strategy(**s) for s in data["pool"]]
        self.last_request = int(data["last_request"])
