"""Steps of the search and their returns along lineages.

A step is the decision vector that made one child. Its return is the child's rise of the path best plus the
discounted mean return of the steps later taken from that child, bootstrapped toward the critic's value of the child;
a mean over follow-up steps, never a sum, so a program is not credited for how often it was chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GAMMA = 0.9
LAMBDA = 0.7


@dataclass
class Step:
    iteration: int
    island: int
    state: str
    child: str | None
    context: str
    gain: float
    records: dict = field(default_factory=dict)
    y: float = 0.0


def steps(ledger, covered: tuple) -> list[Step]:
    """Lineage rows grouped into steps, in decision order, keeping the covered decisions of each step."""
    out: dict[tuple, Step] = {}
    for row in ledger.rows:
        if row.scope != "lineage" or row.state is None:
            continue
        key = (row.iteration, row.island, row.state, row.child)
        step = out.setdefault(key, Step(row.iteration, row.island, row.state, row.child, row.context, row.gain))
        if row.decision in covered:
            step.records[row.decision] = row
    return sorted(out.values(), key=lambda s: s.iteration)


def returns(all_steps: list[Step], value, horizon: int) -> None:
    """Fill each step's return, latest steps first, with value(program, context) the critic's estimate."""
    follow: dict[str, list[Step]] = {}
    for step in all_steps:
        follow.setdefault(step.state, []).append(step)
    for step in reversed(all_steps):
        if step.child is None:
            step.y = step.gain
            continue
        base = value(step.child, step.context)
        later = [s for s in follow.get(step.child, []) if 0 < s.iteration - step.iteration <= horizon]
        tail = base + (LAMBDA * sum(s.y - base for s in later) / len(later) if later else 0.0)
        step.y = step.gain + GAMMA * tail
