"""Islands as schools: each island keeps its own verdict on strategies, findings and procedures.

An item proven on one island is offered to the others first, and each recipient accepts or retires it only on the
claims estimated from its own rows of the ledger.
"""

from __future__ import annotations

from itertools import zip_longest

DECISIONS = ("strategy", "finding", "procedure")
PREFIX = "island="
INHERITED = "h"


def island_context(island: int) -> str:
    return f"{PREFIX}{island}"


def available(items: list, island: int) -> list:
    return [i for i in items if i.status != "retired" and i.local.get(str(island)) != "retired"]


def local_culture(items: list, island: int) -> list:
    return [i for i in items if i.local.get(str(island)) == "culture"]


def trials(items: list, island: int) -> list:
    """Untested items on this island: items proven on another island first, then inherited and own items in turn,
    own items newest first."""
    untested = [i for i in available(items, island) if i.status == "trial" and str(island) not in i.local]
    proven = [i for i in untested if "culture" in i.local.values()]
    inherited = [i for i in untested if i.id.startswith(INHERITED) and i not in proven]
    own = [i for i in reversed(untested) if i not in proven and i not in inherited]
    turns = [i for pair in zip_longest(inherited, own) for i in pair if i is not None]
    return proven + turns


def judge(items: list, theory, decision: str) -> list:
    changed = []
    for claim in theory.claims:
        if claim.decision != decision or not claim.context.startswith(PREFIX) or not claim.decided:
            continue
        item = next((i for i in items if i.id == claim.action), None)
        island, status = claim.context[len(PREFIX):], "culture" if claim.effect > 0 else "retired"
        if item is not None and item.local.get(island) != status:
            item.local[island] = status
            changed.append((item, int(island), status))
    return changed


def verdict(theory, decision: str, action: str) -> str | None:
    """Run-wide status from the decided run-wide claims: culture or retired only when they agree in sign."""
    effects = [c.effect for c in theory.claims if c.decision == decision and c.action == action and c.decided
               and not c.context.startswith(PREFIX)]
    if effects and min(effects) > 0:
        return "culture"
    if effects and max(effects) < 0:
        return "retired"
    return None
