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
    return [i for i in items if i.status not in ("retired", "archived") and i.local.get(str(island)) != "retired"]


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


STATUS = {"culture": "culture", "retired": "retired", "not_established": "tested"}


def settle(items: list, trials) -> list:
    """Pool items take the verdict of their sibling trial. An item that never established an effect stays in the
    randomized delivery as tested; culture, also when a research programme graduated it, only ever falls to retired."""
    changed = []
    for item in items:
        status = STATUS.get(trials.status(item.id))
        if item.status == "culture" and status != "retired":
            continue
        if status is not None and item.status != status:
            item.status = status
            changed.append(item)
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
