"""Islands as schools: each island keeps its own verdict on strategies, findings and procedures.

An item proven on one island is offered to the others first, and each recipient accepts or retires it only on the
claims estimated from its own rows of the ledger.
"""

from __future__ import annotations

DECISIONS = ("strategy", "finding", "procedure")
PREFIX = "island="


def island_context(island: int) -> str:
    return f"{PREFIX}{island}"


def available(items: list, island: int) -> list:
    return [i for i in items if i.status != "retired" and i.local.get(str(island)) != "retired"]


def local_culture(items: list, island: int) -> list:
    return [i for i in items if i.local.get(str(island)) == "culture"]


def trials(items: list, island: int) -> list:
    """Untested items on this island, items proven on another island first."""
    untested = [i for i in available(items, island) if i.status == "trial" and str(island) not in i.local]
    proven = [i for i in untested if "culture" in i.local.values()]
    return proven + [i for i in reversed(untested) if i not in proven]


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
