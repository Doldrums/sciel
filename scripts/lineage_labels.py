"""Lineage-credit labels for the population meta-cognition model (Phase 1).

Reconstructs the parent->child lineage tree from a run journal (each successful
iteration logs parent_id + child_id + child_score) and computes, per program, the
LINEAGE-CREDIT label:

    subtree_max[p]  = best score reached anywhere in p's descendant subtree (incl. p)
    lineage_gain[p] = max(0, subtree_max[p] - score[p])  -- improvement p's lineage produced

This is the distillation TARGET for a population set-transformer: "if selection breeds
from p, how good does its lineage get" -- the Algorithm-Distillation / Evolution-
Transformer credit computed offline over our own realized run histories. Requires the
`child_id` field on iteration events (added in Phase 0) to build the edges.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_data import (DirtyRun, checkpoint_programs, completed_events,
                          evaluated_code)


def build_lineage(path, with_code=False, *, trajectory=None):
    nodes, children = {}, {}
    try:
        events, end = trajectory if trajectory is not None else completed_events(path)
        bound = end.get('segment_id') is not None
        programs = checkpoint_programs(path, end=end) if with_code or bound else {}
    except (OSError, DirtyRun) as exc:
        print(f"skip {path}: {exc}", file=sys.stderr)
        return nodes, children
    for e in events:
        if e.get("event") != "iteration" or e.get("child_score") is None:
            continue
        cid = e.get("child_id")
        if cid is None:
            print(f"skip {path} iteration {e.get('iteration_no')}: missing child_id",
                  file=sys.stderr)
            continue
        code = evaluated_code(e, programs) if with_code else None
        if with_code and code is None:
            print(f"skip {path} child {cid}: missing evaluated source", file=sys.stderr)
            continue  # raw response may not be the grafted/repaired evaluated source
        nodes[cid] = {"score": e["child_score"], "parent": e.get("parent_id"),
                      "island": e.get("island"), "B_evals": e.get("B_evals")}
        if with_code:                          # for embedding-input promise models
            nodes[cid]["code"] = code
            nodes[cid]["cases"] = e.get("cases") or {}
        children.setdefault(e.get("parent_id"), []).append(cid)
    if bound:
        # Migration is an ancestry bridge, not a new mutation/training sample.
        # Only the selected branch's authenticated checkpoint can supply aliases.
        aliases = {pid:program.get('parent_id') for pid,program in programs.items()
                   if program.get('origin')=='migrant'}
        resolved = {}
        try:
            for ident,parent in aliases.items():
                source=programs.get(parent)
                copy=programs[ident]
                if (source is None or copy.get('code')!=source.get('code')
                        or copy.get('score')!=source.get('score') or copy.get('cases')!=source.get('cases')):
                    raise DirtyRun('checkpoint migration is not an exact observed-program copy')
                chain,seen=[],set()
                current=ident
                while current in aliases and current not in resolved:
                    if current in seen:
                        raise DirtyRun('cycle in checkpoint migration aliases')
                    seen.add(current);chain.append(current);current=aliases[current]
                ancestor=resolved.get(current,current)
                for item in chain:
                    resolved[item]=ancestor
                children.setdefault(parent,[]).append(ident)
            for node in nodes.values():
                node['ancestry_parent']=resolved.get(node['parent'],node['parent'])
        except DirtyRun as exc:
            print(f'skip {path}: {exc}',file=sys.stderr)
            return {},{}
    return nodes, children


def _postorder(roots, edges):
    """Iterative DAG traversal, independent of ancestry depth or input order."""
    state = {}
    for root in roots:
        stack = [(root, False)]
        while stack:
            nid, finished = stack.pop()
            if finished:
                state[nid] = 2
                yield nid
                continue
            if state.get(nid) == 2:
                continue
            if state.get(nid) == 1:
                raise ValueError(f"cycle in lineage at {nid!r}")
            state[nid] = 1
            stack.append((nid, True))
            stack.extend((child, False) for child in reversed(edges.get(nid, ())))


def subtree_max(nodes, children):
    """subtree_max[id] = max score over id's descendant subtree (inclusive)."""
    memo = {}
    for nid in _postorder(nodes, children):
        best = nodes[nid]["score"] if nid in nodes else float("-inf")
        memo[nid] = max((memo[child] for child in children.get(nid, ())), default=best)
        memo[nid] = max(best, memo[nid])
    return memo


def _depths(nodes):
    """depth[id] = generations from a root (parent not in nodes = seed/migrant = 0)."""
    parents = {nid: [n.get('ancestry_parent',n['parent'])] for nid, n in nodes.items()
               if n.get('ancestry_parent',n['parent']) in nodes}
    memo = {}
    for nid in _postorder(nodes, parents):
        memo[nid] = memo[parents[nid][0]] + 1 if nid in parents else 0
    return memo


def labels_for(path, with_code=False, *, trajectory=None):
    nodes, children = build_lineage(path, with_code=with_code, trajectory=trajectory)
    try:
        sm = subtree_max(nodes, children)
        dep = _depths(nodes)
    except ValueError as exc:
        print(f"skip {path}: {exc}", file=sys.stderr)
        return []
    out = []
    for nid, n in nodes.items():
        rec = {"id": nid, "score": n["score"], "island": n["island"],
               "B_evals": n["B_evals"], "parent": n["parent"], "gen": dep[nid],
               "subtree_max": sm[nid],
               "lineage_gain": max(0.0, sm[nid] - n["score"])}
        if with_code:
            rec["code"] = n.get("code", ""); rec["cases"] = n.get("cases", {})
        out.append(rec)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        recs = labels_for(f"{d}/sciel_events.jsonl")
        if not recs:
            print(f"{d}: no child_id edges (pre-Phase-0 run?)"); continue
        gains = [r["lineage_gain"] for r in recs]
        print(f"{d}: {len(recs)} programs, mean lineage_gain={sum(gains)/len(gains):.4f}, "
              f"max={max(gains):.4f}")
