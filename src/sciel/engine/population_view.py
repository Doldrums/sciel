"""Read-only view of the population that a candidate gets while it is evaluated.

A step freezes references to the programs it was built from: the island's best, the parent and the run's best.
At evaluation the references become a file of their saved measurements, with the exact object the evaluator
scored for each program, so a child can continue a construction instead of rebuilding it from its code. The saved
records are write-once and content-addressed, so a replayed evaluation sees the same bytes.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = 'sciel-population-view/1'
SIZE = 8
ENV = 'SCIEL_POPULATION'


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def freeze(engine, island: int, parent_id: str) -> list:
    """References, best first; values are read only when the candidate is evaluated."""
    from .research_artifacts import evaluation_id_for_program
    db = engine.db
    programs = list(db.islands[island].top_k(SIZE))
    for pid in (parent_id, db.pop_best_id):
        if pid is not None and pid in db._all and all(p.id != pid for p in programs):
            programs.append(db._all[pid])
    out = []
    for p in sorted(programs, key=lambda p: -p.score):
        try:
            evaluation_id = evaluation_id_for_program(engine, p)
        except ValueError:
            evaluation_id = None
        out.append({'program_id': p.id, 'score': p.score, 'parent': p.id == parent_id,
                    'source_sha256': _sha(p.code), 'evaluation_id': evaluation_id})
    return out


def _shape(value, depth=0) -> str:
    if isinstance(value, dict) and depth < 2:
        return "a dict with keys " + ", ".join(f"{k} ({_shape(v, depth + 1)})" for k, v in value.items())
    if isinstance(value, list):
        kinds = {type(x).__name__ for x in value[:50]}
        item = "numbers" if kinds <= {"int", "float"} else "lists" if kinds == {"list"} else "items"
        return f"a list of {len(value)} {item}"
    return type(value).__name__


def describe(engine, refs: list) -> str:
    """The structure of a program's output in the view, so a child parses it without guessing; the parent's first."""
    for ref in sorted(refs, key=lambda r: not r['parent']):
        output = _output(engine, ref)[1]
        if output is not None:
            return (f"\nPopulation view: each program's output is {_shape(output)}, as the evaluator recorded it "
                    f"(shown for {ref['program_id']}).\n")
    return ""


def _output(engine, ref) -> tuple:
    from .research_artifacts import saved_evaluation_by_hash
    if ref['evaluation_id'] is None:
        return None, None
    try:
        saved = saved_evaluation_by_hash(engine, ref['source_sha256'], ref['evaluation_id'])
    except ValueError:
        return None, None
    if saved['status'] != 'available':
        return None, None
    result = saved['record']['result']
    witness = (result.get('artifacts') or {}).get('full_output')
    return result.get('cases'), None if witness is None else json.loads(witness).get('output')


def materialize(engine, refs: list, path: str) -> str:
    """Write the view for frozen references and return the sha256 of its bytes."""
    programs = []
    for ref in refs:
        cases, output = _output(engine, ref)
        programs.append({'program_id': ref['program_id'], 'score': ref['score'], 'parent': ref['parent'],
                         'cases': cases, 'output': output})
    data = json.dumps({'schema': SCHEMA, 'programs': programs}, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()
    with open(path, 'wb') as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()
