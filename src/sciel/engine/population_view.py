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


def materialize(engine, refs: list, path: str) -> str:
    """Write the view for frozen references and return the sha256 of its bytes."""
    from .research_artifacts import saved_evaluation_by_hash
    programs = []
    for ref in refs:
        cases = output = None
        if ref['evaluation_id'] is not None:
            try:
                saved = saved_evaluation_by_hash(engine, ref['source_sha256'], ref['evaluation_id'])
            except ValueError:
                saved = {'status': 'unavailable'}
            if saved['status'] == 'available':
                result = saved['record']['result']
                cases = result.get('cases')
                witness = (result.get('artifacts') or {}).get('full_output')
                if witness is not None:
                    output = json.loads(witness).get('output')
        programs.append({'program_id': ref['program_id'], 'score': ref['score'], 'parent': ref['parent'],
                         'cases': cases, 'output': output})
    data = json.dumps({'schema': SCHEMA, 'programs': programs}, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()
    with open(path, 'wb') as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()
