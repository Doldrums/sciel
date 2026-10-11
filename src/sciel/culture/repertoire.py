"""Repertoire passed between generations: measured claims, surviving strategies and the learned search models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..swm import PopulationLearner, SearchWorldModel
from ..theory import Claim, Procedure, Strategy, Theory

SCHEMA = "sciel-repertoire/1"
FILE = "repertoire.json"


def export(engine) -> Path:
    inherited = engine.heritage.generation if engine.heritage is not None else 0
    value = {"schema": SCHEMA, "generation": inherited + 1, "run_id": engine.log.run_id,
             "task": engine.task.qualified_name, "task_hash": engine.task.task_hash,
             "claims": [c.__dict__ for c in engine.theory.claims if c.decided],
             "strategies": [{"text": s.text, "status": s.status} for s in engine.strategies.pool
                            if s.status != "retired"],
             "procedures": [p.params() for p in engine.procedures.pool
                            if p.status == "culture" and p.params() != Procedure(id="p0").params()],
             "swm": engine.swm.state(), "learner": engine.learner.state()}
    path = Path(engine.run_dir) / FILE
    path.write_text(json.dumps(value, sort_keys=True))
    return path


class Heritage:
    """An inherited repertoire; it acts only through randomized arms that the recipient's own ledger judges."""

    def __init__(self, path: str, db):
        raw = Path(path).read_bytes()
        value = json.loads(raw)
        if value.get("schema") != SCHEMA:
            raise ValueError("inherit must name a sciel-repertoire/1 file")
        self.path, self.sha256 = str(path), hashlib.sha256(raw).hexdigest()
        self.generation = int(value["generation"])
        self.source = {"run_id": value["run_id"], "task": value["task"], "task_hash": value["task_hash"]}
        self.theory = Theory()
        self.theory.claims = [Claim(**c) for c in value["claims"]]
        self.strategies = [Strategy(id=f"h{i + 1}", text=s["text"], born=0) for i, s in enumerate(value["strategies"])]
        self.procedures = [Procedure(id=f"h{i + 1}", **p) for i, p in enumerate(value["procedures"])]
        self.swm = SearchWorldModel(db, ())
        self.swm.load_state(value["swm"])
        self.learner = PopulationLearner()
        self.learner.load_state(value["learner"])

    @property
    def ready(self) -> bool:
        return self.swm.ready

    def operator(self, parent_id: str, context: str, distribution: dict) -> dict:
        planned = (self.learner.policy(self.swm, parent_id, context, distribution) if self.learner.ready
                   else self.swm.tilt(parent_id, context, "operator", distribution))
        return self.theory.tilt("operator", context, planned)

    def describe(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "generation": self.generation, **self.source,
                "claims": len(self.theory.claims), "strategies": len(self.strategies),
                "procedures": len(self.procedures),
                "swm": self.swm.ready, "learner": self.learner.ready}
