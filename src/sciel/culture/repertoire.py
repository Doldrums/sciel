"""Repertoire passed between generations: measured claims, surviving strategies and the learned search models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..swm import PopulationLearner, SearchWorldModel
from ..theory import Claim, Finding, Procedure, Strategy, Theory
from .methods import export_methods

SCHEMA = "sciel-repertoire/1"
FILE = "repertoire.json"


def export(engine) -> Path:
    inherited = engine.heritage.generation if engine.heritage is not None else 0
    value = {"schema": SCHEMA, "generation": inherited + 1, "run_id": engine.log.run_id,
             "task": engine.task.qualified_name, "task_hash": engine.task.task_hash,
             "claims": [c.__dict__ for c in engine.theory.claims if c.decided],
             "strategies": [{"text": s.text, "donor_status": s.status} for s in engine.strategies.pool
                            if s.status != "retired"],
             "knowledge": [{"text": f.text, "kind": f.kind, "donor_status": f.status} for f in engine.findings.pool
                           if f.status != "retired"],
             "procedures": [p.params() for p in engine.procedures.pool
                            if p.status == "culture" and p.params() != Procedure(id="p0").params()],
             "methods": export_methods(engine),
             "swm": engine.swm.state(), "learner": engine.learner.state()}
    path = Path(engine.run_dir) / FILE
    path.write_text(json.dumps(value, sort_keys=True))
    return path


class Heritage:
    """An inherited repertoire; it acts only through randomized arms that the recipient's own ledger judges."""

    def __init__(self, path: str, db, embedder=None):
        raw = Path(path).read_bytes()
        value = json.loads(raw)
        if value.get("schema") != SCHEMA:
            raise ValueError("inherit must name a sciel-repertoire/1 file")
        self.path, self.sha256 = str(path), hashlib.sha256(raw).hexdigest()
        self.generation = int(value["generation"])
        self.source = {"run_id": value["run_id"], "task": value["task"], "task_hash": value["task_hash"]}
        self.theory = Theory()
        self.theory.claims = [Claim(**c) for c in value["claims"]]
        self.strategies = [Strategy(id=f"h{i + 1}", text=s["text"], born=0, status=s.get("accept", "trial"))
                           for i, s in enumerate(value["strategies"])]
        self.knowledge = [Finding(id=f"h{i + 1}", text=k["text"], cell_id=None, island=0, receipt_id=None, born=0,
                                  kind=k["kind"], status=k.get("accept", "trial"))
                          for i, k in enumerate(value["knowledge"])]
        self.procedures = [Procedure(id=f"h{i + 1}", status=p.pop("accept", "trial"), **p)
                           for i, p in enumerate(dict(p) for p in value["procedures"])]
        self.methods = value["methods"]
        self.swm = SearchWorldModel(db, (), embedder)
        self.swm.load_state(value["swm"])
        self.swm.edges = {}
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
                "procedures": len(self.procedures), "knowledge": len(self.knowledge), "methods": len(self.methods),
                "swm": self.swm.ready, "learner": self.learner.ready}
