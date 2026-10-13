"""Repertoire passed between generations: measured claims, surviving strategies and the learned search models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..rl import StepPolicy
from ..swm import SearchWorldModel
from ..theory import Claim, Finding, Procedure, Strategy, Theory
from .methods import export_methods

SCHEMA = "sciel-repertoire/3"
FILE = "repertoire.json"


def export(engine) -> Path:
    engine.theory.update(engine.ledger)
    engine.trials.update(engine.siblings.pairs)
    engine.strategies.judge(engine.trials)
    engine.findings.judge(engine.trials)
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
             "swm": engine.swm.state(), "policy": engine.policy.state(),
             "evidence": evidence(engine)}
    path = Path(engine.run_dir) / FILE
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def evidence(engine) -> dict:
    """Betting wealth the next generation continues: claims by decision and action, items by text, and the
    theorist-versus-control programme outcomes."""
    texts = {s.id: s.text for s in engine.strategies.pool} | {f.id: f.text for f in engine.findings.pool}
    items = {texts[k]: v for k, v in engine.trials.wealth(engine.siblings.pairs).items() if k in texts}
    return {"claims": dict(engine.theory.wealth), "items": items,
            "programmes": engine.incubator.prior + engine.incubator.outcomes}


class Heritage:
    """An inherited repertoire; it acts only through randomized arms that the recipient's own ledger judges."""

    def __init__(self, path: str, db, embedder=None):
        raw = Path(path).read_bytes()
        value = json.loads(raw)
        if value.get("schema") != SCHEMA:
            raise ValueError("inherit must name a sciel-repertoire/3 file")
        self.path, self.sha256 = str(path), hashlib.sha256(raw).hexdigest()
        self.generation = int(value["generation"])
        self.source = {"run_id": value["run_id"], "task": value["task"], "task_hash": value["task_hash"]}
        self.theory = Theory()
        self.theory.claims = [Claim(**c) for c in value["claims"]]
        self.strategies = [Strategy(id=f"hs{i + 1}", text=s["text"], born=0, status=s.get("accept", "trial"))
                           for i, s in enumerate(value["strategies"])]
        self.knowledge = [Finding(id=f"hf{i + 1}", text=k["text"], cell_id=None, island=0, receipt_id=None, born=0,
                                  kind=k["kind"], status=k.get("accept", "trial"))
                          for i, k in enumerate(value["knowledge"])]
        self.procedures = [Procedure(id=f"hp{i + 1}", status=p.pop("accept", "trial"), **p)
                           for i, p in enumerate(dict(p) for p in value["procedures"])]
        self.methods = value["methods"]
        self.evidence = value["evidence"]
        self.swm = SearchWorldModel(db, (), embedder)
        self.swm.load_state(value["swm"])
        self.swm.edges = {}
        self.policy = StepPolicy()
        self.policy.load_state(value["policy"])
        self.policy.world = self.swm

    @property
    def ready(self) -> bool:
        return self.policy.ready

    def describe(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "generation": self.generation, **self.source,
                "claims": len(self.theory.claims), "strategies": len(self.strategies),
                "procedures": len(self.procedures), "knowledge": len(self.knowledge), "methods": len(self.methods),
                "swm": self.swm.ready, "policy": self.policy.ready}
