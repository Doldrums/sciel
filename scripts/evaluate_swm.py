"""Held-out quality of the search world model against the cell-mean outcome model it replaces in the DR estimates.

For each run, the world model is fitted on the ledger rows of half of the islands and scored on the other half:
log loss and AUC of the lineage-event prediction for the action actually taken.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sciel.causal.estimators import _outcome_model, _q
from sciel.causal.ledger import Ledger
from sciel.engine.db import Database
from sciel.swm import SearchWorldModel
from sciel.theory.claims import DECISIONS


def usable(row) -> bool:
    return row.outcome is not None and bool(row.state) and row.decision in DECISIONS and len(row.distribution) > 1


def log_loss(p: list, y: list) -> float:
    return -sum(math.log(min(max(q if t else 1 - q, 1e-6), 1)) for q, t in zip(p, y)) / len(y)


def auc(p: list, y: list) -> float | None:
    pos = [q for q, t in zip(p, y) if t]
    neg = [q for q, t in zip(p, y) if not t]
    if not pos or not neg:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def evaluate(run: Path, seed: int) -> dict:
    saved = json.loads((run / "checkpoint.json").read_text())
    db = Database(len(saved["db"]["islands"]))
    db.load_state(saved["db"])
    ledger = Ledger()
    ledger.load_state(saved["ledger"])
    rows = [r for r in ledger.rows if usable(r)]
    train, test = Ledger(), [r for r in rows if r.island % 2 == 1]
    train.rows = [r for r in ledger.rows if r.island % 2 == 0]
    world = SearchWorldModel(db, DECISIONS)
    active = world.fit(train, 0, seed)
    if not active or not test:
        return {"run": run.name, "active": False, "gate": world.gate}
    baseline = _outcome_model([r for r in train.rows if usable(r)])
    y = [r.outcome for r in test]
    learned = [world.q(r, r.choice) for r in test]
    kept = [i for i, v in enumerate(learned) if v is not None]
    cells = [_q(baseline, test[i].context, test[i].choice) for i in kept]
    model = [learned[i] for i in kept]
    truth = [y[i] for i in kept]
    return {"run": run.name, "active": True, "gate": world.gate, "train_rows": len(train.rows),
            "test_rows": len(kept), "base_rate": sum(truth) / len(truth),
            "swm": {"log_loss": log_loss(model, truth), "auc": auc(model, truth)},
            "cell_mean": {"log_loss": log_loss(cells, truth), "auc": auc(cells, truth)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run directories with checkpoint.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps([evaluate(Path(run), args.seed) for run in args.runs], indent=2))


if __name__ == "__main__":
    main()
