"""Search world model fitted online to the decision ledger with inverse-propensity weights."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch

from ..components.policy.code_embed import featurize
from . import model

MIN_ROWS = 60
FIT_INTERVAL = 20
MAX_WEIGHT = 20.0
BETA = 8.0
GAMMA = 0.9
BONUS = 1.0
UNCERTAIN_QUANTILE = 0.75


def _context(context: str) -> list:
    fields = dict(part.split("=") for part in context.split("|"))
    return [float(fields.get("phase", 0)), float(fields.get("third", 0)) / 2]


class SearchWorldModel:
    def __init__(self, db, decisions: tuple):
        self.db = db
        self.decisions = decisions
        self.vocab: list[str] = []
        self.mean = self.std = None
        self.net = None
        self.trained_at = None
        self.uncertain_above = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def ready(self) -> bool:
        return self.net is not None

    def _program(self, pid: str) -> np.ndarray | None:
        if pid not in self._cache:
            program = self.db._all.get(pid)
            if program is None:
                return None
            self._cache[pid] = featurize(program.code, program.cases).astype(np.float32)
        return self._cache[pid]

    def _x(self, pid: str, context: str) -> np.ndarray | None:
        state = self._program(pid)
        return None if state is None else np.concatenate([state, _context(context)]).astype(np.float32)

    def _rows(self, ledger) -> list:
        return [r for r in ledger.rows if r.outcome is not None and r.state is not None
                and r.decision in self.decisions and len(r.distribution) > 1
                and self._program(r.state) is not None]

    def due(self, iteration: int) -> bool:
        return self.trained_at is None or iteration - self.trained_at >= FIT_INTERVAL

    def fit(self, ledger, iteration: int, seed: int) -> bool:
        rows = self._rows(ledger)
        self.trained_at = iteration
        if len(rows) < MIN_ROWS:
            return False
        self.vocab = sorted({f"{r.decision}:{a}" for r in rows for a in r.distribution})
        index = {k: i for i, k in enumerate(self.vocab)}
        x = np.stack([self._x(r.state, r.context) for r in rows])
        self.mean, self.std = x.mean(0), x.std(0) + 1e-3
        children = [self._program(r.child) if r.child else None for r in rows]
        width = x.shape[1] - 2
        boot = np.random.default_rng(seed).poisson(1.0, size=(len(rows), model.HEADS)).astype(np.float32)
        data = {"x": (x - self.mean) / self.std,
                "a": np.array([index[f"{r.decision}:{r.choice}"] for r in rows]),
                "y": np.array([r.outcome for r in rows], dtype=np.float32),
                "w": np.array([min(1.0 / r.propensity, MAX_WEIGHT) for r in rows], dtype=np.float32),
                "valid": np.array([r.child is not None for r in rows], dtype=np.float32),
                "boot": boot,
                "child": np.stack([((np.concatenate([c, [0.0, 0.0]]) - self.mean) / self.std
                                    if c is not None else np.zeros(width + 2)) for c in children]
                                  ).astype(np.float32),
                "has_child": np.array([c is not None for c in children])}
        self.net = model.fit(data, len(self.vocab), seed)
        spread = model.predict(self.net, data["x"], data["a"])["epistemic"]
        self.uncertain_above = float(np.quantile(spread, UNCERTAIN_QUANTILE))
        return True

    def predict(self, pid: str, context: str, decision: str, actions: list) -> dict | None:
        x = self._x(pid, context)
        keys = [f"{decision}:{a}" for a in actions]
        if not self.ready or x is None or any(k not in self.vocab for k in keys):
            return None
        index = {k: i for i, k in enumerate(self.vocab)}
        xs = np.repeat(((x - self.mean) / self.std)[None].astype(np.float32), len(actions), axis=0)
        out = model.predict(self.net, xs, np.array([index[k] for k in keys]))
        return {a: {name: float(out[name][i]) for name in out} for i, a in enumerate(actions)}

    def tilt(self, pid: str, context: str, decision: str, distribution: dict) -> dict:
        x = self._x(pid, context)
        keys = [f"{decision}:{a}" for a in distribution]
        if not self.ready or x is None or any(k not in self.vocab for k in keys):
            return dict(distribution)
        index = {k: i for i, k in enumerate(self.vocab)}
        values = model.lookahead(self.net, ((x - self.mean) / self.std).astype(np.float32),
                                 np.array([index[k] for k in keys]), GAMMA, BONUS)
        shifted = np.array(list(distribution.values())) * np.exp(BETA * (values - values.mean()))
        return {a: float(p) for a, p in zip(distribution, shifted / shifted.sum())}

    def value(self, pid: str, context: str, decision: str, actions: list) -> float | None:
        out = self.predict(pid, context, decision, actions)
        return None if out is None else float(np.mean([v["value"] for v in out.values()]))

    def uncertain(self, pid: str, context: str, decision: str, actions: list) -> bool:
        out = self.predict(pid, context, decision, actions)
        return out is not None and np.mean([v["epistemic"] for v in out.values()]) > self.uncertain_above

    def q(self, row, action: str) -> float | None:
        out = self.predict(row.state, row.context, row.decision, [action]) if row.state else None
        return None if out is None else out[action]["value"]

    def state(self) -> dict:
        weights = None
        if self.net is not None:
            buffer = io.BytesIO()
            torch.save(self.net.state_dict(), buffer)
            weights = base64.b64encode(buffer.getvalue()).decode()
        return {"vocab": self.vocab, "trained_at": self.trained_at, "weights": weights,
                "uncertain_above": self.uncertain_above,
                "mean": None if self.mean is None else self.mean.tolist(),
                "std": None if self.std is None else self.std.tolist()}

    def load_state(self, data: dict) -> None:
        self.vocab, self.trained_at = list(data["vocab"]), data["trained_at"]
        self.uncertain_above = data["uncertain_above"]
        self._cache = {}
        self.net = None
        if data["weights"] is not None:
            self.mean = np.asarray(data["mean"], dtype=np.float32)
            self.std = np.asarray(data["std"], dtype=np.float32)
            net = model.Network(len(self.mean), len(self.vocab))
            net.load_state_dict(torch.load(io.BytesIO(base64.b64decode(data["weights"]))))
            self.net = net.eval()
