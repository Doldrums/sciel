"""Search world model fitted online to the decision ledger with inverse-propensity weights."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch

from types import SimpleNamespace

from ..causal.estimators import _outcome_model, _q
from ..components.policy.code_embed import featurize
from . import model

MIN_ROWS = 60
MAX_TRAIN = 3000
FIT_INTERVAL = 20
MAX_WEIGHT = 3.0
MIN_HELD = 50
SHRINKS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
BETA = 8.0
GAMMA = 0.9
BONUS = 1.0
UNCERTAIN_QUANTILE = 0.75


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _log_loss(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _context(context: str) -> list:
    fields = dict(part.split("=") for part in context.split("|"))
    return [float(fields.get("phase", 0)), float(fields.get("third", 0)) / 2]


class SearchWorldModel:
    def __init__(self, db, decisions: tuple):
        self.db = db
        self.decisions = decisions
        self.vocab: list[str] = []
        self.edges: dict[str, list] = {}
        self.mean = self.std = None
        self.net = None
        self.trained_at = None
        self.uncertain_above = None
        self.gate = None
        self.shrink = 0.0
        self.cells = {}
        self._cache: dict[str, np.ndarray] = {}
        self._latents: dict = {}

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

    def _norm(self, pid: str, context: str) -> np.ndarray:
        return ((self._x(pid, context) - self.mean) / self.std).astype(np.float32)

    def _rows(self, ledger) -> list:
        return [r for r in ledger.rows if r.outcome is not None and r.state is not None
                and r.decision in self.decisions and len(r.distribution) > 1
                and self._program(r.state) is not None]

    def _population(self, pid: str) -> list:
        island = self.db._all[pid].island
        return [p.id for p in self.db.islands[island].top_k(model.POPULATION)] if island < len(self.db.islands) else []

    def _lineage(self, pid: str) -> list:
        chain, node = [], self.db._all.get(pid)
        while node is not None and node.parent_id in self.db._all and len(chain) < model.LINEAGE:
            chain.append((node.parent_id, node.id))
            node = self.db._all[node.parent_id]
        return chain[::-1]

    def _batch(self, states: list, contexts: list, populations: list, actions: list) -> dict:
        n, width = len(states), len(self.mean)
        pop = np.zeros((n, model.POPULATION, width), np.float32)
        pop_mask = np.ones((n, model.POPULATION), bool)
        lin = np.zeros((n, model.LINEAGE, width), np.float32)
        lin_a = np.full((n, model.LINEAGE), len(self.vocab))
        lin_r = np.zeros((n, model.LINEAGE), np.float32)
        lin_mask = np.ones((n, model.LINEAGE), bool)
        for i, (pid, context, members) in enumerate(zip(states, contexts, populations)):
            members = [m for m in members if self._program(m) is not None][:model.POPULATION]
            for j, member in enumerate(members):
                pop[i, j], pop_mask[i, j] = self._norm(member, context), False
            for j, (parent, child) in enumerate(self._lineage(pid)):
                action, reward = self.edges.get(child, (len(self.vocab), 0.0))
                lin[i, j], lin_a[i, j], lin_r[i, j], lin_mask[i, j] = self._norm(parent, context), action, reward, False
        return {"x": np.stack([self._norm(p, c) for p, c in zip(states, contexts)]),
                "a": np.asarray(actions), "pop": pop, "pop_mask": pop_mask,
                "lin_x": lin, "lin_a": lin_a, "lin_r": lin_r, "lin_mask": lin_mask}

    @staticmethod
    def _weights(rows) -> np.ndarray:
        """Stabilized inverse-propensity weights: normalized to mean one and capped, since the action is an input."""
        raw = np.array([1.0 / r.propensity for r in rows], dtype=np.float32)
        return np.minimum(raw / raw.mean(), MAX_WEIGHT).astype(np.float32)

    def due(self, iteration: int) -> bool:
        return self.trained_at is None or iteration - self.trained_at >= FIT_INTERVAL

    def fit(self, ledger, iteration: int, seed: int) -> bool:
        rows = self._rows(ledger)[-MAX_TRAIN:]
        self.trained_at = iteration
        if len(rows) < MIN_ROWS:
            self.net = None
            return False
        self.vocab = sorted({f"{r.decision}:{a}" for r in rows for a in r.distribution})
        index = {k: i for i, k in enumerate(self.vocab)}
        self.edges = {r.child: (index[f"operator:{r.choice}"], r.reward) for r in ledger.rows
                      if r.decision == "operator" and r.child and f"operator:{r.choice}" in index}
        x = np.stack([self._x(r.state, r.context) for r in rows])
        self.mean, self.std = x.mean(0), x.std(0) + 1e-3
        children = [r.child if r.child and self._program(r.child) is not None else None for r in rows]
        data = self._batch([r.state for r in rows], [r.context for r in rows],
                           [r.population or self._population(r.state) for r in rows],
                           [index[f"{r.decision}:{r.choice}"] for r in rows])
        data.update(
            y=np.array([r.outcome for r in rows], dtype=np.float32),
            w=self._weights(rows),
            valid=np.array([r.child is not None for r in rows], dtype=np.float32),
            boot=np.random.default_rng(seed).poisson(1.0, size=(len(rows), model.HEADS)).astype(np.float32),
            child=np.stack([self._norm(c, r.context) if c else np.zeros(x.shape[1], np.float32)
                            for c, r in zip(children, rows)]),
            has_child=np.array([c is not None for c in children]))
        islands = sorted({r.island for r in rows})
        out = islands[(iteration // FIT_INTERVAL) % len(islands)]
        held = np.array([i for i, r in enumerate(rows) if r.island == out])
        kept = np.array([i for i, r in enumerate(rows) if r.island != out])
        if len(islands) < 2 or len(kept) < MIN_ROWS // 2 or len(held) < MIN_HELD:
            self.net = None
            return False
        net, _ = model.fit({k: v[kept] for k, v in data.items()}, {k: v[held] for k, v in data.items()},
                           len(self.vocab), seed)
        self.cells = _outcome_model([SimpleNamespace(context=rows[i].context, choice=f"{rows[i].decision}:{rows[i].choice}",
                                                     outcome=rows[i].outcome) for i in kept])
        y = data["y"][held]
        learned = model.predict(net, {k: v[held] for k, v in data.items()})["value"]
        cell = np.array([self._cell(rows[i].context, f"{rows[i].decision}:{rows[i].choice}") for i in held])
        losses = {a: _log_loss(1 / (1 + np.exp(-(a * _logit(learned) + (1 - a) * _logit(cell)))), y) for a in SHRINKS}
        self.shrink = min(SHRINKS, key=lambda a: (losses[a], a))
        self.gate = {"held_out": len(held), "shrink": self.shrink, "swm_log_loss": losses[1.0],
                     "cell_mean_log_loss": losses[0.0], "blend_log_loss": losses[self.shrink]}
        self.net = net if self.shrink > 0 else None
        self._latents = {}
        if self.net is None:
            return False
        spread = model.predict(self.net, data)["epistemic"]
        self.uncertain_above = float(np.quantile(spread, UNCERTAIN_QUANTILE))
        return True

    def _cell(self, context: str, key: str) -> float:
        return _q(self.cells, context, key)

    def _query(self, pid: str, context: str, decision: str, actions: list) -> dict | None:
        keys = [f"{decision}:{a}" for a in actions]
        if not self.ready or self._program(pid) is None or any(k not in self.vocab for k in keys):
            return None
        index = {k: i for i, k in enumerate(self.vocab)}
        n = len(actions)
        return self._batch([pid] * n, [context] * n, [self._population(pid)] * n, [index[k] for k in keys])

    def predict(self, pid: str, context: str, decision: str, actions: list) -> dict | None:
        batch = self._query(pid, context, decision, actions)
        if batch is None:
            return None
        out = model.predict(self.net, batch)
        cell = np.array([self._cell(context, f"{decision}:{a}") for a in actions])
        out["value"] = 1 / (1 + np.exp(-(self.shrink * _logit(out["value"]) + (1 - self.shrink) * _logit(cell))))
        return {a: {name: float(out[name][i]) for name in out} for i, a in enumerate(actions)}

    def encode(self, features: np.ndarray) -> np.ndarray:
        x = np.concatenate([features, [0.0, 0.0]]).astype(np.float32)
        with torch.no_grad():
            return self.net.encoder(torch.tensor(((x - self.mean) / self.std)[None].astype(np.float32)))[0].numpy()

    def latent(self, pid: str) -> np.ndarray:
        key = (self.trained_at, pid)
        if key not in self._latents:
            self._latents[key] = self.encode(self._program(pid))
        return self._latents[key]

    def tilt(self, pid: str, context: str, decision: str, distribution: dict) -> dict:
        batch = self._query(pid, context, decision, list(distribution))
        if batch is None:
            return dict(distribution)
        values = model.lookahead(self.net, {k: v[:1] for k, v in batch.items()}, batch["a"], GAMMA, BONUS)
        shifted = np.array(list(distribution.values())) * np.exp(BETA * self.shrink * (values - values.mean()))
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
                "uncertain_above": self.uncertain_above, "edges": self.edges, "shrink": self.shrink,
                "cells": [[c, a, v] for (c, a), v in self.cells.items()],
                "mean": None if self.mean is None else self.mean.tolist(),
                "std": None if self.std is None else self.std.tolist()}

    def load_state(self, data: dict) -> None:
        self.vocab, self.trained_at = list(data["vocab"]), data["trained_at"]
        self.uncertain_above = data["uncertain_above"]
        self.shrink = float(data["shrink"])
        self.cells = {(c, a): v for c, a, v in data["cells"]}
        self.edges = {k: tuple(v) for k, v in data["edges"].items()}
        self._cache, self._latents = {}, {}
        self.net = None
        if data["weights"] is not None:
            self.mean = np.asarray(data["mean"], dtype=np.float32)
            self.std = np.asarray(data["std"], dtype=np.float32)
            net = model.Network(len(self.mean), len(self.vocab))
            net.load_state_dict(torch.load(io.BytesIO(base64.b64decode(data["weights"]))))
            self.net = net.eval()
