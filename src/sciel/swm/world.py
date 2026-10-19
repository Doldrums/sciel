"""Search world model fitted online to the decision ledger with inverse-propensity weights: it forecasts whether a
step's child raises its lineage's best, which is known when the step is scored and carries signal that the island
event of a whole horizon does not."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch

from types import SimpleNamespace

from ..causal.estimators import _outcome_model, _q
from ..components.policy.code_embed import featurize
from . import model
from .embedding import DIM

MIN_ROWS = 60
MAX_TRAIN = 50000
FIT_INTERVAL = 20
MAX_WEIGHT = 3.0
SHRINKS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
BETA = 8.0
GAMMA = 0.9
BONUS = 1.0
UNCERTAIN_QUANTILE = 0.75
# A run holds a few hundred steps, too few for the 256 token hashes and the frozen embedding as they are: both enter
# through their leading principal components over the run's programs, the readouts stay as they are.
COMPONENTS = 16
DENSE = slice(8, 264)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _log_loss(p, y, w=None) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.average(y * np.log(p) + (1 - y) * np.log(1 - p), weights=w))


def _step(r) -> tuple:
    return r.iteration, r.island, r.state, r.child


def _context(context: str) -> list:
    fields = dict(part.split("=") for part in context.split("|"))
    return [float(fields.get("phase", 0)), float(fields.get("third", 0)) / 2]


class SearchWorldModel:
    def __init__(self, db, decisions: tuple, embedder=None):
        self.db = db
        self.embedder = embedder
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
        self.center = self.basis = None
        self._cache: dict[str, np.ndarray] = {}
        self._latents: dict = {}

    @property
    def ready(self) -> bool:
        return self.net is not None

    def _program(self, pid: str) -> np.ndarray | None:
        if self.basis is None:
            return None
        if pid not in self._cache:
            program = self.db._all.get(pid)
            if program is None:
                return None
            if self.embedder is not None and self.embedder.get(program.code) is None:
                return None
            self._cache[pid] = self.features(program.code, program.cases)
        return self._cache[pid]

    def prefetch(self) -> None:
        if self.embedder is not None:
            self.embedder.prefetch(p.code for p in self.db._all.values())

    def _dense(self, code: str, base: np.ndarray) -> np.ndarray | None:
        """Token hashes and the frozen code embedding, the high-dimensional part of a program's features."""
        if self.embedder is None:
            return base[DENSE]
        vector = self.embedder.get(code)
        return None if vector is None else np.concatenate([base[DENSE], vector])

    def _project(self) -> bool:
        dense = [self._dense(p.code, featurize(p.code, p.cases).astype(np.float32)) for p in self.db._all.values()]
        dense = [d for d in dense if d is not None]
        if len(dense) < COMPONENTS:
            return False
        dense = np.stack(dense)
        self.center = dense.mean(0)
        self.basis = np.linalg.svd(dense - self.center, full_matrices=False)[2][:COMPONENTS].T.astype(np.float32)
        self._cache = {}
        return True

    def features(self, code: str, cases: dict) -> np.ndarray:
        """Behavior and AST readouts plus the principal components of token hashes and the frozen code embedding;
        until the next batched prefetch the embedding is imputed with its mean, which the components map to zero."""
        base = featurize(code, cases).astype(np.float32)
        dense = self._dense(code, base)
        if dense is None:
            dense = np.concatenate([base[DENSE], self.center[DENSE.stop - DENSE.start:]])
        readouts = np.concatenate([base[:DENSE.start], base[DENSE.stop:]])
        return np.concatenate([readouts, (dense - self.center) @ self.basis]).astype(np.float32)

    def _x(self, pid: str, context: str) -> np.ndarray | None:
        state = self._program(pid)
        return None if state is None else np.concatenate([state, _context(context)]).astype(np.float32)

    def _norm(self, pid: str, context: str) -> np.ndarray:
        return ((self._x(pid, context) - self.mean) / self.std).astype(np.float32)

    def _rows(self, ledger) -> list:
        """Rows of scored steps from a recorded parent; island-scope rows have no child of their own to judge."""
        return [r for r in ledger.rows if r.scope == "lineage" and r.state is not None
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

    def _batch(self, states: list, contexts: list, populations: list, actions: list, children=None, prior=None) -> dict:
        """Rows reference a shared table of context-free program features; index len(table) is padding."""
        ids: dict[str, int] = {}

        def slot(pid):
            if pid not in ids:
                ids[pid] = len(ids)
            return ids[pid]

        n = len(states)
        pop = np.full((n, model.POPULATION), -1)
        lin = np.full((n, model.LINEAGE), -1)
        lin_a = np.full((n, model.LINEAGE), len(self.vocab))
        lin_r = np.zeros((n, model.LINEAGE), np.float32)
        for i, (pid, members) in enumerate(zip(states, populations)):
            members = [m for m in members if self._program(m) is not None][:model.POPULATION]
            pop[i, :len(members)] = [slot(m) for m in members]
            for j, (parent, child) in enumerate(self._lineage(pid)):
                lin[i, j] = slot(parent)
                lin_a[i, j], lin_r[i, j] = self.edges.get(child, (len(self.vocab), 0.0))
        child = np.array([slot(c) if c else -1 for c in children]) if children is not None else None
        table = np.zeros((len(ids) + 1, len(self.mean)), np.float32)
        for pid, k in ids.items():
            table[k] = self._norm(pid, "phase=0|third=0")
        pad = len(ids)
        batch = {"x": np.stack([self._norm(p, c) for p, c in zip(states, contexts)]), "a": np.asarray(actions),
                 "table": table, "pop": np.where(pop < 0, pad, pop), "pop_mask": pop < 0,
                 "lin": np.where(lin < 0, pad, lin), "lin_a": lin_a, "lin_r": lin_r, "lin_mask": lin < 0,
                 "prior": np.zeros(n, np.float32) if prior is None else _logit(np.asarray(prior)).astype(np.float32)}
        if child is not None:
            batch["child"] = np.where(child < 0, pad, child)
        return batch

    @staticmethod
    def _shares(rows) -> np.ndarray:
        """Each step counts once: its decision rows share one outcome, so each row carries a share of the step."""
        size: dict = {}
        for r in rows:
            size[_step(r)] = size.get(_step(r), 0) + 1
        return np.array([1.0 / size[_step(r)] for r in rows], dtype=np.float32)

    @staticmethod
    def _weights(rows, shares) -> np.ndarray:
        """Stabilized inverse-propensity weights: normalized to mean one and capped, since the action is an input."""
        raw = np.array([1.0 / r.propensity for r in rows], dtype=np.float32)
        return (np.minimum(raw / raw.mean(), MAX_WEIGHT) * shares / shares.mean()).astype(np.float32)

    def due(self, iteration: int) -> bool:
        return self.trained_at is None or iteration - self.trained_at >= FIT_INTERVAL

    def fit(self, ledger, iteration: int, seed: int) -> bool:
        self.prefetch()
        self.trained_at = iteration
        if not self._project():
            self.net = None
            return False
        rows = self._rows(ledger)[-MAX_TRAIN:]
        if len(rows) < MIN_ROWS:
            self.net = None
            return False
        self.vocab = sorted({f"{r.decision}:{a}" for r in rows for a in r.distribution})
        index = {k: i for i, k in enumerate(self.vocab)}
        self.edges = {r.child: (index[f"operator:{r.choice}"], r.reward) for r in ledger.rows
                      if r.decision == "operator" and r.child and f"operator:{r.choice}" in index}
        # Islands grow separate lineages and populations, so the gate holds out whole islands: two folds by island
        # parity, each scored by the model fitted on the other, as the step policy's critics are.
        parity = np.array([r.island % 2 == 1 for r in rows])
        if min(parity.sum(), (~parity).sum()) < MIN_ROWS // 2:
            self.net = None
            return False
        x = np.stack([self._x(r.state, r.context) for r in rows])
        self.mean, self.std = x.mean(0), x.std(0) + 1e-3
        children = [r.child if r.child and self._program(r.child) is not None else None for r in rows]
        data = self._batch([r.state for r in rows], [r.context for r in rows],
                           [r.population or self._population(r.state) for r in rows],
                           [index[f"{r.decision}:{r.choice}"] for r in rows], children)
        y = np.array([r.gain > 0 for r in rows], dtype=np.float32)
        shares = self._shares(rows)
        data.update(
            y=y,
            reward=np.array([r.reward for r in rows], dtype=np.float32),
            w=self._weights(rows, shares),
            valid=np.array([r.child is not None for r in rows], dtype=np.float32),
            boot=np.random.default_rng(seed).poisson(1.0, size=(len(rows), model.HEADS)).astype(np.float32),
            has_child=np.array([c is not None for c in children]))
        rng = np.random.default_rng(seed)
        steps = np.array([hash(_step(r)) for r in rows])
        keys = [f"{r.decision}:{r.choice}" for r in rows]
        islands = np.array([r.island for r in rows])

        def train(indices, offset):
            """Early stopping holds out whole islands where the fold has several, else whole steps: the gate scores
            other islands, and a step's rows share their outcome and nearly all inputs."""
            groups = islands if len(np.unique(islands[indices])) > 1 else steps
            order = rng.permutation(np.unique(groups[indices]))
            share = np.array([np.isin(groups[indices], order[:k + 1]).mean() for k in range(len(order))])
            held = np.isin(groups[indices], order[:int(np.argmin(np.abs(share - 0.2))) + 1])
            return model.fit(model.rows(data, indices[~held]), model.rows(data, indices[held]), len(self.vocab),
                             seed + offset)[0]

        def cells(indices):
            return _outcome_model([SimpleNamespace(context=rows[i].context, choice=f"{rows[i].decision}:{rows[i].choice}",
                                                   outcome=float(y[i])) for i in indices])

        def base_rate(indices):
            return {("*", "*"): float(np.average(y[indices], weights=shares[indices]))}

        # The reference the network must beat is the better of the base rate and the cell means, both out of fold:
        # cell means alone can lose to the base rate, and a gate against them would pass a model no better than it.
        def forecast(table, indices):
            return np.array([_q(table, rows[i].context, keys[i]) for i in indices])

        folds = [(np.flatnonzero(parity == side), np.flatnonzero(parity != side)) for side in (False, True)]
        cell, base = np.zeros(len(rows)), np.zeros(len(rows))
        for fitted, scored in folds:
            cell[scored] = forecast(cells(fitted), scored)
            base[scored] = forecast(base_rate(fitted), scored)
        cell_loss, base_loss = _log_loss(cell, y, shares), _log_loss(base, y, shares)
        reference, by_cells = (cell, True) if cell_loss < base_loss else (base, False)
        prior_of = cells if by_cells else base_rate
        # The network corrects the reference: on each fold it starts from the reference fitted on that fold.
        learned = np.zeros(len(rows))
        for side, (fitted, scored) in enumerate(folds):
            data["prior"] = _logit(forecast(prior_of(fitted), range(len(rows)))).astype(np.float32)
            learned[scored] = model.predict(train(fitted, side), model.rows(data, scored))["value"]
        losses = {a: _log_loss(1 / (1 + np.exp(-(a * _logit(learned) + (1 - a) * _logit(reference)))), y, shares)
                  for a in SHRINKS}
        self.shrink = min(SHRINKS, key=lambda a: (losses[a], a))
        self.gate = {"held_out": len(rows), "steps": int(round(shares.sum())), "base_rate": float(np.average(y, weights=shares)),
                     "shrink": self.shrink, "reference": "cells" if by_cells else "base_rate",
                     "swm_log_loss": losses[1.0], "cell_mean_log_loss": cell_loss, "base_rate_log_loss": base_loss,
                     "blend_log_loss": losses[self.shrink]}
        self._latents = {}
        if self.shrink == 0:
            self.net = None
            return False
        # The gate judged the procedure out of fold; the acting model is that procedure fitted on every island.
        everything = np.arange(len(rows))
        self.cells = prior_of(everything)
        data["prior"] = _logit(forecast(self.cells, everything)).astype(np.float32)
        self.net = train(everything, 2)
        spread = model.predict(self.net, model.rows(data, everything))["epistemic"]
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
        return self._batch([pid] * n, [context] * n, [self._population(pid)] * n, [index[k] for k in keys],
                           prior=[self._cell(context, k) for k in keys])

    def state_values(self, states: list, contexts: list) -> np.ndarray:
        """TD value of programs under the search, from the world model's critic head."""
        return model.state_values(self.net, self._batch(states, contexts, [self._population(p) for p in states],
                                                        [0] * len(states)))

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

    def latent(self, pid: str) -> np.ndarray | None:
        key = (self.trained_at, pid)
        if key not in self._latents:
            features = self._program(pid)
            if features is None:
                return None
            self._latents[key] = self.encode(features)
        return self._latents[key]

    def tilt(self, pid: str, context: str, decision: str, distribution: dict) -> dict:
        batch = self._query(pid, context, decision, list(distribution))
        if batch is None:
            return dict(distribution)
        values = model.lookahead(self.net, model.rows(batch, slice(0, 1)), batch["a"], GAMMA, BONUS)
        shifted = np.array(list(distribution.values())) * np.exp(BETA * self.shrink * (values - values.mean()))
        return {a: float(p) for a, p in zip(distribution, shifted / shifted.sum())}

    def value(self, pid: str, context: str, decision: str, actions: list) -> float | None:
        out = self.predict(pid, context, decision, actions)
        return None if out is None else float(np.mean([v["value"] for v in out.values()]))

    def uncertain(self, pid: str, context: str, decision: str, actions: list, scale: float = 1.0) -> bool:
        out = self.predict(pid, context, decision, actions)
        return out is not None and np.mean([v["epistemic"] for v in out.values()]) > scale * self.uncertain_above

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
                "center": None if self.center is None else self.center.tolist(),
                "basis": None if self.basis is None else self.basis.tolist(),
                "std": None if self.std is None else self.std.tolist()}

    def load_state(self, data: dict) -> None:
        self.vocab, self.trained_at = list(data["vocab"]), data["trained_at"]
        self.uncertain_above = data["uncertain_above"]
        self.shrink = float(data["shrink"])
        self.cells = {(c, a): v for c, a, v in data["cells"]}
        self.edges = {k: tuple(v) for k, v in data["edges"].items()}
        self._cache, self._latents = {}, {}
        self.net = None
        if data["weights"] is not None and data.get("basis") is not None:
            self.mean = np.asarray(data["mean"], dtype=np.float32)
            self.center = np.asarray(data["center"], dtype=np.float32)
            self.basis = np.asarray(data["basis"], dtype=np.float32)
            self.std = np.asarray(data["std"], dtype=np.float32)
            net = model.Network(len(self.mean), len(self.vocab))
            net.load_state_dict(torch.load(io.BytesIO(base64.b64decode(data["weights"]))))
            self.net = net.eval()
