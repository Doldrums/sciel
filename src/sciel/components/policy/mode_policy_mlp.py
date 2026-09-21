"""Return-conditioned mode policy: a small MLP mapping
(pre-step features + target return-to-go) -> distribution over selection modes.

A "1-step Decision Transformer": the SAME return-conditioning recipe as the Stage-B
online DT (condition on a desired RTG, predict the action), but an MLP over the current
context instead of a transformer over lineage history -- so it trains on the tiny
current corpus. Trained offline by return-conditioned behaviour cloning; served in-run
as a mode_source. numpy-only: no heavy deps at inference; the trainer is plain SGD/Adam.
"""

from __future__ import annotations

import numpy as np
import hashlib
from io import BytesIO
from pathlib import Path

from .mode_features import MODES, N_FEATURES


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class ReturnConditionedModeMLP:
    def __init__(self, hidden: int = 16, seed: int = 0):
        self.hidden = hidden
        self.seed = seed
        self.in_dim = N_FEATURES + 1                     # features + target_rtg
        self.mu = np.zeros(self.in_dim)
        self.sd = np.ones(self.in_dim)
        self.W1 = self.b1 = self.W2 = self.b2 = None

    # ---------- inference ----------
    def _forward(self, Xn):
        h = np.maximum(0.0, Xn @ self.W1 + self.b1)
        return _softmax(h @ self.W2 + self.b2)

    def dist(self, features, target_rtg: float) -> dict:
        x = np.asarray(list(features) + [float(target_rtg)], dtype=float)
        p = self._forward(((x - self.mu) / self.sd)[None, :])[0]
        return {m: float(p[i]) for i, m in enumerate(MODES)}

    # ---------- offline training (return-conditioned BC) ----------
    def fit(self, samples, epochs: int = 400, lr: float = 1e-2, l2: float = 1e-4):
        rng = np.random.RandomState(self.seed)
        X = np.array([list(s["features"]) + [s["rtg"]] for s in samples], dtype=float)
        y = np.array([MODES.index(s["mode"]) for s in samples], dtype=int)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xn = (X - self.mu) / self.sd
        n, d = Xn.shape
        H, K = self.hidden, len(MODES)
        self.W1 = rng.randn(d, H) * np.sqrt(2.0 / d); self.b1 = np.zeros(H)
        self.W2 = rng.randn(H, K) * np.sqrt(2.0 / H); self.b2 = np.zeros(K)
        Y = np.eye(K)[y]
        names = ("W1", "b1", "W2", "b2")
        mom = {p: 0.0 for p in names}; vel = {p: 0.0 for p in names}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for t in range(1, epochs + 1):
            Hpre = Xn @ self.W1 + self.b1
            Ha = np.maximum(0.0, Hpre)
            P = _softmax(Ha @ self.W2 + self.b2)
            dZ2 = (P - Y) / n
            grads = {
                "W2": Ha.T @ dZ2 + l2 * self.W2, "b2": dZ2.sum(0),
            }
            dHpre = (dZ2 @ self.W2.T) * (Hpre > 0)
            grads["W1"] = Xn.T @ dHpre + l2 * self.W1
            grads["b1"] = dHpre.sum(0)
            for p in names:
                mom[p] = beta1 * mom[p] + (1 - beta1) * grads[p]
                vel[p] = beta2 * vel[p] + (1 - beta2) * (grads[p] ** 2)
                mhat = mom[p] / (1 - beta1 ** t)
                vhat = vel[p] / (1 - beta2 ** t)
                setattr(self, p, getattr(self, p) - lr * mhat / (np.sqrt(vhat) + eps))
        return self

    def cross_entropy(self, samples) -> float:
        X = np.array([list(s["features"]) + [s["rtg"]] for s in samples], dtype=float)
        y = np.array([MODES.index(s["mode"]) for s in samples], dtype=int)
        P = self._forward((X - self.mu) / self.sd)
        return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())

    # ---------- artifact ----------
    def save(self, path: str) -> None:
        np.savez(path, hidden=self.hidden, mu=self.mu, sd=self.sd,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    @classmethod
    def load(cls, path: str) -> "ReturnConditionedModeMLP":
        raw = Path(path).read_bytes()
        with np.load(BytesIO(raw), allow_pickle=False) as d:
            m = cls(hidden=int(d["hidden"]))
            m.mu, m.sd = d["mu"], d["sd"]
            m.W1, m.b1, m.W2, m.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        # Bind the bytes actually loaded, not a later version of this path.
        m.artifact_identity = {"format": "sciel-mode-mlp/1",
            "sha256": hashlib.sha256(raw).hexdigest(), "modes": list(MODES),
            "n_features": N_FEATURES, "hidden": m.hidden}
        return m
