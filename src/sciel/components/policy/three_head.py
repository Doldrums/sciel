"""Shared Set-Transformer over program embeddings with three readouts.

The promise head tilts parent selection; eviction ranks removal candidates;
niche predicts soft membership. Heads are trained offline by
scripts/train_three_head.py and remain frozen during search. Configuration
controls whether the engine uses the model."""

from __future__ import annotations

import numpy as np
import hashlib
from io import BytesIO
from pathlib import Path

from .tilt import Tilt

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:                                    # torch optional at import time
    _TORCH = False

if _TORCH:
    class _SetAttnBlock(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

        def forward(self, x):                        # (B, N, d)
            a, _ = self.attn(x, x, x, need_weights=False)
            x = self.ln1(x + a)
            return self.ln2(x + self.ff(x))

    class _ThreeHeadNet(nn.Module):
        def __init__(self, in_dim, n_niche, d=32, heads=4, blocks=2):
            super().__init__()
            self.embed = nn.Linear(in_dim, d)
            self.blocks = nn.ModuleList([_SetAttnBlock(d, heads) for _ in range(blocks)])
            self.promise = nn.Linear(d, 1)
            self.eviction = nn.Linear(d, 1)
            self.niche = nn.Linear(d, n_niche)

        def trunk(self, x):                          # (B, N, in_dim) -> (B, N, d)
            h = self.embed(x)
            for b in self.blocks:
                h = b(h)
            return h

        def forward(self, x):
            h = self.trunk(x)
            return (self.promise(h).squeeze(-1),     # (B, N)
                    self.eviction(h).squeeze(-1),    # (B, N)
                    self.niche(h))                   # (B, N, n_niche)


class ThreeHeadModel:
    """Feature-normalised shared trunk + three heads; fit/predict/save/load.

    Input per program is its embedder-v2 embedding (behavior + code). Labels are a
    dict of the heads present in the data: promise (lineage-credit scalar), eviction
    (1=keep / 0=evict scalar), niche (VQ id, integer). A head with no labels in a
    batch contributes no loss, so the model trains on whatever labels exist."""

    def __init__(self, in_dim: int, n_niche: int = 12, d: int = 32, heads: int = 4,
                 blocks: int = 2, seed: int = 0):
        if not _TORCH:
            raise RuntimeError("ThreeHeadModel requires torch")
        torch.manual_seed(seed)
        self.in_dim = in_dim
        self.n_niche = n_niche
        self.net = _ThreeHeadNet(in_dim, n_niche, d, heads, blocks)
        self.mu = np.zeros(in_dim)
        self.sd = np.ones(in_dim)

    def _norm(self, X):
        return (np.asarray(X, dtype=float) - self.mu) / self.sd

    def fit(self, sets, labels, epochs: int = 200, lr: float = 3e-3):
        """sets: list of (N_i, in_dim). labels: list of dicts with any of
        {'promise':(N,), 'eviction':(N,), 'niche':(N,) int}. Missing head -> no loss."""
        allX = np.concatenate([np.asarray(s, dtype=float) for s in sets], axis=0)
        self.mu, self.sd = allX.mean(0), allX.std(0) + 1e-8
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        bce = torch.nn.BCEWithLogitsLoss()
        ce = torch.nn.CrossEntropyLoss()
        self.net.train()
        for _ in range(epochs):
            for s, lab in zip(sets, labels):
                xb = torch.tensor(self._norm(s), dtype=torch.float32).unsqueeze(0)
                pr, ev, ni = self.net(xb)
                loss = xb.new_zeros(())
                has_loss = False
                if lab.get("promise") is not None:
                    y = torch.tensor(np.asarray(lab["promise"], float), dtype=torch.float32)
                    loss = loss + ((pr.squeeze(0) - y) ** 2).mean(); has_loss = True
                if lab.get("eviction") is not None:
                    y = torch.tensor(np.asarray(lab["eviction"], float), dtype=torch.float32)
                    loss = loss + bce(ev.squeeze(0), y); has_loss = True
                if lab.get("niche") is not None:
                    y = torch.tensor(np.asarray(lab["niche"], np.int64), dtype=torch.long)
                    loss = loss + ce(ni.squeeze(0), y); has_loss = True
                if not has_loss:
                    continue
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict(self, features) -> dict:
        """features: (N, in_dim) -> {'promise':(N,), 'eviction':(N,), 'niche':(N,)}."""
        self.net.eval()
        with torch.no_grad():
            xb = torch.tensor(self._norm(features), dtype=torch.float32).unsqueeze(0)
            pr, ev, ni = self.net(xb)
            return {"promise": pr.squeeze(0).numpy(),
                    "eviction": ev.squeeze(0).numpy(),
                    "niche": ni.squeeze(0).argmax(-1).numpy()}

    def promise_dist(self, features, temperature: float = 1.0) -> np.ndarray:
        z = self.predict(features)["promise"] / max(temperature, 1e-6)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def save(self, path: str) -> None:
        torch.save({"in_dim": self.in_dim, "n_niche": self.n_niche,
                    "mu": self.mu, "sd": self.sd, "state": self.net.state_dict(),
                    "arch": (self.net.embed.out_features,
                             self.net.blocks[0].attn.num_heads, len(self.net.blocks))},
                   path)

    @classmethod
    def load(cls, path: str) -> "ThreeHeadModel":
        raw = Path(path).read_bytes()
        d = torch.load(BytesIO(raw), weights_only=False)
        emb, heads, blocks = d["arch"]
        m = cls(d["in_dim"], n_niche=d["n_niche"], d=emb, heads=heads, blocks=blocks)
        m.mu, m.sd = d["mu"], d["sd"]
        m.net.load_state_dict(d["state"])
        m.artifact_identity = {"format": "sciel-three-head/1",
            "sha256": hashlib.sha256(raw).hexdigest(), "in_dim": m.in_dim,
            "n_niche": m.n_niche, "arch": list(d["arch"])}
        return m


class ThreeHeadTilt:
    """Promise head -> parent-selection tilt. OFF by default: promise as a selection
    override did not hold up in A/B. Frozen; evaluated per parent-select."""

    def __init__(self, model: ThreeHeadModel, embedder, beta: float = 0.3):
        self.model = model
        self.embedder = embedder
        self.beta = beta

    @classmethod
    def load(cls, model_path: str, embed_path: str, beta: float = 0.3) -> "ThreeHeadTilt":
        from .code_embed import Embedder
        return cls(ThreeHeadModel.load(model_path), Embedder.load(embed_path), beta)

    def build(self, view, db) -> Tilt:
        from .code_embed import embed_batch
        progs = [db.get(i) for i in view.ids]
        feats = embed_batch(self.embedder, [(p.code, p.cases) for p in progs])
        w = self.model.promise_dist(feats)
        target = {i: float(wi) for i, wi in zip(view.ids, w)}
        return Tilt(self.beta, lambda _v: target)
