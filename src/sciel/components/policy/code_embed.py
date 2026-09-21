"""Program embedder (arch-v4 #3): behavior + code -> score-organized latent.

Shared by offline training (scripts/) and online inference (learned grid + VQ
niches). The feature vector is the evaluator's behavioral `cases` readout (AutoQD lesson:
behavior >> code text) plus token-hash and AST stats. Two heads on a shared
trunk: contrastive projection + jointly-trained score regressor (the ingredient
that organized the latent in every winning latent-search domain)."""

from __future__ import annotations

import ast
import hashlib
import math
import os
import re
from io import BytesIO
from pathlib import Path

import numpy as np


def _stable_bin(token: str, mod: int) -> int:
    """Deterministic token hashing (builtin hash() is per-process salted, which
    would make embeddings -- and any grid descriptor built on them -- irreproducible
    across runs)."""
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(),
                          "little") % mod

try:
    import torch
    from torch import nn
    _TORCH = True
except ImportError:                                   # pragma: no cover
    _TORCH = False

DIM_HASH = 256
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

# The v2 behavior slots, in the fixed order the shipped embedder was trained on.
# NEVER reorder or insert: the trained model's input layout depends on it.
BEHAVIOR_KEYS = ["combined_score", "sum_radii", "target_ratio", "validity",
                 "min_area_normalized", "value", "distance", "reliability"]

# Evaluators name some of those slots differently (function_minimization emits
# `value_score`/`distance_score`/`reliability_score`). Without this table three of
# its four readouts were silently read as 0.0 -- the behavior input collapsed to
# `combined_score` alone. Canonical name wins; aliases are tried in order.
BEHAVIOR_ALIASES = {
    "value": ("value_score",),
    "distance": ("distance_score", "distance_to_global"),
    "reliability": ("reliability_score", "runs_successfully"),
}

# Behavior added AFTER the v2 embedder was trained. Appended at the TAIL of the
# feature vector precisely so an older model can be fed by truncation without any
# misalignment of the slots it does know (see `embed_batch`). Order is fixed for
# the same reason; append only.
#
# These are shape descriptors of the solution, not restatements of its score --
# that was the point of the change. A task emits the subset that applies to it;
# missing keys read 0.0, which is also what the pre-change journals replay as.
BEHAVIOR_EXTRA_KEYS = [
    "b_dispersion",     # spread of the solution's parts (radii CV, area CV, x-spread)
    "b_boundary_frac",  # how much of the solution sits on the domain boundary
    "b_contact",        # mean contact / neighbour-coupling of the parts
    "b_offset",         # displacement of the solution's centroid from the domain's
    "b_concentration",  # fraction of trials/parts in the dominant mode
    "b_structure",      # task-specific structural readout (rank, discreteness, ...)
]


class VQBook:
    """arch-v4 #4b: an online-growing VQ codebook over unit embeddings -- the niche
    assignment of VQ-Elites, decoupled from the embedder so the codebook logic is
    pure numpy and testable on its own.

    A candidate whose cosine distance to every centroid exceeds `d_min` starts a new
    niche (up to `k_max`); otherwise it joins the nearest. After each growth the
    threshold adapts (AURORA container-size-control) to steer the codebook toward
    `target_k`: over target -> raise d_min (harder to grow), under -> lower it.
    Deterministic -- assignment order follows eval order, no RNG."""

    def __init__(self, d_min: float = 0.35, target_k: int = 6, k_max: int = 12,
                 csc_rate: float = 0.08):
        self.d_min = float(d_min)
        self.target_k = int(target_k)
        self.k_max = int(k_max)
        self.csc_rate = float(csc_rate)
        self.centroids: list = []

    @staticmethod
    def _unit(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32)
        return z / (float(np.linalg.norm(z)) + 1e-8)

    def _nearest(self, z: np.ndarray) -> tuple:
        dots = [float(z @ c) for c in self.centroids]
        j = int(max(range(len(dots)), key=lambda i: dots[i]))
        return j, 1.0 - dots[j]                      # (id, cosine distance)

    def assign(self, z: np.ndarray) -> int:
        """Assign and, if the candidate is far from every niche, grow the codebook.
        The CSC feedback runs on EVERY call (not only on growth) so a too-high initial
        threshold self-corrects downward toward `target_k` instead of stalling at K=1."""
        z = self._unit(z)
        if not self.centroids:
            self.centroids.append(z)
            self._csc()
            return 0
        j, dist = self._nearest(z)
        nid = j
        if dist > self.d_min and len(self.centroids) < self.k_max:
            self.centroids.append(z)
            nid = len(self.centroids) - 1
        self._csc()
        return nid

    def assign_fixed(self, z: np.ndarray) -> int:
        """Nearest niche without growing -- used to re-encode the archive on SH
        milestones so cells stay consistent as the codebook shifts (AURORA re-encode)."""
        z = self._unit(z)
        if not self.centroids:
            return 0
        return self._nearest(z)[0]

    def _csc(self) -> None:
        k = len(self.centroids)
        self.d_min *= (1.0 + self.csc_rate * (k - self.target_k))
        self.d_min = min(max(self.d_min, 1e-3), 2.0)

    def state(self) -> dict:
        return {"d_min": self.d_min, "target_k": self.target_k, "k_max": self.k_max,
                "csc_rate": self.csc_rate,
                "centroids": [c.tolist() for c in self.centroids]}

    def load_state(self, d: dict) -> None:
        self.d_min = d["d_min"]
        self.target_k = d["target_k"]
        self.k_max = d["k_max"]
        self.csc_rate = d["csc_rate"]
        self.centroids = [np.asarray(c, dtype=np.float32) for c in d["centroids"]]


def behavior_value(cases: dict, key: str) -> float:
    """One behavior slot, resolving the evaluator's own spelling via BEHAVIOR_ALIASES."""
    cases = cases or {}
    if key in cases:
        try:
            return float(cases[key])
        except (TypeError, ValueError):
            return 0.0
    for alias in BEHAVIOR_ALIASES.get(key, ()):
        if alias in cases:
            try:
                return float(cases[alias])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def featurize(code: str, cases: dict) -> np.ndarray:
    """[behavior(8) | token-hash(256) | AST(6) | behavior_extra(6)].

    The extra behavior block is LAST so that a model trained on the shorter vector
    can be fed `X[:, :in_dim]` and still receive exactly the features it was trained
    on. Growing the block in the middle would silently shift every token-hash bin."""
    beh = np.array([behavior_value(cases, k) for k in BEHAVIOR_KEYS],
                   dtype=np.float32)
    tok = np.zeros(DIM_HASH, dtype=np.float32)
    toks = _TOKEN_RE.findall(code or "")
    for t in toks:
        tok[_stable_bin(t, DIM_HASH)] += 1.0
    tok /= max(len(toks), 1)
    astv = np.zeros(6, dtype=np.float32)
    try:
        tree = ast.parse(code or "")
        astv[:] = [
            len(code) / 1e4,
            sum(isinstance(x, ast.FunctionDef) for x in ast.walk(tree)) / 10,
            sum(isinstance(x, (ast.For, ast.While)) for x in ast.walk(tree)) / 10,
            sum(isinstance(x, ast.Call) for x in ast.walk(tree)) / 100,
            max((len(list(ast.walk(x))) for x in tree.body), default=0) / 100,
            (code or "").count("np.") / 50,
        ]
    except SyntaxError:
        pass
    extra = np.array([behavior_value(cases, k) for k in BEHAVIOR_EXTRA_KEYS],
                     dtype=np.float32)
    return np.concatenate([beh, tok, astv, extra])


class BehaviorAxis:
    """A FITTED behavior axis for the MAP-Elites grid: centre the latent, project on
    one principal direction, standardize the projection, then squash.

    Why fitted and not a seeded random direction. Measured over 13 673 stored
    programs, the raw projection on the seeded direction had std 0.065 inside a
    range of [-0.19, 0.20]; tanh is a straight line on that interval, so the
    descriptor never spread beyond two of ten bins. Standardizing the projection is
    what makes the squash do any work at all.

    Why the SECOND principal component by default. The grid's other coordinate is
    already the score, and the first principal component of this embedder's latent
    correlates 0.82 with the score -- as a behavior axis it would duplicate the axis
    the cell already has. The fitter therefore picks the leading component whose
    score correlation is below a threshold, which on our corpus is the second.

    The axis is fitted OFFLINE over pooled journals and then frozen: refitting moves
    every archived program's coordinate retroactively, so the artifact carries a
    version and the hash of the embedder it was fitted against, and runs are only
    comparable within one (embedder, axis) pair."""

    FORMAT = "behavior_axis/1"

    def __init__(self, axis, mean, center=0.0, scale=1.0, *, component=1,
                 version="", embed_sha="", corr_score=float("nan"),
                 explained=float("nan"), n_fit=0, corr_worst=float("nan"),
                 corr_worst_task="", fit_task=""):
        self.axis = np.asarray(axis, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(self.axis))
        if not np.isfinite(n) or n <= 0:
            raise ValueError("behavior axis must be a non-zero finite vector")
        self.axis = self.axis / n
        self.mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        if self.mean.shape != self.axis.shape:
            raise ValueError(f"axis {self.axis.shape} and mean {self.mean.shape} disagree")
        self.center = float(center)
        self.scale = float(scale)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"behavior axis scale must be positive, got {self.scale}")
        self.component = int(component)
        self.version = str(version)
        self.embed_sha = str(embed_sha)
        self.corr_score = float(corr_score)
        self.explained = float(explained)
        self.n_fit = int(n_fit)
        # The pooled correlation hides per-task structure: fitted across tasks whose
        # scores live on different scales, an axis can read as decorrelated overall
        # while duplicating the score axis inside one task. A grid only ever sees one
        # task, so the WORST per-task correlation is the number that matters.
        self.corr_worst = float(corr_worst)
        self.corr_worst_task = str(corr_worst_task)
        self.fit_task = str(fit_task)          # non-empty => fitted on one task only

    @property
    def dim(self) -> int:
        return int(self.axis.shape[0])

    def value(self, z: np.ndarray) -> float:
        """Latent -> [0,1] descriptor. Standardize first, squash second."""
        u = float((np.asarray(z, dtype=np.float32).reshape(-1) - self.mean) @ self.axis)
        return float((np.tanh((u - self.center) / self.scale) + 1.0) / 2.0)

    def save(self, path: str) -> None:
        np.savez(path, fmt=np.array(self.FORMAT), axis=self.axis, mean=self.mean,
                 center=np.array(self.center, dtype=np.float64),
                 scale=np.array(self.scale, dtype=np.float64),
                 component=np.array(self.component), version=np.array(self.version),
                 embed_sha=np.array(self.embed_sha),
                 corr_score=np.array(self.corr_score, dtype=np.float64),
                 explained=np.array(self.explained, dtype=np.float64),
                 n_fit=np.array(self.n_fit),
                 corr_worst=np.array(self.corr_worst, dtype=np.float64),
                 corr_worst_task=np.array(self.corr_worst_task),
                 fit_task=np.array(self.fit_task))

    @classmethod
    def load(cls, path: str) -> "BehaviorAxis":
        raw = Path(path).read_bytes()
        with np.load(BytesIO(raw), allow_pickle=False) as d:
            fmt = str(d["fmt"]) if "fmt" in d else ""
            if fmt != cls.FORMAT:
                raise ValueError(f"{path}: expected {cls.FORMAT}, got {fmt!r}")
            axis = cls(d["axis"], d["mean"], float(d["center"]), float(d["scale"]),
                       component=int(d["component"]), version=str(d["version"]),
                       embed_sha=str(d["embed_sha"]), corr_score=float(d["corr_score"]),
                       explained=float(d["explained"]), n_fit=int(d["n_fit"]),
                       corr_worst=float(d["corr_worst"]) if "corr_worst" in d else float("nan"),
                       corr_worst_task=str(d["corr_worst_task"]) if "corr_worst_task" in d else "",
                       fit_task=str(d["fit_task"]) if "fit_task" in d else "")
        axis.artifact_identity = {"format": cls.FORMAT,
                                  "sha256": hashlib.sha256(raw).hexdigest()}
        return axis


def file_sha(path: str, n: int = 12) -> str:
    """Short content hash -- ties a fitted axis to the exact embedder it came from."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


if _TORCH:
    class Embedder(nn.Module):
        def __init__(self, in_dim, out_dim=64):
            super().__init__()
            self.in_dim = int(in_dim)
            self.trunk = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU())
            self.proj = nn.Linear(128, out_dim)
            self.score = nn.Linear(128, 1)

        def embed(self, x):
            z = self.proj(self.trunk(x))
            return z / (z.norm(dim=-1, keepdim=True) + 1e-8)

        def predict_score(self, x):
            return self.score(self.trunk(x)).squeeze(-1)

        @classmethod
        def load(cls, path: str) -> "Embedder":
            raw = Path(path).read_bytes()
            d = torch.load(BytesIO(raw), weights_only=False)
            m = cls(d["in_dim"]); m.load_state_dict(d["state"]); m.eval()
            m.artifact_identity = {"format": "sciel-code-embedder/1",
                "sha256": hashlib.sha256(raw).hexdigest(), "in_dim": m.in_dim}
            return m

    def embed_batch(model: "Embedder", programs) -> np.ndarray:
        """programs: iterable of (code, cases) -> (N, 64) unit embeddings.

        A model trained before BEHAVIOR_EXTRA_KEYS existed declares a shorter
        `in_dim`; since the extra block is at the tail, slicing gives it byte-for-byte
        the features it was trained on. A model wider than the featurizer is a real
        mismatch and raises rather than being zero-padded into nonsense."""
        X = np.stack([featurize(c, cs) for c, cs in programs]).astype(np.float32)
        want = int(getattr(model, "in_dim", X.shape[1]))
        if want > X.shape[1]:
            raise ValueError(
                f"embedder expects {want} features, featurizer produces {X.shape[1]}; "
                "the model is newer than this code")
        if want < X.shape[1]:
            X = X[:, :want]                  # older model: drop the tail block only
        with torch.no_grad():
            return model.embed(torch.tensor(X)).numpy()

    class LearnedDescriptor:
        """The MAP-Elites behavior axis: embed the program, reduce to one number in
        [0,1]. Deterministic, and frozen for the whole run.

        Two modes, and the run journal records which one was used, because the two
        are NOT comparable -- the same program gets a different cell under each.

          fitted  -- a BehaviorAxis artifact: centred latent, principal direction,
                     standardized projection. This is the mode to run in.
          legacy  -- no artifact: the seeded random direction with an unstandardized
                     tanh, i.e. exactly what shipped before. Kept so old runs replay
                     and resume bit-identically, not because it works: measured over
                     13 673 programs it occupies 2 of 10 bins.
        """

        def __init__(self, embedder: "Embedder", out_dim: int = 64, seed: int = 0,
                     axis: "BehaviorAxis | None" = None, axis_sha: str | None = None):
            self.embedder = embedder
            self.axis = axis
            # The artifact's own content hash. `version` is author-supplied and a refit
            # under the same name reuses it, so version alone cannot tell two axes apart.
            self.axis_sha = axis_sha or ""
            if axis is None:
                rng = np.random.default_rng(seed)
                self.w = rng.standard_normal(out_dim).astype(np.float32)
                self.w /= np.linalg.norm(self.w)
            else:
                self.w = None

        @classmethod
        def load(cls, embed_path: str, seed: int = 0,
                 axis_path: str | None = None) -> "LearnedDescriptor":
            """Load configured artifacts exactly; only explicit null selects legacy mode."""
            embedder = Embedder.load(embed_path)
            axis = axis_sha = None
            if axis_path is not None:
                axis = BehaviorAxis.load(axis_path)
                axis_sha = axis.artifact_identity['sha256'][:12]
                sha = embedder.artifact_identity['sha256'][:12]
                if not axis.embed_sha:
                    raise ValueError(
                        f"{axis_path} records no embedder hash, so nothing ties it to "
                        f"{embed_path}; refit it with scripts/fit_behavior_axis.py")
                if axis.embed_sha != sha:
                    raise ValueError(
                        f"{axis_path} was fitted against embedder {axis.embed_sha}, "
                        f"but {embed_path} hashes to {sha}; refit the axis")
                # Width is checked here rather than at the first descriptor call: a
                # mismatch discovered mid-run surfaces as a broadcast error inside
                # db.add, tens of evaluations in.
                probe = embed_batch(embedder, [("x = 1\n", {})])[0]
                if axis.dim != probe.shape[0]:
                    raise ValueError(
                        f"{axis_path} is {axis.dim}-dimensional but {embed_path} "
                        f"produces a {probe.shape[0]}-dimensional latent; refit the axis")
            return cls(embedder, seed=seed, axis=axis, axis_sha=axis_sha)

        @property
        def kind(self) -> str:
            """Identifies the axis strongly enough to compare two runs. `version` is
            author-supplied and a refit that overwrites the same file reuses it, so the
            artifact's content hash is part of the name."""
            if self.axis is None:
                return "legacy_random_raw"
            tag = self.axis.version or "unversioned"
            if self.axis_sha:
                tag = f"{tag}@{self.axis_sha[:8]}"
            return f"pca{self.axis.component + 1}_{tag}"

        @staticmethod
        def _num(x):
            """json.dumps writes bare NaN/Infinity, which is not valid JSON and breaks
            any strict reader of the journal. Unknown metadata is None, not NaN."""
            x = float(x)
            return round(x, 4) if math.isfinite(x) else None

        def describe(self) -> dict:
            """Goes into run_start: a run must say which axis produced its cells.

            `behavior_in_latent` is the honest flag. An embedder trained before
            BEHAVIOR_EXTRA_KEYS existed has a narrower in_dim, so embed_batch slices
            that block off and the evaluator's behaviour readout -- computed, logged,
            and in case_metrics -- never reaches the latent, the axis, or the grid.
            Retraining the embedder is what closes that gap; until then the flag says
            so in every journal instead of leaving it to be discovered."""
            want = len(BEHAVIOR_KEYS) + DIM_HASH + 6 + len(BEHAVIOR_EXTRA_KEYS)
            d = {"kind": self.kind,
                 "behavior_in_latent": int(getattr(self.embedder, "in_dim", want)) >= want}
            if self.axis is not None:
                d.update(component=self.axis.component, version=self.axis.version,
                         embed_sha=self.axis.embed_sha, axis_sha=self.axis_sha,
                         n_fit=self.axis.n_fit, fit_task=self.axis.fit_task,
                         corr_score=self._num(self.axis.corr_score),
                         corr_worst=self._num(self.axis.corr_worst),
                         corr_worst_task=self.axis.corr_worst_task,
                         explained=self._num(self.axis.explained))
            return d

        def __call__(self, code: str, cases: dict) -> float:
            z = embed_batch(self.embedder, [(code, cases)])[0]
            if self.axis is not None:
                return self.axis.value(z)
            return float((np.tanh(z @ self.w) + 1.0) / 2.0)      # [0,1]

    class VQNiches:
        """Adaptive codebook over program embeddings, checkpointed with the population."""

        def __init__(self, embedder: "Embedder", d_min: float = 0.35,
                     target_k: int = 6, k_max: int = 12, csc_rate: float = 0.08):
            self.embedder = embedder
            self.book = VQBook(d_min=d_min, target_k=target_k, k_max=k_max,
                               csc_rate=csc_rate)

        @classmethod
        def load(cls, embed_path: str, **kw) -> "VQNiches":
            return cls(Embedder.load(embed_path), **kw)

        def _embed(self, code: str, cases: dict) -> np.ndarray:
            return embed_batch(self.embedder, [(code, cases)])[0]

        def assign(self, code: str, cases: dict) -> int:
            return self.book.assign(self._embed(code, cases))

        def assign_fixed(self, code: str, cases: dict) -> int:
            return self.book.assign_fixed(self._embed(code, cases))

        @property
        def k(self) -> int:
            return len(self.book.centroids)

        def state(self) -> dict:
            return self.book.state()

        def load_state(self, d: dict) -> None:
            self.book.load_state(d)
