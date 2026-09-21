"""Deterministic geometry for the opt-in container comparison.

This module owns no programs, scores, retention policy, or random selection. All
variants consume the same externally computed, frozen vectors. ``update`` returns
hard cell labels for retention; ``probabilities`` exposes Gibbs responsibilities
for a caller that samples a centre and then a parent. Probability columns follow
``centre_ids``, whose values survive updates and are never reused.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


class Geometry:
    """Fixed reference CVT, admission-only VQ, or deterministic annealing.

    CVT fits a seeded, search-independent reference cloud once. VQ processes each
    program ID once and starts a centre when Euclidean novelty exceeds the frozen
    threshold. Annealing refits the *current active* population, calibrates T0 only
    on the first nondegenerate batch with enough points, and attempts at most one
    split per changed batch. Coincident or unsupported splits are rolled back.
    """

    _VERSION = 1
    _EPS = 1e-10

    def __init__(self, kind: str, dim: int = 64, k: int = 4,
                 max_centres: int = 6, seed: int = 0, alpha: float = .02,
                 split_min_points: int = 4, novelty_threshold: float = .25):
        if kind not in {"cvt", "vq", "annealed"}:
            raise ValueError(f"unknown geometry: {kind!r}")
        if any(isinstance(v, bool) or int(v) != v or v < 1
               for v in (dim, k, max_centres)):
            raise ValueError("dim, k and max_centres must be positive integers")
        if kind == "cvt" and k > max_centres:
            raise ValueError("k cannot exceed max_centres")
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if int(split_min_points) != split_min_points or split_min_points < 4:
            raise ValueError("split_min_points must be an integer >= 4")
        if not np.isfinite(novelty_threshold) or novelty_threshold <= 0:
            raise ValueError("novelty_threshold must be positive and finite")
        self.kind, self.dim, self.k = kind, int(dim), int(k)
        self.max_centres, self.seed = int(max_centres), int(seed)
        self.alpha, self.split_min_points = float(alpha), int(split_min_points)
        self.novelty_threshold = float(novelty_threshold)
        self.centres = np.zeros((1, self.dim), dtype=np.float64)
        self.centre_ids = [0]
        self._counts = [0]
        self._next_id = 1
        self._initialised = False
        self._seen: set[str] = set()
        self.progress = 0.0
        self.t0: float | None = None
        self.temperature: float | None = None if kind == "annealed" else 1.0
        self._last_signature: str | None = None
        self._last_occupancy: list[int] = []
        self._last_mass: list[float] = []
        self._splits = self._rejected_splits = self._pruned = 0
        if kind == "cvt":
            self.centres = self._reference_cvt()
            self.centre_ids = list(range(self.k))
            self._counts = [0] * self.k
            self._next_id = self.k
            self._initialised = True

    def _reference_cvt(self) -> np.ndarray:
        """Lloyd quantization of a seeded reference, never the search population."""
        rng = np.random.default_rng(self.seed)
        if self.dim == 1:
            # The 1-D unit sphere has only two points; use the unit interval.
            reference = np.linspace(-1, 1, max(256, 128 * self.k))[:, None]
        else:
            reference = rng.normal(size=(max(256, 128 * self.k), self.dim))
            reference /= np.linalg.norm(reference, axis=1, keepdims=True)
        centres = [reference[int(rng.integers(len(reference)))].copy()]
        for _ in range(1, self.k):
            distance = self._distances(reference, np.asarray(centres)).min(axis=1)
            centres.append(reference[int(np.argmax(distance))].copy())
        centres = np.asarray(centres)
        for _ in range(24):
            labels = self._distances(reference, centres).argmin(axis=1)
            updated = np.asarray([reference[labels == j].mean(axis=0)
                                  if np.any(labels == j) else centres[j]
                                  for j in range(self.k)])
            if np.max(np.abs(updated - centres)) < 1e-10:
                break
            centres = updated
        return centres

    def _vectors(self, vectors: np.ndarray) -> np.ndarray:
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"vectors must have shape (N, {self.dim})")
        if not np.all(np.isfinite(values)):
            raise ValueError("vectors must be finite")
        return values

    @staticmethod
    def _distances(vectors: np.ndarray, centres: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            distances = np.square(vectors[:, None, :] - centres[None, :, :]).sum(axis=2)
        if not np.all(np.isfinite(distances)):
            raise ValueError("vectors are too large for finite squared distances")
        return distances

    def _responsibilities(self, vectors: np.ndarray, centres: np.ndarray) -> np.ndarray:
        distances = self._distances(vectors, centres)
        # Subtract the closest distance before dividing: cold assignments remain
        # finite even if every absolute distance / temperature would overflow.
        distances -= distances.min(axis=1, keepdims=True)
        with np.errstate(over="ignore", under="ignore"):
            weights = np.exp(-distances / max(self.temperature or 1.0, 1e-12))
        return weights / weights.sum(axis=1, keepdims=True)

    def probabilities(self, vectors: np.ndarray) -> np.ndarray:
        """Return normalized N x K responsibilities in ``centre_ids`` order."""
        values = self._vectors(vectors)
        if not len(values):
            return np.empty((0, len(self.centres)), dtype=np.float64)
        return self._responsibilities(values, self.centres)

    def refresh_active(self, ids: list[str], vectors: np.ndarray) -> dict[str, int]:
        """Refit annealed centres after retention, without another admission.

        This neither calibrates/cools nor splits, and does not train the VQ or
        change admission history. Call after evicting provisional candidates so
        annealed parent selection cannot use a centre supported only by a rejected
        candidate. Fixed CVT and admission-trained VQ only refresh occupancy.
        """
        values = self._vectors(vectors)
        if len(ids) != len(values) or any(not isinstance(i, str) for i in ids):
            raise ValueError("ids must contain one string per vector")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")
        order = sorted(range(len(ids)), key=ids.__getitem__)
        ordered_ids = [ids[i] for i in order]
        values = values[order]
        if not len(values):
            self._last_occupancy = [0] * len(self.centres)
            self._last_mass = [0.0] * len(self.centres)
            return {}
        if self.kind == "annealed":
            self.centres = self._converge(values, self.centres)
            self._prune_inactive(values)
            # An immediate update of the same retained population is a no-op,
            # rather than another opportunity to split without an admission.
            self._last_signature = hashlib.sha256(json.dumps(ordered_ids).encode() +
                                       values.tobytes() + repr(self.progress).encode()).hexdigest()
        hard = self._distances(values, self.centres).argmin(axis=1)
        self._last_occupancy = np.bincount(hard, minlength=len(self.centres)).tolist()
        self._last_mass = self.probabilities(values).mean(axis=0).tolist()
        return {i: self.centre_ids[int(label)] for i, label in zip(ordered_ids, hard)}

    def update(self, ids: list[str], vectors: np.ndarray,
               progress: float) -> dict[str, int]:
        values = self._vectors(vectors)
        if len(ids) != len(values) or any(not isinstance(i, str) for i in ids):
            raise ValueError("ids must contain one string per vector")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique")
        if not np.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("progress must be finite and in [0, 1]")
        self.progress = max(self.progress, float(progress))
        # Canonical order gives identical floating-point reductions after resume
        # and makes a caller's dictionary insertion order irrelevant.
        order = sorted(range(len(ids)), key=ids.__getitem__)
        ordered_ids = [ids[i] for i in order]
        values = values[order]
        signature = hashlib.sha256(json.dumps(ordered_ids).encode() +
                                   values.tobytes() + repr(self.progress).encode()).hexdigest()
        if len(values) and signature != self._last_signature:
            if self.kind == "vq":
                self._update_vq(ordered_ids, values)
            elif self.kind == "annealed":
                self._update_annealed(values)
            self._seen.update(ordered_ids)
        self._last_signature = signature
        if not len(values):
            self._last_occupancy = [0] * len(self.centres)
            self._last_mass = [0.0] * len(self.centres)
            return {}
        hard = self._distances(values, self.centres).argmin(axis=1)
        self._last_occupancy = np.bincount(hard, minlength=len(self.centres)).tolist()
        self._last_mass = self.probabilities(values).mean(axis=0).tolist()
        return {i: self.centre_ids[int(label)] for i, label in zip(ordered_ids, hard)}

    def _update_vq(self, ids: list[str], values: np.ndarray) -> None:
        for program_id, vector in zip(ids, values):
            if program_id in self._seen:
                continue
            if not self._initialised:
                self.centres[0] = vector
                self._counts[0] = 1
                self._initialised = True
                continue
            distances = self._distances(vector[None, :], self.centres)[0]
            j = int(distances.argmin())
            if (distances[j] > self.novelty_threshold ** 2 and
                    len(self.centres) < self.max_centres):
                self.centres = np.vstack([self.centres, vector])
                self.centre_ids.append(self._next_id)
                self._next_id += 1
                self._counts.append(1)
            else:
                self._counts[j] += 1
                self.centres[j] += (vector - self.centres[j]) / self._counts[j]

    @staticmethod
    def _largest_eigen(values: np.ndarray, weights: np.ndarray,
                       centre: np.ndarray) -> tuple[float, np.ndarray]:
        delta = values - centre
        covariance = (delta * weights[:, None]).T @ delta / weights.sum()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        # Eigenvector signs are arbitrary; fix the sign so split IDs are stable.
        if direction[np.argmax(np.abs(direction))] < 0:
            direction = -direction
        return max(0.0, float(eigenvalues[-1])), direction

    def _converge(self, values: np.ndarray, centres: np.ndarray) -> np.ndarray:
        centres = centres.copy()
        for _ in range(96):
            weights = self._responsibilities(values, centres)
            mass = weights.sum(axis=0)
            updated = centres.copy()
            occupied = mass > self._EPS
            updated[occupied] = (weights[:, occupied].T @ values) / mass[occupied, None]
            movement = np.max(np.abs(updated - centres))
            centres = updated
            if movement < 1e-9:
                break
        return centres

    def _prune_inactive(self, values: np.ndarray) -> None:
        # Centres with no hard support cannot protect distinct retained programs.
        # Removing them also catches centres that coalesced as the active pool
        # changed. IDs of the survivors are preserved, never renumbered.
        for _ in range(len(self.centres)):
            hard = self._distances(values, self.centres).argmin(axis=1)
            keep = sorted(set(int(i) for i in hard))
            # Near-coincident centres can win one hard tie each through rounding;
            # hard occupancy alone therefore does not establish distinct niches.
            distinct = []
            for j in keep:
                if (not distinct or
                        np.min(self._distances(self.centres[j:j + 1],
                                               self.centres[distinct])) > 1e-12):
                    distinct.append(j)
            keep = distinct
            if len(keep) == len(self.centres):
                break
            self._pruned += len(self.centres) - len(keep)
            self.centres = self._converge(values, self.centres[keep])
            self.centre_ids = [self.centre_ids[i] for i in keep]
            self._counts = [0] * len(keep)

    def _update_annealed(self, values: np.ndarray) -> None:
        if not self._initialised:
            self.centres[0] = values.mean(axis=0)
            self._initialised = True
        if self.t0 is None and len(values) >= self.split_min_points:
            variance, _ = self._largest_eigen(values, np.ones(len(values)), values.mean(axis=0))
            if variance > self._EPS:
                self.t0 = 4.0 * variance
        if self.t0 is not None:
            self.temperature = self.t0 * self.alpha ** self.progress
        self.centres = self._converge(values, self.centres)
        self._prune_inactive(values)
        if (self.t0 is None or len(values) < self.split_min_points or
                len(self.centres) >= self.max_centres):
            return
        weights = self.probabilities(values)
        unstable = []
        for j in range(len(self.centres)):
            if weights[:, j].sum() < self.split_min_points:
                continue
            variance, direction = self._largest_eigen(values, weights[:, j], self.centres[j])
            if 2.0 * variance > self.temperature * (1 + 1e-6):
                unstable.append((variance, j, direction))
        if not unstable:
            return
        variance, j, direction = max(unstable, key=lambda item: (item[0], -item[1]))
        # A scale-aware perturbation escapes a numerical duplicate fixed point;
        # the old prototype's constant epsilon often reconverged to duplicates.
        epsilon = max(.05 * np.sqrt(variance),
                      .5 * np.sqrt(max(variance - self.temperature / 2, 0)))
        proposal = np.vstack([self.centres, self.centres[j] + epsilon * direction])
        proposal[j] -= epsilon * direction
        proposal = self._converge(values, proposal)
        distances = self._distances(proposal, proposal)
        np.fill_diagonal(distances, np.inf)
        hard = self._distances(values, proposal).argmin(axis=1)
        counts = np.bincount(hard, minlength=len(proposal))
        mass = self._responsibilities(values, proposal).sum(axis=0)
        min_support = max(2, self.split_min_points // 2)
        separation_floor = max(1e-6, .02 * np.sqrt(variance))
        supported = (counts[j] >= min_support and counts[-1] >= min_support and
                     mass[j] >= min_support and mass[-1] >= min_support and
                     np.all(counts > 0))
        if np.min(distances) <= separation_floor ** 2 or not supported:
            self._rejected_splits += 1
            return
        self.centres = proposal
        self.centre_ids.append(self._next_id)
        self._next_id += 1
        self._counts = [0] * len(proposal)
        self._splits += 1

    def diagnostics(self) -> dict[str, Any]:
        separation = None
        if len(self.centres) > 1:
            distances = self._distances(self.centres, self.centres)
            np.fill_diagonal(distances, np.inf)
            separation = float(np.sqrt(distances.min()))
        mass = np.asarray(self._last_mass)
        return {"kind": self.kind, "n_centres": len(self.centres),
                "centre_ids": list(self.centre_ids), "max_centres": self.max_centres,
                "temperature": self.temperature, "t0": self.t0,
                "progress": self.progress, "seen_ids": len(self._seen),
                "occupancy": list(self._last_occupancy), "mass": list(self._last_mass),
                "hard_occupied_centres": sum(n > 0 for n in self._last_occupancy),
                "effective_centres": float(1 / np.square(mass).sum()) if np.any(mass) else 0.0,
                "min_separation": separation, "accepted_splits": self._splits,
                "rejected_splits": self._rejected_splits, "pruned_centres": self._pruned}

    def state(self) -> dict[str, Any]:
        """Return the complete JSON-serializable checkpoint, including chronology."""
        return {"version": self._VERSION,
                "config": {"kind": self.kind, "dim": self.dim, "k": self.k,
                           "max_centres": self.max_centres, "seed": self.seed,
                           "alpha": self.alpha, "split_min_points": self.split_min_points,
                           "novelty_threshold": self.novelty_threshold},
                "centres": self.centres.tolist(), "centre_ids": list(self.centre_ids),
                "counts": list(self._counts), "next_id": self._next_id,
                "initialised": self._initialised, "seen": sorted(self._seen),
                "progress": self.progress, "t0": self.t0, "temperature": self.temperature,
                "last_signature": self._last_signature,
                "last_occupancy": list(self._last_occupancy), "last_mass": list(self._last_mass),
                "splits": self._splits, "rejected_splits": self._rejected_splits,
                "pruned": self._pruned}

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore a checkpoint, including its configuration, without refitting."""
        if state.get("version") != self._VERSION:
            raise ValueError("unsupported geometry state version")
        restored = Geometry(**state["config"])
        centres = restored._vectors(state["centres"])
        ids = state["centre_ids"]
        counts = state["counts"]
        if (not 1 <= len(centres) <= restored.max_centres or len(ids) != len(centres) or
                (restored.kind == "cvt" and len(centres) != restored.k) or
                len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids) or
                state["next_id"] <= max(ids) or len(counts) != len(ids) or
                any(type(n) is not int or n < 0 for n in counts)):
            raise ValueError("invalid geometry centre state")
        if (not np.isfinite(state["progress"]) or not 0 <= state["progress"] <= 1 or
                any(value is not None and (not np.isfinite(value) or value <= 0)
                    for value in (state["t0"], state["temperature"]))):
            raise ValueError("invalid geometry temperature state")
        restored.centres = centres.copy()
        restored.centre_ids, restored._counts = list(ids), list(counts)
        restored._next_id, restored._initialised = state["next_id"], state["initialised"]
        restored._seen = set(state["seen"])
        restored.progress, restored.t0 = state["progress"], state["t0"]
        restored.temperature, restored._last_signature = state["temperature"], state["last_signature"]
        restored._last_occupancy, restored._last_mass = list(state["last_occupancy"]), list(state["last_mass"])
        restored._splits, restored._rejected_splits = state["splits"], state["rejected_splits"]
        restored._pruned = state["pruned"]
        self.__dict__.update(restored.__dict__)
