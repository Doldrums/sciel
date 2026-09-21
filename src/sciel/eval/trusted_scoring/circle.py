"""Pure Circle metric, extracted unchanged from the pinned legacy evaluator."""
import numpy as np


TARGET_VALUE = 2.635


N_CIRCLES = 26


def _validate_packing(centers, radii):
    # Check the represented coordinates in binary64. Arithmetic in float16/32
    # can round an overlap or wall violation away before comparing the tolerance.
    # Do not change the original arrays or their native score reduction dtype.
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = centers.shape[0]
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
            return False
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:
                return False
    return True


_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(centers, radii, valid):
    """Six bounded shape descriptors. An invalid packing has no shape worth
    describing, so it reports zeros -- as does any degenerate input."""
    if not valid:
        return dict(_BEHAVIOR_ZERO)
    try:
        c = np.asarray(centers, dtype=float)
        r = np.asarray(radii, dtype=float)
        n = r.shape[0]
        if n == 0 or c.shape[0] != n or not np.isfinite(c).all() or not np.isfinite(r).all():
            return dict(_BEHAVIOR_ZERO)
        mean_r = float(r.mean())
        dispersion = float(r.std() / mean_r) if mean_r > 0 else 0.0

        wall = np.minimum.reduce([c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 0] - r,
                                  1.0 - c[:, 1] - r])
        boundary_frac = float((wall < 1e-6).mean())

        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        touch = d - (r[:, None] + r[None, :])
        np.fill_diagonal(touch, np.inf)
        contacts = (touch < 1e-6).sum(axis=1)
        # 6 is the hexagonal maximum for equal circles; normalise to [0,1]
        contact = float(np.clip(contacts.mean() / 6.0, 0.0, 1.0))

        total = float(r.sum())
        if total > 0:
            weighted = (c * r[:, None]).sum(axis=0) / total
        else:
            weighted = c.mean(axis=0)
        offset = float(np.clip(np.linalg.norm(weighted - 0.5) / 0.5, 0.0, 1.0))

        k = max(1, n // 4)
        big = np.sort(r)[-k:]
        concentration = float(big.sum() / total) if total > 0 else 0.0

        structure = float(len(np.unique(np.round(r, 4))) / n)

        out = {"b_dispersion": dispersion / (1.0 + dispersion), "b_boundary_frac": boundary_frac,
               "b_contact": contact, "b_offset": offset,
               "b_concentration": concentration, "b_structure": structure}
        return {k2: (float(v) if np.isfinite(v) else 0.0) for k2, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def score(centers, radii):
    # Arrays are constructed only after the public byte decoder validates them.
    valid = _validate_packing(centers, radii)
    sum_radii = float(np.sum(radii)) if valid else 0.0
    target_ratio = float(sum_radii / TARGET_VALUE) if valid else 0.0
    cases = {
        "sum_radii": sum_radii,
        "target_ratio": target_ratio,
        "validity": 1.0 if valid else 0.0,
        "combined_score": target_ratio,
    }
    cases.update(_behavior(centers, radii, valid))
    return {"ok": True, "score": target_ratio, "cases": cases}
