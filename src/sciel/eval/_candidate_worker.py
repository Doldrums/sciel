"""Untrusted candidate-side coercion; the host independently decodes and scores.

This file is an immutable method-bundle entrypoint, not a host evaluator. The
method loader namespaces the sole candidate.py sibling. No scorer, task root,
credential or host path is needed in the bundle. The returned JSON string is
still untrusted; the separate supervisor and host own receipts and scoring.
"""
import json
import os
import sys

import numpy as np

SCHEMA = "sciel-candidate-values/1"
MAX_BYTES = 128 * 1024
N_CIRCLES = 26
MAX_SUPPORT = 2000


def _as_text(value):
    """Format a program-supplied number without ever raising on it."""
    try:
        return f"{float(value):.6f}"
    except Exception:
        try:
            return repr(value)[:80]
        except Exception:
            return "<unprintable reported_sum>"


def _snapshot(centers, radii):
    """Detach the exact finite real arrays used for both scoring and capture.

    Preserve floating dtypes: converting float32 radii to float64 before np.sum
    would silently change the native metric. Reject unsupported/rounded payloads
    instead of reporting geometry different from the one actually measured.
    """
    values = []
    for name, value, shape in (("centers", centers, (N_CIRCLES, 2)),
                               ("radii", radii, (N_CIRCLES,))):
        array = np.array(value, copy=True)
        if array.shape != shape:
            raise ValueError(f"Invalid shapes: {name}={array.shape}, expected {shape}")
        if array.dtype.kind not in "iuf" or array.dtype.itemsize > 8:
            raise ValueError(f"{name} must contain supported real numbers")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        # Python float/int equality checks exact representability for integers.
        if any(x.item() != float(x) for x in array.flat):
            raise ValueError(f"{name} cannot be represented exactly as binary64 JSON")
        # Integer wall/contact arithmetic can wrap (uint8: 128+128 == 0),
        # admitting an impossible packing. Exactly represented integer inputs
        # use float64; floating inputs retain their original reduction dtype.
        if array.dtype.kind in "iu":
            array = array.astype(float)
        values.append(array)
    return values

def _circle(candidate, stage):
    # Preserve the legacy _run conversion before its detached _snapshot.
    centers, radii, reported_sum = candidate.run_packing()
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)
    centers, radii = _snapshot(centers, radii)
    return {
        "centers": centers.astype(float).tolist(),
        "radii": radii.astype(float).tolist(),
        "centers_dtype": centers.dtype.name,
        "radii_dtype": radii.dtype.name,
        # Legacy smoke does not call untrusted numeric/repr formatting hooks.
        "reported_sum": _as_text(reported_sum) if stage == "full" else "",
    }


def _heil(candidate):
    points = np.array(candidate.heilbronn_triangle11(), dtype=np.float64, copy=True)
    if points.shape != (11, 2):
        raise ValueError(f"Invalid shapes: points = {points.shape}, expected (11, 2)")
    if not np.isfinite(points).all():
        raise ValueError("Points must contain only finite coordinates.")
    return {"points": points.tolist()}


def _beat(candidate):
    result = candidate.beat_the_average_measure()
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("Constructor must return a (support, weights) pair.")
    support, weights = result
    support = np.asarray(support, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if support.shape != weights.shape:
        raise ValueError("support and weights must have the same shape.")
    if support.size < 1:
        raise ValueError("Measure must have at least one support point.")
    if support.size > MAX_SUPPORT:
        raise ValueError(f"support has {support.size} points; the limit is {MAX_SUPPORT}")
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        raise ValueError("support/weights contain NaN or infinite values.")
    # Domain validation and both normalization passes belong to trusted scoring.
    return {"support": support.tolist(), "weights": weights.tolist()}


def run(inputs):
    if sys.platform != "linux" or os.geteuid() != 10001 or os.getpid() == 1:
        raise RuntimeError("candidate execution requires the unprivileged Linux container child")
    if type(inputs) is not dict or set(inputs) != {"kind", "stage", "binding_sha256"}:
        raise ValueError("invalid candidate adapter input fields")
    kind, stage, binding = inputs["kind"], inputs["stage"], inputs["binding_sha256"]
    if type(kind) is not str or kind not in {"circle", "heil", "beat"}:
        raise ValueError("unsupported candidate adapter")
    if type(stage) is not str or stage not in {"smoke", "full"}:
        raise ValueError("unsupported candidate invocation stage")
    if (type(binding) is not str or len(binding) != 64
            or any(char not in "0123456789abcdef" for char in binding)):
        raise ValueError("invalid host binding digest")
    # BundleLoader resolves this exact sibling under _sciel_bundle_<method_id>.
    # No candidate code is imported before the container and input guards.
    import candidate
    if kind == "circle":
        values = _circle(candidate, stage)
    elif kind == "heil":
        values = _heil(candidate)
    else:
        values = _beat(candidate)
    encoded = json.dumps({"schema": SCHEMA, "kind": kind, "values": values},
                         sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > MAX_BYTES:
        raise ValueError("candidate values exceed the byte limit")
    return encoded
