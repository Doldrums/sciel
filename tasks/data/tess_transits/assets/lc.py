"""Light-curve tools for transit search: detrending, a NumPy box least squares search and eclipsing-binary checks.

Curves are TESS-SPOC 200-s FFI photometry binned to 10 minutes: time in days, flux normalized to a median of 1.
"""

import numpy as np
from scipy.ndimage import median_filter

BIN_DAYS = 10.0 / 1440.0


def robust_std(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def detrend(time, flux, window=1.0) -> np.ndarray:
    """flux divided by a running median over `window` days, taken on the regular 10-minute grid across gaps."""
    t, f = np.asarray(time, dtype=np.float64), np.asarray(flux, dtype=np.float64)
    i = np.rint((t - t[0]) / BIN_DAYS).astype(np.int64)
    grid = np.full(i[-1] + 1, np.nan)
    grid[i] = f
    filled = np.where(np.isnan(grid), np.interp(np.arange(grid.size), i, f), grid)
    trend = median_filter(filled, size=max(3, int(window / BIN_DAYS) | 1), mode="nearest")[i]
    return f / trend


def period_grid(time, pmin=0.5, pmax=13.0, oversample=2, qmin=0.015) -> np.ndarray:
    """Periods uniform in frequency, spaced so the shortest box shifts by a fraction of its width over the baseline."""
    span = float(np.ptp(time))
    df = qmin / (oversample * span)
    return 1.0 / np.arange(1.0 / pmax, 1.0 / pmin, df)[::-1]


def bls(time, flux, periods=None, durations=(0.04, 0.06, 0.09, 0.13, 0.18, 0.25), nbins=400) -> dict:
    """Box least squares on a detrended curve. Returns arrays over periods of the best power, depth, duration and
    epoch, with power the box signal-to-noise depth / sigma * sqrt(points in transit)."""
    t = np.asarray(time, dtype=np.float64)
    y = 1.0 - np.asarray(flux, dtype=np.float64)
    y = y - np.median(y)
    sigma = robust_std(y) or 1e-6
    periods = period_grid(t) if periods is None else np.asarray(periods, dtype=np.float64)
    t0 = t[0]
    out = {k: np.zeros(periods.size) for k in ("power", "depth", "duration", "epoch")}
    widths = np.asarray(durations, dtype=np.float64)
    for k, p in enumerate(periods):
        phase = ((t - t0) / p) % 1.0
        b = np.minimum((phase * nbins).astype(np.int64), nbins - 1)
        n = np.bincount(b, minlength=nbins).astype(np.float64)
        s = np.bincount(b, weights=y, minlength=nbins)
        cn, cs = np.concatenate([[0.0], np.cumsum(np.tile(n, 2))]), np.concatenate([[0.0], np.cumsum(np.tile(s, 2))])
        w = np.clip(np.rint(widths / p * nbins).astype(np.int64), 1, nbins // 2 - 1)
        idx = np.arange(nbins)[None, :] + w[:, None]
        nn, ss = cn[idx] - cn[None, :nbins], cs[idx] - cs[None, :nbins]
        snr = np.where(nn > 2, ss / np.sqrt(np.maximum(nn, 1.0)) / sigma, -np.inf)
        d, j = np.unravel_index(int(np.argmax(snr)), snr.shape)
        out["power"][k], out["depth"][k] = snr[d, j], ss[d, j] / max(nn[d, j], 1.0)
        out["duration"][k], out["epoch"][k] = widths[d], t0 + (j + w[d] / 2.0) / nbins * p
    out["period"] = periods
    return out


def peaks(result, k=3, alias_tol=0.01) -> list:
    """Up to k (period, epoch, duration, power) at the strongest periods that are not within alias_tol of an
    integer ratio (1, 2, 3, 1/2, 1/3) of one already taken."""
    order = np.argsort(result["power"])[::-1]
    taken = []
    for j in order:
        p = result["period"][j]
        if any(abs(p / q * r - 1.0) < alias_tol for q, *_ in taken for r in (1, 2, 3, 0.5, 1 / 3)):
            continue
        taken.append((float(p), float(result["epoch"][j]), float(result["duration"][j]), float(result["power"][j])))
        if len(taken) == k:
            break
    return taken


def fold(time, period, epoch) -> np.ndarray:
    """Phase in days from the nearest transit, in [-period/2, period/2)."""
    return (np.asarray(time) - epoch + period / 2.0) % period - period / 2.0


def odd_even(time, flux, period, epoch, duration) -> float:
    """Depth difference of odd and even transits in units of its error; large values point to an eclipsing binary
    at twice the period."""
    t, y = np.asarray(time), 1.0 - np.asarray(flux)
    n = np.floor((t - epoch) / period + 0.5).astype(np.int64)
    inside = np.abs(fold(t, period, epoch)) < duration / 2.0
    sigma = robust_std(y) or 1e-6
    odd, even = y[inside & (n % 2 == 1)], y[inside & (n % 2 == 0)]
    if odd.size < 3 or even.size < 3:
        return 0.0
    return float(abs(odd.mean() - even.mean()) / (sigma * np.sqrt(1.0 / odd.size + 1.0 / even.size)))


def secondary(time, flux, period, epoch, duration) -> float:
    """Signal-to-noise of a dip at phase 0.5; a significant one points to an eclipsing binary."""
    t, y = np.asarray(time), 1.0 - np.asarray(flux)
    inside = np.abs(fold(t, period, epoch + period / 2.0)) < duration / 2.0
    sigma = robust_std(y) or 1e-6
    return float(y[inside].mean() / sigma * np.sqrt(inside.sum())) if inside.sum() > 2 else 0.0
