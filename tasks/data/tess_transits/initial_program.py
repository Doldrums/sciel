# EVOLVE-BLOCK-START
import numpy as np
from sciel.task import asset

lc = asset("lc")


def detect(time, flux, flux_err) -> list:
    """
    Search one TESS light curve (10-minute bins, flux normalized to 1) for periodic transits of a planet.

    Returns:
        signals: up to 3 (period_days, epoch_days, duration_days, score) tuples, strongest first; a higher score
        means more confidence that the signal is a transiting planet rather than noise or an eclipsing binary.
    """
    flat = lc.detrend(time, flux, window=1.0)
    found = lc.peaks(lc.bls(time, flat), k=3)
    return [(p, t0, d, power) for p, t0, d, power in found]


# EVOLVE-BLOCK-END
