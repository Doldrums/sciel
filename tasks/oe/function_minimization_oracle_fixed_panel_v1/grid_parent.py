"""Frozen public baseline: grid25 + local axis20 + seeded global5.

Ties keep the first queried point. Each local round freezes its centre before
four clipped axis probes; repetitions still consume queries. No private seed.
Candidate evaluation provides Python standard library, NumPy 2.3.5, SciPy 1.16.3,
scikit-learn 1.7.2, joblib 1.5.2 and threadpoolctl 3.6.0.
Research tools use their separately selected runtime image;
do not assume this candidate dependency contract applies to that image.
The 50-query limit and 5-second window include candidate/library imports.
The trusted bootstrap seeds Python random and legacy numpy.random before import
from the public panel seed; default_rng() needs an explicit seed for repeatability.
"""

import random


def run_search(oracle):
    best, point = float("inf"), None

    def query(x, y):
        nonlocal best, point
        x, y = max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y))
        value = oracle.query(x, y)
        if value < best:
            best, point = value, (x, y)

    for x in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for y in (-1.0, -0.5, 0.0, 0.5, 1.0):
            query(x, y)
    for radius in (0.25, 0.125, 0.0625, 0.03125, 0.015625):
        x, y = point
        for dx, dy in ((radius, 0), (-radius, 0), (0, radius), (0, -radius)):
            query(x + dx, y + dy)
    for _ in range(5):
        query(random.uniform(-1, 1), random.uniform(-1, 1))
