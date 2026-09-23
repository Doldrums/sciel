"""Minimize an unknown function on [-1, 1]^2 using at most 50 oracle queries.

Candidate evaluation provides Python standard library, NumPy 2.3.5, SciPy 1.16.3,
scikit-learn 1.7.2, joblib 1.5.2 and threadpoolctl 3.6.0.
Research tools use their separately selected runtime image;
do not assume this candidate dependency contract applies to that image.
run_search(oracle) must finish within 5 seconds,
including candidate import. Every oracle.query(x,y) reveals one scalar value and
costs one query, including repeated points. No reset is available. Your returned
value is ignored: the evaluator scores the best point actually queried. Local
mathematics and surrogate optimization are allowed; only oracle observations are
limited. The objective's parameters and its analytic minimum are not supplied.
The trusted bootstrap seeds Python random and the legacy numpy.random global
RNG before this file is imported, using the public fixed panel schedule.
New numpy.random.default_rng() instances need an explicit seed for repeatability;
OS entropy and explicit reseeding are not fixed. Library import time counts.
"""

# EVOLVE-BLOCK-START
import random


def search_algorithm(oracle):
    for _ in range(oracle.query_budget):
        oracle.query(random.uniform(-1, 1), random.uniform(-1, 1))


# EVOLVE-BLOCK-END


def run_search(oracle):
    search_algorithm(oracle)
