"""Minimize an unknown function on [-1, 1]^2 using at most 50 oracle queries.

Python standard library only. run_search(oracle) must finish within 5 seconds,
including candidate import. Every oracle.query(x,y) reveals one scalar value and
costs one query, including repeated points. No reset is available. Your returned
value is ignored: the evaluator scores the best point actually queried. Local
mathematics and surrogate optimization are allowed; only oracle observations are
limited. The objective's parameters and its analytic minimum are not supplied.
"""

# EVOLVE-BLOCK-START
import random


def search_algorithm(oracle):
    for _ in range(oracle.query_budget):
        oracle.query(random.uniform(-1, 1), random.uniform(-1, 1))
# EVOLVE-BLOCK-END


def run_search(oracle):
    search_algorithm(oracle)
