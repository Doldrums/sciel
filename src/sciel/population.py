"""The population view a candidate program can read while it is evaluated."""

import json
import os


def population() -> list[dict]:
    """Programs of the island the candidate's step was built from, best first, plus its parent and the run's best:
    dicts with program_id, score, parent (True for the program being changed), cases and output, the exact object
    the evaluator scored for that program. Empty when the engine gives no view."""
    path = os.environ.get("SCIEL_POPULATION")
    if not path:
        return []
    with open(path) as f:
        return json.load(f)["programs"]
