"""Modules a task gives its candidate programs to import while they are evaluated."""

import os

from sciel.candidate_modules import load_candidate


def asset(name: str):
    """The module assets/<name>.py of the task evaluating this program, such as the task's own exact objective, so a
    program builds on it instead of rewriting it. The assets are part of the task's identity."""
    root = os.environ.get("SCIEL_TASK_ROOT")
    if not root:
        raise ImportError("no task is evaluating this program")
    return load_candidate(os.path.join(root, "assets", name + ".py"), name)
