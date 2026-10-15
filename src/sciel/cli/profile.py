"""The default profile: a working configuration from nothing but a task and an API key in the environment."""

from __future__ import annotations

import os
from pathlib import Path

KEYS_FILE = Path.home() / ".config" / "sciel" / "keys"

PROVIDERS = (
    ("ANTHROPIC_API_KEY", "Anthropic", "claude-sonnet-5-5",
     {"provider": "anthropic", "models": ["claude-sonnet-5-5"], "api_key_env": "ANTHROPIC_API_KEY",
      "reasoning_effort": "medium", "max_tokens": 32000, "timeout": 600}),
    ("OPENROUTER_API_KEY", "OpenRouter", "anthropic/claude-sonnet-5.5",
     {"models": ["anthropic/claude-sonnet-5.5"], "weights": [1.0], "api_base": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY", "reasoning_effort": "medium", "max_tokens": 32000, "timeout": 600,
      "model_capabilities": {"anthropic/claude-sonnet-5.5": {"temperature": False}}, "max_retries": 1}),
)

BASE = {"num_islands": 3, "pop_cap": 50, "archive_size": 20, "grid_bins": 10, "inflight_cap": 4, "dup_cap": 4,
        "backstop_k": 10, "schedule_policy": "sh", "milestone_fracs": [0.5, 0.75], "op_temperature": 0.5,
        "op_floor": 0.05, "op_window": 20, "selection_floor": 0.001, "tau_floor": 0.001, "reward_w_pop": 1.0,
        "reward_w_isl": 1.0, "base_temp": 0.7, "confirm_records": False, "checkpoint_on_milestone": True,
        "embed_grid": False, "candidate_novelty": {"enabled": True},
        "swm": {"embedding_api_base": None, "embedding_model": "hashed-tokens"},
        "parallel_search": {"enabled": True, "workers": 4}}


def load_keys():
    """Fill missing provider keys from the user's saved keys file."""
    try:
        lines = KEYS_FILE.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        name, sep, value = line.partition("=")
        if sep and name.strip() and value.strip():
            os.environ.setdefault(name.strip(), value.strip())


def save_key(name: str, value: str):
    KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)
    kept = [line for line in (KEYS_FILE.read_text().splitlines() if KEYS_FILE.exists() else [])
            if not line.startswith(name + "=")]
    fd = os.open(KEYS_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write("\n".join(kept + [f"{name}={value}"]) + "\n")


def detect():
    """The first provider whose key is available, as (env name, label, model, llm config), or None."""
    load_keys()
    for env, label, model, llm in PROVIDERS:
        if os.environ.get(env):
            return env, label, model, dict(llm)
    return None


def default_config(task: str, llm: dict, *, budget: int = 40, seed: int = 0) -> dict:
    from ..config import validate
    return validate({**BASE, "task": task, "budget": budget, "random_seed": seed, "llm": llm})
