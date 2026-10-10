"""YAML inheritance, search defaults, and configuration validation."""

from __future__ import annotations

import copy
import os

import yaml

from .config_validation import validate_experiments

KNOWN_KEYS = {
    "extends", "task", "budget", "decision_epsilon", "num_islands", "pop_cap", "archive_size",
    "grid_bins", "inflight_cap", "dup_cap", "backstop_k", "random_seed",
    "schedule_policy", "milestone_fracs", "op_temperature", "op_floor",
    "op_window", "selection_floor", "tau_floor", "reward_w_pop", "reward_w_isl",
    "llm", "base_temp", "bet_enabled", "confirm_records",
    "checkpoint_on_milestone", "wandb", "sciel_tasks_revision", "op_fixed",
    "selection_policy", "selection_cap", "selection_temperature", "selection_window",
    "sciel_tasks_repo", "c2_event_bets", "plateau_shadow", "policy",
    "sh_min_fair_evals", "log_population",
    "embed_grid", "embed_grid_path", "embed_axis_path",
    "crossover", "portfolio_seed", "dynamic_islands", "max_islands", "spawn_stall",
    "idea_island", "kb_free",
    "refine_stall", "refine_elite_frac", "op_intensify_boost",
    "embed_niche", "embed_niche_path", "niche_d_min", "niche_target_k",
    "niche_k_max", "niche_csc_rate",
    "three_head", "three_head_path", "three_head_beta", "log_evictions",
    "container", "max_iterations", "preset_selection", "preset_adaptive",
    "schedule_horizon", "checkpoint_interval", "warm_start", "candidate_novelty", "parallel_search",
    "evaluation", "research", "research_tools", "idea_two_stage", "idea_card_selection", "repeat_promotion", "repeat_population", "circle_intervention", "circle_generation",
}

# Accepted in old checkpoints, but no longer used.
DEPRECATED_KEYS = {"inject_entrypoints", "reflection_interval"}

DEFAULTS = {
    "num_islands": 3,
    "budget": 100,
    "decision_epsilon": 0.0,
    "random_seed": 0,
    "dup_cap": 4,
    "candidate_novelty": {"enabled": False, "max_resamples": 2, "neighbors": 3,
                          "output_repeats": False, "output_resolution": None},
    "parallel_search": {"enabled": False, "workers": 4, "host_llm_slots": 12,
        "provider_slots": 12, "host_evaluator_slots": 2, "host_research_slots": None, "capacity_dir": None,
        "max_inflight_per_island": 1, "max_inflight_per_parent": 1,
        "portfolio_schedule": "blocking"},
    "backstop_k": 10,
    "schedule_policy": "sh",
    "milestone_fracs": [0.5, 0.75],
    "inflight_cap": 4,
    "bet_enabled": False,
    "confirm_records": False,
    "sh_min_fair_evals": 16,
    "log_population": True,
    "crossover": True,
    "portfolio_seed": True,
    "idea_island": True,
    "dynamic_islands": True,
    "max_islands": 6,
    "spawn_stall": 12,
    # Bias operators toward refinement once the current family stalls.
    "refine_stall": 8,
    "refine_elite_frac": 0.9,
    "op_intensify_boost": 0.5,
    "embed_grid": False,
    "embed_grid_path": "../sciel-artifacts/code_embed_v3.pt",
    # Explicit null selects the legacy random descriptor; configured axes are required.
    "embed_axis_path": "../sciel-artifacts/behavior_axis/{task}_v3.npz",
    "embed_niche": False,
    "embed_niche_path": "../sciel-artifacts/code_embed_v2.pt",
    "niche_d_min": 0.35,
    "niche_target_k": 6,
    "niche_k_max": 12,
    "niche_csc_rate": 0.08,
    "llm": {"max_tokens": 32000},
    "policy": {
        "enabled": True,
        "base_mix": {"base": 0.8, "exploit": 0.2},
        "plateau_shift": True,
        "plateau_mix": {"base": 0.6, "exploit": 0.4},
        "plateau_hold": 40,
        "plateau_threshold": 0.5,
        "plateau_horizon": 40,
        "plateau_min_n": 12,
        "tilt_beta": 0.3,
        "elite_k": 3,
    },
}


def apply_defaults(cfg: dict) -> dict:
    """Layer the espalier DEFAULTS under a user config (deep-merge, same
    semantics as `extends`), dropping DEPRECATED_KEYS first. Idempotent.

    Mode mixes are the exception: a mix is a complete distribution, so a
    user-provided base_mix/plateau_mix replaces the default outright --
    deep-merging would resurrect default modes the config deliberately
    dropped (base_mix {base: 1.0} must mean pure base, not 1.0/0.2)."""
    cfg = {k: v for k, v in cfg.items() if k not in DEPRECATED_KEYS}
    # yaml `policy:` with no value parses to None -- normalize before merging
    for key in ("policy", "llm", "wandb"):
        if key in cfg and cfg[key] is None:
            cfg[key] = {}
    out = _merge(DEFAULTS, cfg)
    validate_experiments(out, DEFAULTS)
    return out


def _merge(base: dict, over: dict, _path: tuple = ()) -> dict:
    """Merge settings, replacing complete policy distributions at every layer."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        path = _path + (k,)
        distribution = path in (("policy", "base_mix"), ("policy", "plateau_mix"))
        if isinstance(v, dict) and isinstance(out.get(k), dict) and not distribution:
            out[k] = _merge(out[k], v, path)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str, _seen: tuple = ()) -> dict:
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"config extends cycle at {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = os.path.join(os.path.dirname(path), parent)
        base = load_config(parent_path, _seen + (path,))
        cfg = _merge(base, cfg)
    return cfg


def validate(cfg: dict) -> dict:
    unknown = set(cfg) - KNOWN_KEYS - DEPRECATED_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    if "-s" in cfg or "seed" in cfg:
        raise ValueError("seed is set only via overlay {random_seed: N}")
    return apply_defaults(cfg)
