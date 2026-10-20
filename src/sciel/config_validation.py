"""Validate search routes and their supported combinations."""

import math



def _validate_parallel_search(cfg, defaults):
    parallel = cfg.get('parallel_search')
    if type(parallel) is not dict or set(parallel) != set(defaults['parallel_search']):
        raise ValueError('parallel_search requires only the supported execution and capacity fields')
    if type(parallel['enabled']) is not bool:
        raise ValueError('parallel_search enabled must be a boolean')
    if parallel['portfolio_schedule'] not in ('blocking', 'interleaved'):
        raise ValueError('parallel_search portfolio_schedule must be blocking or interleaved')
    if parallel['portfolio_schedule'] == 'interleaved' and (
            not parallel['enabled'] or not cfg.get('portfolio_seed')):
        raise ValueError('interleaved portfolio_schedule requires parallel_search and portfolio_seed')
    for key in ('workers', 'host_llm_slots', 'provider_slots', 'host_evaluator_slots',
                'max_inflight_per_island', 'max_inflight_per_parent'):
        if type(parallel[key]) is not int or parallel[key] < 1:
            raise ValueError('parallel_search ' + key + ' must be a positive integer')
    research_slots = parallel['host_research_slots']
    if research_slots is not None and (type(research_slots) is not int or
            not 1 <= research_slots < parallel['host_evaluator_slots']):
        raise ValueError('parallel_search host_research_slots must leave ordinary evaluator capacity')
    directory = parallel['capacity_dir']
    if directory is not None and (type(directory) is not str or not directory.strip()
                                  or '\x00' in directory):
        raise ValueError('parallel_search capacity_dir must be a nonempty path string or null')
    if not parallel['enabled']:
        return
    if type(cfg.get('num_islands')) is not int or cfg['num_islands'] < 1:
        raise ValueError('parallel_search requires positive integer num_islands')
    llm = cfg.get('llm')
    if type(llm) is not dict or type(llm.get('max_retries', 1)) is not int or llm.get('max_retries', 1) != 1:
        raise ValueError('parallel_search requires llm.max_retries=1: no hidden request replay')
    cfg['llm'] = {**llm, 'max_retries': 1}
    novelty = cfg.get('candidate_novelty')
    if type(novelty) is not dict or novelty.get('enabled') is not True:
        raise ValueError('parallel_search requires candidate_novelty enabled')
    if isinstance(cfg.get('repeat_promotion'), dict) and cfg['repeat_promotion'].get('enabled'):
        raise ValueError('parallel_search does not support repeat_promotion')
    container = cfg.get('container')
    if isinstance(container, dict) and container.get('selection') == 'centres':
        raise ValueError('parallel_search does not support centre selection without parent reservation masks')


def validate_experiments(cfg, defaults):
    if "evaluation" in cfg:
        from .eval.candidate_config import validate_policy
        cfg["evaluation"] = validate_policy(cfg["evaluation"])
    llm = cfg.get('llm')
    if isinstance(llm, dict) and llm.get('provider') == 'codex_responses':
        from .engine.llm.codex import validate_config
        cfg['llm'] = llm = validate_config(llm)
    elif isinstance(llm, dict) and llm.get('provider') == 'anthropic':
        from .engine.llm.anthropic_api import validate_config
        cfg['llm'] = llm = validate_config(llm)
    elif isinstance(llm, dict) and llm.get('provider') == 'claude_code':
        from .engine.llm.claude_code import validate_config
        cfg['llm'] = llm = validate_config(llm)
    elif isinstance(llm, dict) and llm.get('provider') not in (None, 'chat_completions'):
        raise ValueError('unsupported llm.provider')
    if isinstance(llm, dict) and 'model_capabilities' in llm:
        from .engine.llm.client import validate_model_capabilities
        models = llm.get('models') or [llm.get('primary_model', 'gemini-2.5-flash-lite')]
        llm['model_capabilities'] = validate_model_capabilities(llm['model_capabilities'], models)
    _validate_parallel_search(cfg, defaults)
    if 'research_tools' in cfg:
        from .research_tools.config import validate_config
        cfg['research_tools'] = validate_config(cfg['research_tools'])
        if cfg['research_tools']['enabled']:
            if not cfg['parallel_search']['enabled']:
                raise ValueError('research_tools requires parallel_search enabled (workers=1 is supported)')
            if not cfg['candidate_novelty']['enabled']:
                raise ValueError('research_tools requires candidate_novelty enabled')
            if cfg['llm'].get('max_retries', 1) != 1:
                raise ValueError('research_tools requires llm.max_retries=1')
    novelty = cfg.get('candidate_novelty')
    if (type(novelty) is not dict or set(novelty) != {'enabled', 'max_resamples', 'neighbors', 'output_repeats', 'output_resolution'}
            or type(novelty['enabled']) is not bool or type(novelty['output_repeats']) is not bool):
        raise ValueError('candidate_novelty requires enabled, max_resamples, neighbors and output_repeats')
    from .eval.near_output import validate_policy
    novelty['output_resolution'] = validate_policy(novelty['output_resolution'])
    if novelty['output_resolution'] is not None and not novelty['output_repeats']:
        raise ValueError('output_resolution requires output_repeats')
    if novelty['output_repeats']:
        if not novelty['enabled']:
            raise ValueError('output_repeats requires candidate_novelty enabled')
        if isinstance(cfg.get('repeat_promotion'), dict) and cfg['repeat_promotion'].get('enabled'):
            raise ValueError('output_repeats does not support repeat_promotion')
    if type(novelty['max_resamples']) is not int or not 0 <= novelty['max_resamples'] <= 2:
        raise ValueError('candidate_novelty max_resamples must be an integer from 0 to 2')
    if type(novelty['neighbors']) is not int or not 1 <= novelty['neighbors'] <= 10:
        raise ValueError('candidate_novelty neighbors must be an integer from 1 to 10')
    cap = cfg.get('max_eval_cpu_seconds')
    if cap is not None and (type(cap) is not int or cap < 1):
        raise ValueError('max_eval_cpu_seconds must be a positive integer or null')
    swm = cfg.get('swm')
    if (type(swm) is not dict or set(swm) != {'embedding_api_base', 'embedding_model'}
            or type(swm['embedding_api_base']) not in (str, type(None)) or type(swm['embedding_model']) is not str):
        raise ValueError('swm requires embedding_api_base (a URL, or null for local hashed embeddings) '
                         'and an embedding_model string')
    if cfg.get('inherit') is not None and (not isinstance(cfg['inherit'], str) or not cfg['inherit']):
        raise ValueError('inherit must be the path of a repertoire file')
    if cfg.get('warm_start') is not None:
        from .engine.warm_start import validate_warm_start
        validate_warm_start(cfg['warm_start'])
        if isinstance(cfg.get('repeat_promotion'), dict) and cfg['repeat_promotion'].get('enabled'):
            raise ValueError('warm_start supports ordinary search, not repeat_promotion')
    if cfg.get('budget') is None:
        if type(cfg.get('schedule_horizon')) is not int or cfg['schedule_horizon'] < 1:
            raise ValueError('unlimited budget requires a positive schedule_horizon')
        if isinstance(cfg.get('repeat_promotion'), dict) and cfg['repeat_promotion'].get('enabled'):
            raise ValueError('unlimited budget supports ordinary search, not repeat_promotion')
    for key in ('schedule_horizon', 'checkpoint_interval'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f'{key} must be a positive integer')
    if 'repeat_promotion' in cfg:
        from .engine.repeat_promotion import validate_config
        cfg['repeat_promotion'] = validate_config(cfg['repeat_promotion'])
        if cfg['repeat_promotion']['enabled']:
            if any(cfg.get(key) for key in ('confirm_records', 'portfolio_seed', 'dynamic_islands')):
                raise ValueError('repeat_promotion requires confirm_records, portfolio_seed and dynamic_islands off')
            if type(cfg.get('inflight_cap')) is not int or cfg['inflight_cap'] != 1:
                raise ValueError('repeat_promotion requires inflight_cap=1')
            if cfg.get('task') not in (None, 'oe/circle_packing'):
                raise ValueError('repeat_promotion supports only canonical oe/circle_packing')
    if cfg.get("max_iterations") is not None and (type(cfg["max_iterations"]) is not int
                                    or cfg["max_iterations"] < 1):
        raise ValueError("max_iterations must be a positive integer")
    for key in ("preset_selection", "preset_adaptive"):
        if key in cfg and type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if cfg.get("preset_selection") and (
            cfg["policy"].get("enabled") or cfg.get("three_head")
            or cfg.get("selection_policy", "novelty_chain") != "novelty_chain"):
        raise ValueError("preset selection requires controller/three_head off and novelty_chain")
    if "container" not in cfg:
        return
    supplied = cfg["container"]
    defaults = {"kind": "legacy", "strict_cap": True, "selection": "shared",
                "k": 4, "max_centres": 6, "seed": 0, "temperature_alpha": .02,
                "split_min_points": 4, "vq_threshold": .25, "novelty_threshold": .98}
    if not isinstance(supplied, dict) or set(supplied) - set(defaults):
        raise ValueError("invalid or unknown container configuration")
    c = {**defaults, **supplied}
    if c["kind"] not in {"legacy", "grid", "pool", "pool_novelty", "cvt", "vq", "annealed"}:
        raise ValueError("unknown container kind")
    if c["strict_cap"] is not True or c["selection"] not in {"shared", "centres"}:
        raise ValueError("container requires strict_cap=true and a known selection policy")
    if c["selection"] == "centres" and (c["kind"] not in {"cvt", "vq", "annealed"}
                                      or cfg.get("preset_selection")):
        raise ValueError("centre selection requires geometry and cannot combine presets")
    if c["selection"] == "centres" and cfg.get("selection_policy", "novelty_chain") != "novelty_chain":
        raise ValueError("centre selection currently requires novelty_chain visit accounting")
    for key in ("k", "max_centres", "split_min_points"):
        if type(c[key]) is not int or c[key] < (4 if key == "split_min_points" else 1):
            raise ValueError(f"invalid container {key}")
    if c["kind"] == "cvt" and c["k"] > c["max_centres"]:
        raise ValueError("container k cannot exceed max_centres")
    if type(c["seed"]) is not int or c["seed"] < 0:
        raise ValueError("container seed must be a nonnegative integer")
    for key in ("temperature_alpha", "vq_threshold", "novelty_threshold"):
        v = c[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"invalid container {key}")
    if c["temperature_alpha"] > 1 or c["novelty_threshold"] > 1:
        raise ValueError("container alpha and cosine threshold must be <= 1")
    if c["kind"] != "legacy" and cfg.get("embed_niche"):
        raise ValueError("experimental geometry replaces embed_niche; disable the legacy codebook")
    cfg["container"] = c
