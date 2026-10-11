"""Validate search routes and their supported combinations."""

import math

from .research_config import validate_research


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
    for route in ('repeat_population', 'repeat_promotion', 'circle_generation', 'circle_intervention'):
        if isinstance(cfg.get(route), dict) and cfg[route].get('enabled'):
            raise ValueError('parallel_search does not support ' + route)
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
    if novelty['enabled']:
        for route in ('repeat_population', 'circle_generation', 'circle_intervention'):
            value = cfg.get(route)
            if isinstance(value, dict) and value.get('enabled'):
                raise ValueError('candidate_novelty does not support ' + route)
    if cfg.get('inherit') is not None and (not isinstance(cfg['inherit'], str) or not cfg['inherit']):
        raise ValueError('inherit must be the path of a repertoire file')
    if cfg.get('warm_start') is not None:
        from .engine.warm_start import validate_warm_start
        validate_warm_start(cfg['warm_start'])
        if any(isinstance(cfg.get(key), dict) and cfg[key].get('enabled')
               for key in ('repeat_promotion', 'repeat_population',
                           'circle_generation', 'circle_intervention')):
            raise ValueError('warm_start supports ordinary search, not repeat or circle-specialized routes')
    if cfg.get('budget') is None:
        if type(cfg.get('schedule_horizon')) is not int or cfg['schedule_horizon'] < 1:
            raise ValueError('unlimited budget requires a positive schedule_horizon')
        if any(isinstance(cfg.get(key), dict) and cfg[key].get('enabled')
               for key in ('repeat_population', 'repeat_promotion')):
            raise ValueError('unlimited budget supports ordinary search, not repeat routes')
    for key in ('schedule_horizon', 'checkpoint_interval'):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f'{key} must be a positive integer')
    if 'repeat_population' in cfg:
        from .engine.repeat_promotion import validate_config
        cfg['repeat_population'] = validate_config(cfg['repeat_population'])
        if cfg['repeat_population']['enabled']:
            forbidden = ('confirm_records', 'portfolio_seed', 'dynamic_islands',
                'idea_island', 'crossover', 'embed_grid', 'embed_niche', 'three_head',
                'preset_selection', 'plateau_shadow')
            if any(cfg.get(key) for key in forbidden):
                raise ValueError('repeat_population requires static pools and legacy adaptation off')
            if (cfg.get('policy', {}).get('enabled') or cfg.get('policy', {}).get('mode_bandit')
                    or cfg.get('policy', {}).get('mode_source') or cfg.get('policy', {}).get('plateau_shadow')):
                raise ValueError('repeat_population requires policy adaptation off')
            companions = ('repeat_promotion', 'circle_intervention', 'idea_two_stage', 'idea_card_selection')
            if any(cfg.get(key) is not None and not isinstance(cfg[key], dict) for key in companions):
                raise ValueError('repeat_population companion policies must be mappings')
            if any((cfg.get(key) or {}).get('enabled') for key in companions):
                raise ValueError('repeat_population cannot combine other repeat/intervention/idea policies')
            if not cfg.get('research', {}).get('enabled'):
                raise ValueError('repeat_population requires research')
            if cfg.get('op_fixed') not in ('diff', 'full_rewrite', 'circle_generate'):
                raise ValueError('repeat_population requires a supported fixed generation operator')
            if cfg.get('task') not in (None, 'oe/circle_packing'):
                raise ValueError('repeat_population supports only pinned Circle tasks')
            if (type(cfg.get('inflight_cap')) is not int or cfg['inflight_cap'] != 1
                    or type(cfg.get('dup_cap')) is not int or cfg['dup_cap'] != 1):
                raise ValueError('repeat_population requires inflight_cap=dup_cap=1')
            if cfg.get('schedule_policy') != 'round_robin' or cfg.get('milestone_fracs') != []:
                raise ValueError('repeat_population requires round_robin without SH/migration')
            if (cfg.get('container', {}).get('kind') != 'pool'
                    or cfg.get('container', {}).get('selection', 'shared') != 'shared'
                    or cfg.get('archive_size', 0) != 0):
                raise ValueError('repeat_population requires bounded pool and archive_size=0')
            for key, minimum in (('num_islands', 1), ('pop_cap', 2), ('budget', 1)):
                value = cfg.get(key, 50 if key == 'pop_cap' else None)
                if type(value) is not int or value < minimum:
                    raise ValueError('invalid repeat_population ' + key)
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
    validate_research(cfg)
    if 'circle_generation' in cfg:
        supplied = cfg['circle_generation']
        if (type(supplied) is not dict or set(supplied) - {'enabled', 'proposal_policy', 'execution_profile'}
                or type(supplied.get('enabled', False)) is not bool):
            raise ValueError('circle_generation requires boolean enabled and an optional proposal_policy')
        cfg['circle_generation'] = {'enabled': supplied.get('enabled', False)}
        if 'execution_profile' in supplied:
            if (not supplied.get('enabled', False)
                    or type(supplied['execution_profile']) is not str
                    or supplied['execution_profile'] not in ('paired_restart_trace/1', 'paired_restart_trace/2')):
                raise ValueError('circle_generation execution_profile requires enabled paired_restart_trace/1 or /2')
            if not any(type(cfg.get(key)) is dict and cfg[key].get('enabled') is True
                       for key in ('repeat_promotion', 'repeat_population')):
                raise ValueError('paired restart tracing requires an enabled repeat route')
            if cfg.get('research', {}).get('initializer_witness', False):
                raise ValueError('paired restart tracing does not support the legacy initializer_witness source contract')
            cfg['circle_generation']['execution_profile'] = supplied['execution_profile']
        if 'proposal_policy' in supplied:
            if (not supplied.get('enabled', False)
                    or supplied['proposal_policy'] != 'first_eligible_of_two'
                    or type(supplied['proposal_policy']) is not str):
                raise ValueError('circle_generation proposal_policy requires enabled first_eligible_of_two')
            cfg['circle_generation']['proposal_policy'] = supplied['proposal_policy']
        if cfg['circle_generation']['enabled']:
            if not cfg.get('research', {}).get('enabled') or cfg.get('op_fixed') != 'circle_generate':
                raise ValueError('circle_generation requires research and op_fixed=circle_generate')
            if cfg.get('task') != 'oe/circle_packing':
                raise ValueError('circle_generation supports only canonical oe/circle_packing')
            if any(cfg.get(k) for k in ('portfolio_seed', 'dynamic_islands', 'idea_island', 'confirm_records')):
                raise ValueError('circle_generation requires portfolio_seed, dynamic_islands, idea_island and confirm_records off')
            if type(cfg.get('inflight_cap')) is not int or cfg['inflight_cap'] != 1:
                raise ValueError('circle_generation requires inflight_cap=1')
            if any(isinstance(cfg.get(k), dict) and cfg[k].get('enabled')
                   for k in ('idea_two_stage', 'idea_card_selection', 'circle_intervention')):
                raise ValueError('circle_generation cannot combine idea or legacy circle_intervention')
    if cfg.get('op_fixed') == 'circle_generate' and not cfg.get('circle_generation', {}).get('enabled'):
        raise ValueError('op_fixed=circle_generate requires enabled circle_generation')
    if cfg.get('research', {}).get('restart_predictions') and not (
            cfg.get('research', {}).get('enabled')
            and cfg.get('circle_generation', {}).get('enabled')
            and cfg['circle_generation'].get('execution_profile') == 'paired_restart_trace/2'):
        raise ValueError('restart_predictions requires enabled Circle paired_restart_trace/2')
    if any(key in cfg.get('research', {}) for key in ('external_history_sha256', 'external_component_sha256', 'external_component_review_sha256', 'external_evaluated_study_sha256', 'external_evaluated_study_authority_sha256', 'external_evaluated_review_sha256', 'source_reflection_config_sha256')) and not (
            cfg['research'].get('enabled') and cfg['research'].get('restart_predictions')
            and cfg.get('circle_generation', {}).get('enabled')
            and cfg['circle_generation'].get('execution_profile') == 'paired_restart_trace/2'
            and 'proposal_policy' not in cfg['circle_generation']):
        raise ValueError('external history requires single-body Circle paired_restart_trace/2 with restart_predictions')
    if cfg.get('research', {}).get('initializer_witness') and not (
            cfg.get('research', {}).get('enabled') and cfg.get('circle_generation', {}).get('enabled')):
        raise ValueError('research initializer_witness requires enabled research and circle_generation')
    if 'circle_intervention' in cfg:
        supplied = cfg['circle_intervention']
        if (type(supplied) is not dict or set(supplied) - {'enabled', 'operation'}
                or type(supplied.get('enabled', False)) is not bool):
            raise ValueError('circle_intervention accepts boolean enabled and an optional operation')
        cfg['circle_intervention'] = {'enabled': supplied.get('enabled', False)}
        if 'operation' in supplied:
            operation = supplied['operation']
            if type(operation) is not str or operation not in (
                    'circle_restart_reference/1', 'circle_restart_scale/1'):
                raise ValueError('unknown circle_intervention operation')
            cfg['circle_intervention']['operation'] = operation
        if cfg['circle_intervention']['enabled']:
            if not cfg.get('research', {}).get('enabled') or cfg.get('op_fixed') != 'intervention':
                raise ValueError('circle_intervention requires research and op_fixed=intervention')
            if any(cfg.get(k) for k in ('portfolio_seed', 'dynamic_islands', 'idea_island', 'confirm_records')):
                raise ValueError('circle_intervention requires portfolio_seed, dynamic_islands, idea_island and confirm_records off')
            if type(cfg.get('inflight_cap')) is not int or cfg['inflight_cap'] != 1:
                raise ValueError('circle_intervention requires inflight_cap=1')
            if any(isinstance(cfg.get(k), dict) and cfg[k].get('enabled')
                   for k in ('idea_two_stage', 'repeat_promotion')):
                raise ValueError('circle_intervention cannot combine idea_two_stage or repeat_promotion')
    if cfg.get('op_fixed') == 'intervention' and not cfg.get('circle_intervention', {}).get('enabled'):
        raise ValueError('op_fixed=intervention requires enabled circle_intervention')
    if 'idea_two_stage' in cfg:
        idea = cfg['idea_two_stage']
        if (type(idea) is not dict or set(idea) - {'enabled', 'card_repair_attempts'}
                or type(idea.get('enabled', False)) is not bool):
            raise ValueError('idea_two_stage accepts boolean enabled and optional card_repair_attempts')
        if (type(idea.get('card_repair_attempts', 0)) is not int
                or not 0 <= idea.get('card_repair_attempts', 0) <= 1):
            raise ValueError('idea_two_stage card_repair_attempts must be an integer from 0 to 1')
        cfg['idea_two_stage'] = {'enabled': idea.get('enabled', False),
            **({'card_repair_attempts': idea['card_repair_attempts']} if 'card_repair_attempts' in idea else {})}
        if cfg['idea_two_stage']['enabled'] and not cfg.get('research', {}).get('enabled'):
            raise ValueError('idea_two_stage requires enabled research for durable no-replay provenance')
    if 'idea_card_selection' in cfg:
        supplied = cfg['idea_card_selection']
        defaults = {'enabled': False, 'beta': .5, 'neighbors': 3}
        if type(supplied) is not dict or set(supplied) - set(defaults):
            raise ValueError('invalid or unknown idea_card_selection configuration')
        card = {**defaults, **supplied}
        if type(card['enabled']) is not bool:
            raise ValueError('idea_card_selection enabled must be boolean')
        beta = card['beta']
        if (isinstance(beta, bool) or not isinstance(beta, (int, float))
                or not math.isfinite(beta) or not 0 < beta <= 1):
            raise ValueError('idea_card_selection beta must be finite and in (0, 1]')
        if type(card['neighbors']) is not int or not 1 <= card['neighbors'] <= 1000:
            raise ValueError('idea_card_selection neighbors must be an integer in [1, 1000]')
        if card['enabled']:
            if not cfg.get('idea_two_stage', {}).get('enabled') or not cfg.get('idea_island'):
                raise ValueError('idea_card_selection requires idea_two_stage and idea_island')
            if cfg.get('op_fixed') not in (None, '', 'idea'):
                raise ValueError('idea_card_selection cannot combine a non-idea fixed operator')
            if (cfg.get('selection_policy', 'novelty_chain') != 'novelty_chain'
                    or cfg.get('container', {}).get('selection', 'shared') != 'shared'
                    or cfg['policy'].get('enabled') or cfg['policy'].get('mode_bandit')
                    or cfg['policy'].get('mode_source') or cfg.get('preset_selection')
                    or cfg.get('three_head')):
                raise ValueError('idea_card_selection requires shared novelty_chain and all other tilts off')
        cfg['idea_card_selection'] = card
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
