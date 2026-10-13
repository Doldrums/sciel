"""Checkpoint policy identity, counters and restore invariants."""
from __future__ import annotations

import json
import hashlib
from copy import deepcopy


PROVENANCE_SCHEMA = "sciel-checkpoint-provenance/1"

class IdentityMismatch(Exception):
    """The run being resumed was produced by a different objective or a different
    diversity axis than the one now on disk.

    Resuming across such a change is silent corruption, not a recoverable error: the
    archived programs were scored by one evaluator and the new children by another,
    or the archive's cells were assigned by one descriptor and new cells by another,
    so the population stops being a single comparable pool. Nothing in the journal
    would say so afterwards -- both identities would just appear in the same run."""

_COMPONENTS = ("db", "schedule", "op_bandit", "reward", "selection", "intensity", "rng", "ledger", "theory", "experiments", "strategies", "swm", "learner", "procedures", "memory", "findings", "siblings", "trials")

_SCALARS = ("iteration_no", "B_evals", "budget", "seed_score", "cost_total", "llm_unpriced_calls",
            "n_events", "n_dups", "n_eval_fail", "n_nochild",
            "_last_improve_eval", "_last_pop_improve", "_last_pop_improve_iteration",
            "n_records", "_observed_trials",
            "n_remeasurements", "n_remeasurement_successes", "n_evaluation_results", "n_output_repeats")

def counters(engine):
    values = {name: getattr(engine, name) for name in _SCALARS if hasattr(engine, name)}
    if hasattr(engine, "db"):
        values["db_counter"] = engine.db._counter
    return values

def _validate_attempt_anchor(state, cfg):
    tools = cfg.get('research_tools') or {}
    required = tools.get('enabled') and 'search_attempt_stall' in tools.get('trigger_sources', [])
    if '_last_pop_improve_iteration' not in state:
        if required:
            raise IdentityMismatch('search attempt stall requires a saved population-record iteration anchor')
        return
    anchor, iteration = state['_last_pop_improve_iteration'], state.get('iteration_no')
    if (type(anchor) is not int or type(iteration) is not int
            or not 0 <= anchor <= iteration):
        raise IdentityMismatch('invalid population-record iteration anchor in checkpoint')

def _validate_research_failures(state, cfg):
    if not (cfg.get('research_tools') or {}).get('enabled'):
        return
    from .research_targets import validate_failure
    try:
        for failure in state.get('lineage_fail', {}).values():
            if failure.get('_failed_code'):
                artifacts = {k:v for k,v in failure.items()
                             if k not in ('_failed_code','_failure_provenance')}
                validate_failure(failure.get('_failure_provenance'), failure['_failed_code'], artifacts)
    except (TypeError, ValueError, AttributeError) as error:
        raise IdentityMismatch('invalid source-bound research failure provenance') from error

def _bootstrap_state(value, cfg):
    """Validate the optional cursor without guessing legacy paid-call history."""
    if value is None:
        return None
    if (type(value) is not dict or value.get('schema') != 'sciel-portfolio-bootstrap/1'
            or not cfg.get('portfolio_seed')
            or type(value.get('islands')) is not int
            or value['islands'] != cfg.get('num_islands', 3)
            or type(value.get('next_island')) is not int
            or not 0 <= value['next_island'] <= value['islands']):
        raise IdentityMismatch('invalid portfolio bootstrap cursor')
    return deepcopy(value)

def _assert_novelty_anchor(log_path, anchor):
    """Do not omit post-checkpoint source history on a nonresearch resume."""
    if (type(anchor) is not dict or type(anchor.get('byte_offset')) is not int
            or anchor['byte_offset'] < 1):
        raise ValueError('candidate novelty requires a bound journal anchor')
    remaining, digest, last = anchor['byte_offset'], hashlib.sha256(), b''
    with open(log_path, 'rb') as source:
        while remaining:
            block = source.read(min(remaining, 1024 * 1024))
            if not block:
                raise ValueError('candidate novelty journal prefix is incomplete')
            remaining -= len(block)
            digest.update(block)
            last = block[-1:]
        if last != b'\n' or digest.hexdigest() != anchor.get('prefix_sha256'):
            raise ValueError('candidate novelty journal prefix differs')
        allowed = {'checkpoint', 'run_end', 'run_start', 'checkpoint_restore'}
        for line in source:
            if not line.endswith(b'\n'):
                raise ValueError('candidate novelty journal has an incomplete tail')
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError('candidate novelty journal has an invalid tail') from exc
            if type(row) is not dict or row.get('event') not in allowed:
                raise ValueError('uncheckpointed candidate activity; refusing incomplete whole-run history')

def _identity(engine) -> dict:
    """What a resume must agree with: the objective and the diversity axis."""
    from .loop import ENGINE_CONTRACT
    identity = {
        "task": getattr(engine.task, "qualified_name", ""),
        "task_hash": getattr(engine.task, "task_hash", ""),
        "engine_contract": ENGINE_CONTRACT,
        "descriptor": getattr(engine, "descriptor_info", {}) or {},
    }
    cfg = getattr(engine, "cfg", {})
    backend = getattr(engine, "evaluation_backend", None)
    if backend is not None:
        identity["evaluation_backend"] = deepcopy(backend.identity)
    llm_cfg = cfg.get('llm', {}) or {}
    provider = getattr(engine, 'llm', None)
    capabilities = getattr(provider, 'model_capabilities', {})
    configured_capabilities = llm_cfg.get('model_capabilities', {})
    transport = getattr(provider, 'transport_identity', None)
    if llm_cfg.get('provider') == 'codex_responses' or (
            type(transport) is dict and transport.get('provider') == 'codex_responses'):
        from .llm.codex import transport_identity
        expected = transport_identity(llm_cfg)
        if transport != expected:
            raise IdentityMismatch('Codex provider differs from its configured transport')
        identity['llm_transport'] = deepcopy(transport)
    elif configured_capabilities or capabilities:
        from .llm.client import validate_model_capabilities
        models = list(getattr(provider, 'models', []))
        actual = validate_model_capabilities(capabilities, models)
        configured = validate_model_capabilities(configured_capabilities, models)
        if actual != configured:
            raise IdentityMismatch('LLM model capabilities differ from the configured transport')
        identity['llm_transport'] = {
            'schema': 'sciel-llm-transport-parameters/1', 'model_capabilities': actual,
            'models': models, 'temperature': getattr(provider, 'temperature', None),
            'api_base': getattr(provider, 'api_base', llm_cfg.get('api_base'))}
    if cfg.get('parallel_search', {}).get('enabled'):
        identity['parallel_search'] = {'contract': 'sciel-parallel-search/1',
            'config': deepcopy(cfg['parallel_search']),
            'population': {key: deepcopy(cfg.get(key)) for key in (
                'num_islands', 'max_islands', 'dynamic_islands', 'pop_cap',
                'schedule_policy', 'schedule_horizon', 'milestone_fracs',
                'sh_min_fair_evals')}}
    observed = getattr(engine, 'observed_changes', None)
    if observed is not None:
        identity['observed_changes'] = deepcopy(observed.identity)
    if cfg.get('candidate_novelty', {}).get('enabled'):
        registry = getattr(engine, 'candidate_novelty', None)
        if registry is None:
            raise IdentityMismatch('enabled candidate novelty has no whole-run registry')
        identity['candidate_novelty'] = {'config': deepcopy(cfg['candidate_novelty']),
                                         'identity': deepcopy(registry.identity)}
    if cfg.get('research_tools', {}).get('enabled'):
        tools = getattr(engine, 'research_tools', None)
        if tools is None:
            raise IdentityMismatch('enabled research tools have no controller')
        identity['research_tools'] = {'config': deepcopy(cfg['research_tools']),
                                      'identity': deepcopy(tools.identity)}
    if cfg.get('candidate_novelty', {}).get('output_repeats'):
        outputs = getattr(engine, 'output_novelty', None)
        if outputs is None:
            raise IdentityMismatch('enabled output novelty has no whole-run registry')
        identity['output_novelty'] = deepcopy(outputs.identity)
    frozen = {}
    controller = getattr(engine, 'controller', None)
    if getattr(controller, 'enabled', False) and getattr(controller, 'mode_mlp', None) is not None:
        frozen['mode_mlp'] = {'artifact': deepcopy(controller.mode_mlp.artifact_identity),
                              'policy': deepcopy(cfg.get('policy', {}))}
    three_head = getattr(engine, 'three_head', None)
    if three_head is not None:
        frozen['three_head'] = {'model': deepcopy(three_head.model.artifact_identity),
            'embedder': deepcopy(three_head.embedder.artifact_identity), 'beta': three_head.beta}
    if frozen:
        identity['frozen_policies'] = frozen
    if cfg.get("warm_start") is not None:
        identity["warm_start"] = deepcopy(cfg["warm_start"])
    if cfg.get("budget", 100) is None:
        identity["unlimited_search"] = {"schedule_horizon": cfg["schedule_horizon"]}
    if cfg.get("container", {}).get("kind", "legacy") != "legacy":
        identity["container"] = {"config": cfg["container"],
            "pop_cap": cfg.get("pop_cap", 50), "grid_bins": cfg.get("grid_bins", 10),
            "num_islands": cfg.get("num_islands", 3), "budget": cfg.get("budget", 100),
            "selection_policy": cfg.get("selection_policy", "novelty_chain"),
            "selection_floor": cfg.get("selection_floor", 1e-3),
            "tau_floor": cfg.get("tau_floor", 1e-3)}
    if cfg.get("preset_selection"):
        identity["presets"] = {"enabled": True, "adaptive": cfg.get("preset_adaptive", True)}
    repeat = getattr(engine, 'repeat_promotion', None)
    if repeat is not None:
        identity['repeat_promotion'] = {'config': cfg['repeat_promotion'], 'identity': repeat.identity}
    return identity

def check_identity(engine, state: dict) -> list[str]:
    """Differences between the checkpoint's identity and the engine's. Empty == safe.

    A checkpoint written before identity was recorded has nothing to compare against,
    so it is accepted -- refusing every historical checkpoint would be worse than the
    risk, and the journal still shows both task hashes to anyone who looks."""
    want = state.get("identity")
    if not want:
        # Historical checkpoint: no identity block. One thing is still knowable from
        # its stored config, and it is the one that matters most right now.
        cfg = state.get("config")
        want = {} if not cfg else {"config_had_axis_key": "embed_axis_path" in cfg}
    have = _identity(engine)
    out = []
    for key in ("task", "task_hash", "engine_contract"):
        if want.get(key) and want.get(key) != have.get(key):
            out.append(f"{key}: checkpoint {want.get(key)!r} != now {have.get(key)!r}")
    for key in ("llm_transport", "warm_start", "unlimited_search", "parallel_search", "candidate_novelty", "output_novelty", "observed_changes", "container", "presets", "frozen_policies", "research_tools", "repeat_promotion"):
        if want.get(key) != have.get(key):
            out.append(f"{key}: checkpoint configuration differs from current configuration")
    a = (want.get("descriptor") or {}).get("kind")
    b = (have.get("descriptor") or {}).get("kind")
    if a:
        if a != b:
            out.append(f"descriptor: checkpoint {a!r} != now {b!r}")
        elif want.get("descriptor") != have.get("descriptor"):
            # A representation swap can retain the same kind (raw embedding or
            # VQ niches) while changing every cell coordinate. Compare the full
            # recorded fingerprint/config, also for the ordinary container.
            out.append("descriptor: checkpoint frozen representation differs")
    elif want.get("container") and want.get("descriptor") != have.get("descriptor"):
        out.append("descriptor: experimental checkpoint frozen representation differs")
    elif want.get("config_had_axis_key") is False and b and b != "legacy_random_raw":
        # A checkpoint written before the descriptor was recorded. Its stored config
        # tells us anyway: `embed_axis_path` did not exist until the fitted axis did,
        # so a config without that key was produced by the legacy random direction.
        out.append(f"descriptor: checkpoint predates the fitted axis "
                   f"(legacy random direction) != now {b!r}")
    return out
