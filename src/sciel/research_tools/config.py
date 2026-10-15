"""Explicit, bounded optional research-tool episodes in the ordinary Engine."""
from copy import deepcopy
import math
import re

DEFAULTS = {
    'enabled': False,
    # Selected-island visits since its last actual frozen episode, never since
    # run start. Healthy pre-trigger visits do not postpone the first episode.
    'interval': 4,
    # Optional run-wide cadence of committed ordinary attempts; bootstrap is
    # excluded. None preserves the stagnation/maintenance-only policy.
    'proactive_every_attempts': None,
    'attempt_stall_threshold': 8,
    'trigger_sources': ['population_stall', 'controller_plateau'],
    'allow_shadow_triggers': False,
    'max_active_cells': 4,
    'max_requests': 2,
    # Scientific trials per opportunity, separate from repair attempts per trial.
    # One final interpretation/close decision is allowed after the final trial.
    'max_scientific_steps': 3,
    # Optional cooperative return to search after a successful foreground trial;
    # the research question remains open for a later opportunity.
    'yield_after_success': False,
    # Queue cell work independently of ordinary island/parent selection.
    'independent_research': False,
    'research_jobs_per_wave': 1,
    'experimental_protocol': False,
    'shared_observations': False,
    'model': None,
    'temperature': 0.4,
    'share_between_islands': True,
    'max_visible_tools': 8,
    'max_source_chars': 32000,
    'max_context_chars': 12000,
    'max_prompt_chars': 64000,
    'max_response_chars': 80000,
    'image_id': None,
    # Environment variable naming a directory mounted read-only at /data for tools.
    'data_env': None,
    'jobs': {'enabled': False, 'max_active': 1, 'max_pending': 16,
             'max_cpu_seconds': 3600.0, 'max_wall_seconds': 7200.0},
    'limits': {
        'wall_seconds': 30.0,
        'cpu_seconds': 20.0,
        'memory_bytes': 512 * 1024 * 1024,
        'pids': 64,
        'max_output_bytes': 65536,
        'max_stderr_bytes': 8192,
    },
}


def validate_config(value=None):
    if value is None:
        value = {}
    if type(value) is not dict or set(value) - set(DEFAULTS):
        raise ValueError('research_tools contains unsupported configuration fields')
    cfg = {**deepcopy(DEFAULTS), **deepcopy(value)}
    jobs = value.get('jobs', {})
    if type(jobs) is not dict or set(jobs) - set(DEFAULTS['jobs']):
        raise ValueError('research_tools.jobs contains unsupported fields')
    cfg['jobs'] = {**DEFAULTS['jobs'], **jobs}
    if type(cfg['jobs']['enabled']) is not bool:
        raise ValueError('research_tools.jobs.enabled must be boolean')
    for key, high in [('max_active', 64), ('max_pending', 10000)]:
        if type(cfg['jobs'][key]) is not int or not 1 <= cfg['jobs'][key] <= high:
            raise ValueError('invalid research_tools.jobs.' + key)
    for key, high in [('max_cpu_seconds', 3600), ('max_wall_seconds', 7200)]:
        limit = cfg['jobs'][key]
        if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit <= high:
            raise ValueError('invalid research_tools.jobs.' + key)
    limits = value.get('limits', {})
    if type(limits) is not dict or set(limits) - set(DEFAULTS['limits']):
        raise ValueError('research_tools.limits contains unsupported resource fields')
    cfg['limits'] = {**DEFAULTS['limits'], **limits}
    for name in ('enabled', 'share_between_islands', 'allow_shadow_triggers', 'yield_after_success',
                 'independent_research', 'experimental_protocol', 'shared_observations'):
        if type(cfg[name]) is not bool:
            raise ValueError('research_tools.' + name + ' must be boolean')
    bounds = {
        'interval': (1, 1000000), 'attempt_stall_threshold': (1, 1000000), 'max_requests': (1, 3),
        'max_scientific_steps': (1, 16),
        'max_active_cells': (1, 4096),
        'research_jobs_per_wave': (1, 64),
        'max_visible_tools': (1, 32), 'max_source_chars': (256, 65536),
        'max_context_chars': (4096, 12000), 'max_prompt_chars': (8000, 256000),
        'max_response_chars': (1024, 512000),
    }
    for name, (low, high) in bounds.items():
        if type(cfg[name]) is not int or not low <= cfg[name] <= high:
            raise ValueError(f'research_tools.{name} must be an integer in [{low}, {high}]')
    if cfg['yield_after_success'] and cfg['max_scientific_steps'] <= 1:
        raise ValueError('research_tools.yield_after_success requires max_scientific_steps greater than 1')
    if cfg['independent_research'] and cfg['max_scientific_steps'] <= 1:
        raise ValueError('research_tools.independent_research requires max_scientific_steps greater than 1')
    if cfg['experimental_protocol'] and (cfg['max_scientific_steps'] <= 1 or not cfg['shared_observations']):
        raise ValueError('research_tools.experimental_protocol requires sequential research and shared_observations')
    proactive = cfg['proactive_every_attempts']
    if proactive is not None and (type(proactive) is not int or proactive <= 0):
        raise ValueError('research_tools.proactive_every_attempts must be a positive integer or null')
    sources = cfg['trigger_sources']
    if (type(sources) is not list or not sources or any(type(item) is not str for item in sources)
            or len(set(sources)) != len(sources)
            or set(sources) - {'population_stall', 'search_attempt_stall', 'controller_plateau', 'opportunity_cost', 'betting_shadow'}):
        raise ValueError('research_tools.trigger_sources must name supported stagnation signals')
    if cfg['model'] is not None and (type(cfg['model']) is not str or not cfg['model'].strip()):
        raise ValueError('research_tools.model must be a model ID or null')
    if (type(cfg['temperature']) not in (int, float)
            or not math.isfinite(cfg['temperature']) or not 0 <= cfg['temperature'] <= 2):
        raise ValueError('research_tools.temperature must be finite in [0, 2]')
    image = cfg['image_id']
    if image is not None and (type(image) is not str or not re.fullmatch(r'sha256:[0-9a-f]{64}', image)):
        raise ValueError('research_tools.image_id must be an immutable sha256 Docker image ID')
    data_env = cfg['data_env']
    if data_env is not None and (type(data_env) is not str or not re.fullmatch(r'[A-Z][A-Z0-9_]*', data_env)):
        raise ValueError('research_tools.data_env must be an environment variable name or null')
    if cfg['enabled'] and image is None:
        raise ValueError('enabled research_tools requires an immutable Docker image_id')
    for name, limit in cfg['limits'].items():
        if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
            raise ValueError('research_tools.limits.' + name + ' must be finite and positive')
        if name not in ('wall_seconds', 'cpu_seconds') and type(limit) is not int:
            raise ValueError('research_tools.limits.' + name + ' must be an integer')
    lim = cfg['limits']
    if (lim['wall_seconds'] > 3600 or lim['cpu_seconds'] > 3600
            or not 2 <= lim['pids'] <= 256 or lim['memory_bytes'] < 16 * 1024 * 1024
            or lim['max_output_bytes'] > 16 * 1024 * 1024
            or lim['max_stderr_bytes'] > 256 * 1024):
        raise ValueError('research_tools limits exceed isolated executor bounds')
    if cfg['jobs']['enabled'] and any(cfg['limits'][key] > cfg['jobs']['max_' + key]
                                         for key in ('cpu_seconds', 'wall_seconds')):
        raise ValueError('default tool limits exceed job envelope')
    return cfg


def library_limits(config):
    """Immutable maximum envelope; invocation budgets remain explicit and smaller."""
    limits = deepcopy(config['limits'])
    if config['jobs']['enabled']:
        for key in ('cpu_seconds', 'wall_seconds'):
            limits[key] = config['jobs']['max_' + key]
    return limits


def effective_limits(config, requested=None):
    requested = {} if requested is None else requested
    if type(requested) is not dict or set(requested) - set(config['limits']):
        raise ValueError('unsupported requested tool limits')
    limits = {**config['limits'], **requested}
    maximum = library_limits(config)
    for key, value in limits.items():
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum[key]:
            raise ValueError('requested ' + key + ' exceeds the configured resource envelope')
        if key not in ('cpu_seconds', 'wall_seconds') and value != config['limits'][key]:
            raise ValueError('only CPU and wall budgets are model selectable')
    return deepcopy(limits)
