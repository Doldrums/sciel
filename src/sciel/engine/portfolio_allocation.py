"""Fixed, checkpointed allocation of initial portfolio and evolution waves.

This policy changes scheduling, never candidate/evaluator behavior. Counts refer
to committed waves and attempted islands, not successful candidates or families.
"""
from copy import deepcopy

SCHEMA = 'sciel-portfolio-allocation/1'


def enabled(cfg):
    return cfg.get('parallel_search', {}).get('portfolio_schedule', 'blocking') == 'interleaved'


def initial():
    return dict(schema=SCHEMA, policy='interleaved', next_kind='portfolio',
                portfolio_waves=0, evolution_waves=0)


def pending(engine):
    cursor = getattr(engine, '_portfolio_bootstrap', None)
    return cursor is not None and cursor['next_island'] < cursor['islands']


def validate(value, cfg, cursor):
    if not enabled(cfg):
        if value is not None:
            raise ValueError('blocking portfolio cannot contain interleaved allocation')
        return None
    if (type(value) is not dict or set(value) != set(initial())
            or value.get('schema') != SCHEMA or value.get('policy') != 'interleaved'
            or value.get('next_kind') not in ('portfolio', 'evolution')
            or any(type(value.get(key)) is not int or value[key] < 0
                   for key in ('portfolio_waves', 'evolution_waves'))):
        raise ValueError('invalid interleaved portfolio allocation')
    p, e = value['portfolio_waves'], value['evolution_waves']
    attempted = cursor['next_island'] if cursor else 0
    total = cursor['islands'] if cursor else cfg['num_islands']
    workers = cfg['parallel_search']['workers']
    if p != (attempted + workers - 1) // workers:
        raise ValueError('portfolio allocation differs from committed attempt cursor')
    if attempted < total:
        if attempted % workers or not ((p == e and value['next_kind'] == 'portfolio')
                or (p == e + 1 and value['next_kind'] == 'evolution')):
            raise ValueError('interleaved portfolio must alternate committed waves')
    elif value['next_kind'] != 'evolution' or e < p - 1:
        raise ValueError('completed portfolio must allocate evolution waves')
    return deepcopy(value)


def eligible_islands(engine):
    islands = [island.idx for island in engine.db.islands]
    if enabled(engine.cfg) and pending(engine):
        return [island for island in islands if island < engine._portfolio_bootstrap['next_island']]
    return islands


def metadata(engine, kind):
    cursor = getattr(engine, '_portfolio_bootstrap', None)
    return dict(wave_kind=kind,
        portfolio_schedule=engine.cfg['parallel_search'].get('portfolio_schedule', 'blocking'),
        portfolio_attempted=cursor['next_island'] if cursor else 0,
        portfolio_total=cursor['islands'] if cursor else 0,
        eligible_islands=eligible_islands(engine),
        **({'allocation': deepcopy(engine._portfolio_allocation)} if enabled(engine.cfg) else {}))


def advance(engine, kind):
    if not enabled(engine.cfg) or kind == 'canonical_seed':
        return
    state = engine._portfolio_allocation
    state[kind + '_waves'] += 1
    state['next_kind'] = ('evolution' if kind == 'portfolio' or not pending(engine) else 'portfolio')
    validate(state, engine.cfg, engine._portfolio_bootstrap)
