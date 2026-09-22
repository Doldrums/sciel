"""Fail closed on native adaptation until its v0.2 lifecycle port is complete."""

def validate(cfg):
    llm = cfg.get('llm') or {}
    if 'online_adaptation' in cfg or 'oracle_evaluation' in cfg or 'continuous_routing' in llm:
        raise ValueError('upstream v0.2 online adaptation/qualification port is not enabled')
    backend = llm.get('backend', 'openai')
    if backend not in ('openai', 'local_openai'):
        raise ValueError('unknown llm.backend')
    if 'generation_seed' in llm:
        from .client import validate_generation_seed
        if backend != 'local_openai':
            raise ValueError('llm.generation_seed requires local_openai')
        validate_generation_seed(llm['generation_seed'])
    if 'managed_deployment' in llm:
        from ...adaptation.deployment import configuration
        configuration(llm)
    if backend == 'local_openai':
        if llm.get('mock') or llm.get('provider') not in (None, 'chat_completions'):
            raise ValueError('local_openai cannot substitute a mock or different provider')
        if (llm.get('local_inference') or {}).get('adapters'):
            raise ValueError('upstream v0.2 adapter inference port is not enabled')
        for key in ('circle_generation', 'circle_intervention', 'idea_two_stage'):
            if (cfg.get(key) or {}).get('enabled'):
                raise ValueError('local upstream generation route is not ported: ' + key)
