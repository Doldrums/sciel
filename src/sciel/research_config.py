"""Research memory, evidence, and auxiliary-call configuration."""


def validate_research(cfg):
    if 'research' in cfg:
        defaults = {'enabled': False, 'reflection_interval': 10, 'max_aux_calls': 32,
                    'max_lessons': 50, 'max_experiments': 200, 'max_fragments': 20,
                    'retrieve_k': 3, 'model': None, 'inject_lessons': True}
        supplied = cfg['research']
        if not isinstance(supplied, dict) or set(supplied) - (set(defaults) | {'compact_lesson_memory', 'conclusion_source_refs', 'conclusion_repair_attempts', 'conclusion_max_tokens', 'lesson_selection', 'initializer_witness', 'initializer_witness_profile', 'batch_only_lessons', 'restart_predictions', 'structured_auxiliary', 'incomplete_auxiliary', 'comparison_reviews', 'comparison_review_text', 'comparison_review_version', 'comparison_review_text_policy', 'external_history_sha256', 'external_component_sha256', 'external_component_review_sha256', 'generated_paired_review_version', 'external_evaluated_study_sha256', 'external_evaluated_study_authority_sha256', 'external_evaluated_review_sha256', 'source_reflection_config_sha256'}):
            raise ValueError('invalid or unknown research configuration')
        research = {**defaults, **supplied}
        # Preserve omitted legacy settings exactly for existing /8 checkpoints.
        # Explicit experiments retain their normalized settings in identity.
        if 'lesson_selection' in research:
            from .knowledge_base.retrieval import validate_selection
            research['lesson_selection'] = validate_selection(research['lesson_selection'])
        if type(research['enabled']) is not bool:
            raise ValueError('research enabled must be boolean')
        if type(research['inject_lessons']) is not bool:
            raise ValueError('research inject_lessons must be boolean')
        if 'batch_only_lessons' in research and type(research['batch_only_lessons']) is not bool:
            raise ValueError('research batch_only_lessons must be boolean')
        if 'restart_predictions' in research and type(research['restart_predictions']) is not bool:
            raise ValueError('research restart_predictions must be boolean')
        if 'structured_auxiliary' in research and type(research['structured_auxiliary']) is not bool:
            raise ValueError('research structured_auxiliary must be boolean')
        for option in ('compact_lesson_memory', 'conclusion_source_refs'):
            if type(research.get(option, False)) is not bool:
                raise ValueError(f'research {option} must be boolean')
        repairs = research.get('conclusion_repair_attempts', 0)
        if type(repairs) is not int or not 0 <= repairs <= 1:
            raise ValueError('conclusion_repair_attempts must be an integer from 0 to 1')
        conclusion_tokens = research.get('conclusion_max_tokens')
        if conclusion_tokens is not None and (type(conclusion_tokens) is not int or conclusion_tokens <= 0):
            raise ValueError('conclusion_max_tokens must be a positive integer or null')
        if research.get('conclusion_source_refs', False) and not research.get('structured_auxiliary', False):
            raise ValueError('conclusion_source_refs requires structured_auxiliary')
        if repairs and not research.get('structured_auxiliary', False):
            raise ValueError('conclusion repairs require structured_auxiliary')
        if 'incomplete_auxiliary' in research:
            if (type(research['incomplete_auxiliary']) is not str
                    or research['incomplete_auxiliary'] not in ('stop', 'skip')):
                raise ValueError('incomplete_auxiliary must be stop or skip')
            if research['incomplete_auxiliary'] == 'skip' and not research.get('structured_auxiliary', False):
                raise ValueError('incomplete_auxiliary=skip requires structured_auxiliary')
        if type(research.get('comparison_reviews', False)) is not bool:
            raise ValueError('research comparison_reviews must be boolean')
        if research.get('comparison_reviews', False) and not research.get('structured_auxiliary', False):
            raise ValueError('comparison_reviews requires structured_auxiliary')
        if 'comparison_review_version' in research:
            if (type(research['comparison_review_version']) is not int
                    or research['comparison_review_version'] not in (1, 2, 3)):
                raise ValueError('comparison_review_version must be 1, 2 or 3')
            if not research.get('comparison_reviews', False):
                raise ValueError('comparison_review_version requires comparison_reviews')
        if 'comparison_review_text' in research:
            if type(research['comparison_review_text']) is not bool:
                raise ValueError('comparison_review_text must be boolean')
            if not research.get('comparison_reviews', False):
                raise ValueError('comparison_review_text requires comparison_reviews')
        if 'comparison_review_text_policy' in research:
            from .hypothesis.review_eligibility import validate_policy
            research['comparison_review_text_policy'] = validate_policy(research['comparison_review_text_policy'])
            if (research.get('comparison_reviews') is not True
                    or type(research.get('comparison_review_text')) is not bool
                    or research.get('batch_only_lessons') is not True):
                raise ValueError('comparison_review_text_policy requires comparison_reviews, explicit '
                                 'comparison_review_text and batch_only_lessons')
        if 'external_history_sha256' in research:
            value = research['external_history_sha256']
            if (type(value) is not str or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('research external_history_sha256 must be a lowercase SHA256')
        if 'external_component_sha256' in research:
            value = research['external_component_sha256']
            if (type(value) is not str or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('research external_component_sha256 must be a lowercase SHA256')
        from .engine.evaluated_study_import import configured_digests as study_digests
        study_digests(research)
        from .engine.evaluated_review_import import configured_digests as evaluated_review_digests
        evaluated_review_digests(research)
        if 'external_component_review_sha256' in research:
            from .engine.generated_paired_review_import import configured_digest
            configured_digest(research)
        if 'generated_paired_review_version' in research:
            from .engine.generated_paired_review_import import configured_version
            configured_version(research)
        if 'initializer_witness' in research and type(research['initializer_witness']) is not bool:
            raise ValueError('research initializer_witness must be boolean')
        if ('initializer_witness_profile' in research and
                (type(research['initializer_witness_profile']) is not str or
                 research['initializer_witness_profile'] not in ('legacy_v1', 'unequal_radii_v1'))):
            raise ValueError('invalid research initializer_witness_profile')
        for key in ('reflection_interval', 'max_aux_calls', 'max_lessons',
                    'max_experiments', 'max_fragments', 'retrieve_k'):
            if key == 'max_aux_calls' and research[key] is None:
                continue
            lower = 1 if key in ('max_lessons', 'max_experiments', 'max_fragments') else 0
            if type(research[key]) is not int or not lower <= research[key] <= 10000:
                raise ValueError(f'invalid research {key}')
        if research['reflection_interval'] > research['max_experiments']:
            raise ValueError('research reflection window exceeds experiment history')
        if (cfg['parallel_search']['enabled'] and research['enabled']
                and research['max_experiments'] < 3 * cfg['parallel_search']['workers']):
            raise ValueError('parallel_search research.max_experiments must retain at least workers * 3 attempts')
        if research['retrieve_k'] > research['max_lessons']:
            raise ValueError('research retrieval exceeds lesson capacity')
        if research['model'] is not None and (
                not isinstance(research['model'], str) or not research['model'].strip()):
            raise ValueError('research model must be a model ID or null')
        if research['enabled']:
            llm = dict(cfg.get('llm') or {})
            if type(llm.get('max_retries', 1)) is not int or llm.get('max_retries', 1) != 1:
                raise ValueError('research requires llm.max_retries=1: no hidden request replay')
            llm['max_retries'] = 1
            cfg['llm'] = llm
        cfg['research'] = research
