"""Authored legacy-format memory covering source support and advice attribution."""


def memory_records():
    records = []
    cases = [
        (2, 'evaluated', 0.5627179948834453),
        (3, 'eval_failed', None),
        (4, 'evaluated', 0.523557102750764),
        (5, 'no_child', None),
        (6, 'evaluated', 0.0),
    ]
    for number, status, delta in cases:
        records.append({
            'id': f'r{number:06d}a1', 'branch': 'fixture-branch',
            'parent_id': 'fixture-parent', 'parent_score': 0.0,
            'score': delta, 'actual_delta': delta, 'status': status,
            'source_sha256': str(number) * 64 if delta is not None else None,
            'proposed_sha256': str(number) * 64,
            'source_repaired': False, 'forecast_verdict': None,
        })
    origins = [
        (3, 'inherited', 'l000003'),
        (4, 'inherited', 'l000003'),
        (5, 'direct', 'l000006'),
        (6, 'direct', 'l000006'),
    ]
    lesson = {
        'id': 'l000006', 'rule': 'Authored fixture interpretation, not a scientific claim.',
        'polarity': 'positive', 'scope': 'branch', 'branch': 'fixture-branch',
        'evidence_ids': ['r000002a1', 'r000004a1'],
        'retirement_eligible': True, 'focus_for_consolidation': False,
        'observed_exposures': {
            'positive': 1, 'negative': 0, 'zero': 1, 'unmeasured': 2, 'total': 4,
            'shown_ids': [f'r{number:06d}a1' for number, _, _ in origins],
            'omitted_count': 0,
            'representatives': [
                {'experiment_id': f'r{number:06d}a1', 'attribution': [
                    {'kind': kind, 'from_lesson_id': ident, 'from_rule': 'Fixture advice.'}
                ]}
                for number, kind, ident in origins
            ],
            'interpretation': 'Association after advice exposure, not causal support.',
        },
    }
    return records, [lesson]
