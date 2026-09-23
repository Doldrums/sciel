"""One explicit two-record scientific policy; historical minimum remains 32.

This changes a corpus-size research policy, never eligibility or engineering
classification. Distinct source bytes are not a claim of behavioral diversity.
"""
SMALL_DATA_POLICY = 'native-small-data-feasibility/1'
PLAN_SCHEMA = 'sciel-deltaserve-plan/3'


def validate(value):
    if type(value) is not str or value != SMALL_DATA_POLICY:
        raise ValueError('unsupported explicit corpus policy')
    return value


def selected(value):
    if type(value) is not dict:
        raise ValueError('corpus policy requires a typed authority mapping')
    if 'corpus_policy' not in value:
        return None
    return validate(value['corpus_policy'])


def fields(value):
    policy = selected(value)
    return {} if policy is None else {'corpus_policy': policy}


def same(*values):
    policies = [selected(value) for value in values]
    if any(value != policies[0] for value in policies):
        raise ValueError('corpus policy missing or substituted across authority bindings')
    return policies[0]


def check_ancestry(row, corpus_policy):
    """Join only typed revision proofs, after ordinary ancestry verification."""
    if corpus_policy is not None:
        validate(corpus_policy)
    resolution = row['provenance'].get('ancestry_resolution')
    if resolution is None:  # Ordinary verifier permits neutral legacy rows.
        return
    for node in resolution.get('auxiliary_bindings', []):
        if node['kind'] == 'adapter_revision':
            inherited = selected(node['proof'])
            if inherited is not None and inherited != corpus_policy:
                raise ValueError('corpus policy differs from verified adapter ancestry')


def check_rows(rows, *, owner, mode):
    from .deltaserve_plan import _engineering
    if (len(rows) != 2 or len({r['episode_id'] for r in rows}) != 2
            or len({r['source_sha256'] for r in rows}) != 2 or _engineering(rows)):
        raise ValueError('small-data policy requires exactly two distinct nonengineering native sources')
    counts = {'local': 0, 'donor': 0}
    for row in rows:
        origin = 'local' if row['owner'] == owner else 'donor'
        if (row.get('origin') != origin or row.get('recipient_owner') != owner
                or row.get('split') != 'train' or row.get('committed') is not True
                or row.get('evaluation_receipt', {}).get('schema') != 'sciel-saved-evaluation/1'):
            raise ValueError('small-data source is not a genuine owner-bound committed TRAIN record')
        counts[origin] += 1
    expected = {'local': 2, 'donor': 0} if mode == 'local' else {'local': 1, 'donor': 1}
    if mode not in ('local', 'mixed') or counts != expected:
        raise ValueError('small-data policy requires own2 or genuine own1 plus donor1')


def check_steps(plan):
    validate(plan.get('corpus_policy'))
    data, selection, steps = plan['dataset'], plan['selection'], plan['steps']
    if (plan['schema'] != PLAN_SCHEMA or data['example_count'] != 2
            or data['minimum_examples'] != 2 or data['corpus_minimum_met'] is not True
            or data['engineering_only'] is not False or selection['epochs'] != 1
            or len(steps) != 2 or len({s['episode_id'] for s in steps}) != 2
            or len({s['source_sha256'] for s in steps}) != 2
            or selection['selected_unique_examples'] != 2):
        raise ValueError('small-data plan must consume exactly two unique sources once, without engineering')
    expected = {'local': 2} if selection['mode'] == 'local' else {'local': 1, 'donor': 1}
    if (selection['mode'] not in ('local', 'mixed')
            or selection['selected_unique_examples_by_origin'] != expected):
        raise ValueError('small-data selected origins differ')
    for step in steps:
        origin = 'local' if step['producer_owner'] == plan['owner'] else 'donor'
        if step['origin'] != origin or step['recipient_owner'] != plan['owner']:
            raise ValueError('small-data consumed source owner/origin differs')
    if any(sum(s['origin'] == origin for s in steps) != count for origin, count in expected.items()):
        raise ValueError('small-data consumed origins differ')
