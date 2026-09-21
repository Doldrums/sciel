"""Authored receipts that exercise bounded delivery of exact observations."""
from copy import deepcopy
import json

from sciel.methods.models import content_hash
from sciel.research_tools.decision_projection import project_decision
from sciel.research_tools.observation_cards import measurement_card
from sciel.research_tools.result_contract import assess_receipt


BUDGET = 10000
SUMMARY_FIELDS = ('prefix_query_counts', 'mean_window_gains', 'mean_prefix_scores')


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def result(observations):
    return dict(schema='sciel-tool-result/1', status='success', observations=observations,
                diagnostics={}, error=None)


def target_binding():
    source = content_hash('authored candidate')
    return dict(target_kind='program', target_source_sha256=source, candidate_id='p2',
        ancestry_parent_id='p1', population_best_id='p2', population_best_source_sha256=source,
        research_targets_sha256=content_hash('authored research targets'))


def trial_rows():
    return [dict(trial_index=i, best_point=[i / 23, -i / 29], best_query=45,
        prefix_best_values=[.8, .6, .4, .3], prefix_scores=[.5, .625, .75, .8125],
        recorded_best_matches=True, recorded_score=.8125,
        score_gains=[.125, .125, .0625], query_values=[i / 17 + j / 31 for j in range(24)])
        for i in range(20)]


def prefix_output():
    rows = trial_rows()
    return result(dict(prefix_query_counts=[17, 31, 43, 50],
        mean_prefix_scores=[.5, .625, .75, .8125], mean_window_gains=[.125, .125, .0625],
        trial_count=20, rejected=[], saved_evaluation_status='available', rows=rows,
        continuation_state={'rows': deepcopy(rows)},
        scope='Observed prefixes only; no causal allocation comparison.'))


def summary_output():
    return result(dict(available=True, mean_final_score=.8125, trials=trial_rows(),
        late_gain_summary={str(cutoff): dict(count=20, positive_gain_count=positive,
            mean_score_gain_to_end=mean, median_score_gain_to_end=median)
            for cutoff, positive, mean, median in [
                (21, 20, .24071716165210624, .2294619395490758),
                (30, 12, .10744551683757093, .08724679678934327),
                (40, 7, 1.25e-12, 0.0), (45, 0, -.001, -.002)]}))


def invocation(index, output):
    tool_id = content_hash({'tool': index})
    receipt = dict(schema='sciel-method-execution/1', invocation_id=content_hash({'invocation': index}),
        status='failed' if output is None else 'ok', dispatch_status='invoked', output=output,
        reason='method_failed' if output is None else None,
        stderr='ValueError: method output exceeds max_output_bytes' if output is None else '',
        resources={'cpu_seconds': (index + 1) / 100, 'wall_seconds': (index + 1) / 50})
    receipt['receipt_sha256'] = content_hash(receipt)
    return dict(action_id=content_hash({'action': index}), cell_id=content_hash('cell'),
        cell_revision_id=content_hash({'revision': index}), question='Where do the measured gains occur?',
        stage='research', replay_policy='exact_observations', target_binding=target_binding(),
        tool_id=tool_id, receipt=receipt, result=assess_receipt(receipt))


def artifact(kind, payload):
    value = dict(schema='sciel-research-tool-artifact/1', kind=kind, payload=payload)
    return content_hash(value), value


def terminal_proposal(receipt_ids):
    reason = 'Observed gains belong to the saved trials; a matched experiment is still needed. '
    action = dict(action='finish', question='Do late queries improve these saved trajectories?',
        reason=reason * 14, next_decision='Compare alternative query allocations with matched seeds.',
        conclusion='supported')
    return dict(action_id=content_hash('terminal action'), evidence_receipt_ids=receipt_ids,
        response=dict(text=encoded(action), cost=None,
            cost_provenance={'source': 'subscription_unpriced'}))


def terminal_card(proposal, proposal_id):
    return dict(schema='sciel-research-check/4', cell_id=content_hash('cell'),
        revision_id=content_hash('terminal revision'), question='No new tool invocation.',
        source=dict(tool_id=None, invocation_id=None, receipt_id=None, **target_binding()),
        observations=dict(status='not_invoked_this_episode', output_excerpt='', resources={}),
        interpretation='Model interpretation is separate from task validation.',
        next_check='Read the proposed next experiment.', limitations=['No new invocation.'],
        replay_comparison=None, decision=project_decision(proposal, proposal_id))


def measurement(ident, record):
    observation = dict(tool_id=record['tool_id'], receipt_id=ident,
        target_binding=record['target_binding'], question=record['question'],
        result=record['result'], replay_policy=record['replay_policy'], **record['receipt'])
    return measurement_card({'max_context_chars': 2200},
        dict(cell_id=record['cell_id'], cell_revision_id=record['cell_revision_id'],
             question=record['question'], stage=record['stage']), observation)


def output_wrapper(record):
    receipt = record['receipt']
    return dict(output=receipt.get('output'), result=record['result'],
        replay_policy=record['replay_policy'], reason=receipt.get('reason'),
        diagnostic=receipt.get('diagnostic'))


def legacy_projection(original, selected):
    """Authored /1 evidence view with explicit omission counts, not a run snapshot."""
    fields = {}
    def visit(value, path=''):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                visit(child, path + '/' + key.replace('~', '~0').replace('/', '~1'))
        else:
            fields[path] = value
    visit(original)
    observations = [path for path in fields if path.startswith('/output/observations/')]
    shown = sum(path in observations for path in selected)
    omitted = len(fields) - len(selected)
    return dict(schema='sciel-tool-result-projection/1', original_sha256=content_hash(original),
        fields=[dict(path=path, value=fields[path]) for path in selected], observations_state='partial',
        omitted_fields=omitted, omitted_observation_fields=len(observations)-shown,
        omitted_paths=[], omitted_paths_omitted=omitted)


def retained_fixture(*, summaries=False):
    middle = result(dict(decoded_successfully=True, saved_evaluation_status='available',
        trials_structure={'count':20, 'query_history':'available'},
        source='Opaque saved input retained for provenance. ' * 150))
    outputs = [None, middle, summary_output() if summaries else prefix_output()]
    artifacts, rows, cards = {}, [], []
    for index, output in enumerate(outputs):
        payload = invocation(index, output)
        ident, saved = artifact('invocation', payload)
        artifacts[ident] = saved
        rows.append(dict(artifact_id=ident, artifact_canonical_sha256=ident,
                         payload_sha256=content_hash(payload), payload=payload))
        cards.append(measurement(ident, payload))
    proposal = terminal_proposal([row['artifact_id'] for row in rows])
    proposal_id, saved = artifact('proposal', proposal)
    artifacts[proposal_id] = saved
    terminal = terminal_card(proposal, proposal_id)
    cards = list(reversed(cards)) + [terminal]
    if not summaries:
        cards[0]['observations']['output_excerpt'] = encoded(legacy_projection(
            output_wrapper(rows[-1]['payload']), ['/output/observations/trial_count']))
    context = dict(schema='sciel-research-cells-context/1', cards=cards,
                   omitted_count=0, omission_reasons={})
    return dict(budget=BUDGET, records=rows, artifacts=artifacts, actual_context=context,
        actual_g3={'context':context}, terminal_card=terminal, terminal_card_sha256=content_hash(terminal),
        proposal=dict(artifact_id=proposal_id, artifact_canonical_sha256=proposal_id,
                      payload_sha256=content_hash(proposal), payload=proposal))
