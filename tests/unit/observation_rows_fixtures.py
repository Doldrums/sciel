"""Authored numerical dictionaries large enough to require whole-row projection."""
from test_research_experimental_protocol import protocol
from test_sequential_research_episode import finish


def observation_rows_case():
    checks = []
    for tolerance in (1e-10, 1e-8, 1e-6):
        checks.append(dict(active_tolerance=tolerance, lp_success=True,
            max_first_order_gain=-1.25e-8, message='Fixture optimizer converged.',
            active_triples=[[i % 9, (i+1) % 9, (i+2) % 9] for i in range(15)],
            direction=[[i / 37, -i / 41] for i in range(11)],
            line_checks=[dict(step=step, min_area_normalized=.0275-step/8,
                strict_domain_valid=True, active_residuals=[(i+1) / 997 for i in range(12)])
                for step in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)]))
    output = dict(schema='sciel-tool-result/1', status='success', error=None,
        diagnostics={'interpretation_limit':'Only the supplied fixed geometry was checked.'},
        observations=dict(available=True, baseline_score=.75, baseline_min_area_normalized=.0275,
            checks=checks, simplex_max_row_sum=1.0, simplex_min_coordinate=0.0,
            continuation_state={'coordinates':[[i / 23, i / 47] for i in range(11)], 'seed':17},
            source_tail='Opaque saved diagnostic input; never executed. ' * 40))
    prior = 'a' * 64
    interpretation = (
        f'Observed receipt {prior} contains three tolerance checks and finite-step measurements. '
        'The reported first-order gain is negative at each tolerance. This supports only the '
        'local diagnostic; neither global optimality nor candidate improvement was measured. '
        'The full directions, residuals and line checks remain in the immutable receipt. '
    ) * 3
    design = protocol()
    design.update(kind='closure', question_key='local-first-order-gain',
        prior_receipt_ids=[prior], budget_action='none',
        budget_reason='The local diagnostic is complete; allocate the next trial to a new start.',
        saved_state_plan='Retain the supplied coordinates and seed without another invocation.')
    terminal = finish(question='Is the supplied placement blocked under the measured local directions?',
        reason=interpretation, conclusion='supported', experimental_protocol=design,
        next_decision='Try a different starting state under the same measured compute budget.')
    return {'output':output, 'terminal_action':terminal}
