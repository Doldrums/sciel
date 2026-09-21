from types import SimpleNamespace as NS
from sciel.engine.research_search_evidence import search_evidence
from sciel.engine.validation import source_sha256
from sciel.engine.evaluation_timing import remember


def test_evidence_is_bound_to_actual_measurements_and_does_not_invent_cpu():
    parent = NS(id='p1', code='def solve(): return 1', parent_id='p0', island=2)
    migrant = NS(id='p2', code=parent.code, parent_id='p1', island=3)
    record = dict(child_program_id='p1', child_source_sha256=source_sha256(parent.code),
                  record_sha256='a'*64, child_observation={'score': 1, 'reached': 'full', 'confirmed': False},
                  score_comparison='changed', geometry_comparison='different')
    engine = NS(db=NS(_all={'p1':parent, 'p2':migrant},pop_best=1),
        observed_changes=NS(record_for_program=lambda _: record), eval_times={'p1':2},
        _per_run_budget=lambda:28,iteration_no=17,B_evals=9,n_evaluation_results=18,n_output_repeats=4,
        _last_pop_improve_iteration=10,_last_pop_improve=6)
    remember(engine, parent, NS(timing=6., cases={'n_trials':3}, reached='confirm',
                               evaluation_id='fixture-evaluation'))
    evidence = search_evidence(engine,parent)
    assert evidence['schema'] == 'sciel-search-evidence/3'
    assert evidence['recent_evaluations'] == []  # a timing ID alone is not authentication
    assert len(evidence['unverified_admissions']) == 1  # migrant is not a measurement
    assert evidence['unverified_admissions'][0]['identity_status'] == 'unknown'
    runtime = evidence['selected_runtime']
    assert runtime['evaluation_stage_wall_seconds'] == 6.
    assert runtime['evaluator_reported_trials'] == 3
    assert runtime['mean_evaluation_stage_wall_seconds_per_reported_trial'] == 2.
    assert runtime['evaluation_id'] == 'fixture-evaluation'
    assert runtime['fraction_of_wall_limit'] is None
    assert 'observed_wall_seconds_per_trial' not in runtime
    assert evidence['exact_recorded_output_repeats'] == 4
    assert 'not CPU time' in evidence['scope']
    assert 'faster helper does not imply more search' in evidence['scope']
    # The fixture omits p0: incomplete ancestry cannot certify another branch.
    assert evidence['selected_lineage']['complete'] is False
