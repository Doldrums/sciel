"""Bounded measured search context for research decisions, not a diagnosis.

The host supplies facts. A cell must still distinguish a time bottleneck from
an exhausted search budget or poor starting states experimentally.
"""
import math

from .validation import source_sha256
from .evaluation_timing import for_program
from .research_artifacts import evaluation_id_for_program, saved_evaluation


ALIAS_LIMIT = 8


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _recent_evaluations(engine, programs, observed, limit):
    """Group admissions before bounding history; authenticate returned rows.

    A seed can occupy many islands after one measurement. Conversely, identical
    code can have genuinely distinct evaluations. Only an explicit evaluation
    identity and exact source bind those placements, never a score/code match.
    Later migration does not move an old measurement to the end of history.
    """
    groups, unknown, aliases = {}, [], []
    observed_admissions = 0
    for program in programs.values():
        record = observed.record_for_program(program.id) if observed else None
        if record is None:
            continue
        source = source_sha256(program.code)
        if record.get('child_source_sha256') != source:
            raise ValueError('search evidence observation source differs from admission')
        if record.get('child_program_id') != program.id:
            # Only a registered explicit migrant alias is evidence of a copy.
            if program.id in getattr(observed, '_aliases', {}):
                aliases.append((program, record))
            continue
        observed_admissions += 1
        ident = evaluation_id_for_program(engine, program)
        if ident is None:
            unknown.append(dict(program_id=program.id, source_sha256=source,
                observation_sha256=record['record_sha256'], evaluation_id=None,
                identity_status='unknown', reason='No exact evaluation identity is bound to this admission.'))
            continue
        key = ident, source
        if key not in groups:
            groups[key] = dict(program=program, record=record, admissions=[])
        groups[key]['admissions'].append(program)
    for program, record in aliases:
        ident = evaluation_id_for_program(engine, program)
        key = ident, source_sha256(program.code)
        if key in groups:
            groups[key]['admissions'].append(program)

    rows = []
    selected = list(groups.items())[-limit:]
    for (ident, source), group in selected:
        program, record = group['program'], group['record']
        saved = saved_evaluation(engine, program.code, ident)
        if saved['reference'] is None:
            # A legacy ID or an absent artifact is not authenticated evidence.
            # Keep every admission explicit; never merge unknown measurements.
            unknown.extend(dict(program_id=p.id, source_sha256=source,
                observation_sha256=record['record_sha256'], evaluation_id=ident,
                identity_status='unknown', reason=saved['reason'])
                for p in group['admissions'])
            continue
        observation = record['child_observation']
        exact = saved['record']
        if exact is not None and observation.get('score') != exact['result']['score']:
            raise ValueError('search evidence observation differs from saved evaluation')
        timing = for_program(engine, program) or {}
        if timing and timing.get('evaluation_id') != ident:
            raise ValueError('search evidence timing differs from saved evaluation identity')
        admissions = group['admissions']
        row = dict(program_id=program.id, parent_program_id=program.parent_id,
            island_id=program.island, source_sha256=source, evaluation_id=ident,
            identity_status='authenticated', saved_evaluation_sha256=saved['reference']['sha256'],
            observation_sha256=record['record_sha256'],
            score=observation.get('score'), score_comparison=record.get('score_comparison'),
            output_comparison=record.get('geometry_comparison'),
            evaluation_stage_wall_seconds=timing.get('evaluation_stage_wall_seconds'),
            evaluator_reported_trials=timing.get('evaluator_reported_trials'),
            mean_evaluation_stage_wall_seconds_per_reported_trial=timing.get(
                'mean_evaluation_stage_wall_seconds_per_reported_trial'),
            reached=exact['reached'] if exact else observation.get('reached'),
            confirmed=exact['confirmed'] if exact else observation.get('confirmed'),
            admissions_count=len(admissions),
            aliases=[dict(program_id=p.id, island_id=p.island, origin=getattr(p, 'origin', None))
                     for p in admissions[1:ALIAS_LIMIT + 1]],
            aliases_omitted=max(0, len(admissions)-1-ALIAS_LIMIT))
        rows.append(row)
    return dict(recent_evaluations=rows,
        recent_evaluations_omitted=max(0, len(groups)-limit),
        unverified_admissions=unknown[-limit:], unverified_admissions_omitted=max(0, len(unknown)-limit),
        admission_stats=dict(program_records_total=len(programs), observed_admissions_total=observed_admissions,
            explicit_migration_aliases_total=len(aliases), bound_evaluation_identities_total=len(groups)))


def search_evidence(engine, parent, *, limit=12):
    if type(limit) is not int or limit < 1:
        raise ValueError('search evidence history limit must be a positive integer')
    programs = getattr(engine.db, '_all', {})
    observed = getattr(engine, 'observed_changes', None)
    history = _recent_evaluations(engine, programs, observed, limit)
    cap = _number(engine._per_run_budget())
    parent_timing = for_program(engine, parent) or {}
    lineage_ids, lineage_sources, cursor, seen = [], [], parent, set()
    while cursor is not None and cursor.id not in seen and len(lineage_ids) < 256:
        seen.add(cursor.id)
        lineage_ids.append(cursor.id)
        lineage_sources.append(source_sha256(cursor.code))
        if cursor.parent_id is None:
            cursor = None
            break
        cursor = programs.get(cursor.parent_id)
        if cursor is None:
            break
    last = programs.get(lineage_ids[-1]) if lineage_ids else None
    lineage_complete = bool(last and last.parent_id is None)
    outputs = getattr(engine, 'output_novelty', None)
    output_counts = outputs.summary() if outputs is not None else {
        'exact_recorded_repeat': getattr(engine, 'n_output_repeats', None)
            if not getattr(engine, 'cfg', {}).get('candidate_novelty', {}).get('output_resolution') else None}
    return dict(schema='sciel-search-evidence/3',
        committed_iteration=engine.iteration_no, accepted_evaluations=engine.B_evals,
        global_best_score=_number(engine.db.pop_best),
        last_record_iteration=getattr(engine, '_last_pop_improve_iteration', None),
        last_record_accepted_evaluation=getattr(engine, '_last_pop_improve', None),
        **history, recent_limit=limit,
        evaluated_outputs=getattr(engine, 'n_evaluation_results', None),
        exact_recorded_output_repeats=output_counts.get('exact_recorded_repeat'),
        non_improving_near_output_repeats=output_counts.get('non_improving_near_repeat', 0),
        selected_lineage=dict(program_ids=lineage_ids, source_sha256s=lineage_sources,
                              complete=lineage_complete),
        selected_runtime=dict(program_id=parent.id, source_sha256=source_sha256(parent.code),
            **{key: parent_timing.get(key) for key in ('evaluation_id', 'reached',
                'evaluation_stage_wall_seconds', 'evaluator_reported_trials', 'trial_count_status',
                'mean_evaluation_stage_wall_seconds_per_reported_trial', 'timing_scope')},
            per_run_wall_limit_seconds=cap, fraction_of_wall_limit=None,
            fraction_unavailable_reason='No comparable candidate-only per-run timing is recorded. '
                'Stage wall time and its arithmetic mean include evaluator overhead.'),
        scope='Recent evaluations are authenticated, unpaired returned measurements associated with admissions. '
              'One evaluation identity/source appears once regardless of seed placements or explicit migrant copies. '
              'Distinct evaluation IDs remain distinct even for identical source and outcomes. '
              'Rows are ordered by first admission of each evaluation; later aliases do not refresh recency. '
              'Admission statistics count placements, not independent measurements. Unverified admissions have unknown '
              'measurement identity and are not counted as evaluations. The omission count covers older bound identities '
              'not materialized in this view. Failed/rejected attempts are absent from this admission-derived history. '
              'Output repeat totals include measured rejected outputs across the run, not semantic equivalence. '
              'Wall timing covers the returned evaluation stage, including evaluator overhead; it is not CPU time. '
              'A reported trial count permits only a stage-inclusive arithmetic mean, not a measured individual '
              'candidate runtime or proof that all runs met a deadline. Trials may overlap or have unequal costs; '
              'no sequential execution is assumed. Unknown counts remain unknown. '
              'A faster helper does not imply more search unless the algorithm actually spends the freed budget. '
              'Scores use the ordinary evaluator scale; source-bound full results and native metrics are in saved_evaluation.')
