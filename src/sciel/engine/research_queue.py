"""Independent research jobs sharing the ordinary wave's durable I/O ledger.

Research retains its studied source and cell owner. Only an explicit transfer
selects a different branch; neither operation consumes ordinary search slots.
The wave still commits atomically after all its jobs finish.
"""
from copy import deepcopy
from dataclasses import dataclass

from ..components.selection.novelty_chain import Selected
from .research_candidate import evaluate_submission, candidate_binding
from .research_cells_context import compact_context
from .research_search_evidence import search_evidence
from .research_targets import freeze_targets


@dataclass
class ResearchJobResult:
    episode: dict
    candidate: object = None


def enabled(engine):
    tools = getattr(engine, 'research_tools', None)
    return tools is not None and (engine.cfg.get('research_tools') or {}).get('independent_research', False)


def _refresh(engine, saved, state, *, work_kind='question', pending_trial=None):
    tools = engine.research_tools
    snapshot = deepcopy(saved)
    snapshot.iteration_no = engine.iteration_no
    snapshot.tool_episode_plan = None
    snapshot.code_hashes = engine.db.code_hashes(snapshot.island)
    parent = engine.db._all.get(snapshot.parent_id)
    if parent is None:
        raise ValueError('queued research ancestry parent is absent from saved history')
    if work_kind == 'method_transfer':
        if not pending_trial or pending_trial.get('blocked_reason'):
            return None
        # Target selection is independent of ordinary parent/island sampling.
        # Keep the author island: visibility, cell affinity and receipts rely on it.
        candidates = {program.id: program for island in engine.db.islands
                      for program in island.programs.values()}
        for candidate in sorted(candidates.values(), key=lambda row: row.id):
            trial = deepcopy(snapshot)
            trial.parent_id, trial.parent_code = candidate.id, candidate.code
            trial.parent_score, trial.parent_card = candidate.score, candidate.card
            trial.artifacts = {}
            trial.research_targets = freeze_targets(engine, candidate, candidate.code)
            trial.research_search_evidence = search_evidence(engine, candidate)
            if tools._independent_branch(trial, pending_trial):
                snapshot, parent = trial, candidate
                break
        else:
            return None
    elif work_kind != 'method_transfer_continuation':
        # Frozen source/output identity is unchanged while interpreting a result.
        # The surrounding search history and evidence access reflect this wave.
        snapshot.research_search_evidence = search_evidence(engine, parent)
    snapshot.research_cells_context = compact_context(tools.context_view(snapshot),
        max_chars=min(tools.config['max_context_chars'], 10000))
    if snapshot.research_context is not None:
        snapshot.research_context['research_cells_context'] = deepcopy(snapshot.research_cells_context)
    return snapshot


def prepare(engine, *, ordinary_jobs=0):
    if not enabled(engine):
        return []
    # Each research job can submit at most one measured candidate. Reserve its
    # entire possible admission before dispatch, never consume an ordinary slot.
    bounds = [bound for bound in (engine.budget, engine._stop_at) if bound is not None]
    remaining = max(0, min(bounds) - engine.B_evals - ordinary_jobs) if bounds else None
    return engine.research_tools.reserve_continuations(
        lambda snapshot, state, **work: _refresh(engine, snapshot, state, **work), max_jobs=remaining)


def execute(engine, snapshot):
    submissions = []
    evaluator = engine._evaluation_callback(snapshot.island)
    novelty = engine._candidate_novelty_callback(snapshot.island)

    def evaluate(source, provenance):
        if len(submissions) >= 3 or any(row.meta.get('evaluation_count', 0) for row in submissions):
            raise ValueError('one measured research candidate is allowed per opportunity')
        candidate, feedback = evaluate_submission(source, provenance, snapshot, evaluator, novelty)
        submissions.append(candidate)
        return feedback

    episode = engine.research_tools.execute(snapshot.tool_episode_plan, snapshot,
        engine.llm, io=engine._parallel_io, candidate_evaluator=evaluate)
    candidate = submissions[-1] if submissions else None
    if candidate is not None:
        candidate.meta['dup_attempts'] = len(submissions) - 1
        candidate.meta['novelty_checks'] = [check for row in submissions
            for check in row.meta.get('novelty_checks', [])]
        attempts = [deepcopy(attempt) for row in submissions
            for attempt in row.meta.get('validation_trace', {}).get('attempts', [])]
        for index, attempt in enumerate(attempts, 1):
            attempt['attempt'] = index
        candidate.meta['validation_trace']['attempts'] = attempts
    return ResearchJobResult(episode, candidate)


def commit(engine, snapshot, outcome):
    child_id = None
    result = outcome.candidate
    binding = candidate_binding(engine.research_tools, outcome.episode, result)
    before = engine.iteration_no
    if result is not None:
        selected = Selected(snapshot.parent_id, [], 1., {'selection': 'independent_research'})
        model = engine.research_tools.config['model'] or snapshot.model
        child_id = engine._on_result(result, selected, 1., {}, model, 1., [],
            'base', 1., 'base', engine.schedule.phase_code(), False)
        engine.log.decision(policy='research_continuation', action='research_candidate',
            cell_id=outcome.episode['plan']['cell_id'],
            action_id=outcome.episode['plan']['action_id'], island=snapshot.island,
            parent_id=snapshot.parent_id, child_program_id=child_id,
            evaluation_id=result.meta.get('evaluation_id'),
            evaluation_count=result.meta.get('evaluation_count', 0),
            ordinary_iteration=engine.iteration_no,
            scope='Ordinary evaluator/novelty admission; no ordinary operator or mode credit.')
    if engine.iteration_no != before:
        raise ValueError('research-only commit changed the ordinary iteration clock')
    engine.research_tools.commit(outcome.episode, child_program_id=child_id,
        link_kind='remeasurement' if result and result.meta.get('remeasurement') else 'mutation',
        completed_iteration=engine.iteration_no,
        **({'candidate_binding': binding, 'child_source': engine.db._all[child_id].code}
           if child_id is not None and binding is not None else {}))
