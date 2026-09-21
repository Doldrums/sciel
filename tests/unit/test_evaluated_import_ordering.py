"""Fresh counters cannot erase local research preceding an evidence import.

The HOST capsule is built from explicitly authored offline evidence. The review is an
explicit synthetic reply; no paid response is repaired or newly accepted.
"""
import copy
from pathlib import Path

import pytest

from sciel.engine import evaluated_study_import as study_import
from sciel.engine import evaluated_review_import as review_import
from test_evaluated_study_integration import (
    artifact, construct as study_construct, import_study, rows, forbid_runtime,
)
from test_evaluated_review_delivery import (
    reflection, construct as review_construct, import_review,
)


LOCAL_PHASES = (
    'attempt', 'structured_response', 'structured_pair', 'structured_hypothesis',
    'proposal_group', 'hypothesis', 'outcome', 'circle_generation', 'idea', 'intervention',
    'aux_request', 'aux_start', 'aux_result', 'aux_skipped', 'aux_stopped',
    'conclusion_rejected', 'conclusion', 'reflection_context', 'reflection',
    'initializer_witness_presentation', 'restart_observation_presentation',
    'initializer_witness_unavailable', 'initializer_witness_intent',
    'initializer_witness_result', 'initializer_witness_registered', 'initializer_witness_stopped',
)


@pytest.mark.parametrize('phase', LOCAL_PHASES)
def test_known_local_phases_are_not_metadata(phase):
    assert study_import._is_local_research(dict(event='research', phase=phase))
    assert not study_import._is_local_research(dict(event='run_start', phase=phase))


def test_actual_auxiliary_prefix_reproduces_the_old_phase_filter_gap():
    # Literal prior implementation: it accepted an aux_start before the import
    # because no listed local phase was present, irrespective of counters.
    prior_local = {'attempt', 'hypothesis', 'outcome', 'circle_generation', 'aux_request'}
    prefix = [dict(event='research', phase='aux_start', kind='reflection'),
              dict(event='research', phase=study_import.EVENT)]
    assert all(i > 1 for i, row in enumerate(prefix)
               if row['event'] == 'research' and row['phase'] in prior_local)
    assert any(study_import._is_local_research(row) for row in prefix[:1])


def test_historical_imports_and_run_metadata_remain_distinct():
    from sciel.engine import external_research, generated_paired_import, generated_paired_review_import
    for phase in (external_research.EVENT, generated_paired_import.EVENT,
                  generated_paired_review_import.EVENT, study_import.EVENT, review_import.EVENT):
        assert not study_import._is_local_research(dict(event='research', phase=phase))
    for event in ('run_start', 'checkpoint', 'run_end'):
        assert not study_import._is_local_research(dict(event=event))


@pytest.mark.parametrize('phase', ['aux_start', 'structured_hypothesis', 'initializer_witness_result'])
@pytest.mark.parametrize('kind', ['study', 'review'])
def test_public_import_rejects_existing_local_prefix_without_appending_or_publishing(
        artifact, reflection, tmp_path, phase, kind):
    f = (study_construct(artifact, tmp_path) if kind == 'study'
         else review_construct(artifact, reflection, tmp_path))
    try:
        if kind == 'review':
            import_study(f, artifact)
        f.log.research(phase=phase, test_fixture='orphan local prefix with fresh controller counters')
        assert f.c.attempts == f.c.outcomes == f.c.aux_calls == f.worker.calls_started == 0
        before = f.c.state()
        journal = Path(f.log.path).read_bytes()
        with pytest.raises(ValueError, match=f'local research preceded {kind} import'):
            import_study(f, artifact) if kind == 'study' else import_review(f, reflection)
        assert f.c.state() == before
        assert Path(f.log.path).read_bytes() == journal
        assert (f.c.knowledge.evaluated_detached_study is None) == (kind == 'study')
        assert f.c.knowledge.evaluated_study_review is None
        assert f.model.calls == f.charges == []
    finally:
        f.log.close()


def test_resume_rejects_auxiliary_and_generated_prefix_even_with_zero_counters(
        artifact, reflection, tmp_path):
    f = review_construct(artifact, reflection, tmp_path)
    try:
        import_study(f, artifact)
        import_review(f, reflection)
        original = rows(f.log)
        assert f.c.attempts == f.c.outcomes == f.c.aux_calls == f.worker.calls_started == 0
        for phase in ('aux_start', 'aux_result', 'aux_skipped', 'aux_stopped', 'reflection_context',
                      'reflection', 'conclusion', 'aux_request', 'idea', 'intervention',
                      'structured_response', 'initializer_witness_presentation'):
            early = copy.deepcopy(original)
            study_index = next(i for i, row in enumerate(early) if row.get('phase') == study_import.EVENT)
            early.insert(study_index, dict(event='research', phase=phase))
            with pytest.raises(ValueError, match='local research preceded study import'):
                study_import.assert_import_journal(f.c, early)
            between = copy.deepcopy(original)
            review_index = next(i for i, row in enumerate(between) if row.get('phase') == review_import.EVENT)
            between.insert(review_index, dict(event='research', phase=phase))
            study_import.assert_import_journal(f.c, between)
            with pytest.raises(ValueError, match='local research preceded review import'):
                review_import.assert_import_journal(f.c, between)
        assert rows(f.log) == original and f.model.calls == []
    finally:
        f.log.close()


def test_ordered_imports_allow_later_activity_and_idempotent_checks(artifact, reflection, tmp_path):
    f = review_construct(artifact, reflection, tmp_path)
    try:
        assert import_study(f, artifact) is True
        assert import_review(f, reflection) is True
        f.log.research(phase='aux_skipped', kind='reflection', reason='synthetic no dispatch')
        prior = Path(f.log.path).read_bytes()
        study_import.assert_import_journal(f.c, rows(f.log))
        review_import.assert_import_journal(f.c, rows(f.log))
        assert import_study(f, artifact) is False
        assert import_review(f, reflection) is False
        assert Path(f.log.path).read_bytes() == prior
        assert f.model.calls == f.charges == []
    finally:
        f.log.close()
