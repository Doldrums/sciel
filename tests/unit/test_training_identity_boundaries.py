"""Run-disjoint offline splits; no training, provider or evaluator execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

import pytest


def module(name):
    path = Path(__file__).resolve().parents[2] / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location('boundary_' + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def legacy_run(path, *, offset=0., evictions=()):
    path.mkdir(parents=True, exist_ok=True)
    rows = [dict(event='run_start', schema_version=1, budget=6,
                 config={'budget': 6, 'milestone_fracs': [], 'schedule_policy': 'round_robin'})]
    for index in range(1, 7):
        island = (index - 1) // 3
        rows.append(dict(event='iteration', iteration_no=index, B_evals=index, island=island,
                         child_id=f'p{index}', parent_id=None, child_score=offset + index / 10,
                         child_code=f'# inert record {index}\n', cases={},
                         pop_best=offset + index / 10, island_best=offset + index / 10,
                         pop_best_pre=offset + (index - 1) / 10,
                         island_best_pre=offset if index in (1, 4) else offset + (index - 1) / 10,
                         mode='base'))
    rows.extend(dict(event='decision', policy='eviction', **event) for event in evictions)
    rows.append(dict(event='run_end', reason='budget', iteration_no=6, B_evals=6, pop_best=offset + .6))
    (path / 'sciel_events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def modern_run(path, run_id, *, score=.5):
    """Real v2 selected-trajectory/checkpoint digest validation, inert records."""
    path.mkdir(parents=True)
    segment = f'segment-{score}'
    rows = [dict(event='run_start', schema_version=2, resumed=False, budget=1,
                 config={'budget': 1, 'milestone_fracs': []}, counters={'iteration_no': 0, 'B_evals': 0}),
            dict(event='iteration', iteration_no=1, B_evals=1, island=0, child_id='p1',
                 child_score=score, pop_best_pre=0., island_best_pre=0., mode='base')]
    for seq, row in enumerate(rows, 1):
        row.update(run_id=run_id, segment_id=segment, event_seq=seq)
    raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    provenance = dict(schema='sciel-checkpoint-provenance/1', run_id=run_id,
                      checkpoint_id=f'checkpoint-{score}', counters={'iteration_no': 1, 'B_evals': 1},
                      journal_anchor={'byte_offset': len(raw), 'prefix_sha256': hashlib.sha256(raw).hexdigest(),
                                      'segment_id': segment, 'event_seq': 2})
    state = {'provenance': provenance, 'iteration_no': 1, 'B_evals': 1,
             'db': {'pop_best_id': 'p1', 'programs': [{'id': 'p1', 'score': score}]}}
    checkpoint = json.dumps(state).encode()
    checkpoint_sha = hashlib.sha256(checkpoint).hexdigest()
    for seq, event in enumerate((
            dict(event='checkpoint', provenance=provenance), dict(event='run_end', reason='budget', pop_best=score)), 3):
        event.update(run_id=run_id, segment_id=segment, event_seq=seq, iteration_no=1, B_evals=1,
                     checkpoint_id=provenance['checkpoint_id'], checkpoint_sha256=checkpoint_sha)
        raw += (json.dumps(event) + '\n').encode()
    (path / 'sciel_events.jsonl').write_bytes(raw)
    (path / 'checkpoint.json').write_bytes(checkpoint)
    return path


@pytest.mark.parametrize('alias_kind', ['duplicate', 'dotdot', 'symlink', 'hardlink'])
def test_one_physical_trajectory_never_crosses_split(tmp_path, alias_kind):
    loader = module('dt_dataloader')
    first = legacy_run(tmp_path / 'first')
    second = legacy_run(tmp_path / 'second', offset=.1)
    if alias_kind == 'duplicate':
        alias = first
    elif alias_kind == 'dotdot':
        alias = first / '..' / first.name
    elif alias_kind == 'symlink':
        alias = tmp_path / 'symlink'
        alias.symlink_to(first, target_is_directory=True)
    else:
        alias = tmp_path / 'hardlink'
        alias.mkdir()
        os.link(first / 'sciel_events.jsonl', alias / 'sciel_events.jsonl')
    train, hold = loader.held_out_split([first, second, alias], holdout_frac=.5)
    assert len(train) == len(hold) == 6
    assert {Path(row['run']).resolve() for row in train} == {second / 'sciel_events.jsonl'}
    assert {Path(row['run']).resolve() for row in hold} == {first / 'sciel_events.jsonl'}
    samples, report = loader.load_corpus([first, second, alias])
    assert len(samples) == 12 and len(report['loaded']) == 2
    assert len(report['skipped']) == 1 and 'duplicate run' in report['skipped'][0][1]


def test_single_duplicated_run_cannot_supply_a_training_and_holdout_corpus(tmp_path):
    loader = module('dt_dataloader')
    run = legacy_run(tmp_path / 'run')
    train, hold = loader.held_out_split([run, run])
    assert train == [] and len(hold) == 6


def test_copied_legacy_selected_trajectory_is_deduplicated(tmp_path):
    loader = module('dt_dataloader')
    run = legacy_run(tmp_path / 'run')
    copied = tmp_path / 'copy'
    shutil.copytree(run, copied)
    assert loader.held_out_split([run, copied])[0] == []


@pytest.mark.parametrize('same_bytes', [True, False])
def test_bound_run_identity_does_not_cross_directories_or_selected_versions(tmp_path, same_bytes):
    loader = module('dt_dataloader')
    run = modern_run(tmp_path / 'run', 'one-run')
    if same_bytes:
        copied = tmp_path / 'copy'
        shutil.copytree(run, copied)
    else:
        copied = modern_run(tmp_path / 'version', 'one-run', score=.8)
    other = modern_run(tmp_path / 'other', 'independent-run')
    train, hold = loader.held_out_split([run, other, copied], holdout_frac=.5)
    assert len(train) == len(hold) == 1
    assert Path(train[0]['run']).parent == other
    assert Path(hold[0]['run']).parent == run


def test_dirty_copy_is_not_promoted_by_another_runs_identity(tmp_path):
    loader = module('dt_dataloader')
    run = modern_run(tmp_path / 'run', 'one-run')
    copied = tmp_path / 'tampered'
    shutil.copytree(run, copied)
    (copied / 'checkpoint.json').write_text('{}')
    samples, report = loader.load_corpus([copied, run])
    assert len(samples) == 1 and len(report['loaded']) == 1
    assert 'checkpoint' in report['skipped'][0][1]


def test_actual_checkpoint_linked_resume_and_prefix_stay_on_one_side(tmp_path):
    loader = module('dt_dataloader')
    original = modern_run(tmp_path / 'prefix', 'resumed-run')
    resumed = tmp_path / 'resumed'
    shutil.copytree(original, resumed)
    old = json.loads((resumed / 'checkpoint.json').read_text())
    old_sha = hashlib.sha256((resumed / 'checkpoint.json').read_bytes()).hexdigest()
    raw = (resumed / 'sciel_events.jsonl').read_bytes()
    segment = 'resumed-segment'
    rows = [dict(event='run_start', schema_version=2, resumed=True, budget=2,
                 config={'budget': 2, 'milestone_fracs': []}, counters=old['provenance']['counters']),
            dict(event='checkpoint_restore', ancestry='bound', identity_mismatches=[],
                 checkpoint_id=old['provenance']['checkpoint_id'], checkpoint_sha256=old_sha,
                 provenance=old['provenance'], restored_counters=old['provenance']['counters']),
            dict(event='iteration', iteration_no=2, B_evals=2, island=0, child_id='p2',
                 child_score=.8, pop_best_pre=.5, island_best_pre=.5, mode='base')]
    for seq, row in enumerate(rows, 1):
        row.update(run_id='resumed-run', segment_id=segment, event_seq=seq)
        raw += (json.dumps(row) + '\n').encode()
    provenance = dict(schema='sciel-checkpoint-provenance/1', run_id='resumed-run',
                      checkpoint_id='resumed-checkpoint', counters={'iteration_no': 2, 'B_evals': 2},
                      journal_anchor={'byte_offset': len(raw), 'prefix_sha256': hashlib.sha256(raw).hexdigest(),
                                      'segment_id': segment, 'event_seq': 3})
    state = {'provenance': provenance, 'iteration_no': 2, 'B_evals': 2,
             'db': {'pop_best_id': 'p2', 'programs': [{'id': 'p1', 'score': .5}, {'id': 'p2', 'score': .8}]}}
    checkpoint = json.dumps(state).encode()
    for seq, event in enumerate((dict(event='checkpoint', provenance=provenance),
                                 dict(event='run_end', reason='budget', pop_best=.8)), 4):
        event.update(run_id='resumed-run', segment_id=segment, event_seq=seq, iteration_no=2, B_evals=2,
                     checkpoint_id=provenance['checkpoint_id'], checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest())
        raw += (json.dumps(event) + '\n').encode()
    (resumed / 'sciel_events.jsonl').write_bytes(raw)
    (resumed / 'checkpoint.json').write_bytes(checkpoint)
    assert len(loader.load_corpus([resumed])[0]) == 2
    train, hold = loader.held_out_split([resumed, original])
    assert train == [] and len(hold) == 2
