"""Regressions for reviewed source selection, A/B, records, SH and resume bugs."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sciel.config import load_config, validate
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.engine.schedule import Schedule
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.stats import PlateauMonitor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from record_ratio import ratio_for
from run_ab_revision import paired_summary


class CounterLLM:
    models = ['counter']
    weights = [1.0]

    def __init__(self):
        self.n = 0

    def choose_model(self, rng):
        return 'counter'

    def generate(self, prompt, model, temperature=None):
        self.n += 1
        return LLMResponse(text=f'```python\ndef run_search():\n    return ({self.n}, 0)\n```',
                           model='counter', cost=0.0, prompt_tokens=0, completion_tokens=0)


def config(**overrides):
    return dict(task='test/toy_quadratic', num_islands=1, budget=20, embed_grid=False,
                portfolio_seed=False, idea_island=False, dynamic_islands=False,
                policy={'enabled': False}, milestone_fracs=[], op_fixed='full_rewrite',
                llm={'mock': True}, **overrides)


def engine(path, cfg=None):
    task = load_task(str(ROOT / 'tasks/test/toy_quadratic'))
    return Engine(cfg or config(), task, str(path), CounterLLM(), python=sys.executable)


@pytest.mark.parametrize('name', ['heilbronn_default', 'circle', 'func_min'])
def test_standard_configs_resolve_audited_local_tasks_without_network(name, tmp_path):
    from test_loaded_descriptor_identity import artifacts

    cfg = validate(load_config(str(ROOT / f'configs/{name}.yaml')))
    cfg['wandb'] = {'enabled': False}
    # Keep the profile's learned storage enabled. Its external model/axis files
    # are supplied explicitly by this authored fixture, not repository artifacts.
    assert cfg['embed_grid'] is True
    embed, embed_hash, axis, _ = artifacts(tmp_path)
    cfg.update(embed_grid_path=str(embed), embed_axis_path=str(axis))
    with patch('huggingface_hub.snapshot_download', side_effect=AssertionError('unexpected HF fetch')):
        e = build_engine(cfg, str(ROOT / 'tasks'), str(tmp_path), llm=CounterLLM())
    try:
        assert e.descriptor_info['embed_sha'] == embed_hash[:12]
        assert e.descriptor_info['kind'].startswith('pca')
        assert Path(e.task.root).is_relative_to(ROOT / 'tasks')
        assert 'mini' not in e.task.tiers
        assert 'b_dispersion' in e.task.case_metrics
        if name == 'heilbronn_default':
            assert e.task.stochastic is True
    finally:
        e.log.close()


def test_explicit_task_root_overrides_pinned_hf_and_provenance(tmp_path):
    cfg = {**config(), 'sciel_tasks_repo': 'old/repo', 'sciel_tasks_revision': 'old-revision'}
    with patch('huggingface_hub.snapshot_download', side_effect=AssertionError('unexpected HF fetch')):
        e = build_engine(cfg, str(ROOT / 'tasks'), str(tmp_path), llm=CounterLLM(),
                         tasks_root_override=True)
    try:
        assert e.cfg['sciel_tasks_repo'] is None
        assert e.cfg['sciel_tasks_revision'] is None
        assert cfg['sciel_tasks_revision'] == 'old-revision'  # caller's config unchanged
    finally:
        e.log.close()


def test_cli_passes_explicit_task_root(tmp_path, monkeypatch):
    from sciel.cli import main as cli
    import yaml
    cfgpath = tmp_path / 'config.yaml'
    cfgpath.write_text(yaml.safe_dump(config()))
    calls = []
    def build(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(run=lambda: {})
    monkeypatch.setattr(cli, 'build_engine', build)
    assert cli.main(['run', str(cfgpath), '--tasks-root', str(ROOT / 'tasks'),
                     '--outputs', str(tmp_path), '--run-id', 'cli']) == 0
    assert calls[0]['tasks_root_override'] is True


def test_records_and_training_schema_exclude_portfolio_improvement(tmp_path):
    e = engine(tmp_path)
    e.seed_score = 0.1
    e.db.add('def run_search(): return (-1, 0)', 0.1, {}, 0, 'seed')
    e.db.add('def run_search(): return (-2, 0)', 0.9, {}, 0, 'portfolio')
    e._evaluate = lambda code: CascadeOutcome(EvalResult(True, 0.5), 0.5, 'full')
    summary = e.run()
    metric = ratio_for(str(tmp_path / 'sciel_events.jsonl'))
    assert metric['records'] == summary['records'] == 0
    assert metric['ratio'] == summary['record_ratio'] == 0
    events = [json.loads(l) for l in (tmp_path / 'sciel_events.jsonl').read_text().splitlines()]
    children = [r for r in events if r['event'] == 'iteration' and 'child_score' in r]
    assert len(children) == 20
    for i, row in enumerate(children):
        assert row['pop_best_pre'] == row['island_best_pre'] == 0.9
        assert row['phase_pre'] == 0
        assert row['progress_pre'] == i / 20
        assert row['child_code'] == e.db.get(row['child_id']).code
    assert events[-1]['records'] == 0


def write_journal(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def child(i, record=False, **kwargs):
    return dict(event='iteration', B_evals=i, child_score=0.5,
                is_pop_event=record, **kwargs)


def test_offline_metric_uses_events_without_population_logging(tmp_path):
    p = tmp_path / 'events.jsonl'
    write_journal(p, [child(i, i in (3, 9)) for i in range(1, 21)]
                  + [{'event': 'run_end', 'B_evals': 20}])
    metric = ratio_for(str(p))
    assert metric['records'] == 2
    assert metric['n'] == 20


def test_offline_metric_rejects_legacy_checkpoint_replay_without_ancestry(tmp_path):
    p = tmp_path / 'events.jsonl'
    rows = [dict(event='iteration', B_evals=i, child_score=0.5, d_pop=0.1 if i > 10 else 0)
            for i in range(1, 26)]
    rows += [child(i, False) for i in range(11, 21)]
    rows += [{'event': 'run_end', 'B_evals': 20}]
    write_journal(p, rows)
    assert ratio_for(str(p)) is None


def test_offline_metric_rejects_population_only_or_missing_events(tmp_path):
    p = tmp_path / 'events.jsonl'
    write_journal(p, [{'event': 'population', 'pop_best': i / 20} for i in range(20)])
    assert ratio_for(str(p)) is None
    write_journal(p, [child(i) for i in range(2, 22)])
    assert ratio_for(str(p)) is None


def test_offline_metric_rejects_open_resumed_segment(tmp_path):
    p = tmp_path / 'events.jsonl'
    rows = [child(i) for i in range(1, 21)]
    rows += [{'event': 'run_end', 'B_evals': 20, 'reason': 'budget'},
             {'event': 'run_start', 'B_evals': 10}, child(11, True)]
    write_journal(p, rows)
    assert ratio_for(str(p)) is None


def test_ab_pooling_excludes_both_arms_of_incomplete_pairs():
    rows = [dict(arm=arm, seed=seed, B_evals=20, records=seed,
                 expected=2.0, reason='budget')
            for arm in ('base', 'head') for seed in (1, 2, 3)]
    rows[1].update(B_evals=5, reason='backstop', records=None, expected=None)
    rows[-1].update(records=None, expected=None)
    summary = paired_summary(rows, [1, 2, 3, 4], 20)
    assert summary['included_seeds'] == [1]
    assert [pair['seed'] for pair in summary['excluded_pairs']] == [2, 3, 4]
    assert summary['arms']['base'] == summary['arms']['head']
    assert summary['arms']['base']['ratio'] == 0.5
    assert set(summary['excluded_pairs'][-1]['reasons']) == {'base', 'head'}


class ScheduleDB:
    scores = {0: .9, 1: .8, 2: .7, 3: .6}
    def __init__(self):
        self.migrated = []
    def island_best(self, i):
        return self.scores[i]
    def migrate_unique_elites(self, source, survivors, **kwargs):
        self.migrated.append(source)
        return 0


def test_successive_halving_cuts_only_live_islands_and_clears_cooldown():
    s = Schedule(3)
    db = ScheduleDB()
    log = SimpleNamespace(sh_cut=lambda **kw: None)
    s.note_saturation(0)
    s.on_milestone(50, db, log)
    assert s.survivors == [0, 1]
    assert s.cooldown == {}
    s.on_milestone(75, db, log)
    assert s.survivors == [0]
    assert db.migrated == [2, 1]


def test_late_spawn_joins_live_halving_pool():
    s = Schedule(3, min_fair_evals=2)
    s.dispatched = {0: 5, 1: 5, 2: 5}
    db = ScheduleDB()
    log = SimpleNamespace(sh_cut=lambda **kw: None)
    s.on_milestone(50, db, log)
    s.register_island(3)
    s.on_milestone(75, db, log)
    assert 2 not in s.survivors
    assert 3 in s.survivors  # new island protected until it has fair exposure
    assert db.migrated.count(2) == 1


def test_resume_restores_spawned_shadow_monitor_and_trial_count(tmp_path):
    cfg = config(plateau_shadow=True)
    original = engine(tmp_path / 'source', cfg)
    restored = engine(tmp_path / 'restored', cfg)
    try:
        original.db.add_island()
        original.schedule.register_island(1)
        original.plateau_monitors[1] = PlateauMonitor()
        original.plateau_monitors[1].observe(True, 0)
        original._observed_trials = 8
        ckpt = checkpoint.save(original, 'test')
        checkpoint.load(restored, ckpt)
        assert restored.plateau_monitors[1].state() == original.plateau_monitors[1].state()
        assert restored._observed_trials == 8
    finally:
        original.log.close()
        restored.log.close()


def test_ab_requires_explicit_base_before_reading_config(tmp_path, monkeypatch, capsys):
    import run_ab_revision as comparison

    def unexpected(*args, **kwargs):
        raise AssertionError('no config, Git, environment or search work before parsing')

    for name in ('load_config', 'git', 'load_env', 'run_arm'):
        monkeypatch.setattr(comparison, name, unexpected)
    with pytest.raises(SystemExit) as rejected:
        comparison.main(['--dry-run', '--outputs', str(tmp_path / 'unused')])
    assert rejected.value.code == 2
    assert 'base_ref' in capsys.readouterr().err
    assert not (tmp_path / 'unused').exists()
    with pytest.raises(SystemExit) as help_request:
        comparison.main(['--help'])
    assert help_request.value.code == 0
    help_text = ' '.join(capsys.readouterr().out.split())
    assert 'explicit local Git revision' in help_text and 'candidate arm is HEAD' in help_text


def test_ab_preflight_imports_both_actual_revisions(tmp_path, monkeypatch):
    import shutil
    import run_ab_revision as comparison
    import yaml

    # Author two revisions from maintained source; never read retired Git history.
    checkout = tmp_path / 'checkout'
    shutil.copytree(ROOT / 'src', checkout / 'src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', '*.dist-info'))
    shutil.copytree(ROOT / 'tasks/test/toy_quadratic', checkout / 'tasks/test/toy_quadratic')
    shutil.copy2(ROOT / 'pyproject.toml', checkout / 'pyproject.toml')

    def fixture_git(*args):
        return subprocess.check_output(
            ['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
             '-c', 'commit.gpgsign=false', '-c', 'core.hooksPath=/dev/null', *args],
            cwd=checkout, text=True).strip()

    fixture_git('init', '--quiet')
    fixture_git('add', 'src', 'tasks', 'pyproject.toml')
    fixture_git('commit', '--quiet', '-m', 'Authored base')
    base_ref = fixture_git('rev-parse', 'HEAD')
    module = checkout / 'src/sciel/__init__.py'
    module.write_text(module.read_text() + '\nAB_FIXTURE_REVISION = "candidate"\n')
    fixture_git('add', 'src/sciel/__init__.py')
    fixture_git('commit', '--quiet', '-m', 'Authored candidate')
    head_ref = fixture_git('rev-parse', 'HEAD')

    monkeypatch.setattr(comparison, 'ROOT', checkout)
    monkeypatch.setattr(comparison, 'load_env', lambda *args: None)

    def no_search(*args):
        raise AssertionError('dry-run must not start a search')

    monkeypatch.setattr(comparison, 'run_arm', no_search)
    cfgpath = tmp_path / 'config.yaml'
    cfgpath.write_text(yaml.safe_dump(config()))
    assert comparison.main(['--dry-run', '--outputs', str(tmp_path / 'comparisons'),
                            base_ref, str(cfgpath), '42']) == 0
    manifest_path, = (tmp_path / 'comparisons').glob('*/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    base, head = manifest['arms']['base'], manifest['arms']['head']
    assert manifest['base_revision'] == base_ref and manifest['head_revision'] == head_ref
    assert base['module'] != head['module']
    assert head['module'] == str(checkout / 'src/sciel/__init__.py')
    assert base['source_sha256'] != head['source_sha256']
    assert base['task_hash'] == head['task_hash']
    assert manifest['seeds'] == [42]
    assert not list(manifest_path.parent.glob('*_s42'))
