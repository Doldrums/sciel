"""Authored version-1 history with one generation and completed C/R stages."""

import copy
from pathlib import Path
from tempfile import TemporaryDirectory

from sciel.engine import external_research
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison
from sciel.run_log import RunLog

from restart_admission_fixture import _receipt
from test_circle_intervention_engine import task_copy
from test_external_history_v2_engine import (
    CascadeOutcome, EvalResult, Model, builder, controller,
    digest, research_config, snapshot,
)


def build_legacy_history():
    with TemporaryDirectory(prefix='sciel-legacy-history-fixture-') as folder:
        return _build(Path(folder))


def _build(path):
    task = task_copy(path / 'task', initial=builder.source_for())
    metric = metric_context(task)
    config = {**research_config(), 'comparison_reviews': False}
    f = controller(path, config, model=Model(generation_start=2), metric=metric)
    parent_source = digest(builder.source_for())
    pairs = [{'seed': seed, 'comparator': _receipt(parent_source, seed),
              'challenger': _receipt('b' * 64, seed)} for seed in (11, 12, 13)]
    first = pairs[0]['comparator']
    prior = build_comparison(namespace='authored-prior', batch_id='dense', branch='p1',
        origin_experiment_id=None, comparator_source_sha256=parent_source,
        challenger_source_sha256='b' * 64, adapter_identity=first['adapter_identity'],
        limits=first['limits'], pairs=pairs, provenance={'synthetic': True}, task_metric=metric)
    f.c.import_comparison(prior)
    outcome = CascadeOutcome(EvalResult(True, .6, {}, timing=.01), .6, 'full', False)
    assert f.worker.run(snapshot(f), f.model, lambda source: outcome, base_temp=.2).ok
    record = f.c.knowledge.experiments[-1]
    challenger = record['source_sha256']
    measured = [{'seed': seed, 'comparator': _receipt(parent_source, seed),
                 'challenger': _receipt(challenger, seed, variant=seed + (1 if seed != 12 else -1))}
                for seed in (11, 12, 13)]
    f.c.import_comparison(build_comparison(namespace='authored-observation', batch_id='mixed',
        branch='p1', origin_experiment_id=record['id'], comparator_source_sha256=parent_source,
        challenger_source_sha256=challenger, adapter_identity=first['adapter_identity'],
        limits=first['limits'], pairs=measured, provenance={'synthetic': True}, task_metric=metric))
    result = {'schema': external_research.SCHEMA}

    def capture(stage):
        state = copy.deepcopy(f.c.state())
        if stage == 'source':
            # Version 1 stored queued auxiliary work outside its source snapshot.
            state['queue'] = []
        result[stage] = {
            'manifest_sha256': canonical_sha({'fixture': 'authored v1 history', 'stage': stage}),
            'state': state, 'journal': Path(f.log.path).read_text(),
        }
        f.log.close()

    def start(stage):
        f.log = RunLog(str(path / f'{stage}.jsonl'))
        f.log.run_start(scope='authored v1 auxiliary stage')
        f.c.log = f.log

    capture('source')
    start('conclusion')
    f.c.complete_step(reflect=False)
    capture('conclusion')
    start('reflection')
    f.model.reflection = {'lessons': [], 'retire_ids': []}
    f.c.complete_step()
    capture('reflection')
    return result
