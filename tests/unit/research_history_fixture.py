"""Authored G/C/R history using fixed measurements and an in-memory model."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sciel.engine import external_research_v2 as history
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.run_log import RunLog

from test_external_history_v2_engine import (
    CascadeOutcome, EvalResult, comparison, controller, research_config, snapshot,
)
from test_research_paired_comparisons import group


def build_history(*, task_metric=None):
    with TemporaryDirectory(prefix='sciel-history-fixture-') as directory:
        return _build(Path(directory), task_metric=task_metric)


def _build(path, *, task_metric=None):
    config = {**research_config(), 'comparison_reviews': False}
    f = controller(path, config, metric=task_metric)
    journals, operations, groups = [], [], []

    def begin_journal(index):
        f.log.close()
        f.log = RunLog(str(path / f'journal-{index}.jsonl'))
        f.log.run_start(scope='authored fixed-measurement fixture')
        f.c.log = f.log

    def finish_journal():
        text = Path(f.log.path).read_text()
        journals.append({'manifest_sha256': canonical_sha({
            'fixture': 'authored research history', 'journal_index': len(journals)}),
            'journal': text})
        return [json.loads(line) for line in text.splitlines()]

    def record_comparison(value):
        f.c.import_comparison(value)
        groups.append(value)
        operations.append({'kind': 'comparison', 'comparison_id': value['id']})

    record_comparison(group(metric=f.c.metric, batch='prior-comparison'))
    for index, score in enumerate((.6, .4)):
        begin_journal(index * 3)
        outcome = CascadeOutcome(EvalResult(True, score, {}, timing=.01), score, 'full', False)
        assert f.worker.run(snapshot(f), f.model, lambda source: outcome, base_temp=.2).ok
        record = f.c.knowledge.experiments[-1]
        operations.append({'kind': 'generation', 'journal': len(journals),
                           'experiment_id': record['id']})
        finish_journal()
        if index == 0:
            record_comparison(comparison(record, 1))

        begin_journal(index * 3 + 1)
        f.c.complete_step(reflect=False)
        rows = finish_journal()
        event = next(row for row in rows if row.get('phase') == 'conclusion')
        operations.append({'kind': 'conclusion', 'journal': len(journals) - 1,
                           'event_seq': event['event_seq']})
        if index == 1:
            value = group(record=record, metric=f.c.metric, batch='second-comparison',
                          radii=(.02, .005, .02))
            record_comparison(value)

        begin_journal(index * 3 + 2)
        f.model.reflection = {'lessons': [], 'retire_ids': []}
        f.c.complete_step()
        rows = finish_journal()
        event = next(row for row in rows if row.get('phase') == 'reflection')
        assert event['error'] is None
        operations.append({'kind': 'reflection', 'journal': len(journals) - 1,
                           'event_seq': event['event_seq'], 'interval': 1})

    final = f.c.state()
    f.log.close()
    first = final['knowledge']['experiments'][0]
    capsule = {
        'schema': history.SCHEMA, 'profile': history.PROFILE, 'metric': f.c.metric,
        'root_parent': {'id': first['parent_id'], 'source_sha256': first['parent_source_sha256']},
        'policy': history.policy_identity(), 'journals': journals,
        'comparison_catalog': {value['id']: copy.deepcopy(value) for value in groups},
        'operations': operations,
        'terminal': {f'{key}_sha256': canonical_sha(final.get(key, {})) for key in (
            'knowledge', 'conclusions', 'comparison_reviews')},
    }
    return capsule, final
