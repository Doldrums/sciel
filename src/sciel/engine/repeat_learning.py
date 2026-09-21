"""Repeat-measurement completion and research comparison bindings."""
from __future__ import annotations

import hashlib
import json
from ..campaign.budget import CampaignStopped
from . import checkpoint
from .runtime import observed_stage


class EngineRepeatLearning:

    @observed_stage("evaluating")
    def _run_repeat_pending(self):
        """Commit ordinary conclusions, then repeats, then any due reflection."""
        imported = self._sync_repeat_comparisons()
        if not self.repeat_promotion.pending or self.repeat_promotion.stopped_reason:
            if imported:
                path = checkpoint.save(self, 'repeat_import')
                self.log.checkpoint(milestone='repeat_import', iteration_no=self.iteration_no,
                                    B_evals=self.B_evals, path=path, **self._saved_checkpoint)
            self._complete_repeat_research()
            return
        for phase in ('before_draws', 'after_draws'):
            if phase == 'after_draws':
                self.repeat_promotion.run_pending()
                self._sync_repeat_comparisons()
            path = checkpoint.save(self, 'repeat_' + phase)
            self.log.checkpoint(milestone='repeat_' + phase, iteration_no=self.iteration_no,
                                B_evals=self.B_evals, path=path, **self._saved_checkpoint)
        self._complete_repeat_research()

    def _complete_repeat_research(self):
        """Flush an owed reflection only beyond a safe terminal repeat boundary.

        This also runs before the budget loop on resume. Pure evidence import
        never dispatches a model, and an unknown/stopped batch cannot do so.
        """
        research, repeat = self.research, self.repeat_promotion
        if research is None or repeat is None or repeat.pending or repeat.stopped_reason:
            return
        interval = research.config['reflection_interval']
        if not interval or research.outcomes - research.reflected_through < interval:
            return
        try:
            self._complete_research_step()
        except CampaignStopped:
            self._finish('research_stopped')
            raise
        # Preserve consumed auxiliary slots and the watermark before another
        # generation. An intervening crash leaves a research journal tail,
        # which checkpoint loading already rejects rather than replaying.
        path = checkpoint.save(self, 'repeat_after_reflection')
        self.log.checkpoint(milestone='repeat_after_reflection', iteration_no=self.iteration_no,
                            B_evals=self.B_evals, path=path, **self._saved_checkpoint)

    def _repeat_research_binding(self, program_id, origin_id=None):
        """Bind exact admitted source/ancestry; never infer a generating attempt."""
        program = self.db.get(program_id)
        source_sha = hashlib.sha256(program.code.encode()).hexdigest()
        node, seen = program, set()
        while node.parent_id and node.parent_id in self.db._all:
            if node.id in seen:
                raise ValueError('cyclic repeat candidate ancestry')
            seen.add(node.id)
            node = self.db._all[node.parent_id]
        origin = self.research.knowledge._evidence().get(origin_id)
        parent = self.db._all.get(program.parent_id)
        # A source can be generated more than once. Only the worker's explicit
        # attempt ID plus its measured child/parent bindings establishes origin.
        if (origin is None or origin.get('status') != 'evaluated'
                or origin.get('source_sha256') != source_sha
                or origin.get('parent_id') != program.parent_id
                or parent is None or origin.get('parent_source_sha256') !=
                hashlib.sha256(parent.code.encode()).hexdigest()
                or origin.get('branch') != node.id or origin.get('score') != program.score):
            origin_id = None
        return {'branch': node.id, 'origin_experiment_id': origin_id,
                'source_sha256': source_sha}

    def _sync_repeat_comparisons(self, coordinator=None):
        """Rebuild terminal evidence from durable H15 receipts before generation.

        Replaying this pure import is idempotent. A crash after the repeat
        decision but before the Engine checkpoint never dispatches another draw.
        """
        coordinator = self.repeat_promotion if coordinator is None else coordinator
        if coordinator is None or self.research is None:
            return 0
        from ..knowledge_base.paired_comparisons import build_comparison, MAX_SOURCE_BYTES, MAX_JSON_BYTES
        imported = 0
        for batch in coordinator.terminal_comparisons():
            binding = batch['research_context']
            if binding is None:
                binding = self._repeat_research_binding(batch['program_id'])
            if binding['source_sha256'] != batch['challenger_source_sha256']:
                raise ValueError('repeat comparison source differs from research binding')
            arguments = dict(namespace=batch['namespace'], batch_id=batch['batch_id'],
                branch=binding['branch'], origin_experiment_id=binding['origin_experiment_id'],
                comparator_source_sha256=batch['comparator_source_sha256'],
                challenger_source_sha256=batch['challenger_source_sha256'],
                adapter_identity=batch['adapter_identity'], limits=batch['limits'],
                pairs=batch['pairs'], provenance=batch['provenance'], task_metric=self.research.metric)
            group = build_comparison(**arguments)
            source_pair = batch['source_pair']
            # Source presentation capacity must not discard otherwise valid
            # measurements. The exact programs remain in the repeat journal's
            # source artifacts even when their optional KB copy cannot fit.
            reason = ('source_byte_limit' if any(len(code.encode('utf-8')) > MAX_SOURCE_BYTES
                       for code in source_pair.values()) else
                      'group_json_limit' if len(json.dumps({**group, 'source_pair': source_pair},
                       sort_keys=True, separators=(',', ':'), allow_nan=False).encode()) > MAX_JSON_BYTES else None)
            if reason is None:
                group = build_comparison(**arguments, source_pair=source_pair)
            else:
                provenance = {**arguments['provenance'], 'source_pair_omission': reason}
                # Even omission metadata must respect the evidence cap. At the
                # exact boundary retain the base measurement unchanged.
                if len(json.dumps({**group, 'provenance': provenance}, sort_keys=True,
                                  separators=(',', ':'), allow_nan=False).encode()) <= MAX_JSON_BYTES:
                    group = build_comparison(**{**arguments, 'provenance': provenance})
            imported += bool(self.research.import_comparison(group))
            self.observed_changes.observe_paired(group)
        return imported
