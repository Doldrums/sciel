"""Repeat-measurement completion around ordinary search steps."""
from __future__ import annotations

from . import checkpoint
from .runtime import observed_stage


class EngineRepeatLearning:

    @observed_stage("evaluating")
    def _run_repeat_pending(self):
        """Checkpoint before and after the pending repeat draws."""
        if not self.repeat_promotion.pending or self.repeat_promotion.stopped_reason:
            return
        for phase in ('before_draws', 'after_draws'):
            if phase == 'after_draws':
                self.repeat_promotion.run_pending()
            path = checkpoint.save(self, 'repeat_' + phase)
            self.log.checkpoint(milestone='repeat_' + phase, iteration_no=self.iteration_no,
                                B_evals=self.B_evals, path=path, **self._saved_checkpoint)
