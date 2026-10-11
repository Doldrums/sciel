"""Engine run ownership, dispatch, stopping and terminal publication."""
from __future__ import annotations

import os
from ..campaign.budget import CampaignStopped
from ..components.selection.novelty_chain import IslandView
from . import checkpoint
from .schedule import Saturated
from .runtime import RuntimeStatus, observed_stage
from .loop_contracts import _evaluator_metadata, _model_names, _read


class EngineLifecycle:

    def run(self) -> dict:
        self.completed_result = None
        if self._existing_run and not getattr(self, '_restored_checkpoint', None):
            self.close()
            raise ValueError('Run directory contains an experiment; use resume or a new run directory')
        if getattr(self, '_accounting_stopped_reason', None):
            self.close()
            raise CampaignStopped(self._accounting_stopped_reason)
        runtime = RuntimeStatus(self.run_dir, lambda: {
            "run_id": self.log.run_id, "segment_id": self.log.segment_id,
            "iteration_no": self.iteration_no, "B_evals": self.B_evals,
            **({'parallel': self._parallel_status} if self._parallel_status is not None else {}),
        })
        self._runtime_status = runtime
        terminal = dict(status='failed', reason='startup_failed', error_type=None)
        try:
            runtime.start()
            try:
                result = self._run_dispatch()
            except BaseException as exc:
                if getattr(exc, 'current_call_cost_unknown', False):
                    self.cost_total += exc.known_cost
                    self._accounting_stopped_reason = 'LLM charge is unknown; reconcile before continuation'
                    self.log.decision(policy='cost_accounting', action='unknown_cost',
                                      cost=None, known_cost=exc.known_cost,
                                      generation_returned=getattr(exc, 'generation_returned', True),
                                      generation_dispatch=getattr(exc, 'generation_dispatch', 'returned'),
                                      llm_error=getattr(exc, 'llm_error', None),
                                      validation_trace=getattr(exc, 'validation_trace', None))
                    self._finish('accounting_stopped')
                elif hasattr(exc, 'unbooked_known_cost'):
                    # A worker can stop after paid, returned proposals without
                    # producing a Result. Preserve those charges exactly once;
                    # unlike the branch above, this does not invent unknown cost.
                    self.cost_total += exc.unbooked_known_cost
                    self.log.decision(policy='cost_accounting', action='processing_stopped',
                        cost=exc.unbooked_known_cost, known_cost=exc.unbooked_known_cost,
                        current_call_cost_unknown=False,
                        generation_returned=getattr(exc, 'generation_returned', True),
                        generation_dispatch=getattr(exc, 'generation_dispatch', 'returned'),
                        stage=getattr(exc, 'stage', None),
                        validation_trace=getattr(exc, 'validation_trace', None))
                    self._finish('processing_stopped')
                raise
        except BaseException as exc:
            status = ("stopped" if isinstance(exc, CampaignStopped) else
                      "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed")
            terminal = dict(status=status, reason=type(exc).__name__, error_type=type(exc).__name__)
            raise
        else:
            # Cleanup may fail after scientific work and final persistence have
            # completed. Preserve that result for auditing callers while still
            # propagating the cleanup error; it must never become a false success.
            self.completed_result = result
            terminal = dict(status='paused' if result.get('reason') == 'paused' else 'completed',
                            reason=result.get('reason'))
            return result
        finally:
            try:
                # Stop the heartbeat even if checkpoint persistence itself fails,
                # before releasing ownership to the next engine instance.
                runtime.finish(**terminal)
            finally:
                self._runtime_status = None
                self.close()

    def _run_dispatch(self) -> dict:
        if self.paired_batches is not None:
            from .repeat_population_support import run
            return run(self)
        return self._run()

    def _run(self) -> dict:
        # task_hash covers only the task files; the prompt-side contract is
        # engine behavior, so its version is journaled separately -- journals
        # with different engine_contract are not prompt-comparable
        from . import loop as _public
        self.log.run_start(seed=self.cfg["random_seed"], budget=self.budget,
                           counters=checkpoint.counters(self),
                           restored_checkpoint=getattr(self, "_restored_checkpoint", None),
                           task=self.task.qualified_name, task_hash=self.task.task_hash,
                           engine_contract=_public.ENGINE_CONTRACT,
                           num_islands=self.cfg["num_islands"],
                           schedule_policy=self.cfg["schedule_policy"],
                           bet_enabled=self.cfg["bet_enabled"], models=_model_names(self.llm),
                           sciel_tasks_revision=self.cfg.get("sciel_tasks_revision", "local"),
                           non_comparable=self.cfg.get("sciel_tasks_revision") is None,
                           descriptor=self.descriptor_info,
                           # the task's declared target, recorded so a run's numbers
                           # can be read against it later without reopening task.yaml
                           known_best=self.task.known_best,
                           tiers=list(self.task.tiers),
                           evaluator=({**_evaluator_metadata(self.task, self.cfg),
                               'backend': 'candidate_only', 'candidate_isolation': 'unprivileged_container_values_only',
                               'scope': 'candidate_container_and_trusted_scorer_container',
                               'execution_identity': self.evaluation_backend.identity}
                               if self.evaluation_backend is not None else _evaluator_metadata(self.task, self.cfg)),
                           config=self.cfg)
        from .recovery import apply_provider_recovery
        apply_provider_recovery(self)
        if self.cfg.get('parallel_search', {}).get('enabled'):
            from .parallel import run_parallel
            return run_parallel(self)
        if self.research is not None and self.research.stopped_reason:
            raise CampaignStopped(self.research.stopped_reason)
        if self.repeat_promotion is not None:
            self.repeat_promotion.start(self.log.run_id)
            self._run_repeat_pending()
        try:
            if self.db.pop_best_id is None:
                self.seed()
            elif self._portfolio_bootstrap is not None:
                warm = self.cfg.get('warm_start')
                self._portfolio_seed(warm['source'] if warm else _read(self.task.initial_program_path))
        except CampaignStopped as exc:
            if self.research is not None and not getattr(exc, 'current_call_cost_unknown', False):
                self._finish('research_stopped')
            raise
        if self.repeat_promotion is not None and not self.repeat_promotion.stopped_reason:
            if self.repeat_promotion.plan_bootstrap(_read(self.task.initial_program_path)):
                self._run_repeat_pending()
        reason = "budget"
        backstop = self.cfg.get("max_iterations")
        if self.budget is not None:
            failure_limit = self.cfg["backstop_k"] * self.budget
            backstop = min(backstop, failure_limit) if backstop is not None else failure_limit
        while self.budget is None or self.B_evals < self.budget:
            if self.repeat_promotion is not None and self.repeat_promotion.stopped_reason:
                reason = 'repeat_stopped'
                break
            islands = list(range(len(self.db.islands)))
            if self._pause_requested() or (self._stop_at is not None and self.B_evals >= self._stop_at):
                reason = "paused"
                break
            if backstop is not None and self.iteration_no >= backstop:
                reason = "backstop"
                break
            try:
                island = self.schedule.pick_island(islands, {i: 0 for i in islands})
            except Saturated:
                reason = "saturated"
                break
            try:
                if self.repeat_promotion is not None:
                    self.log.repeat_search(action='generation_intent',
                        iteration_no=self.iteration_no, B_evals=self.B_evals, island=island)
                self._step(island)
            except CampaignStopped as exc:
                if self.repeat_promotion is not None and self._repeat_candidate is not None:
                    code, cases, program_id = self._repeat_candidate
                    self.repeat_promotion.note_unexamined_candidate(code, cases, program_id,
                        reason='post_evaluation_research_stopped:' + type(exc).__name__)
                    self._repeat_candidate = None
                if self.research is not None and not getattr(exc, 'current_call_cost_unknown', False):
                    self._finish('research_stopped')
                raise
            if self.repeat_promotion is not None and self._repeat_candidate is not None:
                code, cases, program_id = self._repeat_candidate
                self._repeat_candidate = None
                repeat_options = {}
                if self.research is not None:
                    repeat_options['research_context'] = self._repeat_research_binding(
                        program_id, self._repeat_candidate_origin)
                self._repeat_candidate_origin = None
                if self.repeat_promotion.plan_candidate(code, cases, program_id, **repeat_options):
                    self._run_repeat_pending()
            if self.repeat_promotion is not None:
                # A rejected/unplanned candidate or NoChild has no pending
                # numeric batch, but can still complete the ordinary cadence.
                self._complete_repeat_research()
            self._maybe_milestone()
            interval = self.cfg.get("checkpoint_interval")
            if interval and self.iteration_no % interval == 0:
                self._periodic_checkpoint()
        if self.repeat_promotion is not None and self.repeat_promotion.stopped_reason:
            reason = 'repeat_stopped'
        self._finish(reason)
        result = {"reason": reason, "B_evals": self.B_evals,
                "pop_best": self.db.pop_best, "iteration_no": self.iteration_no,
                "cost_total": self.cost_total, "llm_unpriced_calls": self.llm_unpriced_calls, "n_events": self.n_events,
                "n_remeasurements": self.n_remeasurements,
                "n_evaluation_results": self.n_evaluation_results,
                "n_output_repeats": self.n_output_repeats,
                **self.record_stats()}
        if self.research is not None:
            result['research'] = self.research.summary()
        if self.idea is not None:
            result['idea_two_stage'] = self.idea.summary()
        if self.intervention is not None:
            result['circle_intervention'] = self.intervention.summary()
        if self.circle_generation is not None:
            result['circle_generation'] = self.circle_generation.summary()
        if self.repeat_promotion is not None:
            result['repeat_promotion'] = self._repeat_export
        return result

    def request_stop(self):
        """Pause at the next complete iteration, after pending research work."""
        self._stop_requested = True

    def _pause_requested(self):
        if not self._stop_requested:
            try:
                os.unlink(os.path.join(self.run_dir, "STOP"))
            except FileNotFoundError:
                pass
            else:
                self._stop_requested = True
        return self._stop_requested

    def _schedule_horizon(self):
        # Unlimited search keeps the original finite SH/progress schedule; the
        # horizon schedules adaptation, never the termination of the run.
        return self.cfg["schedule_horizon"] if self.budget is None else self.budget

    def _progress(self, evaluations=None):
        count = self.B_evals if evaluations is None else evaluations
        value = count / max(1, self._schedule_horizon())
        return min(1.0, value) if self.budget is None else value

    @observed_stage("checkpointing")
    def _periodic_checkpoint(self):
        path = checkpoint.save(self, "periodic")
        self.log.checkpoint(milestone="periodic", iteration_no=self.iteration_no,
                            B_evals=self.B_evals, path=path, **self._saved_checkpoint)

    @observed_stage("checkpointing")
    def _finish(self, reason: str) -> None:
        path = checkpoint.save(self, "final")
        self.log.checkpoint(milestone="final", iteration_no=self.iteration_no,
                            B_evals=self.B_evals, path=path, reason=reason, **self._saved_checkpoint)
        if self.repeat_promotion is not None:
            from .repeat_support import export
            self._repeat_export = export(self)
        if self.paired_batches is not None:
            from .repeat_population_support import export
            self._repeat_population_export = export(self)
        from ..culture import export as export_repertoire
        repertoire = export_repertoire(self)
        self.log.decision(policy="heritage", action="export", path=str(repertoire),
                          claims=sum(c.decided for c in self.theory.claims),
                          strategies=sum(s.status != "retired" for s in self.strategies.pool))
        self.log.run_end(reason=reason, B_evals=self.B_evals,
                         checkpoint_id=self._saved_checkpoint["checkpoint_id"],
                         checkpoint_sha256=self._saved_checkpoint["checkpoint_sha256"],
                         iteration_no=self.iteration_no, pop_best=self.db.pop_best,
                         seed_score=self.seed_score, cost_total=self.cost_total,
                         llm_unpriced_calls=self.llm_unpriced_calls,
                         n_events=self.n_events, n_dups=self.n_dups,
                         n_eval_fail=self.n_eval_fail, n_nochild=self.n_nochild,
                         n_remeasurements=self.n_remeasurements,
                         n_evaluation_results=self.n_evaluation_results,
                         n_output_repeats=self.n_output_repeats,
                         islands=[self.db.island_best(i) for i in range(len(self.db.islands))],
                         **self.record_stats())
        self.log.close()
        if self.repeat_promotion is not None:
            self.repeat_promotion.close()
        if self.sink is not None:
            self.sink.finish()

    def _island_view(self, island: int) -> IslandView:
        isl = self.db.islands[island]
        ids = list(isl.programs.keys())
        scores = {pid: isl.programs[pid].score for pid in ids}
        return IslandView(ids=ids, scores=scores, inspiration_ids={})
