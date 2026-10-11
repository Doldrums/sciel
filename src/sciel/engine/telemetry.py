"""Run accounting, observations and published progress metrics."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from ..theory.strategies import PREAMBLE
from . import checkpoint


class EngineTelemetry:

    def _record_llm_accounting(self, response):
        # Logical consumption, including WAL replay, occurs on the coordinator.
        # Physical worker traces do not call this. A rollback restores this
        # checkpointed counter before replay, so completed calls count once.
        from .llm.client import response_accounting
        try:
            accounting = response_accounting(response)
        except ValueError:
            return  # The owning role preserves its existing unknown-cost stop.
        self.llm_unpriced_calls += accounting['unpriced_calls']

    def _charge_research(self, cost):
        self.cost_total += cost

    def _llm_trace_context(self, prompt):
        context = dict(getattr(self.log, '_llm_context', {}))
        # A preceding request's context must never label a different call.
        if context.pop('prompt_sha256', None) != hashlib.sha256(prompt.encode('utf-8')).hexdigest():
            context = dict(role='unknown', iteration=self.iteration_no + 1)
        self.log._llm_context = {}
        if prompt.startswith(PREAMBLE):
            context = dict(role='theorist', iteration=self.iteration_no + 1)
        if prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE\n'):
            try:
                header = json.loads(prompt.split('\n', 2)[1])
            except (ValueError, IndexError):
                header = {}
            if type(header) is dict:
                context = dict(role='research_tool', **{key: header[key] for key in
                    ('action_id', 'island_id', 'parent_program_id', 'iteration', 'attempt')
                    if key in header})
        return dict(context, run_id=self.log.run_id, segment_id=self.log.segment_id,
                    journal_event_seq=self.log.event_seq)

    def _log_iteration(self, res, selected, op_prop, op_stat, ev, brier, model,
                       model_prop, mode, mode_prop, *,
                       pre, phase_pre, effective_mode, decision_context=None) -> None:
        extras = {}
        if decision_context is not None:
            extras['decision_context'] = deepcopy(decision_context)
        # These *_pre fields describe marginal reward at commit. The separately
        # frozen decision_context is the state actually available to mode choice.
        self.log.iteration(
            iteration_no=self.iteration_no, B_evals=self.B_evals, island=res.island,
            parent_id=res.parent_id, child_id=getattr(self, "_child_id", None),
            child_code=res.child_code,
            pop_best_pre=pre["pop_best"], island_best_pre=pre["island_best"],
            phase_pre=phase_pre, progress_pre=self._progress(self.B_evals - 1),
            effective_mode=effective_mode,
            parent_propensity=selected.propensity,
            model_used=model, model_propensity=model_prop,
            mode=mode, mode_propensity=mode_prop,
            op_used=res.meta["op_used"], op_propensity=op_prop,
            op_stat=op_stat, meta=res.meta, prompt=res.prompt, response=res.response,
            bet=({"mechanism": res.bet.mechanism, "q": res.bet.q}
                 if res.bet else None), bet_brier=brier, **extras,
            child_score=res.score, cases=res.cases, reached=res.reached,
            confirmed=res.confirmed, raw_first_score=res.raw_first_score,
            r_raw=ev.r_raw, r_norm=ev.r_norm, d_pop=ev.d_pop, d_isl=ev.d_isl,
            is_island_event=ev.is_island_event, is_pop_event=ev.is_pop_event,
            reward_med=ev.med, reward_normalization=deepcopy(ev.normalization),
            cost=res.cost, timing=res.timing,
            pop_best=self.db.pop_best, island_best=self.db.island_best(res.island))

    def record_stats(self) -> dict:
        """Generated-child improvements divided by the harmonic normalization H_n.

        The starting bar includes seeds, portfolio starts and any spawned programs.
        Only successful generated children increment n or the record counter. H_n is
        the expected record count for continuous exchangeable observations without
        a starting bar; ties and the additional starts here mean this ratio is a
        descriptive normalization, not an exact permutation test or p-value.
        """
        n = self.B_evals - getattr(self, 'n_remeasurement_successes', 0)
        expected = sum(1.0 / k for k in range(1, n + 1)) if n else 0.0
        return {"records": self.n_records, "records_expected": round(expected, 4),
                "record_ratio": round(self.n_records / expected, 4) if expected > 0 else 0.0}

    def _per_run_budget(self) -> float:
        """The wall-clock a single run of the candidate actually has.

        Three things can bind and the smallest wins: the smoke stage, which runs the
        program once; the full stage divided by the number of trials it performs; and
        an explicit per-run cap the task declares because its evaluator enforces one
        internally. Quoting the stage timeout alone told a candidate on
        function_minimization it had 30s when each of its ten runs had at most 5."""
        caps = [self.task.smoke_timeout, self.task.full_timeout]
        trials = max(1, int(self.cfg.get("_eval_trials", 0) or self._observed_trials))
        if trials > 1:
            caps.append(self.task.full_timeout / trials)
        if self.task.per_run_timeout:
            caps.append(float(self.task.per_run_timeout))
        return max(1e-9, min(caps))

    def _emit_metrics(self) -> None:
        if self.sink is None:
            return
        gain = self.db.pop_best - self.seed_score
        spd = (gain / self.cost_total if self.cost_total > 0 else 0.0) if not self.llm_unpriced_calls else None
        metrics = {
            "best/pop": self.db.pop_best,
            "events/intensity": self.intensity.upper_bound(),
            "phase/schedule": self.schedule.phase_code(),
            "health/status": self.health,
            "search/record_ratio": self.record_stats()["record_ratio"],
            "cost/usd_total": self.cost_total,
            "cost/unpriced_llm_calls": self.llm_unpriced_calls,
            "cost/score_per_dollar": spd,
            "evaluation/remeasurements": self.n_remeasurements,
        }
        for i in range(len(self.db.islands)):
            metrics[f"best/island_{i}"] = self.db.island_best(i)
        self.sink.on_eval(self.B_evals, metrics)

    def _maybe_milestone(self) -> None:
        for m in self.schedule.milestones_for(self._schedule_horizon()):
            if self.B_evals >= m and m not in self.milestones_fired:
                self.milestones_fired.add(m)
                self.schedule.on_milestone(m, self.db, self.log)
                self.db.reencode_niches()   # arch-v4 #4b: AURORA re-encode of drifted cells
                if (self.cfg.get("checkpoint_on_milestone", True)
                        and getattr(self, '_parallel_transaction', None) is None):
                    path = checkpoint.save(self, m)
                    self.log.checkpoint(milestone=m, iteration_no=self.iteration_no,
                                        B_evals=self.B_evals, path=path, **self._saved_checkpoint)

    def _plateau_shadow(self, island: int, is_event: bool) -> None:
        mon = self.plateau_monitors[island]
        mon.observe(is_event, phase=self.schedule.phase_code())
        if mon.newly_declared():
            self.log.decision(policy="plateau_shadow", action="plateau", island=island,
                              t_seg=mon.t_seg, wealth=round(mon.wealth, 3) if math.isfinite(mon.wealth) else None,
                              log_wealth=mon.log_wealth(),
                              phase=self.schedule.phase_code(), shadow=True)

    def _oc_observe(self, island: int, is_event: bool) -> None:
        mon = self.oc_monitors[island]
        phase = self.schedule.phase_code()
        mon.observe(is_event, phase=phase)
        if mon.newly_declared():
            self.log.decision(policy="opportunity_cost", action="plateau", island=island,
                              forecast=round(mon.forecast(), 4), threshold=mon.threshold,
                              n=mon.n, events=mon.k, phase=phase,
                              shadow=not self.oc_shift)
            if self.oc_shift:
                self.controller.on_plateau(island, self.iteration_no,
                                           islands=len(self.db.islands))

    def _model_propensity(self, model: str) -> float:
        weights = getattr(self.llm, "weights", None)
        models = getattr(self.llm, "models", None)
        if not weights or not models or model not in models:
            return 1.0
        total = sum(weights)
        return weights[models.index(model)] / total if total > 0 else 1.0
