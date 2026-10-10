"""Candidate admission, population credit and post-result learning."""
from __future__ import annotations

from ..components.tail import extract_lesson
from .types import NoChild
from .loop_contracts import (_changed_region, _trunc)


class EngineAdmission:

    def _observe_program(self, program, result, *, parent=None):
        """Retain measured edit facts without changing admission, reward or lessons."""
        from .evaluation_timing import remember
        remember(self, program, result)
        # SH migration copies a measured source under another ID. Bind that
        # explicit ancestry to the old receipt; it is not a fresh measurement.
        def bind_migrant(node, seen):
            if node is None or node.origin != 'migrant':
                return
            if node.id in seen:
                raise ValueError('cyclic observed migration ancestry')
            original = self.db._all.get(node.parent_id)
            if original is None:
                return
            bind_migrant(original, seen | {node.id})
            self.observed_changes.bind_alias(node, original)

        bind_migrant(parent, set())
        measured = getattr(result, 'result', result)
        observation = dict(ok=bool(measured.ok), score=measured.score,
            cases=measured.cases, artifacts=getattr(measured, 'artifacts', {}),
            reached=getattr(result, 'reached', None),
            confirmed=bool(getattr(result, 'confirmed', False)))
        record = self.observed_changes.observe(program, observation, parent=parent)
        if record is not None:
            self.log.decision(policy='observed_change', action=record['status'],
                iteration_no=self.iteration_no, evidence=record)
        return record

    def _novelty_check(self, code, island):
        result = self.novelty_gate.inspect(
            code, (p.code for p in self.db.islands[island].programs.values()))
        self.log.decision(policy="code_novelty", action="accept" if result["accept"] else "reject",
                          island=island, iteration_no=self.iteration_no, **result)
        return result

    def _candidate_novelty_callback(self, island):
        registry = getattr(self, 'candidate_novelty', None)
        if registry is None:
            return ((lambda code: self._novelty_check(code, island))
                    if getattr(self, 'novelty_gate', None) is not None else None)

        def inspect(code):
            result = registry.inspect(code)
            self.log.decision(policy='candidate_novelty',
                action=('remeasurement' if result['remeasurement'] else
                        'accept' if result['accept'] else 'reject'),
                island=island, iteration_no=self.iteration_no, **result)
            return result

        inspect.candidate_registry = True
        inspect.max_resamples = self.cfg['candidate_novelty']['max_resamples']
        return inspect

    def _bind_novelty_program(self, program, *, evaluation_id=None):
        if evaluation_id is not None:
            from .research_artifacts import bind_evaluation_program
            bind_evaluation_program(self, program, evaluation_id)
        registry = getattr(self, 'candidate_novelty', None)
        if registry is not None:
            registry.bind_program(program.code, program.id, program.island, evaluation_id=evaluation_id)
        outputs = getattr(self, 'output_novelty', None)
        if outputs is not None and evaluation_id is not None:
            outputs.bind_program(evaluation_id, program.id)

    def _on_result(self, res, selected, op_prop, op_stat, model, model_prop,
                   injected_ids, mode, mode_prop, eff_mode, obs_phase, mode_on_policy,
                   *, decision_context=None) -> str | None:
        self.health = 0          # the degraded outcomes below raise it
        if isinstance(res, NoChild):
            self.n_nochild += 1
            self.cost_total += getattr(res, "cost", 0.0)   # the calls still cost money
            if res.reason == "dup_cap":
                self.schedule.note_saturation(res.island)
                self.health = 2
            else:
                self.health = 1
            self.log.iteration(iteration_no=self.iteration_no, B_evals=self.B_evals,
                               island=res.island, child=None, reason=res.reason,
                               meta=res.meta, model_used=model, model_propensity=model_prop,
                               mode=mode, mode_propensity=mode_prop,
                               parent_propensity=selected.propensity,
                               op_propensity=op_prop, op_stat=op_stat,
                               cost=getattr(res, "cost", 0.0),
                               **({"decision_context": decision_context} if decision_context else {}))
            # an attempt that produced nothing is still an attempt under this mode
            self.controller.observe(
                obs_phase, eff_mode, None,
                on_policy=mode_on_policy)
            return
        self.cost_total += res.cost
        if res.meta.get('remeasurement'):
            # Re-observing a known source is not evidence for a new mutation or
            # an RL action. A formerly failed source can first enter the pool
            # after a successful measurement, with its own explicit origin.
            program = None
            if res.ok:
                self.B_evals += 1
                self.n_remeasurement_successes += 1
                # Mutation-side remeasurement is admitted only after all prior
                # measurements failed. A failed seed can nevertheless exist in
                # the pool at a sentinel score: retain it as historical evidence
                # and add this first successful observation separately.
                program = self.db.add(res.child_code, res.score, res.cases,
                    res.island, origin='remeasurement')
                self._bind_novelty_program(program, evaluation_id=res.meta.get('evaluation_id'))
                self._observe_program(program, res)
                self.lineage_fail.pop(res.parent_id, None)
            else:
                self.n_eval_fail += 1
                self.health = 1
            self.log.iteration(iteration_no=self.iteration_no, B_evals=self.B_evals,
                island=res.island, child=None, reason='remeasurement',
                measurement_program_id=program.id if program else None,
                measurement_ok=res.ok, measurement_score=res.score if res.ok else None,
                pop_best=self.db.pop_best, n_remeasurements=self.n_remeasurements,
                meta=res.meta, parent_id=res.parent_id,
                model_used=model, cost=res.cost, timing=res.timing,
                prompt=res.prompt, response=res.response,
                artifacts=_trunc(res.artifacts),
                **({"decision_context": decision_context} if decision_context else {}))
            return program.id if program else None
        if not res.ok:
            # Keep the CHILD's source, not just its error. The repair prompt used to
            # show the parent under the heading "FAILED PROGRAM": in 243 of 291 real
            # repair prompts the shown program had a strictly positive score and the
            # traceback belonged to code the model never saw.
            from .research_targets import failed_provenance
            self.lineage_fail[res.parent_id] = {
                **(res.artifacts or {"error": "eval_failed"}),
                "_failed_code": res.child_code or "",
                "_failure_provenance": failed_provenance(res,
                    iteration=self.iteration_no, B_evals=self.B_evals)}
            self.n_eval_fail += 1
            self.health = 1
            self.log.iteration(iteration_no=self.iteration_no, B_evals=self.B_evals,
                               island=res.island, child=None, reason="eval_failed",
                               meta=res.meta, parent_id=res.parent_id,
                               model_used=model, model_propensity=model_prop,
                               mode=mode, mode_propensity=mode_prop,
                               parent_propensity=selected.propensity,
                               op_propensity=op_prop, op_stat=op_stat,
                               prompt=res.prompt, response=res.response,
                               cost=res.cost, timing=res.timing,
                               artifacts=_trunc(res.artifacts),
                               **({"decision_context": decision_context} if decision_context else {}))
            self.controller.observe(
                obs_phase, eff_mode, None,
                on_policy=mode_on_policy)
            return

        pre = self.db.pre_snapshot(res.island)
        ev = self.reward.improvement_event(res.score, pre, origin="gen",
                                            iter_ref=self.iteration_no)
        if not res.meta["escalated"] and res.meta.get('op_on_policy', True):
            self.op_bandit.observe(res.meta["op_used"], ev)
        if not res.meta.get('research_candidate'):
            self.selection.observe(res.parent_id, ev)
        # An iteration that ran under the plateau mix was not drawn from the bandit's
        # own distribution, so it is not evidence about the bandit's arms.
        self.controller.observe(
            obs_phase, eff_mode, ev,
            on_policy=mode_on_policy)
        self.B_evals += 1
        if hasattr(self.db, "progress"):
            self.db.progress = self._progress()
        if ev.is_pop_event:
            self.n_records += 1
        if ev.is_pop_event:
            self._last_improve_eval = self.B_evals
            self._last_pop_improve = self.B_evals
            self._last_pop_improve_iteration = self.iteration_no
        from .prompt.parse import extract_card
        if self.idea is not None and res.meta.get('idea_card') is not None:
            from .idea import card_text
            saved_card = card_text(res.meta['idea_card'])
        else:
            saved_card = ((res.meta.get('hypothesis') or {}).get('text', '')[:200]
                          if self.research is not None else extract_card(res.response or ""))
        child_prog = self.db.add(res.child_code, res.score, res.cases, res.island,
                                 origin=res.origin, parent_id=res.parent_id,
                                 card=saved_card)
        self._bind_novelty_program(child_prog, evaluation_id=res.meta.get('evaluation_id'))
        parent = self.db._all.get(res.parent_id)
        self._observe_program(child_prog, res, parent=parent)
        if self.repeat_promotion is not None:
            self._repeat_candidate = (child_prog.code, dict(child_prog.cases), child_prog.id)
            self._repeat_candidate_origin = res.meta.get('research_id')
        # Retain the existing reported trial-count scheduling hint separately
        # from wall timing. _observe_program records its stage-inclusive scope.
        from .evaluation_timing import trial_count
        self._observed_trials = trial_count((res.cases or {}).get('n_trials')) or 1
        if self.db.evictions:
            for e in self.db.evictions:
                self.log.decision(policy="eviction", action="drop",
                                  iteration_no=self.iteration_no, **e)
        self.db.evictions.clear()
        # Structured observations supersede the legacy ID-only score warning.
        # They follow both source texts, including aliases and evicted programs.
        # Equal scores alone never establish equal behavior or failed mechanisms.
        self.noop_parents.discard(res.parent_id)
        self._child_id = child_prog.id            # lineage edge for the iteration log
        if self.log_population:
            self.log.population(iteration_no=self.iteration_no, B_evals=self.B_evals,
                                phase=obs_phase, **self.db.population_snapshot())
        self.schedule.note_progress(res.island)
        self.lineage_fail.pop(res.parent_id, None)
        if res.meta["dup_attempts"] > 0:
            self.n_dups += 1
        self.intensity.observe(ev.is_island_event)
        if ev.is_island_event:
            self.n_events += 1
        if self.plateau_shadow:
            self._plateau_shadow(res.island, ev.is_island_event)
        if self.oc_monitors is not None:
            self._oc_observe(res.island, ev.is_island_event)
        brier = None
        if res.bet is not None:
            brier = (res.bet.q - (1.0 if ev.is_island_event else 0.0)) ** 2
        branch_brier = None
        if self.kb_on and not res.meta.get('research_candidate'):
            branch_brier = self._run_tail(res, ev, injected_ids, brier, child_prog)
        self._log_iteration(res, selected, op_prop, op_stat, ev, brier, model,
                            model_prop, mode, mode_prop, branch_brier,
                            pre=pre, phase_pre=obs_phase, effective_mode=eff_mode,
                            decision_context=decision_context)
        return child_prog.id

    def _run_tail(self, res, ev, injected_ids, brier, child_prog):
        island, phase = res.island, self.schedule.phase_code()
        if self.c2_on and brier is not None:
            self.branch_cal.note(res.parent_id, brier)
        for lid in injected_ids:
            self.kb.note_exposure(lid, ev.is_island_event,
                                  self.base_rate.rate(island, phase))
        self.base_rate.observe(island, phase, ev.is_island_event)
        if ev.is_island_event:
            if self.c2_on:
                lesson = self._mint_postmortem(res, island)
            else:
                # kb_free: the lesson is the strategy card the LLM already wrote
                # for this event-child -- no extra call, no code-alignment gate
                card = child_prog.card
                lesson = (self.kb.add_lesson(card, card, "", self.iteration_no, align=False)
                          if card else None)
            self.n_postmortems += 1
            self.log.postmortem(iter_ref=self.iteration_no, island=island,
                                mechanism=(res.bet.mechanism if res.bet else ""),
                                lesson=(lesson.text if lesson else None),
                                aligned=lesson is not None)
        return self.branch_cal.mean(res.parent_id) if self.c2_on else None

    def _mint_postmortem(self, res, island):
        parent = self.db.get(res.parent_id)
        mech = res.bet.mechanism if res.bet else ""
        result, cost = extract_lesson(self.llm, self._postmortem_model(),
                                      parent.code, res.child_code, mech)
        self.cost_total += cost
        if result is None:
            return None
        diff = _changed_region(parent.code, res.child_code)
        return self.kb.add_lesson(result[0], result[1], diff, self.iteration_no)

    def _postmortem_model(self):
        models = getattr(self.llm, "models", ["unknown"])
        return models[-1]
