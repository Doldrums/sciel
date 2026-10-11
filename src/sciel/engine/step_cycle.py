"""Preparation, execution and commit phases of one ordinary search step."""
from __future__ import annotations

from copy import deepcopy
from ..causal import context_of
from ..components.policy.mode_features import mode_features
from .operators import usable_inspirations
from .prompt.build import compose_directive
from .schedule import Saturated
from .types import Snapshot
from .runtime import observed_stage


FRESH_SHARE = 0.2
CONFIRM_TOLERANCE = 1e-6
FRESH_DIRECTIVE = ("\nFRESH DESIGN: the reference program below is the task's starting point, not the current best. "
                   "Write a new design from scratch; use the measured knowledge in this prompt, not the existing lineage.\n")


class EngineStepCycle:

    @observed_stage("generating")
    def _step(self, island: int) -> None:
        plan = self._prepare_step(island)
        result = self._execute_step(plan)
        self._commit_step(plan, result)

    def _confirm(self, result, pop_best, parent_score) -> None:
        """Re-measure a new record, or any improving time-dependent program; inconsistent results keep the minimum."""
        from .determinism import time_dependent
        if not getattr(result, 'ok', False) or result.meta.get('remeasurement'):
            return
        clocked = time_dependent(result.child_code)
        if not (result.score > pop_best or (clocked and result.score > parent_score)):
            return
        second = self._confirmation_score(result.child_code)
        consistent = second is not None and abs(second - result.score) <= CONFIRM_TOLERANCE * max(1.0, abs(result.score))
        result.meta['confirmation'] = {'first': result.score, 'second': second, 'time_dependent': clocked,
                                       'consistent': consistent}
        if not consistent:
            result.score = min(result.score, second) if second is not None else 0.0

    def _idea_target(self, context: str) -> str:
        """An idea-island proposal aims at a dense unexplored behavior, at an open question of the theory, or at neither."""
        from ..swm import directive as behavior_directive, propose
        from ..theory.claims import open_question, question_text
        behavior = propose(self.db, self.rng.derive_seed("swm"), "kde")
        claim = open_question(self.theory.claims, context)
        arms = ["none"] + (["behavior"] if behavior else []) + (["claim"] if claim else [])
        choice = self.decisions.sample("idea_target", self.experiments.allocate("idea_target", context,
            self.theory.tilt("idea_target", context, {a: 1 / len(arms) for a in arms})))
        if choice == "behavior":
            return "\nIdea target: " + behavior_directive(behavior) + "\n"
        if choice == "claim":
            return "\nIdea target: design the new strategy so that its result answers this open question: " \
                + question_text(claim) + "\n"
        return ""

    def _planners(self) -> dict:
        arms = [name for name, ready in (("swm", self.swm.ready), ("rl", self.learner.ready),
                ("heritage", self.heritage is not None and self.heritage.ready)) if ready]
        share = self._procedure.base_share
        return {"base": share} | {a: (1 - share) / len(arms) for a in arms} if arms else {"base": 1.0}

    def _plan_operator(self, planner, parent_id, context, distribution):
        if planner == "swm":
            return self.swm.tilt(parent_id, context, "operator", distribution)
        if planner == "rl":
            return self.learner.policy(self.swm, parent_id, context, distribution)
        if planner == "heritage":
            return self.heritage.operator(parent_id, context, distribution)
        return distribution

    def _prepare_step(self, island: int, *, excluded_parents=(), spawn=True, iteration_no=None) -> dict:
        iteration_no = self.iteration_no if iteration_no is None else iteration_no
        self._procedure = self.procedures.get(self.decisions.sample("procedure", self.procedures.distribution(island)))
        self.theory.shift, self.theory.mix = self._procedure.shift, self._procedure.mix
        self._repeat_candidate = None
        self._repeat_candidate_origin = None
        if spawn:
            self._maybe_spawn_island()
        view = self._island_view(island)
        if excluded_parents:
            view.ids[:] = [pid for pid in view.ids if pid not in excluded_parents]
            if not view.ids:
                self.decisions.pending.clear()
                raise Saturated()
        phase = self.schedule.phase_code()
        decision_context = dict(schema='sciel-mode-decision/1', phase=phase,
            progress=self._progress(), pop_best=self.db.pop_best,
            island_best=self.db.island_best(island), B_evals=self.B_evals,
            iteration_no=iteration_no)
        features = None
        if self.controller.mode_mlp is not None:
            # pre-step, feature-parity context for the return-conditioned MLP.
            features = mode_features(phase, decision_context['progress'],
                                     decision_context['pop_best'], decision_context['island_best'])
        context = context_of(decision_context)
        mode, mode_prop, tilt, mix = self.controller.pick_mode(
            island, iteration_no, self.rng.stream("policy"), view, phase, features,
            adjust=lambda m: self.decisions.floor("mode", self.experiments.allocate("mode", context,
                                                                                     self.theory.tilt("mode", context, m))))
        # Capture the action's source before the worker advances iteration_no or
        # result handling changes the plateau window.
        mode_on_policy = (self.controller.enabled
                          and not self.controller.in_plateau(island, iteration_no)
                          and self.controller.mode_mlp is None)
        preset = None
        if self.preset_weights is not None:
            mode_on_policy = False
            from ..components.policy.tilt import Tilt, elite_target, uniform_target
            preset, preset_prop = self.preset_weights.sample(self.rng.stream("preset"))
            mode, mode_prop = preset, preset_prop
            tilt = (Tilt(1.0, uniform_target) if preset == "explore" else
                    Tilt(1.0, lambda v: elite_target(v, 1)) if preset == "exploit" else None)
            mix = self.preset_weights.probabilities()
            self.log.decision(policy="preset_weights", action=preset, propensity=preset_prop,
                              distribution=mix, island=island, iteration_no=iteration_no,
                              adaptive=self.cfg.get("preset_adaptive", True))
        if self.three_head is not None:
            # #4c: promise head of the shared three-head trunk (default off).
            #
            # It must be resolved BEFORE eff_mode is fixed, and it must not silently
            # discard the controller's tilt. Overwriting after the fact made the mode
            # mixes a decoration -- every iteration executed the head's tilt whatever
            # mode was drawn -- while the reward was credited to a mode that never ran.
            head_tilt = self.three_head.build(view, self.db)
            if head_tilt is not None:
                mode_on_policy = False
                tilt = head_tilt
                mode = "exploit"      # a promise tilt IS an exploit, name it honestly
        # Effective mode for reward attribution: a warmup-gated exploit (tilt dropped)
        # actually executed base selection, so its reward belongs to base.
        eff_mode = mode if tilt is not None else "base"
        self.decisions.observe("mode", mix if isinstance(mix, dict) and mode in mix and preset is None
                               else {mode: 1.0}, mode)
        if self.controller.enabled:
            # mix is the distribution actually sampled from this iteration (learned,
            # plateau, or static) -- not the bandit's estimate when plateau overrode it.
            self.log.decision(policy="controller", action=mode, propensity=mode_prop,
                              island=island, iteration_no=iteration_no, phase=phase,
                              mode_dist=mix,
                              plateau=self.controller.in_plateau(island, iteration_no))
        if self.cfg.get("container", {}).get("selection") == "centres":
            selected = self.db.select_parent(island, self.rng.stream("parent"),
                floor=self.cfg.get("selection_floor", 1e-3),
                tau_floor=self.cfg.get("tau_floor", 1e-3), tilt=tilt)
            self.selection.visits[selected.parent_id] = self.selection.visits.get(selected.parent_id, 0) + 1
            self.log.decision(policy="container_parent", action=selected.parent_id,
                propensity=selected.propensity, island=island, iteration_no=iteration_no,
                **selected.debug)
        else:
            selected = self.selection.select(view, tilt=tilt, rng=self.rng.stream("parent"))
        parent_distribution = (None if self.cfg.get("container", {}).get("selection") == "centres"
                               else getattr(self.selection, "last_distribution", None))
        if parent_distribution is not None and selected.parent_id in parent_distribution:
            self.decisions.observe("parent", parent_distribution, selected.parent_id)
        parent = self.db.get(selected.parent_id)
        decision_context["parent_id"] = parent.id
        decision_context["population"] = [p.id for p in self.db.islands[island].top_k(16)]
        planner = self.decisions.sample("planner", self.experiments.allocate("planner", context,
            self.theory.tilt("planner", context, self._planners())))
        fresh_failure = selected.parent_id in self.lineage_fail
        # Freeze usable donor inputs before choosing the operator. Scan the
        # ranked local pool before applying the three-source prompt limit, so
        # exact-source aliases at its head cannot hide a later distinct donor.
        source = self.decisions.sample("inspiration_source", self.experiments.allocate("inspiration_source",
            context, self.theory.tilt("inspiration_source", context,
                {"island": 0.5, "archive": 0.5} if self.memory.ready else {"island": 1.0})))
        donors = (self.db.inspirations(island, selected.parent_id, k=len(self.db.islands[island].programs))
                  if source == "island" else [self.db.get(pid) for pid in self.memory.neighbours(
                      self.swm, parent.id, {parent.id})])
        inspirations = usable_inspirations(parent.id, parent.code,
            [{'program_id': p.id, 'code': p.code, 'score': p.score, 'card': p.card} for p in donors])
        # exploit-on-stall: pop_best has idled for `refine_stall` evals and this
        # parent is within `refine_elite_frac` of the best -> deepen the incumbent
        # (bias diff, hand a reformulation directive) instead of re-exploring.
        # Explicit exploration/diversity modes retain their search scope. Resolve
        # this after all selection overrides, using the same effective mode that
        # receives reward. Forced-operator paths also retain their own intent.
        forced_op = bool(self.cfg.get("op_fixed")) or (
            self._idea_island is not None and island == self._idea_island)
        stall = self.B_evals - self._last_pop_improve
        stalled = stall >= int(self.cfg.get("refine_stall", 8))
        refine_eligible = (not forced_op) and stalled and parent.score >= (
            float(self.cfg.get("refine_elite_frac", 0.9)) * max(self.db.pop_best, 1e-9))
        exploring = eff_mode in {"explore", "card_diversity"}
        refine = self.decisions.flag("refine", refine_eligible and not exploring)
        if self.cfg.get("op_fixed"):
            op, op_prop, op_stat = self.cfg["op_fixed"], 1.0, {"fixed": self.cfg["op_fixed"]}
            op_selection = 'fixed'
        elif self._idea_island is not None and island == self._idea_island:
            # idea island (arch-v4 #2, QDAIF-style): parent selection stands, but
            # the operator is forced to idea-space -- mutate the strategy, then
            # regenerate code from it, instead of editing the current program
            op, op_prop, op_stat = "idea", 1.0, {"idea_island": True}
            op_selection = 'idea_island'
        else:
            op, op_prop, op_stat = self.op_bandit.pick(
                fresh_failure, self.rng.stream("op"), intensify=refine,
                cross_available=bool(inspirations),
                adjust=lambda p: self.decisions.floor("operator", self.experiments.allocate("operator", context,
                    self.theory.tilt("operator", context, self._plan_operator(planner, parent.id, context, p)))))
            op_selection = 'bandit'
        self.decisions.observe("operator", self.op_bandit.last_distribution if op_selection == 'bandit'
                               else {op: 1.0}, op)
        operator_decision = dict(requested=op, effective=op, selection=op_selection,
                                 fallback_reason=None, on_policy=op_selection == 'bandit')
        if op == 'cross' and not inspirations:
            op, op_prop = 'full_rewrite', None
            operator_decision.update(effective=op, fallback_reason='no_usable_crossover_donor',
                                     on_policy=False)
            op_stat = {**op_stat, 'requested': 'cross', 'effective': op,
                       'fallback_reason': operator_decision['fallback_reason'], 'on_policy': False}
            self.log.decision(policy='operator_eligibility', action=op, island=island,
                iteration_no=iteration_no, parent_id=parent.id, **operator_decision)
        if refine:
            self.log.decision(policy="refine", action="intensify", island=island,
                              parent_id=selected.parent_id, stall=stall, op=op,
                              mode=mode, effective_mode=eff_mode,
                              parent_score=round(parent.score, 4),
                              pop_best=round(self.db.pop_best, 4))
        elif refine_eligible and exploring:
            self.log.decision(policy="refine", action="skip", island=island,
                              iteration_no=iteration_no, parent_id=selected.parent_id,
                              stall=stall, op=op, mode=mode, effective_mode=eff_mode,
                              reason="exploration_mode")
        model = self.llm.choose_model(self.rng.stream("prompt"))
        model_prop = self._model_propensity(model)
        models, weights = getattr(self.llm, "models", None), getattr(self.llm, "weights", None)
        self.decisions.observe("model", {m: w / sum(weights) for m, w in zip(models, weights)}
                               if models and weights and model in models else {model: 1.0}, model)
        strategy = self.decisions.sample("strategy", self.experiments.allocate("strategy", context,
            self.theory.tilt("strategy", context, self.strategies.distribution(island))))
        finding = self.decisions.sample("finding", self.experiments.allocate("finding", context,
            self.theory.tilt("finding", context, self.findings.distribution(island))))
        theorist = None
        if self.strategies.due(iteration_no, self._procedure.theorist_interval):
            best = sorted((p for i in self.db.islands for p in i.top_k(1)), key=lambda p: -p.score)[:2]
            theorist = self.strategies.request(iteration_no, task=self.task_prompt, programs=best,
                                               claims=self.theory.cards(context), findings=self.findings.top())
        lessons, injected_ids = self.theory.cards(context), []
        # the binding cap on ONE run of the program
        # Population stall is not evidence that this particular lineage stalled.
        # Competitive parents may receive refinement in base/exploit mode;
        # exploration retains the structural-change push. An active island
        # detector can independently request a structural change.
        controller_plateau = (self.controller.enabled
                              and self.controller.in_plateau(island, iteration_no))
        directive = compose_directive(
            self._per_run_budget(),
            self.eval_times.get(selected.parent_id),
            plateau=(controller_plateau or stalled) and not refine,
            parent_noop=selected.parent_id in self.noop_parents,
            refine=refine, population_stalled=stalled,
            parent_timing_scope='evaluation_stage_mean')
        failure = dict(self.lineage_fail.get(selected.parent_id, {}))
        failed_code = failure.pop("_failed_code", "")
        failure_ref = failure.pop("_failure_provenance", None)
        # error_repair repairs the program that FAILED, which is the child, not the
        # parent that is still sitting in the population with a valid score
        origin = self.decisions.sample("origin", self.experiments.allocate("origin", context,
            self.theory.tilt("origin", context, {"parent": 1 - FRESH_SHARE, "fresh": FRESH_SHARE}
                             if op in ("diff", "full_rewrite", "cross") else {"parent": 1.0})))
        if origin == "fresh":
            from .loop_contracts import _read
            op, op_prop, inspirations = "full_rewrite", None, []
            failed_code = ""
            operator_decision.update(effective=op, fallback_reason='fresh_design', on_policy=False)
            op_stat = {**op_stat, 'effective': op, 'fallback_reason': 'fresh_design', 'on_policy': False}
            directive += FRESH_DIRECTIVE
        repair_code = (failed_code if (op == "error_repair" and failed_code) else
                       _read(self.task.initial_program_path) if origin == "fresh" else parent.code)
        if strategy != "none":
            directive += "\nStrategy to try in this change: " + self.strategies.text(strategy) + "\n"
        directive += self.findings.deliver(finding, island)
        if op == "idea":
            directive += self._idea_target(context)
        directive += self.observed_changes.block(repair_code)
        repair_score = 0.0 if repair_code is not parent.code else parent.score
        snap = Snapshot(
            island=island, op=op, model=model, iteration_no=iteration_no,
            parent_id=selected.parent_id, parent_code=repair_code,
            parent_score=repair_score, inspirations=inspirations,
            artifacts=failure,
            directive=directive, kb_lessons=lessons,
            code_hashes=self.db.code_hashes(island),
            escalation_seed=self.rng.derive_seed("escalation"),
            task_prompt=self.task_prompt, bet_enabled=True,
            required_entrypoints=self.required_entrypoints,
            operator_decision=operator_decision)
        if self.research_tools is not None and origin != "fresh":
            from .research_tool_signals import snapshot_signals
            from .research_targets import freeze_targets
            from .research_search_evidence import search_evidence
            snap.research_search_evidence = search_evidence(self, parent)
            snap.research_targets = freeze_targets(self, parent, repair_code,
                failed=bool(op == "error_repair" and failed_code), failure=failure_ref)
            from .research_cells_context import compact_context
            signals = snapshot_signals(self, island=island, iteration_no=iteration_no, phase=phase,
                mode=mode, effective_mode=eff_mode, refine=refine, forced_operator=forced_op,
                stall=stall, stalled=stalled, controller_plateau=controller_plateau)
            snap.research_cells_context = compact_context(getattr(self.research_tools, 'context_view', self.research_tools.context)(snap),
                max_chars=min(self.cfg['research_tools']['max_context_chars'], 10000))
            self.research_tools.culture.context = context
            self.research_tools.culture.state = parent.id
            snap.tool_episode_plan = self.research_tools.prepare(snap, signals=signals)
        decision_context["decisions"] = self.decisions.drain()
        return {
            "op": op,
            "snap": snap,
            "selected": selected,
            "op_prop": op_prop,
            "op_stat": op_stat,
            "model": model,
            "model_prop": model_prop,
            "injected_ids": injected_ids,
            "mode": mode,
            "mode_prop": mode_prop,
            "eff_mode": eff_mode,
            "phase": phase,
            "decision_context": decision_context,
            "mode_on_policy": mode_on_policy,
            "preset": preset,
            "preset_prop": preset_prop if preset is not None else None,
            "theorist": theorist,
        }

    def _execute_step(self, plan, *, llm=None, evaluate_fn=None):
        from . import loop as _public
        snapshot = plan['snap']
        episode = None
        submitted = []
        evaluator = evaluate_fn or self._evaluation_callback(snapshot.island)
        novelty_check = self._candidate_novelty_callback(snapshot.island)
        def evaluate_tool_candidate(source, provenance):
            if len(submitted) >= 3 or any(row.meta.get('evaluation_count', 0) for row in submitted):
                raise ValueError('one research candidate submission is allowed per episode')
            from .research_candidate import evaluate_submission
            candidate, feedback = evaluate_submission(source, provenance, snapshot,
                evaluator, novelty_check)
            submitted.append(candidate)
            return feedback
        if self.research_tools is not None:
            snapshot = deepcopy(snapshot)
        if snapshot.tool_episode_plan is not None:
            episode = self.research_tools.execute(snapshot.tool_episode_plan, snapshot,
                llm or self.llm, io=getattr(self, '_parallel_io', None),
                candidate_evaluator=evaluate_tool_candidate)
        if self.research_tools is not None:
            from .research_cells_context import merge_context
            from ..theory.findings import fresh
            context_limit = min(self.cfg['research_tools']['max_context_chars'], 10000)
            prior_cards = snapshot.research_cells_context
            fresh_cards = episode.get('cards', []) if episode else []
            # Eligibility precedes the bounded merge: an unsupported fresh
            # interpretation must not evict an older measured observation.
            for_current = []
            for context in (prior_cards, fresh_cards):
                # Authenticate measured host assessments as well as terminal
                # citations before either channel reaches ordinary generation.
                context = self.research_tools.lifecycle.delivery_context(context, episode=episode)
                for_current.append(context)
            cards = merge_context(*for_current, max_chars=context_limit)
            snapshot.research_cells_context = cards
            if episode is not None:
                snapshot.directive += fresh(episode.get('findings', []))
        worker_options = {}
        if novelty_check is not None:
            worker_options["novelty_check"] = novelty_check
        result = submitted[-1] if submitted else _public.run_worker(plan["op"], snapshot, llm or self.llm,
                            evaluator,
                         dup_cap=self.cfg["dup_cap"], base_temp=self.cfg.get("base_temp", 0.7),
                          **worker_options)
        if submitted:
            result.meta['dup_attempts'] = len(submitted) - 1
            result.meta['novelty_checks'] = [check for item in submitted
                for check in item.meta.get('novelty_checks', [])]
            attempts = [deepcopy(attempt) for item in submitted
                for attempt in item.meta.get('validation_trace', {}).get('attempts', [])]
            for index, attempt in enumerate(attempts, 1):
                attempt['attempt'] = index
            result.meta['validation_trace']['attempts'] = attempts
        if snapshot.operator_decision is not None:
            decision = snapshot.operator_decision
            for key, value in dict(op_requested=decision['requested'],
                    op_initial=decision['effective'], op_fallback_reason=decision['fallback_reason'],
                    operator_selection=decision['selection'],
                    op_on_policy=bool(decision['on_policy'] and not result.meta.get('escalated'))).items():
                result.meta.setdefault(key, value)
        if episode is not None:
            result.meta['research_tools'] = episode
        self._confirm(result, plan['decision_context']['pop_best'], snapshot.parent_score)
        if plan.get('theorist'):
            reply = (llm or self.llm).generate(plan['theorist'], snapshot.model)
            result.meta['strategy_proposal'] = reply.text
            result.cost += reply.cost or 0.0
        return result

    def _commit_step(self, plan, res):
        selected = plan["selected"]
        op_prop = plan["op_prop"]
        op_stat = plan["op_stat"]
        model = plan["model"]
        model_prop = plan["model_prop"]
        injected_ids = plan["injected_ids"]
        mode = plan["mode"]
        mode_prop = plan["mode_prop"]
        eff_mode = plan["eff_mode"]
        phase = plan["phase"]
        mode_on_policy = plan["mode_on_policy"]
        if res.meta.get('research_candidate'):
            # The model's research method, not the sampled mutation/mode/preset,
            # generated this exact candidate. Keep its reward off those policies.
            mode_on_policy = False
        preset = plan["preset"]
        preset_prop = plan["preset_prop"]
        pre_best = self.db.pop_best
        candidate_link = None
        if self.research_tools is not None and (res.meta.get('research_tools') is not None
                or getattr(res, 'origin', None) == 'research_tool'):
            from .research_candidate import candidate_binding
            candidate_link = candidate_binding(self.research_tools, res.meta.get('research_tools'), res)
        self.iteration_no += 1
        child_id = self._on_result(res, selected, op_prop, op_stat, model, model_prop, injected_ids,
                        mode, mode_prop, eff_mode, phase, mode_on_policy,
                        decision_context=plan.get('decision_context'))
        if plan.get('theorist'):
            adopted = self.strategies.adopt(res.meta.get('strategy_proposal'), self.iteration_no)
            self.log.decision(policy="strategy", action="propose", iteration_no=self.iteration_no,
                              strategy=adopted.id if adopted else None, text=adopted.text if adopted else None)
        if self.research_tools is not None and res.meta.get('research_tools') is not None:
            added = self.findings.add(res.meta['research_tools'].get('findings', []), self.iteration_no)
            if added:
                self.log.decision(policy="finding", action="add", iteration_no=self.iteration_no,
                                  findings=[{"id": f.id, "text": f.text, "island": f.island} for f in added])
            self.research_tools.commit(res.meta['research_tools'], child_program_id=child_id,
                link_kind='remeasurement' if res.meta.get('remeasurement') else 'mutation',
                **({'candidate_binding': candidate_link, 'child_source': self.db._all[child_id].code}
                   if child_id is not None and candidate_link is not None else {}))
        if (preset is not None and self.cfg.get("preset_adaptive", True)
                and not res.meta.get('remeasurement') and not res.meta.get('research_candidate')):
            self.preset_weights.update(preset, float(self.db.pop_best > pre_best), preset_prop)
        # Once per iteration, whatever the outcome. Emitting only from the success
        # path -- which sets health=0 immediately before -- pinned health/status to 0
        # in every run, so the board could never show a degraded state.
        self._emit_metrics()
