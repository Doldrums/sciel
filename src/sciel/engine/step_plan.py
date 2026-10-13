"""Preparation of one ordinary search step: parent, operator, knowledge and directive, journaled as decisions."""
from __future__ import annotations

from ..causal import context_of
from ..components.policy.mode_features import mode_features
from .operators import usable_inspirations
from .prompt.build import compose_directive
from .schedule import Saturated
from .siblings import ITEMS
from .types import Snapshot


FRESH_SHARE = 0.2
FRESH_DIRECTIVE = ("\nFRESH DESIGN: the reference program below is the task's starting point, not the current best. "
                   "Write a new design from scratch; use the measured knowledge in this prompt, not the existing lineage.\n")


class EngineStepPlan:

    def _seed_score(self) -> float:
        from .loop_contracts import _read
        initial = _read(self.task.initial_program_path)
        return next((p.score for p in self.db._all.values() if p.origin == "seed" and p.code == initial), 0.0)

    def _families(self) -> list:
        """One strategy card per latent archive cell, or per island best before the archive exists."""
        ids = self.memory.elites if self.memory.ready else [i.best_id for i in self.db.islands if i.best_id]
        cards = [self.db.get(pid).card for pid in ids if self.db.get(pid).card]
        return list(dict.fromkeys(cards))[:8]

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
        arms = [name for name, ready in (("swm", self.swm.ready), ("rl", self.learner.current(self.swm)),
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

    def _prepare_step(self, island: int, *, excluded_parents=(), spawn=True, iteration_no=None,
                      block_room=1 + ITEMS) -> dict:
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
        forced = False
        if self.three_head is not None:
            # #4c: promise head of the shared three-head trunk (default off).
            #
            # It must be resolved BEFORE eff_mode is fixed, and it must not silently
            # discard the controller's tilt. Overwriting after the fact made the mode
            # mixes a decoration -- every iteration executed the head's tilt whatever
            # mode was drawn -- while the reward was credited to a mode that never ran.
            head_tilt = self.three_head.build(view, self.db)
            if head_tilt is not None:
                mode_on_policy, forced = False, True
                tilt = head_tilt
                mode = "exploit"      # a promise tilt IS an exploit, name it honestly
        # Effective mode for reward attribution: a warmup-gated exploit (tilt dropped)
        # actually executed base selection, so its reward belongs to base.
        eff_mode = mode if tilt is not None else "base"
        sampled = isinstance(mix, dict) and mode in mix and preset is None and not forced
        self.decisions.observe("mode", mix if sampled else {mode: 1.0}, mode)
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
        else:
            self.decisions.observe("parent", {selected.parent_id: 1.0}, selected.parent_id)
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
        block = self.siblings.draw(self, island, block_room)
        strategy = finding = "none"
        if block is not None:
            self.decisions.observe("strategy", {"none": 1.0}, "none")
        else:
            strategy = self.decisions.sample("strategy", self.experiments.allocate("strategy", context,
                self.theory.tilt("strategy", context, self.strategies.distribution(island))))
            finding = self.decisions.sample("finding", self.experiments.allocate("finding", context,
                self.theory.tilt("finding", context, self.findings.distribution(island))))
        theorist = None
        if self.strategies.due(iteration_no, self._procedure.theorist_interval):
            best = sorted((p for i in self.db.islands for p in i.top_k(1)), key=lambda p: -p.score)[:2]
            theorist = self.strategies.request(iteration_no, task=self.task_prompt, programs=best,
                                               claims=self.theory.cards(context), findings=self.findings.top(),
                                               families=self._families())
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
            decision_context["origin"] = "fresh"
        repair_code = (failed_code if (op == "error_repair" and failed_code) else
                       _read(self.task.initial_program_path) if origin == "fresh" else parent.code)
        if strategy != "none":
            directive += "\nStrategy to try in this change: " + self.strategies.text(strategy) + "\n"
        directive += self.findings.deliver(finding, island)
        if op == "idea":
            directive += self._idea_target(context)
        directive += self.observed_changes.block(repair_code)
        repair_score = (self._seed_score() if origin == "fresh" else
                        0.0 if repair_code is not parent.code else parent.score)
        decision_context["baseline"] = repair_score
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
        if self.research_tools is not None and origin != "fresh" and block is None:
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
            if self.research_tools.config['independent_research']:
                self.research_tools.culture.detach(island, iteration_no)
        decision_context["decisions"] = self.decisions.drain()
        return {
            "op": op,
            "procedure": self._procedure.id,
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
            "block": block,
        }
