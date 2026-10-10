"""Preparation, execution and commit phases of one ordinary search step."""
from __future__ import annotations

import json
from copy import deepcopy
from ..causal import context_of
from ..components.policy.mode_features import mode_features
from .operators import usable_inspirations
from .prompt.build import compose_directive
from .schedule import Saturated
from .types import Snapshot
from .runtime import observed_stage


class EngineStepCycle:

    @observed_stage("generating")
    def _step(self, island: int) -> None:
        plan = self._prepare_step(island)
        result = self._execute_step(plan)
        self._commit_step(plan, result)

    def _prepare_step(self, island: int, *, excluded_parents=(), spawn=True, iteration_no=None) -> dict:
        iteration_no = self.iteration_no if iteration_no is None else iteration_no
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
            adjust=lambda m: self.theory.tilt("mode", context, m))
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
        card_before_visit = None
        card_cfg = self.cfg.get('idea_card_selection', {})
        if card_cfg.get('enabled') and island == self._idea_island:
            from ..components.selection.idea_cards import prepare_card_tilt
            tilt, card_details = prepare_card_tilt(
                view, {pid: self.db.get(pid).card for pid in view.ids}, self.selection.visits,
                beta=card_cfg['beta'], neighbors=card_cfg['neighbors'])
            if tilt is not None:
                mode_on_policy = False
                mode, mode_prop = 'card_diversity', 1.0

            def card_before_visit(pid, distribution, kernel_details):
                self.log.decision(
                    policy='card_diversity', action='tilt' if tilt is not None else 'fallback',
                    island=island, iteration_no=iteration_no,
                    parent_id=pid, propensity=distribution[pid], distribution=distribution,
                    selection_floor=self.selection.floor, tau_floor=self.selection.tau_floor,
                    **kernel_details, **card_details)
        # Effective mode for reward attribution: a warmup-gated exploit (tilt dropped)
        # actually executed base selection, so its reward belongs to base.
        eff_mode = mode if tilt is not None else "base"
        self.decisions.observe("mode", mix if isinstance(mix, dict) and mode in mix else {mode: 1.0}, mode)
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
            if card_before_visit is None:
                selected = self.selection.select(view, tilt=tilt, rng=self.rng.stream("parent"))
            else:
                selected = self.selection.select(view, tilt=tilt, rng=self.rng.stream("parent"),
                                                 before_visit=card_before_visit)
        parent_distribution = (None if self.cfg.get("container", {}).get("selection") == "centres"
                               else getattr(self.selection, "last_distribution", None))
        if parent_distribution is not None and selected.parent_id in parent_distribution:
            self.decisions.observe("parent", parent_distribution, selected.parent_id)
        parent = self.db.get(selected.parent_id)
        fresh_failure = selected.parent_id in self.lineage_fail
        # Freeze usable donor inputs before choosing the operator. Scan the
        # ranked local pool before applying the three-source prompt limit, so
        # exact-source aliases at its head cannot hide a later distinct donor.
        inspirations = usable_inspirations(parent.id, parent.code,
            [{'program_id': p.id, 'code': p.code, 'score': p.score, 'card': p.card}
             for p in self.db.inspirations(island, selected.parent_id,
                 k=len(self.db.islands[island].programs))])
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
                adjust=lambda p: self.theory.tilt("operator", context, p))
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
        decision_context["decisions"] = self.decisions.drain()
        lessons, injected_ids = self.theory.cards(context), []
        if self.kb_on:
            picked = self.kb.retrieve(parent.score)
            lessons += [l.text for l in picked]
            injected_ids = [l.id for l in picked]
        research_context = None
        if self.research is not None:
            research_context = self.research.context(self.db, parent, island, iteration_no + 1,
                **({'reserve_aux_chars': min(self.cfg['research_tools']['max_context_chars'], 10000)}
                   if self.research_tools is not None else {}))
            lessons = research_context['lesson_texts']
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
        repair_code = failed_code if (op == "error_repair" and failed_code) else parent.code
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
            task_prompt=self.task_prompt, bet_enabled=self.bet_on,
            required_entrypoints=self.required_entrypoints,
            operator_decision=operator_decision)
        if research_context is not None:
            snap.research_context = research_context
        if self.idea is not None:
            from .idea import digest
            snap.parent_card = parent.card
            snap.idea_cards = [{"program_id": p.id, "card": p.card,
                                "source_sha256": digest(p.code)}
                               for p in self.db.islands[island].programs.values() if p.card]
        if self.research_tools is not None:
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
            if snap.research_context is not None:
                snap.research_context['research_cells_context'] = deepcopy(snap.research_cells_context)
            snap.tool_episode_plan = self.research_tools.prepare(snap, signals=signals)
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
        }

    def _execute_step(self, plan, *, llm=None, evaluate_fn=None, idea=None):
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
            from .research_cells_context import merge_context, omit_last
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
            delivered_chars = 0
            if snapshot.research_context is not None:
                snapshot.research_context['research_cells_context'] = deepcopy(cards)
                delivered_chars = len(json.dumps(cards, ensure_ascii=True))
            elif cards['cards']:
                formatted = self.research_tools.format_context(cards['cards'])
                # The formatter includes its own heading and scope statement.
                # Count that wrapper too, evicting older observations first.
                while cards['cards'] and len(formatted) > context_limit:
                    omit_last(cards, 'character_limit')
                    formatted = self.research_tools.format_context(cards['cards'])
                if cards['cards']:
                    snapshot.directive += formatted
                    delivered_chars = len(formatted)
            # Untyped episode.context is audit prose, not an authenticated
            # observation channel. Only retained/provenance-bound cards enter G.
        worker_options = {}
        if novelty_check is not None:
            worker_options["novelty_check"] = novelty_check
        if self.research is not None:
            worker_options['research'] = self.research
        if self.idea is not None:
            worker_options['idea'] = idea or self.idea
        if self.intervention is not None:
            worker_options['intervention'] = self.intervention
        if self.circle_generation is not None:
            worker_options['circle_generation'] = self.circle_generation
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
        # Specialized idea/intervention generators return before the ordinary
        # worker loop; retain the same frozen selection attribution for them.
        if snapshot.operator_decision is not None:
            decision = snapshot.operator_decision
            for key, value in dict(op_requested=decision['requested'],
                    op_initial=decision['effective'], op_fallback_reason=decision['fallback_reason'],
                    operator_selection=decision['selection'],
                    op_on_policy=bool(decision['on_policy'] and not result.meta.get('escalated'))).items():
                result.meta.setdefault(key, value)
        if episode is not None:
            result.meta['research_tools'] = episode
        return result

    def _commit_step(self, plan, res, *, complete_research=True):
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
        if self.research_tools is not None and res.meta.get('research_tools') is not None:
            self.research_tools.commit(res.meta['research_tools'], child_program_id=child_id,
                link_kind='remeasurement' if res.meta.get('remeasurement') else 'mutation',
                **({'candidate_binding': candidate_link, 'child_source': self.db._all[child_id].code}
                   if child_id is not None and candidate_link is not None else {}))
        if self.research_tools is not None:
            for record in res.meta.get('method_followup_records', []):
                final_attempt = record['id'] == res.meta.get('research_id')
                admitted = child_id if final_attempt else None
                self.research_tools.record_followup(record, child_program_id=admitted,
                    link_kind='remeasurement' if final_attempt and res.meta.get('remeasurement') else 'mutation',
                    observed_record=self.observed_changes.record_for_program(admitted) if admitted else None)
        if (preset is not None and self.cfg.get("preset_adaptive", True)
                and not res.meta.get('remeasurement') and not res.meta.get('research_candidate')):
            # The evaluated search result already commits above. A later C/R
            # transport failure must not erase its earned learning update.
            self.preset_weights.update(preset, float(self.db.pop_best > pre_best), preset_prop)
        if complete_research and self.research is not None:
            self._complete_research_step(reflect=self.repeat_promotion is None)
        # Once per iteration, whatever the outcome. Emitting only from the success
        # path -- which sets health=0 immediately before -- pinned health/status to 0
        # in every run, so the board could never show a degraded state.
        self._emit_metrics()

    @observed_stage("research")
    def _complete_research_step(self, **options):
        return self.research.complete_step(**options)
