"""Execution and commit phases of one ordinary search step."""
from __future__ import annotations

from copy import deepcopy
from ..campaign.budget import CampaignStopped
from .runtime import observed_stage
from .siblings import ITEMS


CONFIRM_TOLERANCE = 1e-6


class EngineStepCycle:

    @observed_stage("generating")
    def _step(self, island: int) -> None:
        if self.incubator.owner(island) is not None:
            plan = self.incubator.prepare(self, island, self.iteration_no)
            self._commit_step(plan, self._execute_step(plan))
            return
        left = (self.budget - self.B_evals) if self.budget is not None else 1 + ITEMS
        for member in self.siblings.expand(self._prepare_step(island, block_room=min(1 + ITEMS, max(left, 1)))):
            self._commit_step(member, self._execute_step(member))

    def _recheck(self, code: str, score: float, pop_best: float, parent_score: float) -> tuple:
        """Re-measure a new record, or any improving time-dependent program; inconsistent results keep the minimum."""
        from .determinism import time_dependent
        clocked = time_dependent(code)
        if not (score > pop_best or (clocked and score > parent_score)):
            return score, None
        second = self._confirmation_score(code)
        consistent = second is not None and abs(second - score) <= CONFIRM_TOLERANCE * max(1.0, abs(score))
        info = {'first': score, 'second': second, 'time_dependent': clocked, 'consistent': consistent}
        return (score if consistent else min(score, second) if second is not None else 0.0), info

    def _confirm(self, result, pop_best, parent_score) -> None:
        if not getattr(result, 'ok', False) or result.meta.get('remeasurement'):
            return
        admitted, info = self._recheck(result.child_code, result.score, pop_best, parent_score)
        if info is not None:
            result.meta['confirmation'] = {**info, 'admitted': admitted}

    def _sibling_outcome(self, plan, res, child_id) -> None:
        admitted = res.meta.get('confirmation', {}).get('admitted', res.score) if getattr(res, 'ok', False) else None
        if plan.get('block') is None:
            return
        program = self.db._all.get(child_id) if child_id else None
        for pair in self.siblings.record(plan, admitted, program.code if program else None):
            self.log.decision(policy="sibling", action="pair", iteration_no=self.iteration_no, **pair)

    def _settle_probe(self, spec, scores) -> None:
        from .probe import brief, verdict
        assumption = self.assumptions.get(spec['assumption'])
        assumption.status = verdict(scores)
        assumption.probe = {**spec["prediction"], "seeds": spec["seeds"], "program": spec["program"]}
        assumption.evidence.append(f"parent {scores['parent']} vs variant {scores['variant']}")
        self.assumptions.pending = None
        self.log.decision(policy="probe", action=assumption.status, iteration_no=self.iteration_no,
                          assumption=assumption.id, text=assumption.text, scores=scores, program=spec["program"],
                          evaluations=scores["measured"])
        if assumption.status == "refuted":
            for programme in self.incubator.open(self, brief(assumption)):
                self.log.decision(policy="programme", action="open", iteration_no=self.iteration_no,
                                  programme=programme.id, kind=programme.kind, island=programme.island,
                                  strategy=programme.strategy)

    def _add_findings(self, episode) -> None:
        added = self.findings.add((episode or {}).get('findings', []), self.iteration_no)
        if added:
            self.log.decision(policy="finding", action="add", iteration_no=self.iteration_no,
                              findings=[{"id": f.id, "text": f.text, "island": f.island} for f in added])

    def _execute_step(self, plan, *, llm=None, evaluate_fn=None):
        from . import loop as _public
        snapshot = plan['snap']
        episode = None
        submitted = []
        evaluator = evaluate_fn or self._evaluation_callback(snapshot.island, seed=(plan.get('block') or {}).get('seed'))
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
        for key in ('knockout', 'probe'):
            if plan.get(key):
                from .knockout import execute as knockout
                from .probe import execute as probe
                try:
                    measured = (knockout if key == 'knockout' else probe)(self, plan[key], llm or self.llm,
                                                                          snapshot.model)
                except CampaignStopped:
                    raise
                except Exception as error:
                    result.meta[key + '_error'] = type(error).__name__
                    continue
                result.meta[key] = {k: v for k, v in measured.items() if k != 'cost'}
                result.cost += measured['cost']
        for key, field in (('theorist', 'strategy_proposal'), ('assumptions', 'assumption_reply')):
            if plan.get(key):
                from .llm.client import accounted_cost
                try:
                    reply = (llm or self.llm).generate(plan[key], snapshot.model)
                    result.meta[field] = reply.text
                    result.cost += accounted_cost(reply)
                except CampaignStopped:
                    raise
                except Exception as error:
                    result.meta[key + '_error'] = type(error).__name__
        return result

    def _commit_step(self, plan, res):
        self._procedure = self.procedures.get(plan["procedure"])
        self.theory.shift, self.theory.mix = self._procedure.shift, self._procedure.mix
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
        self._sibling_outcome(plan, res, child_id)
        for event in self.incubator.commit(self, plan):
            strategy = next((s for s in self.strategies.pool if s.id == event["strategy"]), None)
            if event.get("graduated") and strategy is not None:
                strategy.status = "culture"
            self.log.decision(policy="programme", action="graduate" if "graduated" in event else "stage",
                              iteration_no=self.iteration_no, **event)
        for programme in self.incubator.open_next(self):
            self.log.decision(policy="programme", action="open", iteration_no=self.iteration_no,
                              programme=programme.id, kind=programme.kind, island=programme.island,
                              strategy=programme.strategy)
        if plan.get('knockout'):
            from .knockout import reliance
            spec = plan['knockout']
            measured = res.meta.get('knockout') or {"scores": {"parent": None, "target": None, "nulls": []}, "measured": 0}
            scores = measured["scores"]
            self.B_evals += measured["measured"]
            assumption = self.assumptions.get(spec['assumption'])
            assumption.reliance = reliance(scores)
            if assumption.reliance is None:
                assumption.status = "unmeasurable"
            self.assumptions.pending = None
            self.log.decision(policy="knockout", action="measured", iteration_no=self.iteration_no,
                              assumption=spec['assumption'], program=spec['program'], targets=spec['targets'],
                              nulls=spec['nulls'], scores=scores, reliance=assumption.reliance,
                              evaluations=measured["measured"])
        if plan.get('probe'):
            measured = res.meta.get('probe') or {"parent": [None], "variant": [None], "measured": 0}
            self.B_evals += measured["measured"]
            self._settle_probe(plan['probe'], measured)
        if plan.get('assumptions'):
            programs = [self.db._all[pid] for pid in plan['assumption_programs'] if pid in self.db._all]
            for assumption in self.assumptions.adopt(res.meta.get('assumption_reply'), programs, self.iteration_no):
                self.log.decision(policy="assumption", action="add", iteration_no=self.iteration_no,
                                  assumption=assumption.id, text=assumption.text,
                                  counterfactual=assumption.counterfactual, units=assumption.units)
        if plan.get('theorist'):
            adopted = self.strategies.adopt(res.meta.get('strategy_proposal'), self.iteration_no)
            self.log.decision(policy="strategy", action="propose", iteration_no=self.iteration_no,
                              strategy=adopted.id if adopted else None, text=adopted.text if adopted else None)
            for programme in (self.incubator.open(self, adopted) if adopted else []):
                self.log.decision(policy="programme", action="open", iteration_no=self.iteration_no,
                                  programme=programme.id, kind=programme.kind, island=programme.island,
                                  strategy=programme.strategy)
        if self.research_tools is not None and res.meta.get('research_tools') is not None:
            self._add_findings(res.meta['research_tools'])
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
