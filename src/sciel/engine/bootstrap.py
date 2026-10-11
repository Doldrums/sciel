"""Canonical baseline, portfolio seeding and dynamic island bootstrap."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from ..campaign.budget import CampaignStopped
from ..causal import context_of
from ..stats import OpportunityCostMonitor, PlateauMonitor
from ..swm import directive, propose
from .prompt.build import compose_directive
from .types import NoChild, Snapshot
from .runtime import observed_stage
from .loop_contracts import (_paradigm_prompt, _read, _trunc, paradigms)


class EngineBootstrap:

    @observed_stage("seeding")
    def seed(self, *, portfolio=True) -> None:
        warm = self.cfg.get('warm_start')
        initial = warm['source'] if warm else _read(self.task.initial_program_path)
        if warm:
            self.log.decision(policy='initialization', action='warm_start',
                              schema=warm['schema'], source_sha256=warm['source_sha256'],
                              parent=warm['parent'], task_hash=self.task.task_hash)
        outcome = self._evaluate(initial)
        if warm and not outcome.result.ok:
            raise CampaignStopped('search requires a valid evaluated initial baseline')
        self.seed_score = outcome.result.score if outcome.result.ok else 0.0
        for isl in range(len(self.db.islands)):
            prog = self.db.add(initial, self.seed_score, outcome.result.cases,
                               isl, origin="seed")
            self._bind_novelty_program(prog, evaluation_id=getattr(outcome, 'evaluation_id', None))
            self._observe_program(prog, outcome)
            self.log.decision(
                policy="initialization", action="seed_program", schema="sciel-initialization/1",
                program_id=prog.id, source=initial,
                source_sha256=hashlib.sha256(initial.encode("utf-8")).hexdigest(),
                island=isl, measured=bool(outcome.result.ok),
                score=outcome.result.score if outcome.result.ok else None,
                seed_score=outcome.result.score if outcome.result.ok else None,
                iteration_no=self.iteration_no, B_evals=self.B_evals)
        if self.cfg.get("portfolio_seed"):
            self._portfolio_bootstrap = {
                "schema": "sciel-portfolio-bootstrap/1",
                "islands": len(self.db.islands), "next_island": 0}
            if portfolio:
                self._portfolio_seed(initial)

    @observed_stage("generating")
    def _paradigm_child(self, base_code: str, paradigm: str, island: int,
                        origin: str) -> bool:
        """One LLM shot at a paradigm-specific program for `island`; adds it on a
        successful eval. Returns True on success. Costs one call + one eval that
        do NOT count toward B_evals (seeding, like the initial eval)."""
        if getattr(self, 'candidate_novelty', None) is not None:
            return self._novelty_paradigm_child(base_code, paradigm, island, origin)
        from .prompt.parse import extract_card, extract_code
        from .validation import TRACE_SCHEMA, trace_attempt, validate_candidate
        prompt = _paradigm_prompt(self.task_prompt, base_code, paradigm)
        prompt += '\n\n' + compose_directive(self._per_run_budget(), None, plateau=False)
        prompt += self.observed_changes.block(base_code)
        model = self.llm.choose_model(self.rng.stream("prompt"))
        try:
            resp = self.llm.generate(prompt, model, 0.9)
        except CampaignStopped:
            raise
        except Exception as exc:
            # No returned response means its current charge is unknown, even if
            # the provider sent an HTTP response. Do not copy remote error bodies.
            attempt = {"attempt": 1, "operator": origin, "temperature": 0.9,
                       "disposition": "generation_failed", "generation_returned": False,
                       "cost": None, "error_type": type(exc).__name__, "decisions": []}
            self.log.decision(
                policy="paradigm_validation", action="generation_failed", origin=origin,
                island=island, paradigm=paradigm[:40], model_used=model, cost=None,
                meta={"validation_trace": {"schema": TRACE_SCHEMA, "profile": "paradigm",
                                            "attempts": [attempt]}})
            from .worker import unknown_request_stop
            raise unknown_request_stop(exc, 0.0) from None
        from .llm.client import accounted_cost
        try:
            response_cost = accounted_cost(resp)
        except ValueError:
            stopped = CampaignStopped('generation cost is unknown')
            stopped.known_cost = 0.0
            stopped.current_call_cost_unknown = True
            raise stopped from None
        self.cost_total += response_cost
        code = extract_code(resp.text)
        validation = validate_candidate(
            code, parent_code=base_code, required_entrypoints=self.required_entrypoints,
            code_hashes=set(), profile="paradigm",
            novelty_check=(lambda candidate: self._novelty_check(candidate, island))
                          if self.novelty_gate is not None else None)
        attempt = trace_attempt(validation, extracted_code=code, response=resp, attempt=1,
                                operator=origin, temperature=0.9, cost=response_cost)
        trace_fields = {"origin": origin, "cost": response_cost, "model_used": model,
                        "meta": {"validation_trace": {"schema": TRACE_SCHEMA,
                                 "profile": "paradigm", "attempts": [attempt]}}}
        if validation.violation is not None:
            self.log.decision(policy="paradigm_validation", action=validation.violation.kind,
                              island=island, paradigm=paradigm[:40], **trace_fields)
            return False
        code = validation.code
        outcome = self._evaluate(code)
        if not outcome.result.ok:
            attempt["evaluation"] = {"ok": False, "disposition": "eval_failed"}
            self.log.decision(policy="paradigm_validation", action="eval_failed",
                              island=island, paradigm=paradigm[:40],
                              artifacts=_trunc(getattr(outcome.result, "artifacts", None)),
                              **trace_fields)
            return False
        attempt["evaluation"] = {"ok": True, "disposition": "admit"}
        prog = self.db.add(code, outcome.result.score, outcome.result.cases,
                           island, origin=origin, card=extract_card(resp.text))
        self._bind_novelty_program(prog, evaluation_id=getattr(outcome, 'evaluation_id', None))
        parent = next((p for p in self.db._all.values() if p.id != prog.id and p.code == base_code), None)
        self._observe_program(prog, outcome, parent=parent)
        self.log.decision(policy="portfolio_seed", action=origin, island=island,
                          score=outcome.result.score, paradigm=paradigm[:40], **trace_fields)
        return True

    def _novelty_paradigm_child(self, base_code, paradigm, island, origin):
        """Portfolio/spawn uses the same bounded novelty retry protocol as search."""
        from . import loop as _public
        model = self.llm.choose_model(self.rng.stream('prompt'))
        parent = next((p for p in self.db.islands[island].programs.values()
                       if p.code == base_code), None)
        if parent is None and self.db.pop_best_id is not None:
            parent = self.db.get(self.db.pop_best_id)
        snap = Snapshot(island=island, op='full_rewrite', model=model,
            iteration_no=self.iteration_no, parent_id=parent.id if parent else '',
            parent_code=base_code, parent_score=parent.score if parent else self.seed_score,
            directive=(f'Bootstrap {origin}: write a new program using {paradigm}.\n'
                       + compose_directive(self._per_run_budget(), None, plateau=False)
                       + self.observed_changes.block(base_code)),
            task_prompt=self.task_prompt, required_entrypoints=self.required_entrypoints)
        result = _public.run_worker('full_rewrite', snap, self.llm, self._evaluation_callback(island),
            dup_cap=1, base_temp=0.9, novelty_check=self._candidate_novelty_callback(island))
        self.cost_total += result.cost
        ok = not isinstance(result, NoChild) and result.ok
        if ok:
            from .prompt.parse import extract_card
            measured_again = result.meta.get('remeasurement', False)
            program = self.db.add(result.child_code, result.score, result.cases, island,
                origin='remeasurement' if measured_again else origin,
                parent_id=None if measured_again or parent is None else parent.id,
                card=extract_card(result.response))
            self._bind_novelty_program(program, evaluation_id=result.meta.get('evaluation_id'))
            self._observe_program(program, result, parent=None if measured_again else parent)
        self.log.decision(policy='portfolio_seed', action=origin, island=island,
            ok=ok, cost=result.cost, meta=result.meta, model_used=model)
        return ok

    def _portfolio_seed(self, initial: str) -> None:
        """Diverse birth: each island additionally gets one paradigm-specific
        start (audit: a run's fate hinges on whether the right solution family
        is ever DISCOVERED; identical starts make that a lottery)."""
        state = self._portfolio_bootstrap
        if state is None:
            return
        while state['next_island'] < state['islands']:
            if self._pause_requested():
                break
            isl = state['next_island']
            paradigm = paradigms(self.task)[isl % len(paradigms(self.task))]
            # A provider stop seals this attempt, including an unknown request
            # or a child whose later conclusion failed. Recovery must not replay
            # it; the remaining directions are still required before search.
            state['next_island'] = isl + 1
            self._paradigm_child(initial, paradigm, isl, origin="portfolio")

    def _maybe_spawn_island(self) -> None:
        """Dynamic niches: when the population best has not improved for
        `spawn_stall` accepted evals, the current niches are exhausted -- found a
        new island seeded by a fresh paradigm rewrite of the population best.

        The plateau-monitor forecast is calibrated for thousand-eval budgets
        (needs ~16+ in-segment evals with k>=1 to fire); at ~80 evals it never
        declares, so the trigger is a direct population-stall counter instead."""
        if not self.cfg.get("dynamic_islands"):
            return
        cap = int(self.cfg.get("max_islands", 6))
        if len(self.db.islands) >= cap:
            return
        window = int(self.cfg.get("spawn_stall", 12))
        stalled = self.B_evals >= window and self.B_evals - self._last_improve_eval >= window
        if not self.decisions.flag("spawn", stalled):
            return
        idx = self.db.add_island()
        self.schedule.register_island(idx)
        self.controller.plateau_until[idx] = -1
        pol = self.cfg["policy"]
        if self.oc_monitors is not None:
            self.oc_monitors[idx] = OpportunityCostMonitor(
                threshold=pol["plateau_threshold"], horizon=pol["plateau_horizon"],
                min_n=pol["plateau_min_n"])
        if self.plateau_monitors is not None and idx not in self.plateau_monitors:
            # the shadow detector's sibling dict was forgotten here, so a spawn with
            # plateau_shadow on killed the run with a KeyError at the island's first
            # successful eval
            self.plateau_monitors[idx] = PlateauMonitor()
        base = self.db._all[self.db.pop_best_id].code if self.db.pop_best_id \
            else _read(self.task.initial_program_path)
        paradigm = paradigms(self.task)[idx % len(paradigms(self.task))]
        context = context_of({"phase": self.schedule.phase_code(), "progress": self._progress()})
        seed = self.rng.derive_seed("swm")
        targets = {m: propose(self.db, seed, m) for m in ("kde", "diffusion")}
        arms = ["paradigm"] + [m for m, t in targets.items() if t is not None]
        source, record = self.decisions.pick("spawn_target", self.experiments.allocate("spawn_target", context,
            self.theory.tilt("spawn_target", context, {a: 1 / len(arms) for a in arms})))
        target = targets.get(source)
        if target is not None:
            paradigm = directive(target)
        ok = self._paradigm_child(base, paradigm, idx, origin="spawn")
        seed = next(iter(self.db.islands[idx].programs.values()), None) if ok else None
        self.ledger.add(iteration=self.iteration_no, island=idx, child=seed.id if seed else None,
                        parent=self.db.pop_best_id, event=False,
                        decision_context={"phase": self.schedule.phase_code(), "progress": self._progress(),
                                          "parent_id": self.db.pop_best_id, "decisions": [record]})
        self.log.decision(policy="spawn_target", action=source, island=idx, target=target, candidates=targets,
                          propensity=record["propensity"])
        if not ok:
            # Fall back to this run's retained baseline, including a warm start.
            # Copying a measured program is migration, never another evaluation.
            warm = self.cfg.get('warm_start')
            initial = warm['source'] if warm else _read(self.task.initial_program_path)
            baseline = next((p for p in self.db._all.values()
                if p.origin == 'seed' and p.parent_id is None and p.code == initial), None)
            if baseline is None:
                raise ValueError('dynamic island fallback requires a retained initial baseline')
            from .research_artifacts import evaluation_id_for_program
            from .evaluation_timing import for_program
            evaluation_id = evaluation_id_for_program(self, baseline)
            if self.candidate_novelty is not None and evaluation_id is None:
                raise ValueError('dynamic island baseline lacks its exact evaluation identity')
            timing = for_program(self, baseline)
            copied = self.db.add(baseline.code, baseline.score, deepcopy(baseline.cases), idx,
                origin='migrant', parent_id=baseline.id, card=baseline.card,
                sig_hash=baseline.sig_hash)
            self._bind_novelty_program(copied, evaluation_id=evaluation_id)
            self.observed_changes.bind_alias(copied, baseline)
            if timing is not None:
                self.evaluation_timings[copied.id] = timing
                self.eval_times[copied.id] = timing['mean_evaluation_stage_wall_seconds_per_reported_trial']
            self.log.decision(policy='dynamic_islands', action='fallback_copy',
                island=idx, program_id=copied.id, reference_program_id=baseline.id,
                source_sha256=hashlib.sha256(copied.code.encode('utf-8')).hexdigest(),
                evaluation_id=evaluation_id, scope='inherited_baseline_no_new_measurement')
        stall = self.B_evals - self._last_improve_eval   # BEFORE the cooldown reset
        self._last_improve_eval = self.B_evals   # cooldown: give the new niche a window
        self.log.decision(policy="dynamic_islands", action="spawn", island=idx,
                          n_islands=len(self.db.islands), seeded=ok, stall=stall)
