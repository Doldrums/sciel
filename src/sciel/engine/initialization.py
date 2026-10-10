"""Engine component construction and initial runtime state."""
from __future__ import annotations

import os
from ..reward import RewardBus
from ..rng import RngHub
from ..run_log import RunLog
from ..stats import IntensityEstimator, OpportunityCostMonitor, PlateauMonitor
from ..components.selection.novelty_chain import NoveltyChainKernel
from ..components.policy.controller import Controller
from ..components.kb.kb import KnowledgeBase
from ..components.tail import BaseRate, BranchCalibration
from .db import Database
from ..causal import Ledger
from ..theory import Experiments, Strategies, Theory
from .decisions import DecisionPolicy
from .operators import OpBandit
from .schedule import Schedule
from .loop_contracts import (
    _internal_contract,
    _read,
    _required_entrypoints,
    _snapshot_config,
)


class EngineInitialization:

    def _initialize(self, cfg, task, run_dir, llm, python, sink):
        from . import loop as _public
        self.cfg = cfg
        self.task = task
        self.run_dir = run_dir
        self.llm = llm
        self.python = python
        self.completed_result = None
        self._parallel_status = None
        self.repeat_promotion = None
        # None also represents legacy checkpoints whose bootstrap progress was
        # not recorded. Never infer unattempted paid work from those checkpoints.
        self._portfolio_bootstrap = None
        self._repeat_candidate = None
        self._repeat_candidate_origin = None
        self._repeat_export = None
        self.repeat_population = None
        self.paired_batches = None
        self._repeat_population_export = None
        if cfg.get('repeat_population', {}).get('enabled'):
            from .repeat_population_support import create_batches
            self.paired_batches = create_batches(cfg, task, run_dir, python)
        if cfg.get('repeat_promotion', {}).get('enabled'):
            from .repeat_support import create, circle_initial_source_policy
            repeat_options = ({'initial_source_policy': circle_initial_source_policy(cfg)}
                              if cfg.get('circle_generation', {}).get('enabled') else {})
            self.repeat_promotion = create(cfg['repeat_promotion'], task, run_dir, python, **repeat_options)
        os.makedirs(run_dir, exist_ok=True)
        # Construction for resume is read-only with respect to recorded config.
        # New starts in an existing journal are rejected before runtime or calls.
        self._existing_run = any(os.path.exists(os.path.join(run_dir, name))
                                 and os.path.getsize(os.path.join(run_dir, name)) > 0
                                 for name in ('sciel_events.jsonl', 'checkpoint.json'))
        if not self._existing_run:
            _snapshot_config(cfg, os.path.join(run_dir, "config.yaml"))
        warm = cfg.get('warm_start')
        if warm is not None:
            from .warm_start import validate_warm_start
            validate_warm_start(warm)
            if any(cfg.get(name, {}).get('enabled') for name in (
                    'repeat_promotion', 'repeat_population', 'circle_generation', 'circle_intervention')):
                raise ValueError('warm_start supports only ordinary search')
        self.rng = RngHub(cfg["random_seed"])
        descriptor_fn = None
        self.descriptor_info = {"kind": "code_length"}
        if cfg.get("embed_grid"):
            try:
                from ..components.policy.code_embed import LearnedDescriptor
                axis_path = cfg.get("embed_axis_path")
                if axis_path:
                    # one axis per task: see the DEFAULTS comment on embed_axis_path
                    axis_path = axis_path.replace(
                        "{task}", task.qualified_name.replace("/", "_"))
                descriptor_fn = LearnedDescriptor.load(
                    cfg["embed_grid_path"], axis_path=axis_path)
            except Exception as e:
                # An enabled learned descriptor must never change storage silently.
                raise RuntimeError(
                    f"behavior descriptor is configured but unusable "
                    f"(embedder {cfg['embed_grid_path']!r}, axis {axis_path!r}): {e}. "
                    f"Install sciel[learning] and configure readable embed_grid_path/"
                    f"embed_axis_path artifacts, or set embed_grid: false to search on the "
                    f"code-length axis deliberately.") from e
            else:
                self.descriptor_info = descriptor_fn.describe()
                self.descriptor_info['embed_sha'] = descriptor_fn.embedder.artifact_identity['sha256'][:12]
        niche_fn = None
        if cfg.get("embed_niche"):
            try:
                from ..components.policy.code_embed import VQNiches
                niche_fn = VQNiches.load(
                    cfg.get("embed_niche_path", cfg["embed_grid_path"]),
                    d_min=cfg.get("niche_d_min", 0.35),
                    target_k=cfg.get("niche_target_k", 6),
                    k_max=cfg.get("niche_k_max", 12),
                    csc_rate=cfg.get("niche_csc_rate", 0.08))
            except Exception as exc:
                raise RuntimeError(
                    "VQ niche storage is configured but unusable "
                    f"(embedder {cfg.get('embed_niche_path', cfg['embed_grid_path'])!r}): {exc}. "
                    "Install sciel[learning] and configure a readable embed_niche_path artifact, "
                    "or set embed_niche: false to use the grid deliberately."
                ) from exc
        if niche_fn is not None:
            # db._cell prefers niche_fn over descriptor_fn, so the axis actually
            # producing the cells is the codebook, not the descriptor. Recording the
            # descriptor here would label VQ-map runs as grid runs, and the resume
            # guard would then accept exactly the axis swap it exists to refuse.
            self.descriptor_info = {"kind": "vq_niche",
                                    "embed_sha": niche_fn.embedder.artifact_identity['sha256'][:12],
                                    "csc_rate": cfg.get('niche_csc_rate', 0.08),
                                    "target_k": cfg.get("niche_target_k", 6),
                                    "k_max": cfg.get("niche_k_max", 12),
                                    "d_min": cfg.get("niche_d_min", 0.35)}
        self.db = Database(cfg["num_islands"], cfg.get("pop_cap", 50),
                           cfg.get("archive_size", 20), cfg.get("grid_bins", 10),
                           descriptor_fn=descriptor_fn, niche_fn=niche_fn)
        container = cfg.get("container", {})
        if container.get("kind", "legacy") != "legacy" and self.paired_batches is None:
            from .container_db import ContainerDatabase
            vector_fn = None
            if container["kind"] in {"cvt", "vq", "annealed"}:
                if descriptor_fn is None:
                    raise ValueError("geometric container requires a usable frozen embedder")
                from ..components.policy.code_embed import embed_batch
                vector_fn = lambda code, cases: embed_batch(
                    descriptor_fn.embedder, [(code, cases)])[0]
            self.db = ContainerDatabase(
                cfg["num_islands"], cfg.get("pop_cap", 50),
                cfg.get("archive_size", 20), cfg.get("grid_bins", 10),
                descriptor_fn=descriptor_fn, container=container, vector_fn=vector_fn)
        if self.paired_batches is not None:
            from .repeat_population_db import RepeatPopulationDatabase
            self.db = RepeatPopulationDatabase(cfg['num_islands'], cfg.get('pop_cap', 50))
            self.descriptor_info = {'kind': 'repeat_ranked_pool', 'ordinary_scores': 'immutable_history'}
        self.novelty_gate = None
        if container.get("kind") == "pool_novelty":
            from ..components.code_novelty import CodeNoveltyGate
            self.novelty_gate = CodeNoveltyGate(container["novelty_threshold"])
        self.candidate_novelty = None
        if cfg.get("candidate_novelty", {}).get("enabled"):
            from .novelty import CandidateNoveltyRegistry
            self.candidate_novelty = CandidateNoveltyRegistry(
                neighbors=cfg["candidate_novelty"]["neighbors"])
        self.n_remeasurements = 0
        self.n_remeasurement_successes = 0
        self.n_evaluation_results = 0
        self.n_output_repeats = 0
        self.output_novelty = None
        if cfg.get('candidate_novelty', {}).get('output_repeats'):
            from .output_novelty import OutputNoveltyRegistry
            from .output_measurements import validate_task_resolution
            resolution = validate_task_resolution(task, cfg['candidate_novelty'].get('output_resolution'))
            self.output_novelty = OutputNoveltyRegistry(identity={
                'task_hash': task.task_hash, 'engine_contract': _public.ENGINE_CONTRACT},
                near_policy=resolution)
        self.preset_weights = None
        if cfg.get("preset_selection"):
            from ..components.policy.preset_weights import PresetWeights
            self.preset_weights = PresetWeights(prior={"base": .8, "exploit": .15, "explore": .05})
        self.schedule = Schedule(cfg["num_islands"], cfg["schedule_policy"],
                                 tuple(cfg["milestone_fracs"]),
                                 cfg['parallel_search']['max_inflight_per_island']
                                 if cfg['parallel_search']['enabled'] else cfg["inflight_cap"],
                                 cfg["sh_min_fair_evals"],
                                 fair_dispatch=cfg['parallel_search']['enabled'])
        self.decisions = DecisionPolicy(cfg.get("decision_epsilon", 0.1), self.rng.stream("decision"))
        self.schedule.decisions = self.decisions
        self.db.decisions = self.decisions
        self.ledger = Ledger()
        self.theory = Theory()
        self.experiments = Experiments()
        self.strategies = Strategies()
        _arms = ["diff", "full_rewrite", "error_repair"] + \
            (["cross"] if cfg.get("crossover") else [])
        self.op_bandit = OpBandit(cfg.get("op_temperature", 0.5),
                                  cfg.get("op_floor", 0.05), cfg.get("op_window", 50),
                                  arms=_arms,
                                  intensify_boost=cfg.get("op_intensify_boost", 0.5))
        sel_policy = cfg.get("selection_policy", "novelty_chain")
        if sel_policy == "novelty_chain":
            self.selection = NoveltyChainKernel(cfg.get("selection_floor", 1e-3),
                                                cfg.get("tau_floor", 1e-3))
        else:
            from ..components.selection.kernels import build_kernel
            self.selection = build_kernel(sel_policy, cfg)
        self.controller = Controller(cfg["policy"], cfg["num_islands"])
        # #4c: shared three-head trunk; its promise head can tilt parent selection
        # (off by default -- selection-override tilt did not pan out in A/B).
        if cfg.get("three_head"):
            from ..components.policy.three_head import ThreeHeadTilt
            self.three_head = ThreeHeadTilt.load(
                cfg["three_head_path"], cfg.get("embed_niche_path", cfg["embed_grid_path"]),
                cfg.get("three_head_beta", 0.3))
        else:
            self.three_head = None
        needs_tilt = self.controller.needs_tilt() or self.three_head is not None
        if needs_tilt and not getattr(self.selection, "supports_tilt", False):
            raise ValueError(
                f"policy modes require a tilt-capable selection kernel; "
                f"selection_policy={sel_policy!r} ignores tilt")
        self.reward = RewardBus(cfg.get("reward_w_pop", 1.0), cfg.get("reward_w_isl", 1.0))
        self.intensity = IntensityEstimator()
        self.log = RunLog(os.path.join(run_dir, "sciel_events.jsonl"))
        from .llm.trace import TracedLLM
        llm = self.llm = TracedLLM(llm, run_dir, self._llm_trace_context, on_response=self._record_llm_accounting)
        self.sink = sink
        self.budget = None if cfg["budget"] is None else int(cfg["budget"])
        self._stop_requested = False
        self.iteration_no = 0
        self.B_evals = 0
        self.milestones_fired: set = set()
        self.lineage_fail: dict = {}
        self.eval_times: dict = {}   # stage-inclusive mean per reported trial, or None
        self.evaluation_timings: dict = {}  # explicit source-bound timing scope/provenance
        self.noop_parents: set = set()   # legacy score-tie warning; not behavior equivalence
        from .observed_changes import ObservedChanges
        from ..hypothesis.measurement import metric_context
        self.observed_changes = ObservedChanges(metric=metric_context(task))
        # idea island (arch-v4 #2): the last island explores idea-space (forced
        # `idea` operator). None disables it.
        self._idea_island = (cfg["num_islands"] - 1) if cfg.get("idea_island") else None
        self._last_improve_eval = 0   # B_evals when pop_best last improved (spawn trigger)
        # refine's own stall counter: reset ONLY on a pop improvement, never on the
        # spawn cooldown, so island spawns do not interrupt the exploit hold.
        self._last_pop_improve = 0
        # Ordinary committed attempts include failed/no-child outcomes. Keep this
        # anchor separate from successful-evaluation plateau statistics.
        self._last_pop_improve_iteration = 0
        self.seed_score = 0.0
        # Generated-child improvements over the current population best. H_n is a
        # descriptive run-length normalization; seeds, ties and adaptive search mean
        # it is not an exact exchangeable null or a substitute for a second arm.
        self.n_records = 0
        self._observed_trials = 1     # trials the evaluator reported on its last full stage
        self.cost_total = 0.0  # Known USD subtotal; excludes explicitly unpriced calls.
        self.llm_unpriced_calls = 0
        self.n_events = 0
        self.n_dups = 0
        self.n_eval_fail = 0
        self.n_nochild = 0
        self.health = 0
        self._stop_at = None
        self.log_population = bool(cfg["log_population"])
        self.plateau_shadow = bool(cfg.get("plateau_shadow", False))
        self.plateau_monitors = ({i: PlateauMonitor() for i in range(cfg["num_islands"])}
                                 if self.plateau_shadow else None)
        pol = cfg["policy"]
        # plateau shift belongs to the policy: `policy: {enabled: false}` must
        # disable the whole layer even though the defaults fill plateau_shift in
        self.oc_shift = bool(pol.get("enabled")) and bool(pol.get("plateau_shift"))
        self.oc_monitors = (
            {i: OpportunityCostMonitor(threshold=pol["plateau_threshold"],
                                       horizon=pol["plateau_horizon"],
                                       min_n=pol["plateau_min_n"])
             for i in range(cfg["num_islands"])}
            if (bool(pol.get("plateau_shadow")) or self.oc_shift) else None)
        self.c2_on = bool(cfg.get("c2_event_bets", False))
        self.bet_on = self.c2_on or bool(cfg.get("bet_enabled", False))
        # kb_free: same lesson KB, but lessons are the free strategy cards already
        # extracted from responses (no postmortem LLM call). RLAD shows even
        # untrained in-context abstractions help; this pays zero extra calls.
        self.kb_free = bool(cfg.get("kb_free", False))
        self.kb_on = self.c2_on or self.kb_free
        self.kb = KnowledgeBase() if self.kb_on else None
        self.branch_cal = BranchCalibration() if self.c2_on else None
        self.base_rate = BaseRate() if self.kb_on else None
        self.research = None
        if cfg.get('research', {}).get('enabled'):
            from .research import ResearchController
            from ..hypothesis.measurement import metric_context
            self.research = ResearchController(cfg['research'], llm, self.log,
                                               self._charge_research, metric=metric_context(task))
        self.research_tools = None
        if cfg.get('research_tools', {}).get('enabled'):
            from ..research_tools.controller import ResearchTools
            tool_options = ({'capacity_config': cfg['parallel_search']}
                            if cfg['research_tools'].get('jobs', {}).get('enabled') else {})
            self.research_tools = ResearchTools(cfg['research_tools'], run_dir,
                task_identity={'task': task.qualified_name, 'task_hash': task.task_hash},
                emit=self.log.research_tool, charge=self._charge_research, **tool_options)
        self.idea = None
        if cfg.get('idea_two_stage', {}).get('enabled'):
            from .idea import TwoStageIdea
            self.idea = TwoStageIdea(self.research,
                card_repair_attempts=cfg['idea_two_stage'].get('card_repair_attempts', 0))
        self.intervention = None
        if cfg.get('circle_intervention', {}).get('enabled'):
            from .intervention import CircleIntervention, DEFAULT_OPERATION
            self.intervention = CircleIntervention(self.research,
                operation=cfg['circle_intervention'].get('operation', DEFAULT_OPERATION))
        self.circle_generation = None
        if cfg.get('circle_generation', {}).get('enabled'):
            from .circle_generation import CircleGeneration
            self.circle_generation = CircleGeneration(self.research,
                proposal_policy=cfg['circle_generation'].get('proposal_policy'),
                execution_profile=cfg['circle_generation'].get('execution_profile'))
        self.n_postmortems = 0
        from ..eval.task_prompt import compose_task_prompt
        self.task_prompt = compose_task_prompt(task, cfg.get("llm", {}).get("system_message"))
        req = set(_required_entrypoints(getattr(task, "evaluator_path", None)))
        try:
            req |= set(_internal_contract(_read(self.task.initial_program_path)))
        except OSError:
            pass
        self.required_entrypoints = sorted(req)
        if self.required_entrypoints:
            self.task_prompt += (
                "\n\nREQUIRED INTERFACE: the evaluator and the fixed wrapper call these "
                "functions by name -- your program MUST keep each one, with the same name and "
                "return signature as in the current program. Renaming or removing any scores 0: "
                f"{', '.join(self.required_entrypoints)}.")
        if self.paired_batches is not None:
            from .repeat_population_support import initialize
            initialize(self)
