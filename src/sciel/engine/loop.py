"""Search orchestration: budget gates, ordered results and scheduling milestones."""

from __future__ import annotations
import ast
import hashlib
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from ..campaign.budget import CampaignStopped
from ..config import apply_defaults
from ..reward import RewardBus
from ..rng import RngHub
from ..run_log import RunLog
from ..stats import IntensityEstimator, OpportunityCostMonitor, PlateauMonitor
from ..components.selection.novelty_chain import IslandView, NoveltyChainKernel
from ..components.policy.controller import Controller
from ..components.policy.mode_features import mode_features
from ..components.kb.kb import KnowledgeBase
from ..components.tail import BaseRate, BranchCalibration, extract_lesson
from ..eval.cascade import cascade_evaluate
from ..eval.facade import EvaluationInfrastructureError
from ..tui.board import WandbSink
from . import checkpoint
from .db import Database, normalized_hash
from .operators import OpBandit, usable_inspirations
from .prompt.build import compose_directive
from .schedule import Saturated, Schedule
from .types import NoChild, Snapshot
from .worker import run_worker
from .runtime import RuntimeStatus, observed_stage
from .loop_contracts import (
    ENGINE_CONTRACT,
    _ENTRY_RE,
    _PARADIGMS,
    _changed_region,
    _evaluator_metadata,
    _internal_contract,
    _model_names,
    _paradigm_prompt,
    _preflight_task,
    _read,
    _required_entrypoints,
    _snapshot_config,
    _trunc,
)

from .initialization import EngineInitialization
from .bootstrap import EngineBootstrap
from .lifecycle import EngineLifecycle
from .step_cycle import EngineStepCycle
from .admission import EngineAdmission
from .evaluation import EngineEvaluation
from .telemetry import EngineTelemetry
from .repeat_learning import EngineRepeatLearning


class Engine(
    EngineInitialization,
    EngineBootstrap,
    EngineLifecycle,
    EngineStepCycle,
    EngineAdmission,
    EngineEvaluation,
    EngineTelemetry,
    EngineRepeatLearning,
):

    def __init__(self, cfg: dict, task, run_dir: str, llm, python: str | None = None,
                 sink: WandbSink | None = None, evaluation_backend=None):
        # Raw callers get the same defaults as the CLI, but invalid task/source
        # combinations must leave no run directory or lock artifact behind.
        cfg = apply_defaults(cfg)
        _preflight_task(cfg, task)
        from ..eval.candidate_backend import create_backend
        self.evaluation_backend = evaluation_backend or create_backend(cfg, task)
        if self.evaluation_backend is not None:
            self.evaluation_backend.assert_compatible(cfg, task)
        if cfg.get('parallel_search', {}).get('enabled'):
            from .parallel_store import recover
            recover(run_dir)
        from .run_lock import RunDirectoryLock
        self._writer_lock = RunDirectoryLock(run_dir)
        try:
            self._initialize(cfg, task, run_dir, llm, python, sink)
        except BaseException:
            self.close()
            raise

    def close(self):
        from contextlib import ExitStack
        with ExitStack() as cleanup:
            for name in ('_writer_lock', 'log', 'repeat_promotion', 'paired_batches'):
                resource = getattr(self, name, None)
                if resource is not None:
                    cleanup.callback(resource.close)
