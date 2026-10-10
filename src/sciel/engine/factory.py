"""Assemble an Engine and its LLM from a resolved config."""

from __future__ import annotations

import os

from ..config import DEFAULTS
from ..eval.registry import load_task
from ..tui.board import WandbSink
from .llm.client import EnsembleLLM
from .llm.mock import MockLLM
from .loop import Engine


def load_env(dotenv_path: str | None = None) -> None:
    if dotenv_path is None:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))))
        dotenv_path = os.path.join(os.path.dirname(here), ".env")
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_tasks_root(cfg: dict, default_root: str) -> str:
    repo = cfg.get("sciel_tasks_repo")
    rev = cfg.get("sciel_tasks_revision")
    if not repo or not rev:
        return default_root
    load_env()
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo, repo_type="dataset", revision=rev,
                             token=os.environ.get("HF_TOKEN"))


def build_llm(cfg: dict):
    lcfg = cfg.get("llm", {}) or {}
    if lcfg.get("mock", False):
        return MockLLM()
    if lcfg.get('provider') == 'codex_responses':
        from .llm.codex import CodexResponsesLLM, validate_config
        lcfg = validate_config(lcfg)
        if lcfg['codex_auth_file'] is None:
            load_env()
        key = os.environ.get(lcfg['api_key_env']) if lcfg['codex_auth_file'] is None else None
        if lcfg['codex_auth_file'] is None and not key:
            raise RuntimeError('no Codex token in configured environment variable')
        return CodexResponsesLLM(lcfg, key)
    load_env()
    api_key = os.environ.get(lcfg.get("api_key_env", "OPENROUTER_API_KEY"), "")
    if not api_key:
        raise RuntimeError("no LLM api key in environment")
    models = lcfg.get("models") or [lcfg.get("primary_model", "gemini-2.5-flash-lite")]
    weights = lcfg.get("weights") or [1.0] * len(models)
    return EnsembleLLM(
        models=models, weights=weights,
        api_base=lcfg.get("api_base", "https://openrouter.ai/api/v1"),
        api_key=api_key, temperature=lcfg.get("temperature", 0.7),
        max_tokens=lcfg.get("max_tokens", DEFAULTS["llm"]["max_tokens"]),
        timeout=lcfg.get("timeout", 120.0),
        max_retries=lcfg.get("max_retries", 1),
        model_capabilities=lcfg.get("model_capabilities", {}),
        reasoning_effort=lcfg.get("reasoning_effort"))


def build_engine(cfg: dict, tasks_root: str, run_dir: str, python: str | None = None,
                 llm=None, campaign: str = "adhoc", arm: str = "A",
                 tasks_root_override: bool = False) -> Engine:
    if tasks_root_override:
        # An explicit CLI path wins over a saved HF source. Clear its provenance as
        # well, so this local run cannot be labelled as using the pinned snapshot.
        cfg = {**cfg, "sciel_tasks_repo": None, "sciel_tasks_revision": None}
    tasks_root = resolve_tasks_root(cfg, tasks_root)
    task_dir = os.path.join(tasks_root, cfg["task"])
    task = load_task(task_dir)
    from ..config import apply_defaults
    from ..eval.candidate_backend import create_backend
    cfg = apply_defaults(cfg)
    evaluation_backend = create_backend(cfg, task)
    if llm is None:
        llm = build_llm(cfg)
    sink = None
    if (cfg.get("wandb") or {}).get("enabled", False):
        seed = cfg["random_seed"]
        sink = WandbSink(cfg, run_name=f"{campaign}/{arm}_s{seed}", group=campaign,
                         job_type=arm, config_snapshot={"seed": seed, "arm": arm},
                         run_dir=run_dir)
    return Engine(cfg, task, run_dir, llm, python=python, sink=sink, evaluation_backend=evaluation_backend)
