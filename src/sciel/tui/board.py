"""Observability subscribers: wandb 7-metric sink and a text narrator.

Failure of any subscriber is swallowed so observability never drops evolution.
"""

from __future__ import annotations

import logging

_log = logging.getLogger("sciel.board")

METRIC_KEYS = ("best/pop", "events/intensity", "phase/schedule",
               "health/status", "cost/usd_total", "cost/score_per_dollar")


class WandbSink:
    def __init__(self, cfg: dict, run_name: str, group: str, job_type: str,
                 config_snapshot: dict, run_dir: str | None = None):
        self.enabled = False
        self._run = None
        wcfg = cfg.get("wandb") or {}
        if not wcfg.get("enabled", False):
            return
        try:
            import os
            os.environ.setdefault("WANDB_MODE", wcfg.get("mode", "offline"))
            os.environ.setdefault("WANDB_SILENT", "true")
            import wandb
            self._wandb = wandb
            self._run = wandb.init(
                project=wcfg.get("project", "sciel-night"),
                name=run_name, group=group, job_type=job_type,
                config=config_snapshot, reinit=True,
                dir=run_dir or os.getcwd())
            self.enabled = True
        except Exception as e:
            _log.debug("wandb init failed: %s", e)

    def on_eval(self, b_evals: int, metrics: dict) -> None:
        if not self.enabled:
            return
        try:
            self._run.log(dict(metrics), step=b_evals)
        except Exception as e:
            _log.debug("wandb log failed: %s", e)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            try:
                self._run.finish()
            except Exception:
                pass
