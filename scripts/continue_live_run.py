"""Start or resume a live experiment; back off after transient provider loss.

Each lost request remains sealed and unpriced. Recovery starts a new request;
it never resends the abandoned one or relaunches evaluation of a saved child.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import signal
import sys
import threading

from sciel.campaign.budget import CampaignStopped
from sciel.cli.main import _run_engine
from sciel.config import load_config, validate
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.parallel_recovery import (inspect_parallel_stop, authorize_parallel_recovery,
                                           has_unabandoned_provider_failure)
from sciel.engine.runtime import RuntimeStatus
from sciel.engine.run_lock import RunDirectoryBusy, RunDirectoryLock


def backoff(run_dir, seconds):
    """Visible, interruptible cooldown; STOP and SIGTERM still pause the run."""
    try:
        owner = RunDirectoryLock(run_dir)
    except RunDirectoryBusy:
        # Another supervisor took over after Engine.close(). It now owns both
        # the checkpoint and telemetry; this invocation must stop publishing.
        return False
    try:
        return _owned_backoff(run_dir, seconds)
    finally:
        owner.close()


def _owned_backoff(run_dir, seconds):
    current = json.loads((run_dir / 'checkpoint.json').read_text())
    stopped = threading.Event()
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in old:
        signal.signal(sig, lambda *_: stopped.set())
    runtime = RuntimeStatus(run_dir, lambda: dict(
        run_id=current.get('provenance', {}).get('run_id'),
        segment_id=current.get('provenance', {}).get('journal_anchor', {}).get('segment_id'),
        iteration_no=current.get('iteration_no'),
        B_evals=current.get('B_evals')))
    try:
        runtime.start()
        with runtime.phase('provider_backoff'):
            for _ in range(seconds):
                if stopped.wait(1) or (run_dir / 'STOP').exists():
                    return False
        return not stopped.is_set() and not (run_dir / 'STOP').exists()
    finally:
        runtime.finish('paused' if stopped.is_set() or (run_dir / 'STOP').exists() else 'running',
                       reason='provider_backoff')
        for sig, handler in old.items():
            signal.signal(sig, handler)


def ensure_credentials(cfg):
    """File-based Codex auth is read by its adapter immediately before HTTP."""
    llm = cfg['llm']
    if llm.get('mock'):
        return
    codex = llm.get('provider') == 'codex_responses'
    if codex and llm.get('codex_auth_file') is not None:
        return
    key_name = llm.get('api_key_env', 'CODEX_TOKEN' if codex else 'OPENROUTER_API_KEY')
    if not os.environ.get(key_name):
        label = 'Codex token' if codex else 'OpenRouter API key'
        os.environ[key_name] = getpass.getpass(label + ' (hidden): ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--abandon-provider-failure', action='store_true',
                       help='Explicitly abandon eligible provider requests from an earlier stopped invocation')
    start.add_argument('--fresh-config', type=Path,
                       help='Start from the canonical seed in a new or empty run directory')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    ckpt = run_dir / 'checkpoint.json'
    fresh = args.fresh_config is not None
    if fresh:
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            parser.error('A fresh run requires a new or empty output directory')
        cfg = validate(load_config(str(args.fresh_config)))
        if cfg.get('warm_start') is not None:
            parser.error('A fresh run cannot use warm_start')
    else:
        cfg = validate(json.loads(ckpt.read_text())['config'])
    if cfg.get('budget') is not None or cfg.get('max_iterations') is not None:
        parser.error('This supervisor is only for explicitly unlimited live runs')
    ensure_credentials(cfg)
    if not cfg.get('parallel_search', {}).get('enabled'):
        parser.error('This supervisor requires parallel_search')
    recover = args.abandon_provider_failure
    failures = 0
    initial_resume = not fresh
    while True:
        engine = build_engine(cfg, str(root / 'tasks'), str(run_dir), python=sys.executable)
        before = 0
        try:
            if not fresh:
                checkpoint.load(engine, str(ckpt))
            if (initial_resume and not recover
                    and has_unabandoned_provider_failure(str(ckpt))):
                parser.error('Parallel checkpoint has an existing unresolved provider failure; '
                             'use --abandon-provider-failure to explicitly abandon it')
            before = engine.iteration_no
            if recover:
                authorize_parallel_recovery(engine, str(ckpt), inspect_parallel_stop(str(ckpt)))
            fresh = False
            result = _run_engine(engine)
            print(json.dumps(result), flush=True)
            return 0
        except CampaignStopped:
            if engine._stop_requested:
                # A graceful pause returns normally. A simultaneous stop request
                # cannot certify this failure or erase an unknown paid outcome.
                raise
            receipt = inspect_parallel_stop(str(ckpt))
            if not receipt.get('auto_recoverable'):
                raise
            authorize_parallel_recovery(engine, str(ckpt), receipt)
            failures = 1 if engine.iteration_no > before else failures + 1
            delay = min(300, 30 * (2 ** min(failures - 1, 4)))
            print(f'Provider failure sealed; continuation after {delay}s without retrying the failed call. '
                  'Unknown charge retained in recovery history.', flush=True)
        finally:
            engine.close()
        if not backoff(run_dir, delay):
            return 0
        recover = False
        initial_resume = False


if __name__ == '__main__':
    raise SystemExit(main())
