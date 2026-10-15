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

from ..campaign.budget import CampaignStopped
from ..config import load_config, validate
from ..engine import checkpoint
from ..engine.parallel_recovery import (inspect_parallel_stop, authorize_parallel_recovery,
                                        has_unabandoned_provider_failure)
from ..engine.recovery import ProviderRecoveryError
from ..engine.runtime import RuntimeStatus
from ..engine.run_lock import RunDirectoryBusy, RunDirectoryLock


class SuperviseError(ValueError):
    """The run cannot start as asked; the message says what to do instead."""


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
    old = ({sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
           if threading.current_thread() is threading.main_thread() else {})
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
    llm = cfg.get('llm') or {}
    if llm.get('mock'):
        return
    provider = llm.get('provider')
    if provider == 'codex_responses' and llm.get('codex_auth_file') is not None:
        return
    key_name, label = {'codex_responses': ('CODEX_TOKEN', 'Codex token'),
                       'anthropic': ('ANTHROPIC_API_KEY', 'Anthropic API key')}.get(
        provider, ('OPENROUTER_API_KEY', 'OpenRouter API key'))
    key_name = llm.get('api_key_env', key_name)
    if not os.environ.get(key_name):
        os.environ[key_name] = getpass.getpass(label + ' (hidden): ')


def start_config(run_dir, fresh_config):
    """The validated configuration a new or resumed run will use."""
    if fresh_config is not None:
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            raise SuperviseError('a fresh run needs a new or empty output directory')
        return validate(load_config(str(fresh_config)) if not isinstance(fresh_config, dict) else fresh_config)
    ckpt = run_dir / 'checkpoint.json'
    if not ckpt.exists():
        raise SuperviseError(f'no checkpoint in {run_dir}; start it with a config first')
    return validate(json.loads(ckpt.read_text())['config'])


def _explain(run_dir, cfg) -> str | None:
    """Say why a parallel wave stopped, from the provider errors its calls recorded."""
    from ..engine.parallel_recovery import _calls
    try:
        wave = json.loads((run_dir / '.parallel/current.json').read_text())['wave_id']
        errors = [row.get('error') or {} for row in _calls(run_dir / '.parallel' / f'wave-{wave}')
                  if row.get('status') == 'error']
    except (OSError, ValueError, KeyError, TypeError):
        errors = []
    statuses = {e.get('http_status') for e in errors}
    key = cfg.get('llm', {}).get('api_key_env', 'the API key')
    if statuses & {401, 403}:
        return (f'the model provider rejected {key} (HTTP {min(statuses & {401, 403})}); '
                'check the key, or remove it from ~/.config/sciel/keys and start again to enter it anew')
    if statuses & {400, 404}:
        return (f'the model provider refused the request (HTTP {min(statuses & {400, 404})}); '
                f'check that the model {cfg.get("llm", {}).get("models")} exists for this provider')
    kinds = sorted({e.get('cause_type', 'unknown') + (f" {e['http_status']}" if e.get('http_status') else '')
                    for e in errors if e})
    return 'a model call failed in a way that cannot be retried: ' + ', '.join(kinds) if kinds else None


def supervise(run_dir, cfg, *, tasks_root, fresh, recover=False, log=print, tasks_root_override=False):
    """Run the engine to its end, continuing after each transient provider failure of a parallel run."""
    from . import main as cli
    run_dir = Path(run_dir).resolve()
    ckpt = run_dir / 'checkpoint.json'
    parallel = bool(cfg.get('parallel_search', {}).get('enabled'))
    failures = 0
    initial_resume = not fresh
    while True:
        engine = cli.build_engine(cfg, str(tasks_root), str(run_dir), python=sys.executable,
                                  tasks_root_override=tasks_root_override)
        before = 0
        try:
            if not fresh:
                checkpoint.load(engine, str(ckpt))
            if parallel and initial_resume and not recover and has_unabandoned_provider_failure(str(ckpt)):
                raise SuperviseError('the checkpoint ends on an unresolved provider failure; '
                                     'resume with --abandon-provider-failure to give up that request')
            before = getattr(engine, "iteration_no", 0)
            if recover:
                authorize_parallel_recovery(engine, str(ckpt), inspect_parallel_stop(str(ckpt)))
            fresh = False
            return cli._run_engine(engine)
        except CampaignStopped:
            if engine._stop_requested or not parallel:
                # A graceful pause returns normally. A simultaneous stop request
                # cannot certify this failure or erase an unknown paid outcome.
                raise
            try:
                receipt = inspect_parallel_stop(str(ckpt))
            except ProviderRecoveryError:
                reason = _explain(run_dir, cfg)
                if reason is None:
                    raise
                raise SuperviseError(reason) from None
            if not receipt.get('auto_recoverable'):
                raise
            authorize_parallel_recovery(engine, str(ckpt), receipt)
            failures = 1 if engine.iteration_no > before else failures + 1
            delay = min(300, 30 * (2 ** min(failures - 1, 4)))
            log(f'Provider failure sealed; continuation after {delay}s without retrying the failed call. '
                'Unknown charge retained in recovery history.')
        finally:
            close = getattr(engine, 'close', None)
            if callable(close):
                close()
        if not backoff(run_dir, delay):
            return {'reason': 'paused'}
        recover = False
        initial_resume = False


def main(argv=None):
    from .main import _default_tasks_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    start = parser.add_mutually_exclusive_group()
    start.add_argument('--abandon-provider-failure', action='store_true',
                       help='Explicitly abandon eligible provider requests from an earlier stopped invocation')
    start.add_argument('--fresh-config', type=Path,
                       help='Start from the canonical seed in a new or empty run directory')
    parser.add_argument('--tasks-root', default=None)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    try:
        cfg = start_config(run_dir, args.fresh_config)
        ensure_credentials(cfg)
        result = supervise(run_dir, cfg, tasks_root=args.tasks_root or _default_tasks_root(),
                           fresh=args.fresh_config is not None, recover=args.abandon_provider_failure,
                           log=lambda m: print(m, flush=True), tasks_root_override=args.tasks_root is not None)
    except SuperviseError as exc:
        print(f'SuperviseError: {exc}', file=sys.stderr, flush=True)
        return 2
    print(json.dumps(result), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
