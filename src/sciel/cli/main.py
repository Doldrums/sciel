"""Command-line interface for running and managing experiments."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time

def build_engine(*args, **kwargs):
    """Load search dependencies only when building an engine."""
    from ..engine.factory import build_engine as factory
    return factory(*args, **kwargs)


def _default_tasks_root() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    return os.path.join(here, "tasks")


class TaskRootError(ValueError):
    """An unavailable local task bundle, reported as a CLI usage error."""


def _tasks_root(args, *, task_name=None) -> str:
    root = args.tasks_root if args.tasks_root is not None else _default_tasks_root()
    guidance = ("Pass --tasks-root /path/to/tasks containing <suite>/<name>/task.yaml. "
                "Task bundles are not included in the installed wheel.")
    if not os.path.isdir(root):
        raise TaskRootError(f"Task root is not a directory: {root!r}. {guidance}")
    if task_name is not None and not os.path.isfile(os.path.join(root, task_name, "task.yaml")):
        raise TaskRootError(f"Task {task_name!r} was not found in task root {root!r}. {guidance}")
    return root


def _resolve_config(config_path: str, seed: int | None) -> dict:
    from ..config import load_config, validate
    cfg = validate(load_config(config_path))
    if seed is not None:
        cfg["random_seed"] = seed
    return cfg


def _run_engine(engine):
    if not hasattr(engine, 'request_stop') or threading.current_thread() is not threading.main_thread():
        return engine.run()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def stop(signum, frame):
        engine.request_stop()
    try:
        for sig in previous:
            signal.signal(sig, stop)
        return engine.run()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def cmd_run(args) -> int:
    cfg = _resolve_config(args.config, args.seed)
    if args.budget is not None:
        cfg["budget"] = args.budget
    tasks_root = args.tasks_root or _default_tasks_root()
    stamp = args.run_id or f"{cfg['task'].replace('/', '_')}_{int(time.time())}_s{cfg['random_seed']}"
    run_dir = os.path.join(args.outputs, stamp)
    if os.path.exists(run_dir) and os.listdir(run_dir):
        raise ValueError('Run directory is not empty; use resume or a new --run-id')
    if getattr(args, 'warm_start_from', None):
        from ..engine.warm_start import from_run
        cfg['warm_start'] = from_run(args.warm_start_from, cfg['task'])
    engine = build_engine(cfg, tasks_root, run_dir, python=sys.executable,
                          campaign=args.campaign, arm=args.arm,
                          tasks_root_override=args.tasks_root is not None)
    try:
        summary = _run_engine(engine)
    finally:
        close = getattr(engine, 'close', None)
        if callable(close):
            close()
    print(json.dumps(summary))
    return 0


def cmd_resume(args) -> int:
    """Resume from the checkpoint's own config.

    The run directory's config.yaml used to be loaded and validated first and then
    overwritten by the checkpoint's copy on the next line: the value was never used,
    yet a validation error in that unused file aborted the resume -- which defeats
    DEPRECATED_KEYS, the mechanism that exists to keep old snapshots resumable."""
    from ..config import validate
    from ..engine import checkpoint
    run_dir = args.run_dir
    ckpt = os.path.join(run_dir, "checkpoint.json")
    with open(ckpt) as f:
        cfg = validate(json.load(f)["config"])
    tasks_root = args.tasks_root or _default_tasks_root()
    engine = build_engine(cfg, tasks_root, run_dir, python=sys.executable,
                          tasks_root_override=args.tasks_root is not None)
    try:
        checkpoint.load(engine, ckpt,
                        allow_identity_mismatch=getattr(args, "force_identity", False))
        summary = _run_engine(engine)
    except checkpoint.IdentityMismatch as e:
        print(str(e), file=sys.stderr)
        return 2
    finally:
        close = getattr(engine, 'close', None)
        if callable(close):
            close()
    print(json.dumps(summary))
    return 0


def cmd_tasks(args) -> int:
    from ..eval.registry import determinism_check, discover, load_task
    if args.action == "hash" and not args.name:
        print("usage: sciel tasks hash <suite>/<name>", file=sys.stderr)
        return 2
    tasks_root = _tasks_root(args, task_name=args.name if args.action == "hash" else None)
    if args.action in ("list", "validate"):
        tasks = discover(tasks_root)
        if not tasks:
            raise TaskRootError(
                f"No tasks found in task root {tasks_root!r}. "
                "Pass --tasks-root /path/to/tasks containing <suite>/<name>/task.yaml. "
                "Task bundles are not included in the installed wheel.")
    if args.action == "list":
        for t in tasks:
            print(f"{t.qualified_name}\t{t.task_hash[:12]}\t{t.origin}")
        return 0
    if args.action == "validate":
        ok = True
        for t in tasks:
            dc = determinism_check(t, python=sys.executable,
                                   program_path=args.probe)
            status = "OK" if dc["valid"] else "FAIL"
            ok = ok and dc["valid"]
            print(f"{status}\t{t.qualified_name}\t{dc}")
            if not dc["informative"]:
                print(f"    note: both runs of {dc['probe']} scored 0.0, so this "
                      f"proves nothing about whether the task is deterministic. "
                      f"Re-check with --probe <a real evolved program>.")
        return 0 if ok else 1
    if args.action == "hash":
        t = load_task(os.path.join(tasks_root, args.name))
        print(t.task_hash)
        return 0
    return 2




def cmd_stop(args) -> int:
    from ..engine.runtime import process_matches
    try:
        with open(os.path.join(args.run_dir, 'runtime.json')) as stream:
            runtime = json.load(stream)
        if runtime.get('status') != 'running' or process_matches(runtime) is not True:
            raise ValueError('no verified local running process')
        with open(os.path.join(args.run_dir, 'STOP'), 'a'):
            pass
    except (OSError, ValueError, TypeError):
        print('No verified running experiment at this path.', file=sys.stderr)
        return 2
    print('Pause requested. The current iteration will finish and save a checkpoint.')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sciel")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run")
    pr.add_argument("config")
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--budget", type=int, default=None)
    pr.add_argument("--outputs", default="outputs")
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--tasks-root", default=None)
    pr.add_argument("--campaign", default="adhoc")
    pr.add_argument("--arm", default="A")
    pr.add_argument('--warm-start-from', metavar='RUN_DIR',
                    help='start a new run from a checkpoint best source, freshly evaluated')
    pr.set_defaults(func=cmd_run)

    prs = sub.add_parser("resume")
    prs.add_argument("--force-identity", action="store_true",
                     help="resume even though the task or the diversity axis changed "
                          "since the checkpoint was written (the archive and the new "
                          "children will not be comparable)")
    prs.add_argument("run_dir")
    prs.add_argument("--tasks-root", default=None)
    prs.set_defaults(func=cmd_resume)

    pt = sub.add_parser("tasks")
    pt.add_argument("action", choices=["list", "validate", "hash"])
    pt.add_argument("name", nargs="?", default=None)
    pt.add_argument("--probe", default=None,
                    help="program to run twice for the determinism check; the default "
                         "is the task's seed, which for a constant seed proves nothing")
    pt.add_argument("--tasks-root", default=None)
    pt.set_defaults(func=cmd_tasks)

    ps = sub.add_parser("stop", help="Pause a local experiment after its current iteration")
    ps.add_argument("run_dir")
    ps.set_defaults(func=cmd_stop)

    # The guard has to run BEFORE parse_args: argparse rejects an unknown option
    # first, so the explanatory message was unreachable. It also read sys.argv rather
    # than the argv it was given, so it was wrong under programmatic calls.
    source = list(argv) if argv is not None else sys.argv[1:]
    if source[:1] == ["run"] and "-s" in source:
        raise SystemExit("seed via --seed / overlay {random_seed: N}, not -s")
    args = p.parse_args(argv)
    try:
        return args.func(args)
    except TaskRootError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
