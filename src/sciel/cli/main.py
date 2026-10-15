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
    from pathlib import Path
    from .live import default_mode, follow, spawn, write_config
    from .start import StartError, next_run_dir, resolve
    from .supervise import SuperviseError, ensure_credentials, supervise
    try:
        cfg, tasks_root, stem = resolve(args.config, tasks_root=args.tasks_root or _default_tasks_root(),
                                        budget=args.budget, seed=args.seed)
        ensure_credentials(cfg)
    except (StartError, SuperviseError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.run_id:
        run_dir = Path(args.outputs, args.run_id).resolve()
    elif args.outputs != ".sciel/runs":
        run_dir = Path(args.outputs, f"{stem}_{int(time.time())}_s{cfg['random_seed']}").resolve()
    else:
        run_dir = next_run_dir(stem)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Run directory is not empty; use resume or a new --run-id')
    if getattr(args, 'warm_start_from', None):
        from ..engine.warm_start import from_run
        cfg['warm_start'] = from_run(args.warm_start_from, cfg['task'])
    mode = args.watch or default_mode()
    if mode == "none":
        summary = supervise(run_dir, cfg, tasks_root=tasks_root, fresh=True,
                            log=lambda m: print(m, file=sys.stderr, flush=True))
        print(json.dumps(summary))
        return 0
    child = spawn(run_dir, fresh_config=write_config(cfg, run_dir), tasks_root=tasks_root)
    return follow(run_dir, child, mode)


def cmd_task(args) -> int:
    from ..taskbuilder import interactive
    from .start import StartError, workdir
    try:
        return interactive(args.folder, tasks_root=str(workdir() / "tasks"), note=args.note)
    except (StartError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def cmd_watch(args) -> int:
    from pathlib import Path
    from .live import follow
    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "sciel_events.jsonl").exists():
        print(f"No run at {run_dir}", file=sys.stderr)
        return 2
    return follow(run_dir, None, "web" if args.cmd == "web" else ("both" if args.web else "terminal"))


def cmd_resume(args) -> int:
    """Resume from the checkpoint's own config.

    The run directory's config.yaml used to be loaded and validated first and then
    overwritten by the checkpoint's copy on the next line: the value was never used,
    yet a validation error in that unused file aborted the resume -- which defeats
    DEPRECATED_KEYS, the mechanism that exists to keep old snapshots resumable."""
    from pathlib import Path
    from ..engine import checkpoint
    from .live import default_mode, follow, spawn
    from .supervise import SuperviseError, ensure_credentials, start_config, supervise
    run_dir = Path(args.run_dir).resolve()
    try:
        cfg = start_config(run_dir, None)
        ensure_credentials(cfg)
    except SuperviseError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    mode = args.watch or default_mode()
    from .start import _tasks_root_of
    tasks_root = args.tasks_root or _tasks_root_of(run_dir, _default_tasks_root())
    if mode != "none" and not args.force_identity:
        child = spawn(run_dir, tasks_root=tasks_root, recover=args.abandon_provider_failure)
        return follow(run_dir, child, mode)
    if args.force_identity:
        engine = build_engine(cfg, tasks_root, str(run_dir), python=sys.executable,
                              tasks_root_override=args.tasks_root is not None)
        try:
            checkpoint.load(engine, str(run_dir / "checkpoint.json"), allow_identity_mismatch=True)
            summary = _run_engine(engine)
        finally:
            engine.close()
    else:
        try:
            summary = supervise(run_dir, cfg, tasks_root=tasks_root, fresh=False,
                                recover=args.abandon_provider_failure,
                                log=lambda m: print(m, file=sys.stderr, flush=True))
        except checkpoint.IdentityMismatch as e:
            print(str(e), file=sys.stderr)
            return 2
        except SuperviseError as exc:
            print(str(exc), file=sys.stderr)
            return 2
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
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    watch_help = "where to follow the run: both (default in a terminal), terminal, web, or none"
    pr = sub.add_parser("run", help="Start a run: a config file, a .py file with solve() and score(), or a task")
    pr.add_argument("config", metavar="target")
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--budget", type=int, default=None, help="number of evaluations")
    pr.add_argument("--watch", choices=("both", "terminal", "web", "none"), default=None, help=watch_help)
    pr.add_argument("--outputs", default=".sciel/runs", help="where run directories go")
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--tasks-root", default=None)
    pr.add_argument('--warm-start-from', metavar='RUN_DIR',
                    help='start a new run from a checkpoint best source, freshly evaluated')
    pr.set_defaults(func=cmd_run)

    prs = sub.add_parser("resume", help="Continue a paused or stopped run")
    prs.add_argument("--force-identity", action="store_true",
                     help="resume even though the task or the diversity axis changed "
                          "since the checkpoint was written (the archive and the new "
                          "children will not be comparable)")
    prs.add_argument("--abandon-provider-failure", action="store_true",
                     help="give up the provider request the run stopped on and continue")
    prs.add_argument("--watch", choices=("both", "terminal", "web", "none"), default=None, help=watch_help)
    prs.add_argument("run_dir")
    prs.add_argument("--tasks-root", default=None)
    prs.set_defaults(func=cmd_resume)

    pt2 = sub.add_parser("task", help="Turn a folder of materials into a checked task, then search it")
    pt2.add_argument("folder")
    pt2.add_argument("--note", default="", help="what you want from the task, in a sentence")
    pt2.set_defaults(func=cmd_task)

    pw = sub.add_parser("watch", help="Follow a run in the terminal")
    pw.add_argument("run_dir")
    pw.add_argument("--web", action="store_true", help="also open the web page")
    pw.set_defaults(func=cmd_watch)

    pweb = sub.add_parser("web", help="Open a run's live page in the browser")
    pweb.add_argument("run_dir")
    pweb.set_defaults(func=cmd_watch)

    pt = sub.add_parser("tasks")
    pt.add_argument("action", choices=["list", "validate", "hash"])
    pt.add_argument("name", nargs="?", default=None)
    pt.add_argument("--probe", default=None,
                    help="program to run twice for the determinism check; the default "
                         "is the task's seed, which for a constant seed proves nothing")
    pt.add_argument("--tasks-root", default=None)
    pt.set_defaults(func=cmd_tasks)

    ps = sub.add_parser("stop", help="Pause a run after its current step")
    ps.add_argument("run_dir")
    ps.set_defaults(func=cmd_stop)

    # The guard has to run BEFORE parse_args: argparse rejects an unknown option
    # first, so the explanatory message was unreachable. It also read sys.argv rather
    # than the argv it was given, so it was wrong under programmatic calls.
    source = list(argv) if argv is not None else sys.argv[1:]
    if not source:
        from .start import menu
        return menu(_default_tasks_root())
    if source[0].endswith(".py") and os.path.isfile(source[0]):
        source = ["run"] + source
        argv = source
    if source[:1] == ["_engine"]:
        from .supervise import main as supervise_main
        return supervise_main(source[1:])
    if source[:1] == ["run"] and "-s" in source:
        raise SystemExit("seed via --seed / overlay {random_seed: N}, not -s")
    args = p.parse_args(argv)
    try:
        return args.func(args)
    except TaskRootError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
