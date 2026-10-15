"""Starting without a config: pick a provider from the environment, turn the target into a task, ask the rest."""

from __future__ import annotations

import getpass
import json
import os
import sys
from pathlib import Path

from rich.console import Console

AMBER, MUTED, OK = "#E8A33D", "#8A8A8E", "#3DAE7C"


class StartError(ValueError):
    """The target cannot be started; the message says why."""


def workdir() -> Path:
    return Path.cwd() / ".sciel"


def next_run_dir(stem: str) -> Path:
    runs = workdir() / "runs"
    n = 1
    while (runs / f"{stem}-{n}").exists():
        n += 1
    return runs / f"{stem}-{n}"


def ask_provider(console: Console):
    """Use a key from the environment or the saved keys file; otherwise ask for one once."""
    from .profile import PROVIDERS, detect, save_key
    found = detect()
    if found is not None:
        return found
    if not sys.stdin.isatty():
        raise StartError("set ANTHROPIC_API_KEY or OPENROUTER_API_KEY")
    console.print(f"[{MUTED}]No model key found. Which provider?[/]")
    for i, (env, label, model, _) in enumerate(PROVIDERS, 1):
        console.print(f"  [{AMBER}]{i}[/] {label} [{MUTED}]({model}, key in {env})[/]")
    choice = console.input(f"[{AMBER}]›[/] ").strip() or "1"
    if choice not in {str(i) for i in range(1, len(PROVIDERS) + 1)}:
        raise StartError("no provider chosen")
    env, label, model, llm = PROVIDERS[int(choice) - 1]
    key = getpass.getpass(f"{env} (hidden): ").strip()
    if not key:
        raise StartError("no key given")
    os.environ[env] = key
    if console.input(f"[{MUTED}]save it in ~/.config/sciel/keys for next time? [Y/n][/] ").strip().lower() != "n":
        save_key(env, key)
    return env, label, model, dict(llm)


def resolve(target: str, *, tasks_root: str, budget=None, seed=None, console=None):
    """(config, tasks root, run name) for a config file, a single-file program, or a task name."""
    from ..config import load_config, validate
    from .profile import default_config
    path = Path(target)
    if path.suffix in (".yaml", ".yml", ".json") and path.is_file():
        cfg = validate(load_config(str(path)))
        if seed is not None:
            cfg["random_seed"] = seed
        if budget is not None:
            cfg["budget"] = budget
        return cfg, tasks_root, cfg["task"].replace("/", "_")
    console = console or Console()
    _, _, _, llm = ask_provider(console)
    options = {"budget": budget or 40, "seed": seed or 0}
    if path.suffix == ".py" and path.is_file():
        from ..local_task import build
        root = workdir() / "tasks"
        return default_config(build(path, root), llm, **options), str(root), path.stem
    if (Path(tasks_root) / target / "task.yaml").is_file():
        return default_config(target, llm, **options), tasks_root, target.replace("/", "_")
    raise StartError(f"{target} is not a config file, a .py file with solve() and score(), "
                     f"or a task under {tasks_root}")


def _runs():
    rows = []
    for run in sorted((workdir() / "runs").glob("*/runtime.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            status = json.loads(run.read_text()).get("status")
        except (OSError, ValueError):
            continue
        rows.append((run.parent, status))
    return rows


def menu(tasks_root: str) -> int:
    """The interactive start screen shown by a bare `sciel`."""
    from .. import __version__
    from ..demo import CIRCLES
    from .live import follow, spawn, write_config
    console = Console()
    console.print(f"[bold {AMBER}]SCIEL[/]  [{MUTED}]search that gets better at searching · {__version__}[/]\n")
    try:
        env, label, model, _ = ask_provider(console)
    except StartError as exc:
        console.print(str(exc))
        return 2
    console.print(f"[{OK}]✓[/] [{MUTED}]model[/]    {env} → {model} via {label}")
    console.print(f"[{OK}]✓[/] [{MUTED}]profile[/]  default · 3 islands · 40 evaluations")
    console.print(f"[{MUTED}]·[/] [{MUTED}]saves to[/] ./.sciel   run sciel here again to come back\n")
    runs = [(d, s) for d, s in _runs() if s != "completed"]
    choices = [("1", "demo: 26 circles in a square", "a few minutes"),
               ("2", "a program of mine", "a .py file with solve() and score()")]
    if runs:
        choices.append(("3", f"continue {runs[0][0].name}", runs[0][1]))
    console.print(f"[{MUTED}]WHAT SHOULD IT IMPROVE?[/]")
    for key, title, hint in choices:
        console.print(f"  [{AMBER}]{key}[/]  {title:<36} [{MUTED}]{hint}[/]")
    pick = console.input(f"[{AMBER}]›[/] ").strip() or "1"
    console.print(f"\n[{MUTED}]WATCH IN[/]  [{AMBER}]t[/] terminal   [{AMBER}]w[/] browser   [{AMBER}]b[/] both [{MUTED}](default)[/]")
    mode = {"t": "terminal", "w": "web"}.get(console.input(f"[{AMBER}]›[/] ").strip().lower(), "both")
    if pick == "3" and runs:
        run_dir, status = runs[0]
        child = None if status == "running" else spawn(run_dir, tasks_root=_tasks_root_of(run_dir, tasks_root))
        return follow(run_dir, child, mode)
    if pick == "2":
        target = console.input(f"[{MUTED}]path to the .py file:[/] ").strip()
    elif pick == "1":
        target = str(CIRCLES)
    else:
        console.print("no such choice")
        return 2
    try:
        cfg, root, stem = resolve(target, tasks_root=tasks_root, console=console)
    except (StartError, OSError, ValueError) as exc:
        console.print(str(exc))
        return 2
    run_dir = next_run_dir(stem)
    return follow(run_dir, spawn(run_dir, fresh_config=write_config(cfg, run_dir), tasks_root=root), mode)


def _tasks_root_of(run_dir: Path, default: str) -> str:
    """Local tasks live under .sciel/tasks; everything else under the configured task root."""
    try:
        task = json.loads((run_dir / "checkpoint.json").read_text())["config"]["task"]
    except (OSError, ValueError, KeyError):
        return default
    return str(workdir() / "tasks") if task.startswith("local/") else default
