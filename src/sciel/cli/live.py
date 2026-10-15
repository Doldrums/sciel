"""Run the engine in its own process and follow it in the terminal, the browser, or both.

Closing a viewer never stops the run: the engine keeps going in the background and
`sciel watch`, `sciel web` or `sciel stop` reach it again by its directory.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console

MODES = ("both", "terminal", "web", "none")


def log_path(run_dir: Path) -> Path:
    return run_dir.parent / (run_dir.name + ".log")


def spawn(run_dir: Path, *, fresh_config=None, tasks_root=None, recover=False) -> subprocess.Popen:
    """Start the engine supervisor as a detached child that writes beside the run directory."""
    command = [sys.executable, "-m", "sciel.cli.main", "_engine", str(run_dir)]
    if fresh_config is not None:
        command += ["--fresh-config", str(fresh_config)]
    if tasks_root is not None:
        command += ["--tasks-root", str(tasks_root)]
    if recover:
        command.append("--abandon-provider-failure")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path(run_dir), "ab")
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=os.environ.copy())


def _reason(lines) -> str | None:
    """The exception line that ended the engine, without the traceback above it."""
    found = [line.strip() for line in lines
             if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Stopped|Interrupt|Exit)\b:?", line.strip())]
    if found:
        return found[-1]
    rest = [line.strip() for line in lines if line.strip()]
    return rest[-1] if rest else None


def _tail(path: Path, lines=12) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def follow(run_dir: Path, child, mode: str) -> int:
    """Show the run in the chosen viewer; return the engine's exit code, or 0 if it is still running."""
    from ..ui.web import request_pause, serve
    console = Console()
    web = {}

    def engine():
        code = child.poll() if child is not None else None
        tail = _tail(log_path(run_dir), 30).splitlines() if code not in (None, 0) else []
        return {"exit": code, "log": str(log_path(run_dir)), "tail": tail, "reason": _reason(tail)}

    def open_web():
        if "url" not in web:
            web["url"], web["server"] = serve(run_dir, engine=engine)
        else:
            import webbrowser
            webbrowser.open(web["url"])
        return web["url"]

    if mode in ("both", "web"):
        open_web()
    try:
        if mode in ("both", "terminal") and sys.stdin.isatty() and sys.stdout.isatty():
            from ..ui.terminal import watch
            watch(run_dir, web_url=web.get("url"), open_web=open_web, pause=lambda: request_pause(run_dir),
                  engine=engine)
        else:
            console.print(f"[#8A8A8E]web[/] {web.get('url', '—')}   [#8A8A8E]log[/] {log_path(run_dir)}   "
                          "[#8A8A8E]Ctrl-C leaves the run going[/]")
            while child is None or child.poll() is None:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    return _farewell(console, run_dir, child)


def _farewell(console, run_dir: Path, child) -> int:
    code = child.poll() if child is not None else None
    summary = _summary(run_dir)
    if child is None:
        try:
            status = json.loads((run_dir / "runtime.json").read_text()).get("status")
        except (OSError, ValueError):
            status = None
        if status != "running":
            console.print(f"The run is {status or 'not running'}. {summary}\n  {run_dir}")
            return 0
    if code is None:
        console.print(f"The run continues in the background.\n"
                      f"  [#E8A33D]sciel watch[/] {run_dir}   follow it again\n"
                      f"  [#E8A33D]sciel web[/] {run_dir}     open the page\n"
                      f"  [#E8A33D]sciel stop[/] {run_dir}    pause after the current step")
        return 0
    if code == 0:
        console.print(f"Run finished. {summary}\n  {run_dir}")
    else:
        reason = _reason(_tail(log_path(run_dir), 30).splitlines())
        console.print(f"[#E5484D]The engine stopped with an error.[/] {summary}\n  {reason or ''}\n"
                      f"  full log: {log_path(run_dir)}")
    return code


def _summary(run_dir: Path) -> str:
    from ..ui.state import RunState
    try:
        snap = RunState(run_dir).refresh()
    except (OSError, ValueError):
        return ""
    run = snap["run"]
    if run.get("best") is None:
        return ""
    seed = run.get("seed_score")
    gain = f" (+{run['best'] - seed:.3f} over seed)" if seed is not None else ""
    return f"Best {run['best']:.4f}{gain} after {run['B_evals']} evaluations, ${run.get('cost') or 0:.2f}."


def default_mode() -> str:
    return "both" if sys.stdout.isatty() else "none"


def write_config(cfg: dict, run_dir: Path) -> Path:
    """Keep the exact starting configuration beside the run, where a fresh start can read it."""
    path = run_dir.parent / (run_dir.name + ".config.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1))
    return path
