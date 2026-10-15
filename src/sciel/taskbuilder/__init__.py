"""Materials to task: read a folder, draft problem.py with a model, audit it, and hand it to the search."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ..local_task import build
from .audit import audit
from .draft import Drafter
from .materials import digest

ACCENT, MUTED, INK, OK, BAD = "#64D2FF", "#8A8A8E", "#E8E8EA", "#3DAE7C", "#E5484D"
REPAIRS = 2
MARKS = {"✓": OK, "!": ACCENT, "✗": BAD}


def _steps(console, done):
    line = Text()
    for i, name in enumerate(("read", "draft", "audit", "search")):
        mark, color = ("✓", OK) if i < done else ("›", ACCENT) if i == done else ("·", MUTED)
        line.append(f"{mark} {name}   ", style=color)
    console.print(line)


def _label(console, text):
    console.print(Text(text.upper(), style=MUTED))


def _audit_table(result):
    table = Table.grid(padding=(0, 2))
    for mark, name, detail in result["rows"]:
        table.add_row(Text(mark, style=MARKS.get(mark, MUTED)), Text(name, style=INK), Text(detail, style=MUTED))
    return table


def run(folder, *, tasks_root, note="", console=None):
    """Return (task name, audit result, problem path), or raise ValueError with what went wrong."""
    from ..cli.start import ask_provider
    from ..engine.factory import build_llm
    console = console or Console()
    folder = Path(folder).resolve()
    console.print(Text.assemble(("SCIEL", f"bold {ACCENT}"), ("  task ", MUTED), str(folder)))
    _steps(console, 0)
    text, rows = digest(folder)
    _label(console, "read")
    for row in rows[:12]:
        console.print(Text.assemble((row["name"], INK), (f"  {row['size']:,} B  ", MUTED), (row["took"][:70], MUTED)))
    if len(rows) > 12:
        console.print(Text(f"… {len(rows) - 12} more files", style=MUTED))
    env, label, model, llm_cfg = ask_provider(console)
    drafter = Drafter(build_llm({"llm": llm_cfg}), llm_cfg["models"][0])
    _steps(console, 1)
    with console.status(f"drafting with {model}…", spinner="dots"):
        draft = drafter.draft(folder, text, note=note)
    drafts = Path(tasks_root).parent / "drafts" / folder.name
    drafts.mkdir(parents=True, exist_ok=True)
    problem = drafts / "problem.py"
    for attempt in range(REPAIRS + 1):
        problem.write_text(draft["source"])
        name = build(problem, tasks_root)
        _steps(console, 2)
        with console.status("auditing: running the seed and the reference…", spinner="dots"):
            result = audit(Path(tasks_root) / name)
        if result["error"] is None or attempt == REPAIRS:
            break
        console.print(Text(f"the seed failed; asking the model to fix it ({attempt + 1}/{REPAIRS})", style=ACCENT))
        with console.status("repairing…", spinner="dots"):
            draft = drafter.repair(draft["source"], result["error"])
    _label(console, "draft")
    console.print(Text(draft["about"] or "(no description)", style=INK))
    console.print(Text(str(problem), style=MUTED))
    for question in draft["questions"]:
        console.print(Text.assemble(("? ", ACCENT), (question, INK)))
    _label(console, "audit")
    console.print(_audit_table(result))
    console.print(Text(result["verdict"], style=OK if result["admitted"] else BAD))
    console.print(Text(f"drafting cost ${drafter.cost:.2f}", style=MUTED))
    return name, result, problem


def interactive(folder, *, tasks_root, note=""):
    """`sciel task FOLDER`: build, then let the user start the search, edit the draft and re-audit, or stop."""
    from ..cli.live import follow, spawn, write_config
    from ..cli.profile import default_config
    from ..cli.start import ask_provider, next_run_dir
    console = Console()
    name, result, problem = run(folder, tasks_root=tasks_root, note=note, console=console)
    if not sys.stdin.isatty():
        console.print(f"task {name} under {tasks_root}; start it with: sciel run {problem}")
        return 0 if result["admitted"] else 1
    while True:
        prompt = Text.assemble(("enter", ACCENT), " start search   ", ("e", ACCENT), " edit draft   ", ("q", ACCENT), " quit ")
        choice = console.input(prompt).strip().lower()
        if choice == "q" or (choice == "" and result["error"] is not None):
            console.print(f"Saved {problem}. Start it later with: sciel run {problem}")
            return 0 if result["admitted"] else 1
        if choice == "e":
            subprocess.run(shlex.split(os.environ.get("EDITOR", "vi")) + [str(problem)])
            name = build(problem, tasks_root)
            with console.status("auditing…", spinner="dots"):
                result = audit(Path(tasks_root) / name)
            console.print(_audit_table(result))
            console.print(Text(result["verdict"], style=OK if result["admitted"] else BAD))
            continue
        _, _, _, llm = ask_provider(console)
        run_dir = next_run_dir(Path(folder).name)
        cfg = default_config(name, llm)
        return follow(run_dir, spawn(run_dir, fresh_config=write_config(cfg, run_dir), tasks_root=tasks_root), "both")
