"""The terminal dashboard: a quiet three-column screen that follows a run and draws its views in braille."""

from __future__ import annotations

import os
import select
import sys
import threading
import time
from contextlib import contextmanager

from rich.console import Console, Group
from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .render import braille, Braille
from .state import RunState
from .views import all_views, island_color

INK, MUTED, DIM, RULE = "#E8E8EA", "#8A8A8E", "#5A5A5E", "#2C2C2E"
AMBER, UP, DOWN = "#E8A33D", "#3DAE7C", "#E5484D"
VIEWS = (("1", "culture"), ("2", "landscape"), ("3", "code"), ("4", "edits"))
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
TONE = {"best": UP, "drop": DOWN, "warn": AMBER, "finding": AMBER, "cell": INK, "child": INK}
STAGES = {"seeding": "evaluating the seed", "evaluating": "evaluating", "generating": "writing programs",
          "checkpointing": "saving", "provider_backoff": "waiting for the provider", "finished": "finished"}
STATUS = {"culture": (UP, "holds"), "tested": ("#0A84FF", "tested"), "trial": (AMBER, "trial"),
          "retired": (MUTED, "dropped")}


def _clock(seconds):
    if seconds is None:
        return ""
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds < 3600 else f"{seconds // 3600}h{seconds % 3600 // 60:02d}"


def _ago(stamp):
    from .state import _seconds
    at = _seconds(stamp)
    return None if at is None else max(0, time.time() - at)


def _f(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _seconds_or_now(stamp):
    from .state import _seconds
    at = _seconds(stamp)
    return at if at is not None else time.time()


def _label(text):
    return Text(text.upper(), style=Style(color=MUTED, bold=False))


def _canvas_text(canvas: Braille) -> Text:
    out = Text()
    for i, row in enumerate(canvas.cells()):
        if i:
            out.append("\n")
        run, style = "", None
        for ch, fg, bg in row:
            cell = Style(color=fg or DIM, bgcolor=bg)
            if cell != style and run:
                out.append(run, style)
                run = ""
            style = cell
            run += ch
        if run:
            out.append(run, style)
    return out


def _sparkline(history, seed, width, height=3):
    canvas = Braille(width, height)
    if history:
        values = [v for _, v in history] + ([seed] if seed is not None else [])
        low, high = min(values), max(values)
        span = (high - low) or 1.0
        end = max(history[-1][0], 1)
        y = lambda v: 0.05 + 0.9 * (v - low) / span
        points = []
        for (x, v), nxt in zip(history, history[1:] + [[end, history[-1][1]]]):
            points += [[x / end, y(v)], [nxt[0] / end, y(v)]]
        for a, b in zip(points, points[1:]):
            canvas.line(a, b, INK)
        if seed is not None:
            canvas.line([0, y(seed)], [1, y(seed)], DIM, dashed=True)
    return _canvas_text(canvas)


class Dashboard:
    def __init__(self, run_dir, web_url=None, open_web=None, pause=None, engine=None):
        self.state = RunState(run_dir)
        self.engine = engine
        self.view = "culture"
        self.web_url = web_url
        self.open_web = open_web
        self.pause = pause
        self.message = None
        self.quit = False
        self.snap = None
        self.figures = None
        self._figure_key = None
        self._refreshed = 0.0

    def _header(self, run, width):
        live = run.get("status") == "running"
        stage = STAGES.get(run.get("stage"), run.get("stage"))
        status = (Text("● live" + (f" · {stage}" if stage else ""), style=UP) if live
                  else Text(str(run.get("status") or "unknown"), style=AMBER))
        if run.get("stage") == "provider_backoff":
            status = Text("● waiting for the provider", style=AMBER)
        failed = self._failure()
        if failed:
            status = Text("● engine stopped with an error", style=DOWN)
        left = Text.assemble(("SCIEL", Style(color=AMBER, bold=True)), "  ", (str(run.get("task") or ""), INK),
                             ("  run " + str(run.get("run_id") or "")[:8], MUTED))
        budget = f"/{run['budget']}" if run.get("budget") else ""
        eta = run.get("eta_seconds")
        right = Text.assemble(("evals ", MUTED), f"{run.get('B_evals', 0)}{budget}", ("   spent ", MUTED),
                              f"${run.get('cost') or 0:.2f}",
                              *((("   eta ", MUTED), f"{int(eta // 60)}m") if eta and live else ()), "   ", status)
        pad = max(1, width - left.cell_len - right.cell_len)
        return Text.assemble(left, " " * pad, right)

    def _left(self, snap, width):
        run = snap["run"]
        best, seed = run.get("best"), run.get("seed_score")
        gain = None if best is None or seed is None else best - seed
        score = Text.assemble((_f(best), Style(color=INK, bold=True)), "   ",
                              ((("▲ +" if gain >= 0 else "▼ ") + _f(abs(gain)), UP if gain >= 0 else DOWN)
                               if gain is not None else ("", MUTED)))
        parts = [_label("score"), score, _sparkline(snap["history"], seed, width),
                 Text(f"seed {_f(seed)}", style=MUTED), Text("")]
        best_id = snap.get("best")
        best_program = next((p for p in snap["programs"] if p["id"] == best_id), None)
        cases = sorted((best_program or {}).get("cases", {}).items())[:5]
        if cases:
            parts.append(_label(f"best program {best_id}"))
            for key, value in cases:
                parts.append(Text.assemble((f"{key[:18]:<19}", MUTED), f"{value:.4g}"))
            parts.append(Text(""))
        parts.append(_label("islands"))
        table = Table.grid(padding=(0, 1))
        for column in ("id", "mode", "best", "now"):
            table.add_column(column, no_wrap=True, overflow="ellipsis", max_width=width - 18 if column == "now" else None)
        for island in snap["islands"]:
            table.add_row(Text(f"I{island['id']}", style=island_color(island["id"])),
                          Text(str(island.get("mode") or "—"), style=INK), _f(island.get("best")),
                          Text(str(island.get("idea") or island.get("op") or ""), style=MUTED))
        parts.append(table)
        return Group(*parts)

    def _failure(self):
        info = self.engine() if self.engine is not None else None
        return info if info and info.get("exit") not in (None, 0) else None

    def _status_panel(self, snap, width):
        """What the center shows instead of a view: a failed engine, or one that has not produced a program yet."""
        if not snap["programs"]:
            stage = STAGES.get(snap["run"].get("stage"))
            if snap["run"].get("status") == "starting":
                text = "starting the engine: loading the task and the model"
            else:
                text = f"{stage or 'working'}: the first programs appear after the seed is scored"
            return Group(_label("starting"), Text(text, style=INK),
                         Text("the seed is evaluated first; with a slow evaluator this takes a minute", style=MUTED))
        return None

    def _center(self, snap, width, height):
        panel = self._status_panel(snap, width)
        if panel is not None:
            return panel
        tabs = Text()
        for key, name in VIEWS:
            on = name == self.view
            tabs.append(key + " ", style=AMBER)
            tabs.append(name, style=Style(color=INK if on else MUTED, underline=on))
            tabs.append("   ")
        fig = self.figures[self.view]
        stats = fig.get("stats") or {}
        if self.view == "culture" and stats:
            tabs.append(f"  passed on {stats['transfers']} · rose {stats['worked']} (drawn)", style=MUTED)
        events = snap["events"][-5:]
        room = max(6, height - len(events) - 6)
        if self.view == "culture":
            rows = min(room, width // 2)
            cols = rows * 2
        else:
            rows, cols = room, width
        if self.view == "culture":
            fig = {**fig, "items": [i for i in fig["items"] if not (i["t"] == "line" and i.get("dash"))]}
        art = _canvas_text(braille(fig, cols, rows))
        tape = [Text.assemble((str(e.get("t") or "")[11:19], DIM), "  ", (e["text"], TONE.get(e.get("kind"), MUTED)))
                for e in reversed(events)]
        legend = Text(fig.get("caption") or "", style=MUTED, overflow="ellipsis", no_wrap=True)
        return Group(tabs, Text(""), art, legend, Text(""), *self._now(snap), Text(""), _label("events"), *tape)

    def _now(self, snap):
        """One line per job of the current wave, with a spinner and a ticking clock while it works."""
        act = snap.get("activity") or {}
        frame = SPIN[int(time.time() * 8) % len(SPIN)]
        quiet = _ago(act.get("last") or (snap["events"][-1]["t"] if snap["events"] else None))
        head = Text.assemble(("NOW", MUTED), (f"   wave {act['wave']}" if act.get("wave") else "", MUTED),
                             (f"   last activity {_clock(quiet)} ago" if quiet is not None else "", DIM))
        rows = [head]
        for job in act.get("jobs", []):
            busy = job["doing"] not in ("done", "queued")
            clock = _clock(time.time() - _seconds_or_now(job["since"]) if busy else job["seconds"])
            who = f"I{job['island']}" if job.get("island") is not None else job["job"]
            what = f"{job['what']} of {job['parent']}" if job.get("parent") else str(job["what"])
            rows.append(Text.assemble((f"{who:<4}", island_color(job.get("island")) if job.get("island") is not None
                                       else MUTED), (f"{what[:26]:<27}", INK),
                                      (f"{frame} " if busy else ("✓ " if job["doing"] == "done" else "· "),
                                       AMBER if busy else (UP if job["doing"] == "done" else DIM)),
                                      (f"{job['doing']:<18}", INK if busy else MUTED), (clock, MUTED)))
        if len(rows) == 1:
            stage = STAGES.get(snap["run"].get("stage"))
            rows.append(Text(f"{frame} {stage or 'working'}", style=AMBER))
        return rows

    def _right(self, snap, width):
        parts = [_label("what the search learned")]
        rank = {"culture": 0, "tested": 1, "trial": 2, "retired": 3}
        findings = sorted(snap["findings"], key=lambda f: rank.get(f.get("status"), 2))[:5]
        if not findings:
            parts.append(Text("nothing yet", style=DIM))
        for f in findings:
            color, word = STATUS.get(f.get("status"), (MUTED, f.get("status") or ""))
            parts.append(Text.assemble((f"{word:<8}", color), (str(f.get("text") or "")[:width * 2 - 10], INK)))
        decided = [c for c in snap["claims"] if c.get("rejected") and (c.get("low", 0) > 0 or c.get("high", 0) < 0)]
        for c in sorted(decided, key=lambda c: -abs(c.get("effect", 0)))[:3]:
            sign = UP if c["effect"] > 0 else DOWN
            parts.append(Text.assemble(("claim   ", sign), f"{c['decision']}={c['action']} ",
                                       (f"{c['effect']:+.2f} [{c['low']:+.2f}, {c['high']:+.2f}]", MUTED)))
        parts += [Text(""), _label("research cells")]
        cells = snap["cells"][-4:]
        if not cells:
            parts.append(Text("none open", style=DIM))
        for cell in reversed(cells):
            parts.append(Text.assemble((f"{cell['id']}  ", MUTED), (str(cell.get("question") or "")[:width * 2 - 10], INK)))
            parts.append(Text(f"        {cell.get('state')}", style=MUTED))
        return Group(*parts)

    def _footer(self, width):
        keys = Text()
        for key, label in (("1-4", "views"), ("w", "open web"), ("p", "pause run"), ("q", "quit viewer")):
            keys.append(key, style=AMBER)
            keys.append(f" {label}   ", style=MUTED)
        tail = Text(self.message or (f"web {self.web_url}" if self.web_url else ""), style=MUTED)
        return Text.assemble(keys, " " * max(1, width - keys.cell_len - tail.cell_len), tail)

    def _error_screen(self, failed, width):
        lines = [Text(""), _label("the engine stopped"), Text(""),
                 Text(failed.get("reason") or "it exited with code " + str(failed["exit"]), style=DOWN), Text(""),
                 Text(f"full log  {failed['log']}", style=MUTED),
                 Text("fix the cause and start the run again; q leaves this screen", style=MUTED)]
        return Group(*lines)

    def render(self, width, height):
        if self.snap is None or time.monotonic() - self._refreshed >= 1.0:
            self.snap = self.state.refresh()
            self._refreshed = time.monotonic()
        failed = self._failure()
        if failed:
            rule = Text("─" * width, style=RULE)
            return Group(self._header(self.snap["run"], width), rule, self._error_screen(failed, width), rule,
                         self._footer(width))
        key = (len(self.snap["programs"]), len(self.snap["transfers"]), len(self.snap["edits"]),
               len(self.snap["findings"]), sum(bool(p.get("alive")) for p in self.snap["programs"]))
        if key != self._figure_key:
            self.figures = all_views(self.snap, self.state.code)
            self._figure_key = key
        left_w, right_w = 36, 46
        center_w = max(30, width - left_w - right_w - 6)
        body = Table.grid(padding=(0, 2), expand=True)
        body.add_column(width=left_w)
        body.add_column(width=center_w)
        body.add_column(width=right_w)
        body.add_row(self._left(self.snap, left_w), self._center(self.snap, center_w, height - 5),
                     self._right(self.snap, right_w))
        rule = Text("─" * width, style=RULE)
        return Group(self._header(self.snap["run"], width), rule, body, rule, self._footer(width))

    def key(self, ch):
        names = dict(VIEWS)
        if ch in names:
            self.view = names[ch]
        elif ch == "q":
            self.quit = True
        elif ch == "w" and self.open_web is not None:
            self.web_url = self.open_web()
            self.message = f"opened {self.web_url}"
        elif ch == "p" and self.pause is not None:
            self.message = "pausing after this step" if self.pause() else "the run is not running here"


@contextmanager
def _cbreak():
    if not sys.stdin.isatty():
        yield False
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def watch(run_dir, *, web_url=None, open_web=None, pause=None, alive=None, engine=None):
    """Follow a run until the viewer quits, or until `alive()` turns false and the final state is drawn."""
    console = Console()
    board = Dashboard(run_dir, web_url, open_web, pause, engine)
    with _cbreak() as keys, Live(console=console, screen=True, auto_refresh=False, transient=False) as live:
        last = 0.0
        while not board.quit:
            if time.monotonic() - last >= 0.25:
                size = console.size
                live.update(board.render(size.width, size.height), refresh=True)
                last = time.monotonic()
                if alive is not None and not alive():
                    break
            if keys and select.select([sys.stdin], [], [], 0.05)[0]:
                board.key(os.read(sys.stdin.fileno(), 1).decode(errors="ignore"))
                last = 0.0
            elif not keys:
                time.sleep(0.2)
    return board.snap
