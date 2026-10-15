"""The terminal dashboard: what each step tried, what came of it, and what the run is doing now.

Left: score with a labelled chart, islands, failures. Centre: the attempts table and the selected attempt in full
(the model's idea, the error, the diff against its parent), or the best program. Right: work in flight, findings,
research cells, events.
"""

from __future__ import annotations

import os
import select
import sys
import textwrap
import time
from contextlib import contextmanager

from rich.console import Console, Group
from rich.live import Live
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .render import Braille
from .state import RunState, _seconds
from .views import island_color

INK, MUTED, DIM, RULE = "#E8E8EA", "#8A8A8E", "#5A5A5E", "#2C2C2E"
AMBER, UP, DOWN = "#E8A33D", "#3DAE7C", "#E5484D"
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
STAGES = {"seeding": "evaluating the seed", "evaluating": "evaluating", "generating": "writing programs",
          "checkpointing": "saving", "provider_backoff": "waiting for the provider", "finished": "finished"}
STATUS = {"culture": (UP, "holds"), "tested": ("#0A84FF", "tested"), "trial": (AMBER, "trial"),
          "retired": (MUTED, "dropped")}
OPS = {"diff": "edit", "full_rewrite": "rewrite", "cross": "crossover", "idea": "new idea", "error_repair": "repair"}
LEFT, RIGHT = 38, 46


def _f(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _clock(seconds):
    if seconds is None:
        return ""
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds < 3600 else f"{seconds // 3600}h{seconds % 3600 // 60:02d}"


def _label(text):
    return Text(text.upper(), style=MUTED)


def _canvas_text(canvas: Braille) -> list[Text]:
    rows = []
    for row in canvas.cells():
        line = Text()
        for ch, fg, bg in row:
            line.append(ch, Style(color=fg or DIM, bgcolor=bg))
        rows.append(line)
    return rows


def _result(attempt):
    """The outcome column: the score change, or why there is no score."""
    status = attempt["status"]
    if status == "failed":
        return Text("✗ failed", style=DOWN)
    if attempt["delta"] is None:
        return Text(f"  {_f(attempt['score'])}", style=MUTED)
    if status == "better":
        return Text(f"▲ {_f(attempt['score'])} +{attempt['delta']:.3g}", style=UP)
    if status == "worse":
        return Text(f"▼ {_f(attempt['score'])} {attempt['delta']:.3g}", style=DOWN)
    return Text(f"= {_f(attempt['score'])}", style=MUTED)


class Dashboard:
    def __init__(self, run_dir, web_url=None, open_web=None, pause=None, engine=None):
        self.state = RunState(run_dir)
        self.web_url, self.open_web, self.pause, self.engine = web_url, open_web, pause, engine
        self.message = None
        self.quit = False
        self.snap = None
        self.selected = 0
        self.show_best = False
        self._refreshed = 0.0

    def _failure(self):
        info = self.engine() if self.engine is not None else None
        return info if info and info.get("exit") not in (None, 0) else None

    def _header(self, run, width):
        live = run.get("status") == "running"
        stage = STAGES.get(run.get("stage"), run.get("stage"))
        status = (Text("● live" + (f" · {stage}" if stage else ""), style=UP) if live
                  else Text(str(run.get("status") or "unknown"), style=AMBER))
        if self._failure():
            status = Text("● engine stopped with an error", style=DOWN)
        left = Text.assemble(("SCIEL", Style(color=AMBER, bold=True)), "  ", (str(run.get("task") or ""), INK),
                             ("  run " + str(run.get("run_id") or "")[:8], MUTED))
        budget = f"/{run['budget']}" if run.get("budget") else ""
        eta = run.get("eta_seconds")
        right = Text.assemble(("evals ", MUTED), f"{run.get('B_evals', 0)}{budget}", ("   spent ", MUTED),
                              f"${run.get('cost') or 0:.2f}",
                              *((("   left ", MUTED), _clock(eta)) if eta and live else ()), "   ", status)
        return Text.assemble(left, " " * max(1, width - left.cell_len - right.cell_len), right)

    def _chart(self, snap, width):
        """Best score so far on each island, against attempts, with its scale and legend."""
        attempts = [a for a in snap["attempts"] if a["score"] is not None]
        seed = snap["run"].get("seed_score")
        islands = sorted({a["island"] for a in attempts if a["island"] is not None})
        if not attempts:
            return [Text("no scores yet", style=DIM)]
        best, series = {}, {i: [] for i in islands}
        for n, a in enumerate(attempts, 1):
            if a["score"] > best.get(a["island"], float("-inf")):
                best[a["island"]] = a["score"]
            for i in islands:
                if i in best:
                    series[i].append((n, best[i]))
        values = [v for s in series.values() for _, v in s] + ([seed] if seed is not None else [])
        low, high = min(values), max(values)
        span = (high - low) or abs(high) or 1.0
        axis = 7
        cols, rows = width - axis - 1, 6
        canvas = Braille(cols, rows)
        y = lambda v: 0.06 + 0.88 * (v - low) / span
        total = len(attempts)
        if seed is not None:
            canvas.line([0, y(seed)], [1, y(seed)], DIM, dashed=True)
        for i in islands:
            points = series[i]
            for (n1, v1), (n2, v2) in zip(points, points[1:]):
                canvas.line([n1 / total, y(v1)], [n2 / total, y(v1)], island_color(i))
                canvas.line([n2 / total, y(v1)], [n2 / total, y(v2)], island_color(i))
            if points:
                canvas.dot(points[-1][0] / total, y(points[-1][1]), island_color(i))
        out = []
        for r, line in enumerate(_canvas_text(canvas)):
            tick = (f"{high:.4g}" if r == 0 else f"{low:.4g}" if r == rows - 1 and high > low else "")
            out.append(Text.assemble((f"{tick:>{axis}}", DIM), ("┤" if tick else "│", DIM), line))
        out.append(Text.assemble(" " * axis, ("└" + "─" * cols, DIM)))
        middle = cols - len(str(total)) - 1
        out.append(Text.assemble(" " * (axis + 1), ("1", DIM), (f"{'attempts':^{middle}}", DIM), (str(total), DIM)))
        legend = Text(" " * (axis + 1))
        for i in islands:
            legend.append("━", style=island_color(i))
            legend.append(f" I{i}  ", style=MUTED)
        if seed is not None:
            legend.append("┄ seed", style=DIM)
        out.append(legend)
        return out

    def _left(self, snap):
        run = snap["run"]
        best, seed = run.get("best"), run.get("seed_score")
        gain = None if best is None or seed is None else best - seed
        parts = [_label("best score"),
                 Text.assemble((_f(best, 4), Style(color=INK, bold=True)), "   ",
                               ((f"{gain:+.4g} over seed {_f(seed, 4)}", UP if gain and gain > 0 else MUTED)
                                if gain is not None else ("", MUTED))), Text(""),
                 _label("best per island over attempts")]
        parts += self._chart(snap, LEFT) + [Text("")]
        attempts = snap["attempts"]
        table = Table.grid(padding=(0, 1))
        for justify in ("left", "right", "right", "right", "right"):
            table.add_column(justify=justify, no_wrap=True)
        table.add_row(*(Text(h, style=DIM) for h in ("ISL", "BEST", "TRIES", "BETTER", "FAILED")))
        for island in snap["islands"]:
            own = [a for a in attempts if a["island"] == island["id"]]
            table.add_row(Text(f"I{island['id']}", style=island_color(island["id"])), _f(island.get("best"), 4),
                          str(len(own)), Text(str(sum(a["status"] == "better" for a in own)), style=UP),
                          Text(str(sum(a["status"] == "failed" for a in own)), style=DOWN))
        parts += [_label("islands"), table, Text(""), _label("why candidates fail")]
        if not snap["failures"]:
            parts.append(Text("no failures", style=DIM))
        for message, count in snap["failures"][:4]:
            for i, line in enumerate(textwrap.wrap(message, LEFT - 5)[:2]):
                parts.append(Text.assemble((f"{count:>3}× " if i == 0 else "     ", DOWN), (line, INK)))
        return Group(*parts)

    def _attempts(self, snap, width, rows):
        attempts = list(reversed(snap["attempts"]))
        self.selected = min(self.selected, max(0, len(attempts) - 1))
        start = max(0, self.selected - rows + 1)
        table = Table.grid(padding=(0, 1))
        for name in ("n", "isl", "step", "op", "result", "cost", "idea"):
            table.add_column(no_wrap=True, overflow="ellipsis", max_width=max(12, width - 66) if name == "idea" else None)
        table.add_row(*(Text(h, style=DIM) for h in ("#", "ISL", "CHILD ← PARENT", "OP", "RESULT", "COST", "IDEA")))
        for i, a in enumerate(attempts[start:start + rows], start):
            step = f"{a['child'] or '—'} ← {a['parent'] or 'seed'}"
            note = a.get("error") or a.get("idea") or a.get("reply") or ""
            table.add_row(Text(("› " if i == self.selected else "  ") + str(a["n"]), style=AMBER if i == self.selected else DIM),
                          Text(f"I{a['island']}", style=island_color(a["island"])), Text(step, style=INK),
                          Text(OPS.get(a["op"], a["op"] or ""), style=MUTED), _result(a),
                          Text(f"${a['cost']:.3f}" if a.get("cost") else "", style=DIM),
                          Text(note, style=DOWN if a.get("error") else MUTED),
                          style=Style(bgcolor="#1C1C1E") if i == self.selected else None)
        return table, attempts

    def _detail(self, attempt, width, height):
        if attempt is None:
            return [Text("no attempts yet", style=DIM)]
        title = Text.assemble((f"#{attempt['n']}  ", DIM), (f"I{attempt['island']}  ", island_color(attempt["island"])),
                              (f"{attempt['child'] or 'no program'} ← {attempt['parent'] or 'seed'}  ", INK),
                              (f"{OPS.get(attempt['op'], attempt['op'] or '')}  ", MUTED), _result(attempt),
                              (f"   took {_clock(attempt['seconds'])}" if attempt.get("seconds") else "", DIM))
        lines = [_label("selected attempt"), title]
        idea = attempt.get("idea") or attempt.get("reply")
        if idea:
            lines += [Text(line, style=INK) for line in textwrap.wrap("idea: " + idea, width)[:3]]
        if attempt.get("error"):
            lines += [Text(line, style=DOWN) for line in textwrap.wrap("error: " + attempt["error"], width)[:3]]
        diff = self.state.diff(attempt["child"]) if attempt.get("child") else []
        room = max(0, height - len(lines) - 2)
        if diff:
            added = sum(l.startswith("+") and not l.startswith("+++") for l in diff)
            removed = sum(l.startswith("-") and not l.startswith("---") for l in diff)
            lines += [Text(""), Text.assemble(("DIFF AGAINST THE PARENT", MUTED), (f"   +{added} −{removed} lines", DIM))]
            for line in diff[2:2 + room]:
                color = (UP if line.startswith("+") else DOWN if line.startswith("-")
                         else AMBER if line.startswith("@@") else MUTED)
                lines.append(Text(line[:width], style=color, no_wrap=True))
            if len(diff) - 2 > room:
                lines.append(Text(f"… {len(diff) - 2 - room} more lines; the web page shows the whole diff", style=DIM))
        elif attempt.get("child"):
            lines.append(Text("same code as the parent", style=DIM))
        return lines

    def _center(self, snap, width, height):
        table_rows = max(4, min(10, height // 3))
        table, attempts = self._attempts(snap, width, table_rows)
        parts = [_label("attempts, newest first"), table, Text("")]
        room = height - table_rows - 5
        if self.show_best:
            best = snap.get("best") or {}
            code = self.state.source.get(best.get("id"), "")
            parts.append(Text.assemble(("BEST PROGRAM  ", MUTED), (f"{best.get('id', '—')} · {_f(best.get('score'), 4)}", INK),
                                       ("   b returns to the attempt", DIM)))
            parts.append(Syntax("\n".join(code.splitlines()[:room]), "python", theme="ansi_dark",
                                background_color="default", line_numbers=True))
        else:
            parts += self._detail(attempts[self.selected] if attempts else None, width, room)
        return Group(*parts)

    def _now(self, snap):
        act = snap.get("activity") or {}
        frame = SPIN[int(time.time() * 8) % len(SPIN)]
        if snap["run"].get("status") not in ("running", "starting"):
            return [_label("now"), Text(f"the run is {snap['run'].get('status') or 'not running'}", style=AMBER)]
        last = _seconds(act.get("last") or (snap["events"][-1]["t"] if snap["events"] else None))
        rows = [Text.assemble(("NOW", MUTED), (f"   wave {act['wave']}" if act.get("wave") else "", DIM),
                              (f"   last activity {_clock(time.time() - last)} ago" if last else "", DIM))]
        for job in act.get("jobs", []):
            busy = job["doing"] not in ("done", "queued")
            started = _seconds(job["since"])
            clock = _clock(time.time() - started if busy and started else job["seconds"])
            who = f"I{job['island']}" if job.get("island") is not None else ""
            what = f"{job['what']} of {job['parent']}" if job.get("parent") else str(job["what"])
            rows.append(Text.assemble((f"{who:<4}", island_color(job.get("island")) if who else MUTED),
                                      (f"{what[:20]:<21}", INK),
                                      (f"{frame} " if busy else ("✓ " if job["doing"] == "done" else "· "),
                                       AMBER if busy else (UP if job["doing"] == "done" else DIM)),
                                      (f"{job['doing'][:13]:<14}", INK if busy else MUTED), (clock, MUTED)))
        if len(rows) == 1:
            rows.append(Text(f"{frame} {STAGES.get(snap['run'].get('stage')) or 'working'}", style=AMBER))
        return rows

    def _right(self, snap):
        parts = self._now(snap) + [Text(""), _label("what the search learned")]
        rank = {"culture": 0, "tested": 1, "trial": 2, "retired": 3}
        findings = sorted(snap["findings"], key=lambda f: rank.get(f.get("status"), 2))[:4]
        if not findings:
            parts.append(Text("nothing yet", style=DIM))
        for f in findings:
            color, word = STATUS.get(f.get("status"), (MUTED, f.get("status") or ""))
            for i, line in enumerate(textwrap.wrap(str(f.get("text") or ""), RIGHT - 9)[:2]):
                parts.append(Text.assemble((f"{word if i == 0 else '':<9}", color), (line, INK)))
        parts += [Text(""), _label("research cells")]
        cells = snap["cells"][-3:]
        if not cells:
            parts.append(Text("none open", style=DIM))
        for cell in reversed(cells):
            parts.append(Text.assemble((f"{cell['id']}  ", MUTED), (str(cell.get("question") or "")[:RIGHT - 8], INK)))
            parts.append(Text(f"        {cell.get('state')}", style=MUTED))
        parts += [Text(""), _label("events")]
        for e in reversed(snap["events"][-6:]):
            parts.append(Text.assemble((str(e.get("t") or "")[11:19], DIM), "  ", (e["text"][:RIGHT - 10], MUTED)))
        return Group(*parts)

    def _footer(self, width):
        keys = Text()
        for key, label in (("↑↓", "choose attempt"), ("b", "best program"), ("w", "web"), ("p", "pause run"),
                           ("q", "quit viewer")):
            keys.append(key, style=AMBER)
            keys.append(f" {label}   ", style=MUTED)
        tail = Text(self.message or (f"web {self.web_url}" if self.web_url else ""), style=MUTED)
        return Text.assemble(keys, " " * max(1, width - keys.cell_len - tail.cell_len), tail)

    def _starting(self, snap):
        stage = STAGES.get(snap["run"].get("stage"))
        text = ("starting the engine: loading the task and the model" if snap["run"].get("status") == "starting"
                else f"{stage or 'working'}: attempts appear here once the first programs are scored")
        return Group(_label("starting"), Text(text, style=INK),
                     Text("the seed is evaluated first; with a slow evaluator this takes a minute", style=MUTED))

    def render(self, width, height):
        if self.snap is None or time.monotonic() - self._refreshed >= 1.0:
            self.snap = self.state.refresh()
            self._refreshed = time.monotonic()
        snap = self.snap
        rule = Text("─" * width, style=RULE)
        failed = self._failure()
        if failed:
            body = Group(Text(""), _label("the engine stopped"), Text(""),
                         Text(failed.get("reason") or f"it exited with code {failed['exit']}", style=DOWN), Text(""),
                         Text(f"full log  {failed['log']}", style=MUTED),
                         Text("fix the cause and start the run again; q leaves this screen", style=MUTED))
            return Group(self._header(snap["run"], width), rule, body, rule, self._footer(width))
        center_w = max(40, width - LEFT - RIGHT - 6)
        center = self._center(snap, center_w, height - 5) if snap["attempts"] else self._starting(snap)
        body = Table.grid(padding=(0, 2), expand=True)
        body.add_column(width=LEFT)
        body.add_column(width=center_w)
        body.add_column(width=RIGHT)
        body.add_row(self._left(snap), center, self._right(snap))
        return Group(self._header(snap["run"], width), rule, body, rule, self._footer(width))

    def key(self, ch):
        if ch in ("\x1b[A", "k"):
            self.selected = max(0, self.selected - 1)
            self.show_best = False
        elif ch in ("\x1b[B", "j"):
            self.selected += 1
            self.show_best = False
        elif ch == "b":
            self.show_best = not self.show_best
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
                data = os.read(sys.stdin.fileno(), 8).decode(errors="ignore")
                board.key(data if data.startswith("\x1b") else data[:1])
                last = 0.0
            elif not keys:
                time.sleep(0.2)
    return board.snap
