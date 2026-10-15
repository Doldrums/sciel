"""Draw a view figure as SVG for the web page or as braille cells for the terminal."""

from __future__ import annotations

import math
from html import escape

BG = "#000000"
PLOT = "#0B0B0C"
AXIS = "#2C2C2E"
LABEL = "#98989D"


def _fmt(v):
    return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"


def svg(fig, width=520, height=520) -> str:
    plot = fig.get("kind") != "disk"
    left, right, top, bottom = (58, 10, 8, 34) if plot else (22, 22, 22, 22)
    w, h = width - left - right, height - top - bottom
    X = lambda x: left + x * w
    Y = lambda y: top + (1 - y) * h
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{escape(fig.get("caption", ""))}">',
           '<defs><marker id="ah" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" '
           f'orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="{LABEL}"/></marker></defs>']
    if plot:
        out.append(f'<rect x="{left}" y="{top}" width="{w}" height="{h}" fill="{PLOT}" stroke="{AXIS}"/>')
    for item in fig["items"]:
        t = item["t"]
        if t == "ring":
            out.append(f'<circle cx="{X(item["x"]):.1f}" cy="{Y(item["y"]):.1f}" r="{item["r"] * w:.1f}" '
                       f'fill="#0B0B0C" stroke="{item["c"]}"/>')
        elif t == "cell":
            out.append(f'<rect x="{X(item["x"]):.1f}" y="{Y(item["y"] + item["h"]):.1f}" width="{item["w"] * w + 0.5:.1f}" '
                       f'height="{item["h"] * h + 0.5:.1f}" fill="{_blend(item["c"], item["o"], PLOT)}" shape-rendering="crispEdges"/>')
        elif t == "blob":
            out.append(f'<circle cx="{X(item["x"]):.1f}" cy="{Y(item["y"]):.1f}" r="{item["r"] * w:.1f}" '
                       f'fill="{item["c"]}" fill-opacity="{item["o"]}"/>')
        elif t == "band":
            b = item["w"]
            pts = [(0.04, 0.04 + b), (0.96, 0.96 + b), (0.96, 0.96 - b), (0.04, 0.04 - b)]
            out.append('<polygon points="' + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
                       + f'" fill="{item["c"]}" fill-opacity="{item["o"]}"/>')
        elif t == "line":
            pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in item["pts"])
            dash = ' stroke-dasharray="5 4"' if item.get("dash") else ""
            arrow = ' marker-end="url(#ah)"' if item.get("arrow") else ""
            out.append(f'<polyline points="{pts}" fill="none" stroke="{item["c"]}" stroke-opacity="{item.get("o", 1)}" '
                       f'stroke-width="{item.get("w", 1)}"{dash}{arrow}/>')
        elif t == "dot":
            r = max(2.0, item["r"] * w)
            paint = (f'fill="{item["c"]}" stroke="#000" stroke-width="0.8"' if item.get("fill")
                     else f'fill="#000" stroke="{item["c"]}" stroke-width="1.1"')
            tip = f'<title>{escape(item["id"])}</title>' if item.get("id") else ""
            out.append(f'<circle cx="{X(item["x"]):.1f}" cy="{Y(item["y"]):.1f}" r="{r:.1f}" {paint}>{tip}</circle>')
        elif t == "text":
            out.append(f'<text x="{X(item["x"]):.1f}" y="{Y(item["y"]) + 4:.1f}" fill="{item["c"]}" font-size="11" '
                       f'text-anchor="{item.get("anchor", "start")}">{escape(item["s"])}</text>')
    if plot and fig.get("axes") and fig.get("ranges"):
        (x0, x1), (y0, y1) = fig["ranges"]
        ax, ay = fig["axes"]
        out.append(f'<g fill="{LABEL}" font-size="10">'
                   f'<text x="{left}" y="{top + h + 14}">{_fmt(x0)}</text>'
                   f'<text x="{left + w}" y="{top + h + 14}" text-anchor="end">{_fmt(x1)}</text>'
                   f'<text x="{left + w / 2}" y="{top + h + 28}" text-anchor="middle">{escape(ax)}</text>'
                   f'<text x="{left - 6}" y="{top + h}" text-anchor="end">{_fmt(y0)}</text>'
                   f'<text x="{left - 6}" y="{top + 9}" text-anchor="end">{_fmt(y1)}</text>'
                   f'<text x="{10}" y="{top + h / 2}" text-anchor="middle" transform="rotate(-90 10 {top + h / 2})">'
                   f'{escape(ay)}</text></g>')
    out.append("</svg>")
    return "".join(out)


_DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def _blend(color, alpha, ground="#000000"):
    a = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    g = [int(ground[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(g[k] + (a[k] - g[k]) * alpha):02x}" for k in range(3))


class Braille:
    """A character grid where each cell holds 2x4 dots, one foreground and one background colour."""

    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.bits = [[0] * cols for _ in range(rows)]
        self.fg = [[None] * cols for _ in range(rows)]
        self.bg = [[None] * cols for _ in range(rows)]
        self.text = {}

    def dot(self, x, y, color):
        px, py = round(x * (2 * self.cols - 1)), round((1 - y) * (4 * self.rows - 1))
        if 0 <= px < 2 * self.cols and 0 <= py < 4 * self.rows:
            r, c = py // 4, px // 2
            self.bits[r][c] |= _DOTS[py % 4][px % 2]
            self.fg[r][c] = color

    def line(self, a, b, color, dashed=False):
        steps = max(2, int(max(abs(b[0] - a[0]) * 2 * self.cols, abs(b[1] - a[1]) * 4 * self.rows)) + 1)
        for i in range(steps + 1):
            if dashed and (i // 3) % 2:
                continue
            t = i / steps
            self.dot(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, color)

    def disc(self, x, y, color, radius_cells=0.6):
        rx, ry = radius_cells / self.cols, radius_cells / self.rows * 0.5
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i * i + j * j <= 5:
                    self.dot(x + i * rx / 2, y + j * ry / 2, color)

    def shade(self, x, y, w, h, color, alpha):
        for r in range(int((1 - y - h) * self.rows), int((1 - y) * self.rows) + 1):
            for c in range(int(x * self.cols), int((x + w) * self.cols) + 1):
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    self.bg[r][c] = _blend(color, min(1.0, alpha))

    def label(self, x, y, s, color):
        r, c = round((1 - y) * (self.rows - 1)), round(x * (self.cols - 1))
        for i, ch in enumerate(s):
            if 0 <= r < self.rows and 0 <= c + i < self.cols:
                self.text[(r, c + i)] = (ch, color)

    def cells(self):
        """Rows of (character, foreground, background)."""
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.text:
                    ch, fg = self.text[(r, c)]
                else:
                    ch, fg = chr(0x2800 + self.bits[r][c]), self.fg[r][c]
                row.append((ch, fg, self.bg[r][c]))
            yield row


def braille(fig, cols, rows) -> Braille:
    canvas = Braille(cols, rows)
    for item in fig["items"]:
        t = item["t"]
        if t == "cell":
            canvas.shade(item["x"], item["y"], item["w"], item["h"], item["c"], item["o"] * 0.8)
        elif t == "blob":
            canvas.shade(item["x"] - item["r"] / 2, item["y"] - item["r"] / 2, item["r"], item["r"], item["c"],
                         item["o"] * 1.6)
        elif t == "ring":
            for k in range(240):
                a = 2 * math.pi * k / 240
                canvas.dot(item["x"] + item["r"] * math.cos(a), item["y"] + item["r"] * math.sin(a), item["c"])
        elif t == "line":
            color = item["c"] if item.get("o", 1) > 0.6 else _blend(item["c"], 0.75)
            pts = item["pts"]
            for a, b in zip(pts, pts[1:]):
                canvas.line(a, b, color, item.get("dash", False))
        elif t == "dot":
            canvas.disc(item["x"], item["y"], item["c"] if item.get("fill") else _blend(item["c"], 0.7),
                        0.9 if item.get("fill") else 0.6)
        elif t == "text":
            canvas.label(item["x"], item["y"], item["s"], item["c"])
    return canvas
