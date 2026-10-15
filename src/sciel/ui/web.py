"""A local web page for one run: a small HTTP server on 127.0.0.1 that serves the page and its live state."""

from __future__ import annotations

import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .render import svg
from .state import RunState
from .views import all_views

PORT = 7777
REFRESH_SECONDS = 1.5
SIZES = {"culture": (560, 560), "landscape": (460, 320), "code": (460, 320), "edits": (460, 320)}


def request_pause(run_dir) -> bool:
    """Ask a live local run to pause after its current iteration, as `sciel stop` does."""
    from ..engine.runtime import process_matches
    try:
        runtime = json.loads((Path(run_dir) / "runtime.json").read_text())
    except (OSError, ValueError):
        return False
    if runtime.get("status") != "running" or process_matches(runtime) is not True:
        return False
    with open(Path(run_dir) / "STOP", "a"):
        pass
    return True


class Live:
    """The run's state, refreshed at most every REFRESH_SECONDS however many viewers ask."""

    def __init__(self, run_dir, engine=None):
        self.state = RunState(run_dir)
        self.engine = engine
        self._views = (None, None)
        self.lock = threading.Lock()
        self.at = 0.0
        self.body = b"{}"

    def program(self, pid: str) -> dict:
        """A program's code, its parent's code and the diff between them."""
        with self.lock:
            state = self.state
            program = state.programs.get(pid) or {}
            return {"id": pid, "parent": program.get("parent"), "code": state.source.get(pid),
                    "parent_code": state.source.get(program.get("parent")), "diff": state.diff(pid, context=3)}

    def payload(self) -> bytes:
        with self.lock:
            if time.monotonic() - self.at >= REFRESH_SECONDS:
                snap = self.state.refresh()
                key = (self.state._offset, self.state._checkpoint_mtime)
                if self._views[0] != key:
                    views = all_views(snap, self.state.code)
                    self._views = (key, {name: {"svg": svg(fig, *SIZES[name]), "caption": fig.get("caption"),
                                                "stats": fig.get("stats")} for name, fig in views.items()})
                snap["views"] = self._views[1]
                snap["programs"] = [{k: v for k, v in p.items() if k != "cases"} for p in snap["programs"]]
                snap["claims"] = [c for c in snap["claims"] if c.get("rejected") and (c.get("low", 0) > 0 or c.get("high", 0) < 0)]
                snap["edits"] = []
                snap["engine"] = self.engine() if self.engine is not None else None
                self.body = json.dumps(snap, default=str).encode()
                self.at = time.monotonic()
            return self.body


def _handler(live, run_dir):
    page = files("sciel.ui").joinpath("static/index.html").read_bytes()
    script = files("sciel.ui").joinpath("static/engine.js").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _local(self):
            """Refuse requests a foreign page could forge: a rebound Host, or a cross-site Origin on a write."""
            port = self.server.server_address[1]
            allowed = {f"localhost:{port}", f"127.0.0.1:{port}"}
            if self.headers.get("Host") not in allowed:
                return False
            origin = self.headers.get("Origin")
            return self.command == "GET" or origin in {f"http://{host}" for host in allowed}

        def do_GET(self):
            if not self._local():
                return self._send(403, b"forbidden", "text/plain")
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/engine.js":
                self._send(200, script, "text/javascript; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, live.payload(), "application/json")
            elif self.path.startswith("/api/program"):
                pid = parse_qs(urlsplit(self.path).query).get("id", [""])[0]
                self._send(200, json.dumps(live.program(pid)).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if not self._local():
                return self._send(403, b"forbidden", "text/plain")
            if self.path == "/api/pause":
                ok = request_pause(run_dir)
                self._send(200 if ok else 409, json.dumps({"paused": ok}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, *args):
            pass

    return Handler


def serve(run_dir, port=PORT, open_browser=True, engine=None):
    """Start the page in a background thread; return its URL and the server."""
    live = Live(run_dir, engine)
    handler = _handler(live, run_dir)
    for candidate in range(port, port + 50):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), handler)
            break
        except OSError:
            continue
    else:
        raise OSError(f"no free port in {port}..{port + 49}")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="sciel-web").start()
    url = f"http://localhost:{server.server_address[1]}"
    if open_browser:
        webbrowser.open(url)
    return url, server
