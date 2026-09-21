"""Process-bound telemetry checks; no experiment, model or evaluator imports."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from sciel.engine.runtime import RuntimeStatus, observed_stage, process_identity, process_matches


class RuntimeStatusTests(unittest.TestCase):
    def test_heartbeat_is_atomic_process_bound_and_stops_at_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = dict(run_id="run", segment_id="segment", iteration_no=0, B_evals=0)
            writer = RuntimeStatus(directory, lambda: metadata, interval=.02)
            path = Path(directory) / "runtime.json"
            writer.start()
            try:
                initial = json.loads(path.read_text())
                self.assertEqual(initial["schema"], "sciel-runtime/1")
                self.assertEqual(initial["status"], "running")
                self.assertEqual(initial["pid"], os.getpid())
                self.assertTrue(process_matches(initial))
                deadline = time.monotonic() + 2
                observed = set()
                while time.monotonic() < deadline:
                    # Concurrent reads must never see the writer's partial JSON.
                    current = json.loads(path.read_text())
                    observed.add(current["heartbeat_at"])
                    if len(observed) > 2:
                        break
                    threading.Event().wait(.005)
                self.assertGreater(len(observed), 2)
                metadata.update(iteration_no=2, B_evals=1)
                with writer.phase("generating"):
                    self.assertEqual(json.loads(path.read_text())["stage"], "generating")
                    with writer.phase("evaluating"):
                        self.assertEqual(json.loads(path.read_text())["stage"], "evaluating")
                    self.assertEqual(json.loads(path.read_text())["stage"], "generating")
                writer.finish("completed", reason="budget")
                final = json.loads(path.read_text())
                self.assertEqual(final["status"], "completed")
                self.assertEqual(final["stage"], "finished")
                self.assertEqual(final["iteration_no"], 2)
                self.assertEqual(final["reason"], "budget")
                self.assertIsNotNone(final["finished_at"])
                content = path.read_bytes()
                threading.Event().wait(.05)
                self.assertEqual(content, path.read_bytes())
                self.assertFalse(list(Path(directory).glob(".runtime-*.tmp")))
            finally:
                if writer._thread.is_alive():
                    writer.finish("interrupted")

    def test_dead_reused_or_remote_process_is_never_confirmed(self):
        record = dict(hostname=socket.gethostname(), pid=os.getpid(),
                      process_identity=process_identity(os.getpid()))
        self.assertTrue(process_matches(record))
        self.assertIsNone(process_matches(dict(record, hostname="unrelated-host")))
        self.assertFalse(process_matches(dict(record, process_identity=dict(kind="wrong", value="wrong"))))
        self.assertIsNone(process_matches(dict(record, process_identity=None)))
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        process.wait(timeout=5)
        self.assertFalse(process_matches(dict(record, pid=process.pid)))

    def test_failed_publication_does_not_stop_work_and_phase_restores_after_error(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = RuntimeStatus(directory, lambda: {}, interval=60)
            writer.start()
            try:
                with self.assertRaisesRegex(ValueError, "fixture"):
                    with writer.phase("evaluating"):
                        raise ValueError("fixture")
                self.assertEqual(json.loads(writer.path.read_text())["stage"], "starting")
                with patch("sciel.engine.runtime.os.replace", side_effect=OSError("fixture")):
                    writer._publish()
                self.assertFalse(list(Path(directory).glob(".runtime-*.tmp")))
                writer.finish("failed", error_type="ValueError")
                final = json.loads(writer.path.read_text())
                self.assertEqual(final["status"], "failed")
                self.assertEqual(final["error_type"], "ValueError")
            finally:
                if writer._thread.is_alive():
                    writer.finish("interrupted")

    def test_stage_decorator_preserves_result_exception_and_unobserved_methods(self):
        class Subject:
            @observed_stage("evaluating")
            def evaluate(self, value):
                return value + 1
        self.assertEqual(Subject().evaluate(4), 5)
        with tempfile.TemporaryDirectory() as directory:
            subject = Subject()
            subject._runtime_status = RuntimeStatus(directory, lambda: {}, interval=60)
            subject._runtime_status.start()
            try:
                self.assertEqual(subject.evaluate(4), 5)
            finally:
                subject._runtime_status.finish("completed")

    def test_cli_import_and_help_need_no_engine_dependencies(self):
        repo = Path(__file__).resolve().parents[2]
        code = """
import importlib.abc, sys
class Forbid(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','pydantic','yaml','openai','torch'}:
            raise RuntimeError('heavy dependency imported: ' + fullname)
sys.meta_path.insert(0, Forbid())
from sciel.cli.main import main
main(['--help'])
"""
        result = subprocess.run([sys.executable, "-B", "-c", code],
            cwd=repo, env=dict(os.environ, PYTHONPATH=str(repo / "src")),
            capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage: sciel", result.stdout)
        self.assertNotIn("dashboard", result.stdout)


if __name__ == "__main__":
    unittest.main()
