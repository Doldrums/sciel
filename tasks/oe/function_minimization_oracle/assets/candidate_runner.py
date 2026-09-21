"""Untrusted-side adapter: contains no objective parameters or scoring code."""

import importlib.util
import sys

from oracle_client import OracleClient


def main():
    oracle = OracleClient()
    # Ordinary candidate print output goes to stderr; direct fd writes remain
    # untrusted protocol input and are validated by the host.
    sys.stdout = sys.stderr
    sys.path.insert(0, "/input")
    spec = importlib.util.spec_from_file_location("candidate", "/input/candidate.py")
    program = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = program
    spec.loader.exec_module(program)
    if not hasattr(program, "run_search"):
        raise RuntimeError("missing run_search")
    program.run_search(oracle)
    oracle.finish()


if __name__ == "__main__":
    main()
