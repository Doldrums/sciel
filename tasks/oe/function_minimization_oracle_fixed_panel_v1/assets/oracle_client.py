"""Public client. Its local counters and return values are never trusted."""

import json
import os


class OracleClient:
    def __init__(self):
        hello = self._read()
        if hello.get("protocol") != "sciel-hidden-oracle-fixed-panel/1":
            raise RuntimeError("oracle protocol mismatch")
        if (
            hello.get("bootstrap") != "sciel-oracle-public-seed/1"
            or type(hello.get("algorithm_seed")) is not int
            or not 0 <= hello["algorithm_seed"] < 2**32
        ):
            raise RuntimeError("oracle public seed bootstrap mismatch")
        self.algorithm_seed = hello["algorithm_seed"]
        self.bounds = tuple(hello["bounds"])
        self.query_budget = hello["query_budget"]

    def initialized(self):
        self._write(
            {
                "op": "initialized",
                "bootstrap": "sciel-oracle-public-seed/1",
                "algorithm_seed": self.algorithm_seed,
            }
        )
        if self._read() != {"status": "initialized"}:
            raise RuntimeError("oracle bootstrap rejected")

    @staticmethod
    def _read():
        data = bytearray()
        while not data.endswith(b"\n"):
            part = os.read(0, 1)
            if not part or len(data) >= 4096:
                raise RuntimeError("oracle connection closed")
            data.extend(part)
        return json.loads(data)

    @staticmethod
    def _write(payload):
        data = (
            json.dumps(payload, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        )
        while data:
            data = data[os.write(1, data) :]

    def query(self, x, y):
        self._write({"op": "query", "point": [float(x), float(y)]})
        result = self._read()
        if result.get("status") != "ok":
            raise RuntimeError("oracle request rejected")
        return result["value"]

    def finish(self):
        self._write({"op": "finish"})
        result = self._read()
        if result.get("status") != "finished":
            raise RuntimeError("oracle finish rejected")
