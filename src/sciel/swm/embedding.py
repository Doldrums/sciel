"""Frozen code embeddings for the world model, cached by source hash.

With an endpoint they come from an OpenAI-compatible embeddings server. Without one they are hashed counts of
the program's tokens and token pairs: no server and no weights, and programs that share code stay close.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request

import numpy as np

DIM = 1024
MAX_CHARS = 16000
BATCH = 32
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9_]")


def hashed(code: str) -> np.ndarray:
    """Signed feature hashing of token unigrams and bigrams, log-scaled."""
    tokens = _TOKEN.findall(code[:MAX_CHARS * 4])
    vector = np.zeros(DIM, dtype=np.float32)
    counts = {}
    for gram in tokens + [a + " " + b for a, b in zip(tokens, tokens[1:])]:
        counts[gram] = counts.get(gram, 0) + 1
    for gram, n in counts.items():
        h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "little")
        vector[h % DIM] += (1.0 if (h >> 63) else -1.0) * (1.0 + math.log(n))
    return vector


class CodeEmbedder:
    def __init__(self, api_base: str | None, model: str, *, mock: bool = False):
        self.api_base, self.model, self.mock = api_base, model, mock
        self._cache: dict[str, np.ndarray] = {}

    def check(self) -> None:
        if self.mock or self.api_base is None:
            return
        try:
            self("def probe():\n    return 0\n")
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"the code-embedding server at {self.api_base} is not reachable ({exc}); start it, "
                               "or set swm.embedding_api_base: null to use local hashed embeddings") from None

    def _remote(self, codes: list) -> list:
        body = json.dumps({"input": [c[:MAX_CHARS] for c in codes], "model": self.model}).encode()
        request = urllib.request.Request(self.api_base.rstrip("/") + "/embeddings", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            rows = sorted(json.loads(response.read())["data"], key=lambda row: row["index"])
        vectors = [np.asarray(row["embedding"], dtype=np.float32) for row in rows]
        if len(vectors) != len(codes) or any(v.shape != (DIM,) for v in vectors):
            raise ValueError(f"code embeddings do not match the {len(codes)} requested inputs of dimension {DIM}")
        return vectors

    @staticmethod
    def _key(code: str) -> str:
        return hashlib.sha256(code.encode()).hexdigest()

    def prefetch(self, codes) -> None:
        """Embed every uncached source in batches; the endpoint's per-call overhead dominates single requests."""
        missing = list(dict.fromkeys(c for c in codes if self._key(c) not in self._cache))
        for start in range(0, len(missing), BATCH):
            chunk = missing[start:start + BATCH]
            if self.mock:
                vectors = [np.random.default_rng(int(self._key(c)[:16], 16)).normal(size=DIM).astype(np.float32)
                           for c in chunk]
            elif self.api_base is None:
                vectors = [hashed(c) for c in chunk]
            else:
                vectors = self._remote(chunk)
            for code, vector in zip(chunk, vectors):
                self._cache[self._key(code)] = vector / (np.linalg.norm(vector) + 1e-8)

    def get(self, code: str) -> np.ndarray | None:
        return self._cache.get(self._key(code))

    def __call__(self, code: str) -> np.ndarray:
        self.prefetch([code])
        return self._cache[self._key(code)]
