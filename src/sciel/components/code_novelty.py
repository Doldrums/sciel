"""Deterministic, code-only lexical novelty before an expensive evaluation.

The descriptor is deliberately independent of scores, evaluator behaviour and
learned embeddings. Cosine proximity is a heuristic for similar source code,
not a claim of equivalent behaviour. Numeric literals and identifiers remain
visible; comments and insignificant formatting do not.
"""

from __future__ import annotations

import hashlib
import io
import math
import tokenize
from collections import deque
from collections.abc import Iterable
from itertools import islice


DESCRIPTOR_VERSION = "code-token-ngram-blake2b-512-v1"
_DIMENSIONS = 512
_IGNORED = {tokenize.COMMENT, tokenize.NL, tokenize.ENDMARKER, tokenize.ENCODING}
_STRUCTURAL = {tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}


def _descriptor(code: str) -> list[float]:
    """Stream token 1/2/3-grams into a signed, fixed-size feature vector."""
    if not isinstance(code, str):
        raise TypeError("code must be a string")
    values = [0.0] * _DIMENSIONS
    recent: deque[bytes] = deque(maxlen=3)
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type in _IGNORED:
                continue
            # The indentation *structure* matters, its number of spaces does not.
            value = "" if token.type in _STRUCTURAL else token.string
            recent.append(f"{tokenize.tok_name[token.type]}:{value}".encode("utf-8"))
            items = list(recent)
            for length in range(1, len(items) + 1):
                # Length framing avoids ambiguity even for strings containing NUL.
                parts = items[-length:]
                payload = b"".join(len(part).to_bytes(8, "big") + part for part in parts)
                digest = hashlib.blake2b(payload, digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % _DIMENSIONS
                sign = 1.0 if digest[4] & 1 else -1.0
                values[bucket] += sign * length
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Syntax validation owns malformed programs. A partial token prefix must
        # not cause this optional gate to reject a different complete program.
        return [0.0] * _DIMENSIONS
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


class CodeNoveltyGate:
    """Compare against current container members, with no retained hidden state.

    ``max_entries`` checks only the first N supplied references; callers choose
    their deterministic reference order. At most two descriptors are retained at
    once, so memory does not grow with the number of references.
    """

    descriptor_version = DESCRIPTOR_VERSION

    def __init__(self, threshold: float = 0.98, max_entries: int | None = None):
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("novelty threshold must be finite and in (0, 1]")
        if max_entries is not None and (
            isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1
        ):
            raise ValueError("max_entries must be a positive integer or None")
        self.threshold = threshold
        self.max_entries = max_entries

    def inspect(self, code: str, reference_codes: Iterable[str]) -> dict:
        candidate = _descriptor(code)
        result = {
            "accept": True,
            "similarity": 0.0,
            "matched_index": None,
            "descriptor_version": self.descriptor_version,
        }
        if not any(candidate):
            return result
        references = (
            reference_codes if self.max_entries is None
            else islice(reference_codes, self.max_entries)
        )
        best = -1.0
        for index, code in enumerate(references):
            reference = _descriptor(code)
            if not any(reference):
                continue
            # Equality is exact on deterministic vectors, including threshold=1.
            similarity = 1.0 if candidate == reference else max(
                -1.0, min(1.0, sum(a * b for a, b in zip(candidate, reference)))
            )
            if result["matched_index"] is None or similarity > best:
                best = similarity
                result["similarity"] = similarity
                result["matched_index"] = index
        result["accept"] = result["matched_index"] is None or best < self.threshold
        return result
