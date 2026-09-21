"""A lexical card-space proxy; never evidence of semantic equivalence.

Each of the three structured fields contributes an equally weighted unit
bag-of-tokens vector. Token order outside quoted literals is intentionally
discarded. Missing/legacy cards are not invented orthogonal strategies.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re


CONTRACT = "sciel-card-space/1"
_TOKEN = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r'(?:[^\W\d]|_)\w*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|'
    r'<=|>=|!=|==|\*\*|//|->|[^\w\s]', re.UNICODE)


def _digest(value):
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def _card(value):
    # Reuse the generation contract, including its duplicate-key and literal
    # rules. The local import avoids a selection/engine initialization cycle.
    from ...engine.idea import FIELDS, card_text, parse_card

    if value is None or (type(value) is str and not value):
        return None, {"kind": "missing", "sha256": None}
    if type(value) is dict:
        if (set(value) != set(FIELDS) or any(type(value[k]) is not str or
                len(value[k]) > limit for k, limit in FIELDS.items())):
            return None, {"kind": "malformed", "sha256": None}
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    elif type(value) is str:
        raw = value
    else:
        return None, {"kind": "malformed", "sha256": None}
    metadata = {"sha256": _digest(raw), "chars": len(raw)}
    if len(raw) > 8192:
        return None, {**metadata, "kind": "oversized"}
    parsed = parse_card(raw)
    if parsed is None:
        stripped = raw.strip()
        kind = ("missing" if not stripped else "malformed" if type(value) is dict or
                stripped.startswith(("{", "[", "IDEA_CARD_JSON:")) else "legacy")
        return None, {**metadata, "kind": kind}
    return parsed, {**metadata, "kind": "structured",
                    "canonical_sha256": _digest(card_text(parsed))}


def card_vector(value):
    """Return a bounded, normalized sparse vector and explicit input status.

    Operators, case and complete quoted literals are retained as tokens. This
    still cannot distinguish all contradictory statements or paraphrases and
    must not be used as proof that two strategies have the same meaning.
    """
    parsed, metadata = _card(value)
    if parsed is None:
        return {}, metadata
    vector = {}
    for field in sorted(parsed):
        counts = Counter(_TOKEN.findall(parsed[field]))
        norm = math.sqrt(sum(count * count for count in counts.values()))
        if norm:
            vector.update({(field, token): count / norm for token, count in counts.items()})
    norm = math.sqrt(math.fsum(v * v for v in vector.values()))
    if not norm:
        return {}, {**metadata, "kind": "unrepresented"}
    return {token: count / norm for token, count in vector.items()}, metadata


def cosine_distance(first, second):
    """Cosine distance for nonempty nonnegative unit vectors from card_vector."""
    if not first or not second:
        raise ValueError("missing card vectors have no measured distance")
    dot = math.fsum(value * second.get(token, 0.0) for token, value in first.items())
    distance = min(1.0, max(0.0, 1.0 - dot))
    return 0.0 if distance < 1e-12 else distance


def card_novelty(cards, k=3):
    """Mean k-nearest distances within the supplied active population only.

    Legacy/missing inputs receive the mean observed novelty, not a premium for
    being incomparable. Fewer than two typed cards or an all-zero geometry
    explicitly disables the tilt. Callers must then execute the original
    selector unchanged. No archive, score, visits, RNG or cache is read here.
    """
    if type(k) is not int or k < 1:
        raise ValueError("card neighbors must be a positive integer")
    if type(cards) is not dict or any(type(key) is not str or not key for key in cards):
        raise ValueError("active cards require nonempty program IDs")
    vectors, metadata = {}, {}
    for ident in sorted(cards):
        vector, info = card_vector(cards[ident])
        metadata[ident] = info
        if vector:
            vectors[ident] = vector
    typed = sorted(vectors)
    measured = {}
    neighbors = min(k, max(0, len(typed) - 1))
    if neighbors:
        for ident in typed:
            distances = sorted(cosine_distance(vectors[ident], vectors[other])
                               for other in typed if other != ident)
            measured[ident] = math.fsum(distances[:neighbors]) / neighbors
    neutral = math.fsum(measured.values()) / len(measured) if measured else 0.0
    available = bool(neighbors and max(measured.values(), default=0.0) > 0)
    return {"schema": CONTRACT, "available": available,
            "reason": None if available else "too_few_structured_cards" if len(typed) < 2
                      else "zero_lexical_diversity",
            "novelty": {ident: measured.get(ident, neutral) for ident in sorted(cards)},
            "typed_ids": typed, "neighbors": neighbors,
            "neutral_novelty": neutral, "card_metadata": metadata,
            "interpretation": "Lexical card-space diversity; not verified algorithm-family discovery."}
