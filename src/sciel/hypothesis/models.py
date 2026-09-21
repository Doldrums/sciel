"""Data exchanged by the research protocol; scores remain evaluator-owned."""

from dataclasses import asdict, dataclass

CONTRACT = "sciel-research/12"


@dataclass(frozen=True)
class Hypothesis:
    """A forecast interval for child score minus parent score, not absolute score."""

    text: str
    predicted_low: float
    predicted_high: float
    reasoning: str
    confidence: float
    restart_predictions: dict | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        if self.restart_predictions is None:
            del value['restart_predictions']  # Keep historical five-field hashes exact.
        return value


@dataclass(frozen=True)
class Conclusion:
    verdict: str
    explanation: str
    lessons: list[dict]
    mechanism_review: list[dict] | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        if self.mechanism_review is None:
            del value['mechanism_review']  # Preserve historical three-field records exactly.
        return value


@dataclass(frozen=True)
class Reflection:
    lessons: list[dict]
    retire_ids: list[str]
    comparison_reviews: list[dict] | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        if self.comparison_reviews is None:
            del value['comparison_reviews']  # Keep historical two-field records exact.
        return value
