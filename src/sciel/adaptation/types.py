"""Immutable identities shared by generation, datasets and training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_hash(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def require_hash(value: str, name: str = "hash") -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def require_revision(value: str, name: str = "revision") -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40,64}", value) is None:
        raise ValueError(f"{name} must be a pinned commit hash, not a floating branch")
    return value


def require_identifier(value: str, name: str = "identifier") -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise ValueError(f"invalid {name}")
    return value


@dataclass(frozen=True)
class InferenceSpec:
    owner: str
    role: str
    base_model: str
    base_revision: str
    tokenizer_revision: str
    chat_template_sha256: str
    serving_alias: str
    adapter_id: str | None = None
    adapter_sha256: str | None = None
    dataset_sha256: str | None = None

    def __post_init__(self):
        require_identifier(self.owner, "owner")
        if any(not isinstance(v, str) or not v.strip()
               for v in (self.role, self.base_model, self.serving_alias)):
            raise ValueError("role, base model and serving alias must be nonempty")
        require_revision(self.base_revision, "base_revision")
        require_revision(self.tokenizer_revision, "tokenizer_revision")
        require_hash(self.chat_template_sha256, "chat_template_sha256")
        values = (self.adapter_id, self.adapter_sha256, self.dataset_sha256)
        if any(v is not None for v in values):
            require_identifier(self.adapter_id, "adapter_id")
            require_hash(self.adapter_sha256, "adapter_sha256")
            require_hash(self.dataset_sha256, "dataset_sha256")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "InferenceSpec":
        if not isinstance(value, dict):
            raise ValueError("inference spec must be a mapping")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("invalid inference spec fields") from exc

    @property
    def digest(self) -> str:
        return content_hash(self.to_dict())
