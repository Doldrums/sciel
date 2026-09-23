"""Response-only tokenization and whole-example budgets, without ML imports."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenizedExample:
    episode_id: str
    origin: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def loss_tokens(self) -> int:
        # Causal LM logits at t predict labels at t+1.
        return sum(label != IGNORE_INDEX for label in self.labels[1:])


@dataclass(frozen=True)
class NativeResponseEncoding:
    """Exact full response inputs with a masked assistant forecast prefix."""

    example: TokenizedExample
    target_binding_sha256: str
    chat_template_sha256: str
    rendered_prefix_sha256: str
    user_prefix_tokens: int
    masked_assistant_prefix_tokens: int
    supervised_assistant_tokens: int

    def receipt(self) -> dict:
        from .native_target import POLICY
        from .types import content_hash
        return dict(schema="sciel-native-response-tokenization/1", policy=POLICY,
            target_binding_sha256=self.target_binding_sha256,
            chat_template_sha256=self.chat_template_sha256,
            rendered_prefix_sha256=self.rendered_prefix_sha256,
            user_prefix_tokens=self.user_prefix_tokens,
            masked_assistant_prefix_tokens=self.masked_assistant_prefix_tokens,
            supervised_assistant_tokens=self.supervised_assistant_tokens,
            supervised_eos_tokens=1, loss_tokens=self.example.loss_tokens,
            sequence_tokens=len(self.example.input_ids),
            input_ids_sha256=content_hash(list(self.example.input_ids)),
            labels_sha256=content_hash(list(self.example.labels)))


@dataclass(frozen=True)
class TrainingPlan:
    updates: tuple[tuple[TokenizedExample, ...], ...]
    loss_tokens_by_origin: dict[str, int]
    token_limits_by_origin: dict[str, int]
    stop_reason: str

    @property
    def loss_tokens(self) -> int:
        return sum(self.loss_tokens_by_origin.values())


def template_sha256(tokenizer: Any) -> str:
    template = tokenizer.get_chat_template()
    if not isinstance(template, str) or not template:
        raise ValueError("An explicit, nonempty tokenizer chat template is required")
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def encode_response_example(
    tokenizer: Any, row: Mapping[str, Any], *, max_sequence_tokens: int
) -> TokenizedExample:
    """Keep the exact code target, masking the entire non-thinking prompt.

    Rendering the assistant message through a chat template can strip leading
    newlines from code. Instead render the generation prefix, append the exact
    target, and add the tokenizer's EOS. Reject unstable BPE boundaries rather
    than silently labelling a prompt token or tokenizing the target separately.
    """
    prompt, target = row.get("prompt"), row.get("target")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Training prompt must be a nonempty string")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Training target must be a nonempty string")
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or eos < 0:
        raise ValueError("Tokenizer must define an EOS token ID")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prefix, str):
        raise ValueError("Chat template did not return a string")
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    combined_ids = list(tokenizer.encode(prefix + target, add_special_tokens=False))
    if not prefix_ids or combined_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError(
            f"Unstable prompt/target token boundary in episode {row.get('episode_id')}; "
            "cannot construct a response-only mask safely"
        )
    if len(combined_ids) <= len(prefix_ids):
        raise ValueError("Target produced no tokens")
    ids = tuple(combined_ids + [eos])
    if len(ids) > max_sequence_tokens:
        raise ValueError(
            f"Episode {row.get('episode_id')} needs {len(ids)} tokens, above "
            f"max_sequence_tokens={max_sequence_tokens}; no truncation was applied"
        )
    labels = (IGNORE_INDEX,) * len(prefix_ids) + ids[len(prefix_ids):]
    origin = row.get("origin", "local")
    if origin not in {"local", "donor"}:
        raise ValueError(f"Unknown example origin: {origin}")
    return TokenizedExample(str(row["episode_id"]), origin, ids, labels)


def encode_native_response_example(
    tokenizer: Any, row: Mapping[str, Any], *, target_binding: dict,
    binding_inputs: Mapping[str, Any], expected_chat_template_sha256: str,
    max_sequence_tokens: int,
) -> NativeResponseEncoding:
    """Encode native-research-code-suffix/1 without changing the legacy path.

    The caller selects the actual pinned tokenizer artifacts. The expected
    template hash is checked here, not taken from the tokenizer as its own pin.
    Native admission inputs are rechecked by the binder; checkpoint authenticity,
    ancestry, eligibility and model/runtime qualification remain outside this
    encoder. Forecast tokens condition the model but have no direct loss labels.
    """
    from .native_target import validate_native_response_target
    from .types import require_hash, require_identifier, sha256_bytes

    validate_native_response_target(target_binding, **binding_inputs)
    episode_id = require_identifier(row.get("episode_id"), "episode_id")
    origin = row.get("origin", "local")
    if origin not in {"local", "donor"}:
        raise ValueError(f"Unknown example origin: {origin}")
    prompt = row.get("prompt")
    if type(prompt) is not str or not prompt.strip():
        raise ValueError("Training prompt must be a nonempty string")
    if type(max_sequence_tokens) is not int or max_sequence_tokens < 1:
        raise ValueError("max_sequence_tokens must be a positive integer")
    require_hash(expected_chat_template_sha256, "expected chat template hash")
    actual_template = template_sha256(tokenizer)
    if actual_template != expected_chat_template_sha256:
        raise ValueError("Tokenizer template differs from the pinned template")
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError("Tokenizer must define an EOS token ID")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if type(prefix) is not str or not prefix:
        raise ValueError("Chat template did not return a nonempty string")
    target = target_binding["target"]
    start, end = target_binding["target_policy"]["supervised_assistant_suffix"]
    if end != len(target):
        raise ValueError("Native target must supervise a complete response suffix")

    def encode(text):
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if not ids or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Tokenizer returned invalid token IDs")
        return ids

    prefix_ids = encode(prefix)
    masked_ids = encode(prefix + target[:start])
    combined_ids = encode(prefix + target)
    if combined_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError(f"Unstable user-prefix/response token boundary in episode {episode_id}")
    if combined_ids[:len(masked_ids)] != masked_ids:
        raise ValueError(f"Unstable masked-forecast/suffix token boundary in episode {episode_id}")
    if not len(prefix_ids) < len(masked_ids) < len(combined_ids):
        raise ValueError("Native response needs both masked forecast and supervised suffix tokens")
    # Stable prefixes alone would not detect a normalizing tokenizer. No source
    # or whitespace may be silently changed by the actual token representation.
    decode_options = dict(skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if (tokenizer.decode(combined_ids, **decode_options) != prefix + target
            or tokenizer.decode(combined_ids[len(prefix_ids):], **decode_options) != target
            or tokenizer.decode(combined_ids[len(masked_ids):], **decode_options) != target[start:]):
        raise ValueError("Tokenizer does not preserve the exact native response and suffix")
    if eos in combined_ids[len(masked_ids):]:
        raise ValueError("Native response suffix already contains an EOS token")
    ids = tuple(combined_ids + [eos])
    if len(ids) > max_sequence_tokens:
        raise ValueError(f"Episode {episode_id} needs {len(ids)} tokens, above "
                         f"max_sequence_tokens={max_sequence_tokens}; no truncation was applied")
    labels = (IGNORE_INDEX,) * len(masked_ids) + ids[len(masked_ids):]
    example = TokenizedExample(episode_id, origin, ids, labels)
    return NativeResponseEncoding(example, target_binding["sha256"], actual_template,
        sha256_bytes(prefix.encode("utf-8")), len(prefix_ids),
        len(masked_ids) - len(prefix_ids), len(combined_ids) - len(masked_ids))


def collate_response_examples(
    examples: Sequence[TokenizedExample], *, pad_token_id: int
) -> dict[str, list[list[int]]]:
    """Right padding masks labels independently of the EOS/pad token value."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(item.input_ids) for item in examples)
    return {
        "input_ids": [list(e.input_ids) + [pad_token_id] * (width - len(e.input_ids)) for e in examples],
        "attention_mask": [[1] * len(e.input_ids) + [0] * (width - len(e.input_ids)) for e in examples],
        "labels": [list(e.labels) + [IGNORE_INDEX] * (width - len(e.labels)) for e in examples],
    }


def plan_training(
    examples: Sequence[TokenizedExample], *, mode: str, max_loss_tokens: int,
    max_optimizer_steps: int, gradient_accumulation_steps: int, epochs: int,
    seed: int, donor_fraction: float = 0.5,
) -> TrainingPlan:
    """Plan whole examples under independent token and update upper bounds.

    Mixed jobs reserve separate local/donor token quotas. Unused quotas are
    reported, never transferred to another origin. Each origin has a seeded,
    shuffled stream; its first example that cannot fit ends that stream, so
    budgeting never silently selects shorter later examples. A final incomplete
    accumulation group is an update with its actual loss-token denominator.
    """
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in (
        max_loss_tokens, max_optimizer_steps, gradient_accumulation_steps, epochs
    )):
        raise ValueError("All training budgets and epochs must be positive integers")
    if mode not in {"local", "mixed"}:
        raise ValueError("Dataset mode must be local or mixed")
    if not examples:
        raise ValueError("Verified training dataset is empty")
    if mode == "mixed" and not 0 < donor_fraction < 1:
        raise ValueError("Mixed donor_fraction must lie strictly between 0 and 1")
    if any(e.loss_tokens <= 0 for e in examples):
        raise ValueError("Each training example must have response loss tokens")
    origins = {e.origin for e in examples}
    expected = {"local", "donor"} if mode == "mixed" else {"local"}
    if origins != expected:
        raise ValueError(f"Dataset origins {origins} do not match mode={mode}")
    donor_limit = int(max_loss_tokens * donor_fraction) if mode == "mixed" else 0
    limits = {"local": max_loss_tokens - donor_limit}
    if mode == "mixed":
        limits["donor"] = donor_limit
        if min(limits.values()) <= 0:
            raise ValueError("Mixed token budget must allocate tokens to both origins")
    rng = random.Random(seed)
    streams: dict[str, list[TokenizedExample]] = {o: [] for o in limits}
    for _ in range(epochs):
        for origin in streams:
            epoch = [e for e in examples if e.origin == origin]
            rng.shuffle(epoch)
            streams[origin].extend(epoch)
    used = {o: 0 for o in limits}
    positions = {o: 0 for o in limits}
    blocked: set[str] = set()
    selected: list[TokenizedExample] = []
    max_examples = max_optimizer_steps * gradient_accumulation_steps
    while len(selected) < max_examples:
        ready = [o for o in streams if o not in blocked and positions[o] < len(streams[o])]
        if not ready:
            break
        origin = min(ready, key=lambda o: used[o] / limits[o])
        item = streams[origin][positions[origin]]
        if used[origin] + item.loss_tokens > limits[origin]:
            blocked.add(origin)
            continue
        positions[origin] += 1
        selected.append(item)
        used[origin] += item.loss_tokens
    if not selected or (mode == "mixed" and any(n == 0 for n in used.values())):
        raise ValueError("Budget cannot fit a complete example from every required origin")
    updates = tuple(tuple(selected[i:i + gradient_accumulation_steps])
                    for i in range(0, len(selected), gradient_accumulation_steps))
    reason = ("optimizer_step_budget" if len(selected) == max_examples else
              "loss_token_budget" if blocked else "dataset_exhausted")
    return TrainingPlan(updates, used, limits, reason)
