"""Explicit trusted setup and strict fixed-panel loading; no implicit creation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat

from . import oracle_runtime as runtime

BENCHMARK = "function_minimization_oracle_fixed_panel_v1"
STUDY_SCHEMA = "sciel-oracle-study/1"
BANK_SCHEMA = "sciel-oracle-fixed-bank/1"
PANEL_SCHEMA = "sciel-oracle-fixed-panel/1"
GENERATOR = "sciel-quadratic-waves/1"
SCORER = "best-queried-value-failed-zero/1"
COUNTS = dict(train=80, dev=20, offline_audit=20, online_audit=20)
PANEL_SIZE = 20


def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require_hash(value):
    need(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value),
        "invalid commitment hash",
    )
    return value


def generator_sha256():
    # Complete versioned generator/broker file, not a caller version string.
    root = Path(runtime.__file__).parent
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("oracle_runtime.py", "oracle_protocol.py")})


def algorithm_seed(index):
    need(type(index) is int and 0 <= index < 80, "invalid public seed index")
    return 13007 + 1009 * index


def panel(split, bank_hash, instance_hashes, number):
    need(
        split in COUNTS
        and type(number) is int
        and 0 <= number < COUNTS[split] // PANEL_SIZE,
        "invalid split/panel index",
    )
    require_hash(bank_hash)
    need(len(instance_hashes) == COUNTS[split], "bank instance inventory differs")
    for h in instance_hashes:
        require_hash(h)
    indices = list(range(number * PANEL_SIZE, (number + 1) * PANEL_SIZE))
    body = dict(
        schema=PANEL_SCHEMA,
        benchmark=BENCHMARK,
        protocol=runtime.PROTOCOL,
        generator=GENERATOR,
        generator_sha256=generator_sha256(),
        split=split,
        bank_sha256=bank_hash,
        panel_index=number,
        indices=indices,
        instance_sha256=[instance_hashes[i] for i in indices],
        algorithm_seeds=[algorithm_seed(i) for i in indices],
        query_budget=runtime.QUERY_BUDGET,
        scorer=SCORER,
    )
    return {**body, "sha256": digest(body)}


def validate_panel(value, *, expected_split):
    need(
        type(value) is dict
        and set(value)
        == {
            "schema",
            "benchmark",
            "protocol",
            "generator",
            "generator_sha256",
            "split",
            "bank_sha256",
            "panel_index",
            "indices",
            "instance_sha256",
            "algorithm_seeds",
            "query_budget",
            "scorer",
            "sha256",
        },
        "invalid panel fields",
    )
    need(
        expected_split in COUNTS and value["split"] == expected_split,
        "panel split mismatch",
    )
    require_hash(value["bank_sha256"])
    require_hash(value["sha256"])
    need(
        type(value["panel_index"]) is int
        and 0 <= value["panel_index"] < COUNTS[expected_split] // PANEL_SIZE,
        "invalid panel index",
    )
    indices = list(
        range(
            value["panel_index"] * PANEL_SIZE, (value["panel_index"] + 1) * PANEL_SIZE
        )
    )
    need(
        value["indices"] == indices
        and all(type(i) is int for i in value["indices"])
        and value["algorithm_seeds"] == [algorithm_seed(i) for i in indices]
        and all(type(i) is int for i in value["algorithm_seeds"]),
        "panel ordering/seed schedule differs",
    )
    need(
        type(value["instance_sha256"]) is list
        and len(value["instance_sha256"]) == PANEL_SIZE
        and len(set(value["instance_sha256"])) == PANEL_SIZE,
        "invalid panel instance inventory",
    )
    for h in value["instance_sha256"]:
        require_hash(h)
    need(
        value["schema"] == PANEL_SCHEMA
        and value["benchmark"] == BENCHMARK
        and value["protocol"] == runtime.PROTOCOL
        and value["generator"] == GENERATOR
        and value["generator_sha256"] == generator_sha256()
        and value["scorer"] == SCORER
        and type(value["query_budget"]) is int
        and value["query_budget"] == runtime.QUERY_BUDGET
        and value["sha256"]
        == digest({k: v for k, v in value.items() if k != "sha256"}),
        "panel identity differs",
    )
    return value


def _private_root(directory):
    path = Path(directory).absolute()
    need(
        ".." not in path.parts and path == path.resolve(strict=True) and path.is_dir(),
        "private study root must be a canonical directory without symlinks",
    )
    need(
        stat.S_IMODE(path.stat().st_mode) == 0o700,
        "private study directory must have mode 0700",
    )
    return path


def _read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        need(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o600
            and info.st_size <= 2 * 1024 * 1024,
            "private study file must be bounded regular mode0600",
        )
        raw = stream.read()
    return json.loads(
        raw,
        object_pairs_hook=runtime._unique_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def _write(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sync(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _instances(seed, split, count):
    return [
        runtime.instance_from_seed(
            hashlib.sha256(
                bytes.fromhex(seed) + split.encode() + i.to_bytes(8, "big")
            ).digest()
        )
        for i in range(count)
    ]


def create_study(directory):
    """One explicit setup. Existing/partial directories are never reused.

    This API is not called by evaluation or importing the task. Tests supply
    only deterministic synthetic entropy; real study creation is separate.
    """
    root = Path(directory).absolute()
    need(
        root.parent == root.parent.resolve(strict=True) and ".." not in root.parts,
        "study parent must be canonical",
    )
    root.mkdir(mode=0o700, exist_ok=False)
    _sync(root.parent)
    _private_root(root)
    seeds = [secrets.token_bytes(32) for _ in COUNTS]
    need(
        all(type(s) is bytes and len(s) == 32 for s in seeds)
        and len(set(seeds)) == len(COUNTS),
        "independent 256-bit split seeds required",
    )
    splits, panels, seen = {}, {}, set()
    for (split, count), seed in zip(COUNTS.items(), seeds):
        instances = _instances(seed.hex(), split, count)
        hashes = [digest(x) for x in instances]
        need(
            len(set(hashes)) == count and not (seen & set(hashes)),
            "study instance commitments overlap",
        )
        seen.update(hashes)
        body = dict(
            schema=BANK_SCHEMA,
            benchmark=BENCHMARK,
            generator=GENERATOR,
            generator_sha256=generator_sha256(),
            split=split,
            count=count,
            seed=seed.hex(),
            instances=instances,
        )
        _write(root / (split + ".json"), body)
        splits[split] = dict(count=count, sha256=digest(body), instance_sha256=hashes)
        for number in range(count // PANEL_SIZE):
            panels[f"{split}-{number}"] = panel(split, digest(body), hashes, number)
    manifest = dict(
        schema=STUDY_SCHEMA,
        benchmark=BENCHMARK,
        generator=GENERATOR,
        generator_sha256=generator_sha256(),
        splits=splits,
        panels=panels,
    )
    _sync(root)
    _write(root / ".manifest.pending", manifest)
    # Exclusive publication of the final marker after every bank is durable.
    os.link(root / ".manifest.pending", root / "manifest.json", follow_symlinks=False)
    _sync(root)
    (root / ".manifest.pending").unlink()
    _sync(root)
    return {"manifest": manifest, "manifest_sha256": digest(manifest)}


def validate_contract(manifest, manifest_sha256, panel_name, panel_sha256, *, expected_split):
    """Public manifest validation; no private bank or candidate execution."""
    require_hash(manifest_sha256)
    require_hash(panel_sha256)
    need(
        expected_split in COUNTS and type(panel_name) is str,
        "invalid requested split/panel",
    )
    need(
        type(manifest) is dict
        and set(manifest)
        == {"schema", "benchmark", "generator", "generator_sha256", "splits", "panels"}
        and digest(manifest) == manifest_sha256
        and manifest["schema"] == STUDY_SCHEMA
        and manifest["benchmark"] == BENCHMARK
        and manifest["generator"] == GENERATOR
        and manifest["generator_sha256"] == generator_sha256(),
        "study manifest identity differs",
    )
    need(
        type(manifest["splits"]) is dict
        and set(manifest["splits"]) == set(COUNTS)
        and type(manifest["panels"]) is dict,
        "invalid study split inventory",
    )
    expected_panels, seen = {}, set()
    for split, count in COUNTS.items():
        binding = manifest["splits"][split]
        need(
            type(binding) is dict
            and set(binding) == {"count", "sha256", "instance_sha256"}
            and type(binding["count"]) is int
            and binding["count"] == count,
            "split count differs",
        )
        hashes = binding["instance_sha256"]
        need(
            type(hashes) is list
            and len(hashes) == len(set(hashes)) == count
            and not (set(hashes) & seen),
            "study instance inventory overlaps",
        )
        seen.update(hashes)
        for number in range(count // PANEL_SIZE):
            expected_panels[f"{split}-{number}"] = panel(
                split, binding["sha256"], hashes, number
            )
    need(
        canonical(manifest["panels"]) == canonical(expected_panels)
        and panel_name in expected_panels,
        "study panel inventory differs",
    )
    selected = validate_panel(
        expected_panels[panel_name], expected_split=expected_split
    )
    need(selected["sha256"] == panel_sha256, "selected panel commitment differs")
    return selected


def load_contract(directory, manifest_sha256, panel_name, panel_sha256, *, expected_split):
    root = _private_root(directory)
    manifest = _read(root / "manifest.json")
    selected = validate_contract(manifest, manifest_sha256, panel_name, panel_sha256,
                                 expected_split=expected_split)
    return manifest, selected


def load_panel(directory, manifest_sha256, panel_name, panel_sha256, *, expected_split):
    """Read only the specified split bank; never open the other private roles."""
    root = _private_root(directory)
    manifest, selected = load_contract(root, manifest_sha256, panel_name, panel_sha256,
                                      expected_split=expected_split)
    bank = _read(root / (expected_split + ".json"))
    need(
        type(bank) is dict
        and set(bank)
        == {
            "schema",
            "benchmark",
            "generator",
            "generator_sha256",
            "split",
            "count",
            "seed",
            "instances",
        }
        and bank["schema"] == BANK_SCHEMA
        and bank["benchmark"] == BENCHMARK
        and bank["generator"] == GENERATOR
        and bank["generator_sha256"] == generator_sha256()
        and bank["split"] == expected_split
        and type(bank["count"]) is int
        and bank["count"] == COUNTS[expected_split]
        and digest(bank) == selected["bank_sha256"],
        "bank role/hash/identity differs",
    )
    require_hash(bank["seed"])
    need(
        canonical(bank["instances"])
        == canonical(_instances(bank["seed"], expected_split, bank["count"])),
        "private bank differs from pinned generator",
    )
    need(
        [digest(x) for x in bank["instances"]]
        == manifest["splits"][expected_split]["instance_sha256"],
        "private bank instance commitments differ",
    )
    return [bank["instances"][i] for i in selected["indices"]], selected
