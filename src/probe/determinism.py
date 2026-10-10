"""Deterministic serialization, hashing and seeding.

Every hash PROBE publishes (protocol hashes, decision-sequence hashes, truth
commitments) is ``sha256(canonical_json(obj))``. Canonical JSON is UTF-8, keys
sorted, no insignificant whitespace, and floats written with Python's
shortest round-trip ``repr``, which is identical on Windows and Linux.

Seeds come from ``numpy.random.SeedSequence`` keyed by a *path* of strings or
ints (e.g. ``("microclimate", "R1", 17, "world")``). The same path always gives
the same stream, which is what makes common random numbers across policies
possible: world noise is keyed without the policy name, policy randomness with it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

__all__ = [
    "canonical_json",
    "content_hash",
    "make_rng",
    "rng_state_hash",
    "seed_sequence",
    "sha256_hex",
    "stable_int",
    "to_jsonable",
]


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays, dataclasses, pydantic models, tuples and sets
    into plain JSON types. Non-finite floats are rejected: they must never reach a
    published hash silently."""
    if obj is None or isinstance(obj, bool | str):
        return obj
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        f = float(obj)
        if not math.isfinite(f):
            raise ValueError(f"non-finite float {f!r} cannot be canonicalized")
        return f
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if hasattr(obj, "model_dump") and callable(obj.model_dump):  # pydantic v2
        return to_jsonable(obj.model_dump(mode="json"))
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"canonical JSON keys must be str, got {type(k).__name__}")
            out[k] = to_jsonable(v)
        return out
    if isinstance(obj, set | frozenset):
        return sorted(to_jsonable(x) for x in obj)
    if isinstance(obj, Sequence):
        return [to_jsonable(x) for x in obj]
    raise TypeError(f"cannot canonicalize object of type {type(obj).__name__}")


def canonical_json(obj: Any) -> bytes:
    return json.dumps(
        to_jsonable(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """``sha256(canonical_json(obj))`` as lowercase hex."""
    return sha256_hex(canonical_json(obj))


def stable_int(key: str | int, bits: int = 32) -> int:
    """Map a string or int to a stable non-negative int (for SeedSequence spawn keys).
    Python's ``hash()`` is salted per process, so it is never used for this."""
    if isinstance(key, int):
        if key < 0:
            raise ValueError("seed path ints must be non-negative")
        return key
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[: bits // 8], "big")


def seed_sequence(root: int, *path: str | int) -> np.random.SeedSequence:
    """A SeedSequence for ``root`` and a key path. Independent of call order."""
    return np.random.SeedSequence(entropy=root, spawn_key=tuple(stable_int(p) for p in path))


def make_rng(root: int, *path: str | int) -> np.random.Generator:
    """A PCG64 generator keyed by ``(root, *path)``. PCG64 output is
    platform-independent for a given seed."""
    return np.random.Generator(np.random.PCG64(seed_sequence(root, *path)))


def rng_state_hash(rng: np.random.Generator) -> str:
    """Hash of the generator's bit-generator state, logged with every decision so a
    replay can prove it consumed randomness identically."""
    return content_hash(rng.bit_generator.state)
