"""Sealed hidden truth (handover §12 H.1, H.11).

The truth of a synthetic world (its true structure and parameters) is wrapped in
a :class:`SealedTruth`. Its ``repr``/``str`` are redacted, and its payload is
only reachable through :meth:`SealedTruth.open` with an :class:`EvaluatorKey`.
The key is created by evaluator code (the benchmark scorer, tests); policies and
the run loop never receive one.

This is API discipline inside one Python process, not cryptographic isolation:
the guarantee is enforced by construction plus tests that scan every
agent-facing output for truth values. The commitment hash is logged at world
creation so a later audit can prove the truth was fixed in advance.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from probe.determinism import content_hash, to_jsonable

__all__ = ["EvaluatorKey", "SealedTruth"]


class EvaluatorKey:
    """Capability token. Only evaluator code constructs one."""

    __slots__ = ("purpose",)

    def __init__(self, purpose: str) -> None:
        if not purpose:
            raise ValueError("state why the truth is being opened")
        self.purpose = purpose

    def __repr__(self) -> str:
        return f"EvaluatorKey({self.purpose!r})"


class SealedTruth:
    """Hidden truth for one world.

    ``true_hypothesis_id`` is the registered hypothesis the world was built to
    satisfy, or ``None`` when the truth lies outside the hypothesis set (the
    correct verdict is then "none fits"/residual). ``salt`` makes the commitment
    non-guessable from a small parameter space; it is part of the opened payload.
    Families derive it deterministically from the world seed (e.g.
    ``make_rng(seed, family, regime, "salt").bytes(16).hex()``) so that a
    same-seed rerun reproduces the commitment exactly.
    """

    __slots__ = ("__payload", "__true_id", "_commitment", "world_id")

    def __init__(
        self,
        world_id: str,
        true_hypothesis_id: str | None,
        payload: Mapping[str, Any],
        salt: str,
    ) -> None:
        if len(salt) < 16:
            raise ValueError("salt must be >= 16 chars (derive it from the world seed)")
        body = {
            "world_id": world_id,
            "true_hypothesis_id": true_hypothesis_id,
            "payload": to_jsonable(payload),
            "salt": salt,
        }
        self.world_id = world_id
        self.__true_id = true_hypothesis_id
        self.__payload = MappingProxyType(body)
        self._commitment = content_hash(body)

    @property
    def commitment(self) -> str:
        return self._commitment

    def open(self, key: EvaluatorKey) -> Mapping[str, Any]:
        if not isinstance(key, EvaluatorKey):
            raise PermissionError("hidden truth can only be opened with an EvaluatorKey")
        return self.__payload

    def true_hypothesis_id(self, key: EvaluatorKey) -> str | None:
        if not isinstance(key, EvaluatorKey):
            raise PermissionError("hidden truth can only be opened with an EvaluatorKey")
        return self.__true_id

    def verify(self, opened: Mapping[str, Any]) -> bool:
        """Check an opened payload against the commitment."""
        return content_hash(dict(opened)) == self._commitment

    def __repr__(self) -> str:
        return f"SealedTruth(world_id={self.world_id!r}, commitment={self._commitment[:12]}…)"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("SealedTruth is not picklable; pass world seeds across processes")
