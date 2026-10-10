"""Action spaces of the system lab: finite, deterministic candidate lists.

Rate limiters (:class:`RateActionSpace`): "idle ``wait`` ticks, then send
``burst`` simultaneous requests". The action takes ``wait + 1`` ticks of wall
time. Under the default ``wall`` cost model it costs exactly that (cost unit
``tick``), so costs are heterogeneous: long waits are what reveal refill
behaviour, and they are expensive. The ``uniform`` model (R2) charges one unit
(``action``) per pattern.

Caches (:class:`CacheActionSpace`): replay a fixed key sequence (a *key-reuse
pattern*). Every access takes one unit of time and costs one ``access``; all
patterns of a regime have the same length, so costs are uniform.

Candidates are rebuilt from the current clock at every step, sorted by action id;
an action whose end would pass the campaign horizon is listed as infeasible.
Action ids are zero-padded so that lexicographic order is natural order, and
are identical on every platform (they are the tie-break keys).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from probe.engine.ir import CovariateSpec, OutcomeSpec
from probe.envs.base import Action, Candidate

__all__ = ["CacheActionSpace", "RateActionSpace", "cache_patterns"]


def _wait_class(w: int) -> str:
    return "short" if w <= 2 else "mid" if w <= 8 else "long"


def _burst_class(b: int) -> str:
    return "small" if b <= 2 else "mid" if b <= 6 else "large"


@dataclass(frozen=True)
class RateActionSpace:
    waits: tuple[int, ...]
    bursts: tuple[int, ...]
    horizon: int = 20_000
    """Campaign length in ticks; actions ending later are infeasible."""
    cost_model: Literal["wall", "uniform"] = "wall"
    """``wall``: cost = wall time ``wait + 1`` ticks. ``uniform``: every action costs 1
    (time still advances by ``wait + 1``)."""

    def __post_init__(self) -> None:
        if not self.waits or not self.bursts:
            raise ValueError("need at least one wait and one burst size")
        if min(self.waits) < 0 or min(self.bursts) < 1:
            raise ValueError("waits must be >= 0 and bursts >= 1")
        if max(self.waits) >= 1000 or max(self.bursts) >= 100:
            raise ValueError("waits < 1000 and bursts < 100 keep ids fixed-width")

    @property
    def cost_unit(self) -> str:
        return "tick" if self.cost_model == "wall" else "action"

    def cost(self, wait: int, burst: int) -> float:
        return float(self.duration(wait, burst)) if self.cost_model == "wall" else 1.0

    @property
    def max_count(self) -> int:
        return max(self.bursts)

    def outcome(self) -> OutcomeSpec:
        return OutcomeSpec(
            name="accepted",
            kind="count",
            unit="requests",
            max_count=self.max_count,
            description="Reported number of requests in the burst that the service admitted.",
        )

    def covariates(self) -> tuple[CovariateSpec, ...]:
        return (
            CovariateSpec(name="burst", kind="numeric", unit="requests", description="Burst size."),
            CovariateSpec(
                name="clock", kind="numeric", unit="tick", description="Clock at action start."
            ),
            CovariateSpec(
                name="wait", kind="numeric", unit="tick", description="Idle ticks before the burst."
            ),
        )

    @staticmethod
    def action_id(wait: int, burst: int) -> str:
        return f"wait={wait:03d}|burst={burst:02d}"

    @staticmethod
    def stratum(wait: int, burst: int) -> str:
        return f"wait={_wait_class(wait)}|burst={_burst_class(burst)}"

    @staticmethod
    def duration(wait: int, burst: int) -> int:
        return wait + 1

    def population(self) -> tuple[str, ...]:
        return tuple(sorted({self.stratum(w, b) for w in self.waits for b in self.bursts}))

    @cached_property
    def _index(self) -> dict[str, tuple[int, int]]:
        return {
            self.action_id(w, b): (w, b)
            for w in sorted(set(self.waits))
            for b in sorted(set(self.bursts))
        }

    def _candidate(self, wait: int, burst: int, clock: int) -> Candidate:
        feasible = clock + self.duration(wait, burst) <= self.horizon
        return Candidate(
            action=Action(id=self.action_id(wait, burst), params={"wait": wait, "burst": burst}),
            record={"wait": wait, "burst": burst, "clock": clock},
            cost=self.cost(wait, burst),
            stratum=self.stratum(wait, burst),
            feasible=feasible,
            infeasible_reason=None if feasible else "beyond campaign horizon",
        )

    def candidate(self, action_id: str, clock: int) -> Candidate | None:
        """The candidate with this id at ``clock`` (``None`` if no such action)."""
        wb = self._index.get(action_id)
        return None if wb is None else self._candidate(wb[0], wb[1], clock)

    def candidates(self, clock: int) -> list[Candidate]:
        return [self._candidate(w, b, clock) for _, (w, b) in sorted(self._index.items())]


def cache_patterns(keys: str, length: int) -> tuple[tuple[str, str], ...]:
    """Key-reuse patterns ``(family, sequence)`` over ``keys`` of exactly ``length``
    accesses, for two rotations of the key alphabet. Families: ``loop`` (cycle over
    m keys), ``scan`` (distinct keys), ``hot`` (a key accessed three times, then m
    others, then again), ``refresh`` (one key re-accessed every g + 1 steps between
    fresh keys). Duplicates are dropped; the order is deterministic."""
    if len(set(keys)) != len(keys) or not all("a" <= c <= "z" for c in keys):
        raise ValueError("keys must be distinct lowercase letters")
    u = len(keys)
    half = u // 2
    rotations = [keys, keys[half:] + keys[:half]]
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    def add(family: str, seq: str) -> None:
        if len(seq) == length and seq not in seen:
            seen.add(seq)
            out.append((family, seq))

    for ks in rotations:
        if u >= length:
            add("scan", ks[:length])
        for m in range(2, u + 1):
            add("loop", (ks[:m] * length)[:length])
        for m in range(2, max(3, length - 3)):
            body = ks[0] * 3 + ks[1 : 1 + m] + ks[0]
            add("hot", (body + ks[1:] * length)[:length])
        for gap in (1, 2, 3):
            seq, fresh = "", iter(ks[1:] * length)
            while len(seq) < length:
                seq += ks[0] + "".join(next(fresh) for _ in range(gap))
            add("refresh", seq[:length])
    return tuple(out)


@dataclass(frozen=True)
class CacheActionSpace:
    patterns: tuple[tuple[str, str], ...]
    """``(family, key sequence)`` pairs; all sequences have the same length."""
    horizon: int = 20_000
    """Campaign length in accesses."""

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ValueError("need at least one pattern")
        lengths = {len(s) for _, s in self.patterns}
        if len(lengths) != 1:
            raise ValueError("all patterns of a regime must have the same length (uniform cost)")
        seqs = [s for _, s in self.patterns]
        if len(set(seqs)) != len(seqs):
            raise ValueError("duplicate patterns")

    @property
    def cost_unit(self) -> str:
        return "access"

    @property
    def length(self) -> int:
        return len(self.patterns[0][1])

    @property
    def max_count(self) -> int:
        return self.length

    def keys(self) -> str:
        return "".join(sorted({c for _, s in self.patterns for c in s}))

    def outcome(self) -> OutcomeSpec:
        return OutcomeSpec(
            name="hits",
            kind="count",
            unit="accesses",
            max_count=self.max_count,
            description="Reported number of cache hits in the access sequence.",
        )

    def covariates(self) -> tuple[CovariateSpec, ...]:
        return (
            CovariateSpec(
                name="clock", kind="numeric", unit="access", description="Accesses so far."
            ),
            CovariateSpec(
                name="keys",
                kind="categorical",
                levels=tuple(sorted(s for _, s in self.patterns)),
                description="Key sequence replayed against the cache, one letter per access.",
            ),
        )

    @staticmethod
    def action_id(seq: str) -> str:
        return f"keys={seq}"

    def population(self) -> tuple[str, ...]:
        return tuple(sorted({f"pattern={f}" for f, _ in self.patterns}))

    @cached_property
    def _index(self) -> dict[str, tuple[str, str]]:
        return {self.action_id(seq): (family, seq) for family, seq in self.patterns}

    def _candidate(self, family: str, seq: str, clock: int) -> Candidate:
        feasible = clock + len(seq) <= self.horizon
        return Candidate(
            action=Action(id=self.action_id(seq), params={"keys": seq}),
            record={"keys": seq, "clock": clock},
            cost=float(len(seq)),
            stratum=f"pattern={family}",
            feasible=feasible,
            infeasible_reason=None if feasible else "beyond campaign horizon",
        )

    def candidate(self, action_id: str, clock: int) -> Candidate | None:
        """The candidate with this id at ``clock`` (``None`` if no such action)."""
        fs = self._index.get(action_id)
        return None if fs is None else self._candidate(fs[0], fs[1], clock)

    def candidates(self, clock: int) -> list[Candidate]:
        return [self._candidate(f, q, clock) for _, (f, q) in sorted(self._index.items())]
