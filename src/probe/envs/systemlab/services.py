"""Planted mystery services: deterministic, scalar reference simulators.

These are the *truth side* of the system lab. They are written as plain,
one-request-at-a-time state machines, deliberately independently of the
vectorized grid mechanisms in :mod:`probe.envs.systemlab.mechanisms`; tests check
that the two agree exactly (noise-free) on random action sequences.

Time is integer **ticks**, so every limiter is exact integer arithmetic and
identical on every platform. A rate-limiter action is "idle for ``wait`` ticks,
then send a burst of ``n`` simultaneous requests"; :meth:`RateLimiter.burst`
processes the ``n`` requests in order at one instant and returns how many were
admitted. A cache action is a key sequence; :meth:`Cache.run` returns the hit
count.

Rate limiters (all start "full", i.e. idle since the beginning of time):

* **token bucket** ``(capacity C, period P)``: one token is added every ``P``
  ticks, continuously, up to ``C`` tokens; a request consumes a token. Tracked as
  an integer level in token-ticks, ``L ∈ [0, C·P]``; admit iff ``L ≥ P``.
* **leaky meter** (GCRA, the "leaky bucket as a meter") ``(capacity C, period P)``:
  theoretical arrival time ``TAT``; admit iff ``TAT − t ≤ (C − 1)·P``, then
  ``TAT ← max(TAT, t) + P``. Observationally *equivalent* to the token bucket
  with the same ``(C, P)`` (R5 relies on this; it is tested).
* **fixed window** ``(limit, window W, phase q)``: windows ``[jW − o, (j+1)W − o)``
  with offset ``o = q·W/4`` (``q`` in quarters); at most ``limit`` admits per window.
* **sliding window** (log) ``(limit, window W)``: admit iff fewer than ``limit``
  admits in the half-open interval ``(t − W, t]``.

Out-of-set services (truths for R3; no registered hypothesis has their form):

* **dual** limiter: a request is admitted only if *both* inner limiters admit it
  (e.g. a burst token bucket plus a fixed-window quota, as many public APIs do);
  it then consumes from both.
* **penalty box**: an inner limiter, and after a rejection *by the inner limiter*
  every request is rejected for ``penalty`` ticks (requests during the penalty
  are rejected without extending it).
* **deploy**: one service until tick ``at``, then a *fresh* instance of another
  (R3: a different algorithm; R7: the same algorithm with new parameters).

Caches (``policy``, ``size``): hit iff the key is resident. On a miss with a
full cache the victim is the least recently used (``lru``), the earliest
inserted (``fifo``), the least frequently used with ties broken by least recent
use (``lfu``; in-cache counts, reset on eviction), the most recently used
(``mru``), or chosen by the second-chance ``clock`` sweep (a hit sets the
frame's reference bit, a new key enters with the bit clear, the hand clears set
bits until it finds a clear one and evicts there). ``mru`` and ``clock`` have no
grid mechanism: they are the out-of-set, finite-state truths of R11.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from probe.determinism import to_jsonable

__all__ = [
    "CACHE_POLICIES",
    "Cache",
    "DeployService",
    "DualLimiter",
    "FixedWindow",
    "LeakyMeter",
    "PenaltyBox",
    "RateLimiter",
    "ServiceSpec",
    "SlidingWindow",
    "TokenBucket",
    "build_service",
]

CACHE_POLICIES = ("clock", "fifo", "lfu", "lru", "mru")
"""All simulated policies. Only lru, fifo and lfu have a grid mechanism; clock and
mru are out-of-set truths (R11)."""


def _pos_int(name: str, v: Any) -> int:
    if isinstance(v, bool) or int(v) != v or int(v) < 1:
        raise ValueError(f"{name} must be a positive integer, got {v!r}")
    return int(v)


# --------------------------------------------------------------------------- rate limiters


class RateLimiter(ABC):
    """A deterministic limiter. ``advance_to`` must be called with non-decreasing ``t``."""

    kind: str

    def __init__(self) -> None:
        self._t = 0

    def advance_to(self, t: int) -> None:
        if t < self._t:
            raise ValueError(f"time went backwards ({t} < {self._t})")
        self._t = int(t)
        self._on_time(self._t)

    @abstractmethod
    def _on_time(self, t: int) -> None: ...

    @abstractmethod
    def can_admit(self) -> bool: ...

    @abstractmethod
    def admit(self) -> None: ...

    def request(self) -> bool:
        """One request at the current time: admit it if possible."""
        if self.can_admit():
            self.admit()
            return True
        return False

    def burst(self, t: int, n: int) -> int:
        """``n`` simultaneous requests at tick ``t``; returns the number admitted."""
        if n < 0:
            raise ValueError("burst size must be >= 0")
        self.advance_to(t)
        return sum(1 for _ in range(n) if self.request())

    def clone(self) -> RateLimiter:
        return copy.deepcopy(self)


class TokenBucket(RateLimiter):
    kind = "token_bucket"

    def __init__(self, capacity: int, period: int) -> None:
        super().__init__()
        self.capacity = _pos_int("capacity", capacity)
        self.period = _pos_int("period", period)
        self._level = self.capacity * self.period  # token-ticks
        self._t_level = 0

    def _on_time(self, t: int) -> None:
        full = self.capacity * self.period
        self._level = min(full, self._level + (t - self._t_level))
        self._t_level = t

    def can_admit(self) -> bool:
        return self._level >= self.period

    def admit(self) -> None:
        self._level -= self.period


class LeakyMeter(RateLimiter):
    kind = "leaky_meter"

    def __init__(self, capacity: int, period: int) -> None:
        super().__init__()
        self.capacity = _pos_int("capacity", capacity)
        self.period = _pos_int("period", period)
        self._tat = 0

    def _on_time(self, t: int) -> None:
        pass

    def can_admit(self) -> bool:
        return self._tat - self._t <= (self.capacity - 1) * self.period

    def admit(self) -> None:
        self._tat = max(self._tat, self._t) + self.period


class FixedWindow(RateLimiter):
    kind = "fixed_window"

    def __init__(self, limit: int, window: int, phase: int = 0) -> None:
        super().__init__()
        self.limit = _pos_int("limit", limit)
        self.window = _pos_int("window", window)
        if int(phase) != phase or not 0 <= int(phase) <= 3:
            raise ValueError("phase is in quarters of the window: 0, 1, 2 or 3")
        self.phase = int(phase)
        self._offset = self.phase * self.window // 4
        self._win = self._offset // self.window
        self._count = 0

    def _on_time(self, t: int) -> None:
        w = (t + self._offset) // self.window
        if w != self._win:
            self._win = w
            self._count = 0

    def can_admit(self) -> bool:
        return self._count < self.limit

    def admit(self) -> None:
        self._count += 1


class SlidingWindow(RateLimiter):
    kind = "sliding_window"

    def __init__(self, limit: int, window: int) -> None:
        super().__init__()
        self.limit = _pos_int("limit", limit)
        self.window = _pos_int("window", window)
        self._log: deque[int] = deque()

    def _on_time(self, t: int) -> None:
        while self._log and self._log[0] <= t - self.window:
            self._log.popleft()

    def can_admit(self) -> bool:
        return len(self._log) < self.limit

    def admit(self) -> None:
        self._log.append(self._t)


class DualLimiter(RateLimiter):
    kind = "dual"

    def __init__(self, first: RateLimiter, second: RateLimiter) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def _on_time(self, t: int) -> None:
        self.first.advance_to(t)
        self.second.advance_to(t)

    def can_admit(self) -> bool:
        return self.first.can_admit() and self.second.can_admit()

    def admit(self) -> None:
        self.first.admit()
        self.second.admit()


class PenaltyBox(RateLimiter):
    kind = "penalty_box"

    def __init__(self, inner: RateLimiter, penalty: int) -> None:
        super().__init__()
        self.inner = inner
        self.penalty = _pos_int("penalty", penalty)
        self._blocked_until = 0

    def _on_time(self, t: int) -> None:
        self.inner.advance_to(t)

    def can_admit(self) -> bool:
        return self._t >= self._blocked_until and self.inner.can_admit()

    def admit(self) -> None:
        self.inner.admit()

    def request(self) -> bool:
        if self._t < self._blocked_until:
            return False  # in the penalty box; does not extend the penalty
        if self.inner.can_admit():
            self.inner.admit()
            return True
        self._blocked_until = self._t + self.penalty
        return False


class DeployService(RateLimiter):
    """``before`` until tick ``at`` (exclusive), then a fresh ``after``."""

    kind = "deploy"

    def __init__(self, before: RateLimiter, after: RateLimiter, at: int) -> None:
        super().__init__()
        if int(at) != at or at < 0:
            raise ValueError("deploy time must be a non-negative integer tick")
        self.before = before
        self.after = after
        self.at = int(at)

    def _active(self) -> RateLimiter:
        return self.after if self._t >= self.at else self.before

    def _on_time(self, t: int) -> None:
        self._active().advance_to(t)

    def can_admit(self) -> bool:
        return self._active().can_admit()

    def admit(self) -> None:
        self._active().admit()

    def request(self) -> bool:
        return self._active().request()


# --------------------------------------------------------------------------- caches


class Cache:
    """A deterministic cache over string keys (see module docstring for policies)."""

    kind = "cache"

    def __init__(self, policy: str, size: int) -> None:
        if policy not in CACHE_POLICIES:
            raise ValueError(f"unknown cache policy {policy!r}; expected one of {CACHE_POLICIES}")
        self.policy = policy
        self.size = _pos_int("size", size)
        self._order: OrderedDict[str, None] = OrderedDict()  # lru: recency, fifo: insertion
        self._freq: dict[str, int] = {}
        self._last: dict[str, int] = {}
        self._frames: list[str] = []  # clock
        self._ref: list[bool] = []
        self._hand = 0
        self._tick = 0

    def access(self, key: str) -> bool:
        tick = self._tick
        self._tick += 1
        if self.policy == "clock":
            return self._access_clock(key)
        if self.policy in ("lfu", "mru"):
            if key in self._freq:
                self._freq[key] += 1
                self._last[key] = tick
                return True
            if len(self._freq) >= self.size:
                if self.policy == "lfu":
                    victim = min(self._freq, key=lambda k: (self._freq[k], self._last[k]))
                else:
                    victim = max(self._last, key=lambda k: self._last[k])
                del self._freq[victim]
                del self._last[victim]
            self._freq[key] = 1
            self._last[key] = tick
            return False
        if key in self._order:
            if self.policy == "lru":
                self._order.move_to_end(key)
            return True
        if len(self._order) >= self.size:
            self._order.popitem(last=False)
        self._order[key] = None
        return False

    def _access_clock(self, key: str) -> bool:
        if key in self._frames:
            self._ref[self._frames.index(key)] = True
            return True
        if len(self._frames) < self.size:
            self._frames.append(key)
            self._ref.append(False)
            return False
        while self._ref[self._hand]:
            self._ref[self._hand] = False
            self._hand = (self._hand + 1) % self.size
        self._frames[self._hand] = key
        self._ref[self._hand] = False
        self._hand = (self._hand + 1) % self.size
        return False

    def run(self, keys: str) -> int:
        """Access each character of ``keys`` in order; return the number of hits."""
        return sum(1 for k in keys if self.access(k))

    def resident(self) -> frozenset[str]:
        if self.policy == "clock":
            return frozenset(self._frames)
        if self.policy in ("lfu", "mru"):
            return frozenset(self._freq)
        return frozenset(self._order)

    def state_key(self) -> tuple[Any, ...]:
        """A hashable key of everything the cache's future input/output behaviour
        depends on (absolute access counters abstracted away): two caches of the same
        policy and size with equal keys answer every key sequence identically.

        ``lru``/``fifo``: residents in recency/insertion order; ``mru``: the resident
        set and the most recently used key; ``clock``: frames, reference bits and
        hand. These four have finitely many keys (finite-state machines). ``lfu``
        keeps unbounded in-cache counts, so its key space is infinite."""
        if self.policy == "clock":
            return ("clock", self.size, tuple(self._frames), tuple(self._ref), self._hand)
        if self.policy in ("lru", "fifo"):
            return (self.policy, self.size, tuple(self._order))
        by_recency = sorted(self._last, key=lambda k: self._last[k])
        if self.policy == "mru":
            return ("mru", self.size, tuple(sorted(self._freq)), tuple(by_recency[-1:]))
        return ("lfu", self.size, tuple((k, self._freq[k]) for k in by_recency))

    def clone(self) -> Cache:
        return copy.deepcopy(self)


# --------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class ServiceSpec:
    """A JSON-able description of a planted service (lives inside the sealed truth).

    ``kind`` is one of ``token_bucket``, ``leaky_meter``, ``fixed_window``,
    ``sliding_window``, ``dual``, ``penalty_box``, ``deploy`` or ``cache``.
    Composite kinds nest specs in ``children`` (``dual``: first, second;
    ``penalty_box``: inner; ``deploy``: before, after).
    """

    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)
    children: Mapping[str, ServiceSpec] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "params": to_jsonable(dict(self.params))}
        if self.children:
            out["children"] = {k: v.as_dict() for k, v in sorted(self.children.items())}
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ServiceSpec:
        """Inverse of :meth:`as_dict` (evaluator side: rebuild the planted service from
        an opened :class:`~probe.truth.SealedTruth` payload)."""
        children = {k: cls.from_dict(v) for k, v in dict(d.get("children", {})).items()}
        return cls(str(d["kind"]), dict(d.get("params", {})), children)

    def is_cache(self) -> bool:
        return self.kind == "cache"


def build_service(spec: ServiceSpec) -> RateLimiter | Cache:
    """A fresh service instance (in its initial state) from a spec."""
    p = dict(spec.params)
    k = spec.kind
    if k == "token_bucket":
        return TokenBucket(p["capacity"], p["period"])
    if k == "leaky_meter":
        return LeakyMeter(p["capacity"], p["period"])
    if k == "fixed_window":
        return FixedWindow(p["limit"], p["window"], p.get("phase", 0))
    if k == "sliding_window":
        return SlidingWindow(p["limit"], p["window"])
    if k == "cache":
        return Cache(str(p["policy"]), p["size"])
    if k == "dual":
        return DualLimiter(_limiter(spec, "first"), _limiter(spec, "second"))
    if k == "penalty_box":
        return PenaltyBox(_limiter(spec, "inner"), p["penalty"])
    if k == "deploy":
        return DeployService(_limiter(spec, "before"), _limiter(spec, "after"), p["at"])
    raise ValueError(f"unknown service kind {k!r}")


def _limiter(spec: ServiceSpec, child: str) -> RateLimiter:
    try:
        sub = spec.children[child]
    except KeyError as e:
        raise ValueError(f"{spec.kind}: missing child spec {child!r}") from e
    svc = build_service(sub)
    if not isinstance(svc, RateLimiter):
        raise ValueError(f"{spec.kind}: child {child!r} must be a rate limiter")
    return svc
