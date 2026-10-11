"""Reset-capable input/output oracle for automata-learning baselines (P11, AALpy).

Active automata learning (L*, KV) needs a different interaction model from the
forward-only environment: *membership queries* that start from a reset service.
:class:`MealyOracle` offers exactly that on fresh, noise-free copies of a planted
deterministic service, and counts its cost (steps and resets) so the benchmark can
charge the baseline in the regime's cost unit.

Input alphabets:

* caches: one letter per key (``"a"``, ``"b"``, ...); output ``"hit"``/``"miss"``.
* rate limiters: ``"req"`` (one request at the current tick; ``"ok"``/``"rej"``)
  and ``"tick"`` (advance the clock by one; output ``"-"``). Several ``"req"`` in a
  row form a burst at one instant.

The service is held privately and never exposed; only the input/output behaviour
is observable, as for any black-box client. AALpy is a development dependency, so
:func:`aalpy_sul` imports it lazily.

Evaluator side (needs the planted service, i.e. an opened sealed truth):
:func:`mealy_counterexample` decides *exactly* whether a learned Mealy machine is
equivalent to a planted cache, by breadth-first search over the product of the
learned machine and the cache's behavioural states (:meth:`Cache.state_key`); it
returns a shortest distinguishing input word, or ``None``. :func:`reachable_states`
counts the cache's reachable behavioural states (an upper bound on the size of its
minimal Mealy machine). Random-walk equivalence oracles, as used inside L*/KV,
can stop early with a wrong machine; this check cannot.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from probe.envs.systemlab.services import Cache, RateLimiter

__all__ = ["MealyOracle", "aalpy_sul", "mealy_counterexample", "reachable_states"]


class MealyOracle:
    RATE_ALPHABET: tuple[str, ...] = ("req", "tick")

    def __init__(
        self,
        factory: Callable[[], RateLimiter | Cache],
        alphabet: Sequence[str],
        *,
        step_cost: float = 1.0,
        reset_cost: float = 1.0,
    ) -> None:
        if step_cost < 0 or reset_cost < 0:
            raise ValueError("costs must be >= 0")
        self.__factory = factory
        self.__svc: RateLimiter | Cache = factory()
        self._t = 0
        self.alphabet: tuple[str, ...] = tuple(alphabet)
        self.step_cost = float(step_cost)
        self.reset_cost = float(reset_cost)
        self.n_steps = 0
        self.n_resets = 0

    @property
    def cost(self) -> float:
        return self.n_steps * self.step_cost + self.n_resets * self.reset_cost

    def reset(self) -> None:
        self.__svc = self.__factory()
        self._t = 0
        self.n_resets += 1

    def step(self, letter: str) -> str:
        if letter not in self.alphabet:
            raise ValueError(f"{letter!r} is not in the input alphabet {self.alphabet}")
        self.n_steps += 1
        svc = self.__svc
        if isinstance(svc, Cache):
            return "hit" if svc.access(letter) else "miss"
        if letter == "tick":
            self._t += 1
            return "-"
        return "ok" if svc.burst(self._t, 1) == 1 else "rej"

    def query(self, word: Sequence[str]) -> list[str]:
        """Reset, then run ``word``; the outputs, one per input."""
        self.reset()
        return [self.step(x) for x in word]

    def __repr__(self) -> str:
        return (
            f"MealyOracle(alphabet={self.alphabet}, steps={self.n_steps}, resets={self.n_resets})"
        )


def aalpy_sul(oracle: MealyOracle) -> Any:
    """Wrap ``oracle`` as an ``aalpy.base.SUL`` (``pre`` resets, ``step`` forwards)."""
    from aalpy.base import SUL  # dev dependency: imported only when used

    class _SystemLabSUL(SUL):
        def __init__(self, inner: MealyOracle) -> None:
            super().__init__()
            self.inner = inner

        def pre(self) -> None:
            self.inner.reset()

        def post(self) -> None:
            pass

        def step(self, letter: Any) -> Any:
            if letter is None:
                return None
            return self.inner.step(letter)

    return _SystemLabSUL(oracle)


def _cache_factory(factory: Callable[[], RateLimiter | Cache]) -> Cache:
    c = factory()
    if not isinstance(c, Cache):
        raise TypeError("exact equivalence checks are implemented for caches only")
    return c


def _out(c: Cache, letter: str) -> str:
    return "hit" if c.access(letter) else "miss"


def reachable_states(
    factory: Callable[[], RateLimiter | Cache], alphabet: Sequence[str], *, limit: int = 100_000
) -> int:
    """Number of behavioural states of the planted cache reachable from its initial
    state over ``alphabet`` (raises ``RuntimeError`` beyond ``limit``, e.g. for LFU,
    whose in-cache counts are unbounded)."""
    c0 = _cache_factory(factory)
    seen = {c0.state_key()}
    queue = deque([c0])
    while queue:
        c = queue.popleft()
        for a in alphabet:
            d = c.clone()
            d.access(a)
            k = d.state_key()
            if k not in seen:
                if len(seen) >= limit:
                    raise RuntimeError(f"more than {limit} reachable states")
                seen.add(k)
                queue.append(d)
    return len(seen)


def mealy_counterexample(
    model: Any,
    factory: Callable[[], RateLimiter | Cache],
    alphabet: Sequence[str],
    *,
    limit: int = 1_000_000,
) -> list[str] | None:
    """A shortest input word on which ``model`` and the planted cache disagree, or
    ``None`` when they are equivalent on every word over ``alphabet``.

    ``model`` is a Mealy machine with AALpy's shape: ``initial_state``, and per state
    ``transitions[letter]`` and ``output_fun[letter]`` (outputs ``"hit"``/``"miss"``).
    Exact: the product of the model's states and the cache's behavioural states is
    finite for LRU, FIFO, MRU and CLOCK and is explored completely (``RuntimeError``
    beyond ``limit`` product states)."""
    c0 = _cache_factory(factory)
    s0 = model.initial_state
    seen = {(id(s0), c0.state_key())}
    queue: deque[tuple[Any, Cache, tuple[str, ...]]] = deque([(s0, c0, ())])
    while queue:
        s, c, word = queue.popleft()
        for a in alphabet:
            d = c.clone()
            want = _out(d, a)
            if a not in s.transitions or s.output_fun.get(a) != want:
                return [*word, a]
            t = s.transitions[a]
            k = (id(t), d.state_key())
            if k not in seen:
                if len(seen) >= limit:
                    raise RuntimeError(f"more than {limit} product states")
                seen.add(k)
                queue.append((t, d, (*word, a)))
    return None
