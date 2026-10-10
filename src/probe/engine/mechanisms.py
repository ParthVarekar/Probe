"""Registry of in-tree mechanisms for ``finite_grid`` hypotheses.

A ``GridHypothesis`` names a mechanism; it never carries code. Mechanisms are
ordinary reviewed Python registered at import time (e.g. by
``probe.envs.systemlab.mechanisms``). There is deliberately no way to register a
mechanism from IR, an LLM proposal or user input.

Mechanisms are **vectorized over grid points**: ``params`` is the column table
from :meth:`GridHypothesis.grid_table` (each value an array of length G), and
state is whatever arrays the mechanism needs, with leading dimension G.

Contract:

* ``initial_state(params) -> state`` (arrays with leading dimension G).
* ``predict(state, params, record) -> probs`` of shape (G, S) over the outcome
  support 0..S-1, where S = max_count + 1. This must not mutate ``state``.
* ``advance(state, params, record) -> new_state``: the state after the action in
  ``record`` was executed. State dynamics depend on the action and the
  parameters, **not** on the noisy observed outcome, so the posterior over grid
  points stays exact.
* Pure and deterministic. No randomness, no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from probe.engine.ir import Record

__all__ = ["Mechanism", "get_mechanism", "register_mechanism", "registered_mechanisms"]


class Mechanism(Protocol):
    name: str
    support_size: int

    def initial_state(self, params: Mapping[str, np.ndarray]) -> Any: ...

    def predict(
        self, state: Any, params: Mapping[str, np.ndarray], record: Record
    ) -> np.ndarray: ...

    def advance(self, state: Any, params: Mapping[str, np.ndarray], record: Record) -> Any: ...


_REGISTRY: dict[str, Callable[[int], Mechanism]] = {}


def register_mechanism(
    name: str,
) -> Callable[[Callable[[int], Mechanism]], Callable[[int], Mechanism]]:
    """Register a factory ``f(support_size) -> Mechanism`` under ``name``."""

    def deco(factory: Callable[[int], Mechanism]) -> Callable[[int], Mechanism]:
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"mechanism {name!r} already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def get_mechanism(name: str, support_size: int) -> Mechanism:
    try:
        factory = _REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"unknown mechanism {name!r}; registered: {sorted(_REGISTRY)}") from e
    return factory(support_size)


def registered_mechanisms() -> list[str]:
    return sorted(_REGISTRY)
