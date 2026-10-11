"""Policy registry: ids, human names and construction (handover §12 Part I).

``make_policy("P8")`` returns a fresh policy object; keyword arguments go to the
policy's constructor (e.g. ``make_policy("P9", max_plan_length=8)``). The oracle
P10 needs the opened truth and is built by evaluator code only:
``make_policy("P10", opened_truth=world.truth.open(key))``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from probe.select.policies import (
    EIGPerCostPolicy,
    FixedGridPolicy,
    GuardedEIGPolicy,
    MaxPredictiveVariancePolicy,
    ModelEIGPolicy,
    OraclePolicy,
    ParameterBALDPolicy,
    PolicyBase,
    RandomPolicy,
    StaticMaximinPolicy,
    StratifiedPolicy,
)

__all__ = ["POLICY_CLASSES", "POLICY_IDS", "POLICY_NAMES", "make_policy"]

POLICY_CLASSES: Mapping[str, type[PolicyBase]] = MappingProxyType(
    {
        "P1": RandomPolicy,
        "P2": FixedGridPolicy,
        "P3": StratifiedPolicy,
        "P4": MaxPredictiveVariancePolicy,
        "P5": ParameterBALDPolicy,
        "P6": ModelEIGPolicy,
        "P7": EIGPerCostPolicy,
        "P8": GuardedEIGPolicy,
        "P9": StaticMaximinPolicy,
        "P10": OraclePolicy,
    }
)

POLICY_IDS: tuple[str, ...] = tuple(POLICY_CLASSES)
"""``("P1", ..., "P10")``."""

POLICY_NAMES: Mapping[str, str] = MappingProxyType(
    {pid: cls.name for pid, cls in POLICY_CLASSES.items()}
)


def make_policy(policy_id: str, **kwargs: Any) -> PolicyBase:
    """A fresh policy ``policy_id`` (one of :data:`POLICY_IDS`)."""
    try:
        cls = POLICY_CLASSES[policy_id]
    except KeyError as e:
        raise KeyError(f"unknown policy {policy_id!r}; known: {list(POLICY_IDS)}") from e
    if cls is OraclePolicy and "opened_truth" not in kwargs:
        raise TypeError("P10 (oracle) needs opened_truth=SealedTruth.open(EvaluatorKey(...))")
    return cls(**kwargs)
