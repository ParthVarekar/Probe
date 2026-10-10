"""Stopping-rule contract (handover §12 H.6).

The verdict vocabulary is shared by the run loop, the benchmark scorer and,
later, the claim-guard. "Inconclusive" and "none fits" are first-class
findings, not failures.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["StopDecision", "StopReason", "Verdict"]

Verdict = Literal["hypothesis", "none_fits", "inconclusive", "undecided"]
"""``hypothesis``: one registered hypothesis is supported (see ``leader``).
``none_fits``: the residual leads or a calibrated inadequacy alarm fired.
``inconclusive``: the hypotheses cannot be told apart with the available actions.
``undecided``: the budget or step limit ran out first."""

StopReason = Literal[
    "decision_reached",
    "none_fits",
    "inconclusive",
    "budget_exhausted",
    "max_steps",
    "user_stopped",
    "error",
]


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: StopReason | None = None
    verdict: Verdict | None = None
    leader: str | None = None
    """Hypothesis id with the highest posterior at the time of the decision."""
    details: Mapping[str, Any] = field(default_factory=dict)
    """JSON-able numbers behind the decision (posterior, e-values, thresholds)."""

    @staticmethod
    def go_on(leader: str | None = None, **details: Any) -> StopDecision:
        return StopDecision(stop=False, leader=leader, details=details)
