"""Selection-policy contract (handover §12 H.5, Part I policies P1-P10).

Each step the run loop computes, *once and identically for every policy*, the
engine scores of every feasible candidate (:class:`ScoredCandidates`). A policy
only chooses among them. This keeps comparisons fair (equal information, equal
stopping rules) and makes every decision explainable: the full score table,
the propensity, the tie set and the RNG state go into the decision record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from probe.engine.ir import HypothesisSet
from probe.engine.predictive import Predictive
from probe.envs.base import Candidate, EnvMetadata
from probe.run.protocol import RunProtocol

__all__ = [
    "Decision",
    "DecisionContext",
    "DecisionMode",
    "Policy",
    "PolicyInit",
    "ScoredCandidates",
    "argmax_with_ties",
    "stable_key",
]

DecisionMode = Literal["model", "audit", "warmup", "schedule", "random", "oracle"]


@dataclass(frozen=True)
class ScoredCandidates:
    """Engine-computed quantities for the feasible candidates of one step, aligned by
    index. Candidates are sorted by ``action.id``."""

    candidates: tuple[Candidate, ...]
    cost: np.ndarray
    u_model: np.ndarray
    """Model-indicator EIG U_M(d), nats."""
    u_model_err: np.ndarray
    """Numerical error estimate of U_M (0 for discrete outcomes)."""
    u_param: np.ndarray
    """Posterior-weighted parameter EIG Σ_m π_m U_θ|m(d), nats."""
    pred_var: np.ndarray
    """Variance of the model-averaged predictive."""
    predictives: tuple[Predictive, ...]
    """One batch per hypothesis (hypothesis-set order), each over ``candidates``."""

    @property
    def n(self) -> int:
        return len(self.candidates)

    def action_ids(self) -> list[str]:
        return [c.action.id for c in self.candidates]


@dataclass(frozen=True)
class PolicyInit:
    """What a policy may know before step 0 (for pre-planned schedules such as the
    fixed grid P2 and the static maximin design P9)."""

    metadata: EnvMetadata
    hypothesis_set: HypothesisSet
    protocol: RunProtocol
    initial_candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class DecisionContext:
    step: int
    hypothesis_set: HypothesisSet
    weights: np.ndarray
    """Current posterior over hypotheses, hypothesis-set order."""
    scored: ScoredCandidates
    budget_remaining: float
    spent: float
    protocol: RunProtocol
    stratum_counts: Mapping[str, int]
    """Observations taken so far per stratum key (warm-up included)."""
    population: tuple[str, ...]
    alarm: bool
    """Calibrated misspecification alarm currently raised (H.7)."""
    history: tuple[str, ...]
    """Action ids executed so far, in order."""


@dataclass(frozen=True)
class Decision:
    step: int
    policy_id: str
    policy_version: str
    chosen_index: int
    """Index into ``DecisionContext.scored.candidates``."""
    action_id: str
    score: float
    scores: np.ndarray
    """The policy's own score per candidate (NaN where the policy has none)."""
    propensity: float
    """P(choose this action | context) under the policy's randomization (1 if deterministic)."""
    tie_set: tuple[str, ...]
    mode: DecisionMode
    rng_state_before: str
    rng_state_after: str
    reason: str
    """Human-readable "why this next", generated from the numbers, never free-form."""


class Policy(Protocol):
    id: str
    version: str

    def reset(self, init: PolicyInit) -> None:
        """Called once before step 0."""
        ...

    def decide(self, ctx: DecisionContext, rng: np.random.Generator) -> Decision: ...


def stable_key(action_id: str) -> str:
    """Platform-independent tie-break key for an action id."""
    return hashlib.sha256(action_id.encode("utf-8")).hexdigest()


def argmax_with_ties(
    scores: np.ndarray, action_ids: Sequence[str], rel_tol: float
) -> tuple[int, tuple[str, ...]]:
    """Deterministic argmax. Every candidate within ``rel_tol`` (relative to the
    max magnitude, with a tiny absolute floor) of the maximum is in the tie set; the
    winner is the tie-set member with the smallest :func:`stable_key`. NaN scores
    never win. This makes decisions identical across BLAS/OS differences smaller
    than the tolerance (handover §12 Part O risk 7)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.shape != (len(action_ids),):
        raise ValueError("scores and action_ids must align")
    finite = np.isfinite(s)
    if not finite.any():
        raise ValueError("no finite scores to choose from")
    best = float(np.max(s[finite]))
    tol = rel_tol * max(abs(best), 1e-300) + 1e-300
    ties = [i for i in range(len(s)) if finite[i] and s[i] >= best - tol]
    winner = min(ties, key=lambda i: stable_key(action_ids[i]))
    return winner, tuple(sorted(action_ids[i] for i in ties))
