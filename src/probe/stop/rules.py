"""Pre-declared stopping rules (handover §12 H.6).

:class:`StoppingRule` turns the protocol (:class:`~probe.run.protocol.RunProtocol`)
and a snapshot of the run (:class:`StopState`) into a
:class:`~probe.stop.base.StopDecision`. It is evaluated *identically for every
policy* (the run loop builds the state from engine quantities, never from a
policy's own scores), before each selection step and after the last update.

The rules are checked in the H.6 order and the **first** that holds decides:

1. **Decision.** Let m̂ be the posterior leader.

   * m̂ is a registered (non-residual) hypothesis with π_m̂ ≥ π*, and, when
     ``protocol.require_eprocess``, the universal-inference e-process
     E_t(m̂ : m′) ≥ 1/α for every rival m′ whose model class does not contain
     m̂'s (``StopState.nests``; a nesting rival also contains the truth when m̂
     does, so that e-process cannot grow; see :mod:`probe.stop.eprocess`)
     → verdict ``hypothesis``, reason ``decision_reached``.
   * The residual can never be "the decision": if the residual leads with
     π ≥ π* (after ``min_obs_before_none_fits`` observations) → verdict
     ``none_fits``, reason ``none_fits``.

2. **Budget exhausted.** No budget left, no feasible (affordable) action, or the
   environment is exhausted → ``undecided`` / ``budget_exhausted``; the step
   limit ``max_steps`` reached → ``undecided`` / ``max_steps``.
3. **Nothing informative left.** After ``min_obs_before_inconclusive``
   observations, with no hypothesis at π*, the largest model-indicator
   information rate over the feasible candidates, max_d U_M(d)/c(d), is below
   ``eps_stop`` → ``inconclusive`` (the hypotheses cannot be told apart with the
   available actions; report it with the identifiability analysis).
4. **Inadequacy.** After ``min_obs_before_none_fits`` observations the residual
   leads the posterior, or a calibrated alarm has fired persistently (the
   :class:`~probe.diagnostics.alarms.AlarmMonitor` applies the persistence) →
   ``none_fits``.
5. **User or safety stop** → ``undecided`` / ``user_stopped``.

Otherwise the run goes on. Every decision carries, in ``details``, the numbers
behind it and the outcome of every check, as canonical-JSON-able values
(non-finite e-values as strings, see :func:`~probe.stop.eprocess.json_float`).

Determinism: the leader is the largest weight; weights within the protocol's
relative tie tolerance of the maximum tie, and a tie is resolved to the first
non-residual hypothesis in hypothesis-set order (the residual "leads" only when
it is strictly ahead beyond the tolerance), so ulp-level differences in the
posterior cannot flip a verdict between platforms except exactly at a threshold.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from probe.run.protocol import RunProtocol
from probe.stop.base import StopDecision, StopReason, Verdict
from probe.stop.eprocess import json_float

__all__ = ["StopState", "StoppingRule", "information_rate"]

_WEIGHT_SUM_TOL = 1e-9


def information_rate(
    u_model: np.ndarray | Sequence[float],
    cost: np.ndarray | Sequence[float],
    u_err: np.ndarray | Sequence[float] | None = None,
) -> float | None:
    """max_d (U_M(d) + err(d)) / c(d) over the feasible candidates, in nats per cost
    unit (``StopState.max_score``); ``None`` when there are no candidates.

    The numerical error estimate of U_M is *added*, so "nothing informative left"
    is only concluded when even the upper bound of the information rate is below
    ``eps_stop`` (the conservative side). Pass the arrays of
    :class:`~probe.select.base.ScoredCandidates` (``u_model``, ``cost``,
    ``u_model_err``)."""
    u = np.asarray(u_model, dtype=np.float64)
    c = np.asarray(cost, dtype=np.float64)
    if u.shape != c.shape or u.ndim != 1:
        raise ValueError("u_model and cost must be 1-D arrays of equal length")
    if u.size == 0:
        return None
    if u_err is not None:
        e = np.asarray(u_err, dtype=np.float64)
        if e.shape != u.shape:
            raise ValueError("u_err must align with u_model")
        u = u + np.maximum(e, 0.0)
    if np.any(~np.isfinite(u)) or np.any(u < 0.0):
        raise ValueError("information values must be finite and non-negative")
    if np.any(~np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("costs must be finite and positive")
    return float(np.max(u / c))


@dataclass(frozen=True)
class StopState:
    """What the stopping rule sees at one point of the run (built by the run loop)."""

    step: int
    """Executed actions so far (warm-up included); the index of the next decision."""
    hypothesis_ids: tuple[str, ...]
    weights: np.ndarray
    """Current (possibly tempered) posterior over hypotheses, set order."""
    residual_id: str | None
    """Id of the residual hypothesis (``None`` if the set has none)."""
    n_obs: int
    """Non-missing observations so far."""
    budget_remaining: float
    feasible_count: int
    """Feasible candidates for the next step that fit in the remaining budget."""
    max_score: float | None
    """:func:`information_rate` over those candidates (``None`` if there are none)."""
    log_e: np.ndarray | None = None
    """(K, K) log E_t(i : j) (:meth:`EProcessTracker.log_e`), ``None`` if untracked."""
    nests: np.ndarray | None = None
    """(K, K) bool, ``nests[i, j]`` when hypothesis j's model class contains i's
    (:func:`probe.diagnostics.identifiability.nesting_matrix`); ``None`` = no
    nesting known, every rival needs its e-process."""
    alarm: bool = False
    """A calibrated misspecification alarm is raised (persistence already applied)."""
    exhausted: bool = False
    """The environment can offer no further action."""
    user_stop: bool = False
    """A user or safety stop was requested."""

    def __post_init__(self) -> None:
        ids = tuple(self.hypothesis_ids)
        k = len(ids)
        if k == 0 or len(set(ids)) != k:
            raise ValueError("hypothesis ids must be non-empty and unique")
        w = np.asarray(self.weights, dtype=np.float64)
        if w.shape != (k,) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError(f"weights must be {k} finite non-negative numbers")
        if abs(float(w.sum()) - 1.0) > _WEIGHT_SUM_TOL:
            raise ValueError("weights must sum to 1")
        if self.residual_id is not None and self.residual_id not in ids:
            raise ValueError(f"residual {self.residual_id!r} is not a hypothesis")
        if self.log_e is not None:
            le = np.asarray(self.log_e, dtype=np.float64)
            if le.shape != (k, k) or np.any(np.isnan(le)):
                raise ValueError(f"log_e must be a ({k}, {k}) matrix without NaN")
        if self.nests is not None and np.asarray(self.nests).shape != (k, k):
            raise ValueError(f"nests must be a ({k}, {k}) matrix")
        if self.step < 0 or self.n_obs < 0 or self.feasible_count < 0:
            raise ValueError("step, n_obs and feasible_count must be non-negative")
        if math.isnan(self.budget_remaining):
            raise ValueError("budget_remaining is NaN")
        if self.max_score is not None and not (
            math.isfinite(self.max_score) and self.max_score >= 0.0
        ):
            raise ValueError("max_score must be finite and non-negative (or None)")


class StoppingRule:
    """The H.6 stopping rules for one protocol (see the module docstring)."""

    def __init__(self, protocol: RunProtocol) -> None:
        self.protocol = protocol
        self.log_e_threshold = -math.log(protocol.alpha)
        """log(1/α): an e-process at or above it rejects the rival."""

    # -- helpers -------------------------------------------------------------
    def _leader(self, state: StopState) -> tuple[int, bool]:
        """(leader index, residual strictly leads)."""
        w = np.asarray(state.weights, dtype=np.float64)
        best = float(np.max(w))
        tol = self.protocol.tie_rel_tol * max(best, 1e-300)
        ties = [i for i in range(w.size) if w[i] >= best - tol]
        res = state.residual_id
        non_res = [i for i in ties if state.hypothesis_ids[i] != res]
        if non_res:
            return non_res[0], False
        return ties[0], True  # only the residual is (tied for) the maximum

    def _eprocess_check(
        self, state: StopState, leader: int
    ) -> tuple[bool | None, dict[str, float | str], list[str]]:
        """(all required e-processes ≥ 1/α, log E per required rival, exempt rivals).
        ``None`` when the e-processes are required but not tracked."""
        ids = state.hypothesis_ids
        nests = None if state.nests is None else np.asarray(state.nests, dtype=bool)
        exempt = [
            ids[j]
            for j in range(len(ids))
            if j != leader and nests is not None and bool(nests[leader, j])
        ]
        required = [j for j in range(len(ids)) if j != leader and ids[j] not in exempt]
        if state.log_e is None:
            return None, {}, exempt
        le = np.asarray(state.log_e, dtype=np.float64)
        values = {ids[j]: json_float(float(le[leader, j])) for j in required}
        passed = all(float(le[leader, j]) >= self.log_e_threshold for j in required)
        return passed, values, exempt

    # -- the rule ------------------------------------------------------------
    def evaluate(self, state: StopState) -> StopDecision:
        p = self.protocol
        ids = state.hypothesis_ids
        w = np.asarray(state.weights, dtype=np.float64)
        leader, residual_leads = self._leader(state)
        leader_id = ids[leader]
        leader_w = float(w[leader])
        is_res = leader_id == state.residual_id
        res_w = float(w[ids.index(state.residual_id)]) if state.residual_id is not None else 0.0
        at_star = leader_w >= p.pi_star
        none_fits_ready = state.n_obs >= p.min_obs_before_none_fits

        e_passed: bool | None = None
        e_values: dict[str, float | str] = {}
        e_exempt: list[str] = []
        if not is_res:
            e_passed, e_values, e_exempt = self._eprocess_check(state, leader)

        # 1. decision
        decision_ok = at_star and not is_res and (not p.require_eprocess or bool(e_passed))
        residual_decision = at_star and is_res and none_fits_ready
        # 2. budget / steps
        no_action = state.budget_remaining <= 0.0 or state.feasible_count == 0 or state.exhausted
        max_steps = state.step >= p.max_steps
        # 3. nothing informative left
        inconclusive = (
            state.n_obs >= p.min_obs_before_inconclusive
            and not at_star
            and state.max_score is not None
            and state.max_score < p.eps_stop
        )
        # 4. inadequacy
        inadequate = none_fits_ready and (residual_leads or state.alarm)

        details: dict[str, Any] = {
            "step": state.step,
            "n_obs": state.n_obs,
            "posterior": {h: float(v) for h, v in zip(ids, w, strict=True)},
            "leader": leader_id,
            "leader_weight": leader_w,
            "pi_star": p.pi_star,
            "residual_id": state.residual_id,
            "residual_weight": res_w,
            "residual_leads": residual_leads,
            "alpha": p.alpha,
            "require_eprocess": p.require_eprocess,
            "log_e_threshold": self.log_e_threshold,
            "eprocess_tracked": state.log_e is not None,
            "log_e_vs_rivals": e_values,
            "eprocess_exempt_nesting": e_exempt,
            "eprocess_passed": e_passed,
            "max_score": state.max_score,
            "eps_stop": p.eps_stop,
            "budget_remaining": json_float(state.budget_remaining),
            "feasible_count": state.feasible_count,
            "exhausted": state.exhausted,
            "max_steps": p.max_steps,
            "min_obs_before_inconclusive": p.min_obs_before_inconclusive,
            "min_obs_before_none_fits": p.min_obs_before_none_fits,
            "alarm": state.alarm,
            "user_stop": state.user_stop,
            "checks": {
                "decision": decision_ok,
                "residual_decision": residual_decision,
                "no_action": no_action,
                "max_steps": max_steps,
                "inconclusive": inconclusive,
                "inadequate": inadequate,
                "user_stop": state.user_stop,
            },
        }

        def stop(reason: StopReason, verdict: Verdict, rule: str) -> StopDecision:
            return StopDecision(
                stop=True,
                reason=reason,
                verdict=verdict,
                leader=leader_id,
                details={**details, "rule": rule},
            )

        if decision_ok:
            return stop("decision_reached", "hypothesis", "1_decision")
        if residual_decision:
            return stop("none_fits", "none_fits", "1_residual_at_pi_star")
        if no_action:
            return stop("budget_exhausted", "undecided", "2_budget_exhausted")
        if max_steps:
            return stop("max_steps", "undecided", "2_max_steps")
        if inconclusive:
            return stop("inconclusive", "inconclusive", "3_nothing_informative")
        if inadequate:
            return stop(
                "none_fits",
                "none_fits",
                "4_alarm" if state.alarm and not residual_leads else "4_residual_leads",
            )
        if state.user_stop:
            return stop("user_stopped", "undecided", "5_user_stop")
        return StopDecision(stop=False, leader=leader_id, details={**details, "rule": None})
