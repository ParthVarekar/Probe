"""Selection policies P1-P10 (handover §12 H.5, H.7, H.12, Part I).

Every policy chooses one of the feasible candidates of
:class:`~probe.select.base.ScoredCandidates`, the table computed once per step by
:func:`~probe.select.scoring.score_candidates` and shared by all policies. A
:class:`~probe.select.base.Decision` records the policy's own score per candidate,
the chosen index, the exact propensity P(choose this action | context), the tie
set, the mode, the hash of the generator state before and after, a reason string
built only from the numbers, and policy-specific ``details``.

========  ===========================  =====================================================
id        name                         rule
========  ===========================  =====================================================
P1        random                       uniform over the feasible set; propensity 1/n
P2        fixed grid                   pre-planned stratum round-robin schedule (reset);
                                       nearest feasible when the planned action is not
P3        stratified                   least-sampled stratum (ties uniform), uniform within
P4        max predictive variance      argmax Var of the model-averaged predictive
P5        parameter-BALD               argmax Σ_m π_m U_θ|m
P6        model-indicator EIG          argmax U_M (unguarded, cost-blind)
P7        EIG / cost                   argmax U_M / c
P8        guarded EIG / cost           ε: audit design; 1 − ε: argmax U / c (claimed policy)
P9        static maximin design        pre-planned maximin-JS design (reset), in order
P10       oracle                       argmax G / c, G = expected log-odds gain of the truth
========  ===========================  =====================================================

All argmax choices use :func:`~probe.select.base.argmax_with_ties` with the
protocol's δ (``tie_rel_tol``): every candidate within δ of the best is in the tie
set and the smallest :func:`~probe.select.base.stable_key` wins, so decisions are
immune to ulp-level platform differences.

**Randomness.** Only P1, P3 and P8 draw from the generator the run loop passes
(keyed with the policy id, so world noise stays common across policies). P1 makes
one ``integers`` draw; P3 and the audit design make two (stratum, then member);
P8 first makes one ``random`` draw for the branch, skipped when ε is 0 or 1, so
P8 with ε = 0 consumes nothing (exactly P7) and with ε = 1 consumes exactly what
the audit design does.

**Guarded policy P8** (the claimed policy, H.5 and H.7 fallback (a)). With
probability ε the action comes from the audit design (stratified over the
declared population, :func:`~probe.select.designs.audit_draw`, inclusion
probability q(d)); otherwise from argmax U(d)/c(d) with U = U_M + λ·Σ_m π_m U_θ|m
(ε = ``protocol.audit_eps``, λ = ``protocol.lambda_param``). While the calibrated
misspecification alarm is raised (``ctx.alarm``) ε is raised to
``alarm_audit_eps`` (default 1: collection becomes the audit design). The logged
propensity is the exact mixture probability

    P(d) = ε·q(d) + (1 − ε)·1[d = d*],   d* = the model branch's argmax,

whichever branch produced d; ``mode`` records the branch ("audit" or "model").

Scores per policy (``Decision.scores``): P1/P3 the selection probabilities;
P2/P9 minus the record distance to the planned action; P4 the mixture variance;
P5 u_param; P6 U_M; P7 U_M/c; P8 U/c; P10 G/c.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from probe.determinism import rng_state_hash
from probe.run.protocol import RunProtocol
from probe.select.base import (
    Decision,
    DecisionContext,
    DecisionMode,
    PolicyInit,
    argmax_with_ties,
)
from probe.select.designs import (
    MaximinPlan,
    RecordMetric,
    audit_draw,
    audit_probabilities,
    fixed_grid_schedule,
    least_sampled_draw,
    maximin_plan,
)
from probe.select.oracle import truth_log_odds_gain
from probe.truth import EvaluatorKey, SealedTruth

__all__ = [
    "POLICY_VERSION",
    "EIGPerCostPolicy",
    "FixedGridPolicy",
    "GuardedEIGPolicy",
    "MaxPredictiveVariancePolicy",
    "ModelEIGPolicy",
    "NoCandidatesError",
    "OraclePolicy",
    "ParameterBALDPolicy",
    "PolicyBase",
    "RandomPolicy",
    "StaticMaximinPolicy",
    "StratifiedPolicy",
]

POLICY_VERSION = "0.1.0"
_PROPENSITY_SLACK = 1e-12


class NoCandidatesError(ValueError):
    """A policy was asked to choose from an empty feasible set (the run loop stops
    before that happens)."""


@dataclass(frozen=True)
class _Choice:
    index: int
    scores: np.ndarray
    propensity: float
    mode: DecisionMode
    reason: str
    tie_set: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- formatting


def _g(x: float) -> str:
    """Three significant digits; NaN and ±inf spelled out."""
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return f"{x:.3g}"


def _runner_up(scores: np.ndarray, idx: int, ids: Sequence[str], fmt: str = "{}") -> str:
    """'next best <id> at <score>' from the numbers (or that there is none)."""
    s = np.asarray(scores, dtype=np.float64).copy()
    s[idx] = np.nan
    finite = np.isfinite(s) | np.isposinf(s)
    if not finite.any():
        return "no other candidate"
    j = int(np.flatnonzero(finite)[np.argmax(s[finite])])
    return f"next best {ids[j]} at {fmt.format(_g(s[j]))}"


def _tie_text(ties: Sequence[str], rel_tol: float) -> str:
    if len(ties) <= 1:
        return ""
    return f"; {len(ties)} candidates tied within δ = {rel_tol:g}, broken by stable key"


def _read_only(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=np.float64)
    out.setflags(write=False)
    return out


# --------------------------------------------------------------------------- base


class PolicyBase:
    """Shared plumbing: reset bookkeeping, generator hashes, validation and the
    :class:`Decision` record. Subclasses implement :meth:`_choose`."""

    id: ClassVar[str] = ""
    name: ClassVar[str] = ""
    version: str = POLICY_VERSION
    _needs_reset: ClassVar[bool] = False

    def __init__(self) -> None:
        self._init: PolicyInit | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, config={self.config()!r})"

    # -- contract -----------------------------------------------------------
    def reset(self, init: PolicyInit) -> None:
        self._init = init
        self._on_reset(init)

    def decide(self, ctx: DecisionContext, rng: np.random.Generator) -> Decision:
        scored = ctx.scored
        if scored.n == 0:
            raise NoCandidatesError(f"{self.id}: no feasible candidate to choose from")
        if self._needs_reset and self._init is None:
            raise RuntimeError(f"{self.id}: reset() must be called before decide()")
        before = rng_state_hash(rng)
        ch = self._choose(ctx, rng)
        after = rng_state_hash(rng)
        if not 0 <= ch.index < scored.n:
            raise AssertionError(f"{self.id}: chosen index {ch.index} out of range")
        if not 0.0 < ch.propensity <= 1.0 + _PROPENSITY_SLACK:
            raise AssertionError(f"{self.id}: propensity {ch.propensity!r} outside (0, 1]")
        scores = _read_only(ch.scores)
        if scores.shape != (scored.n,):
            raise AssertionError(f"{self.id}: scores must align with the candidates")
        ids = scored.action_ids()
        return Decision(
            step=ctx.step,
            policy_id=self.id,
            policy_version=self.version,
            chosen_index=ch.index,
            action_id=ids[ch.index],
            score=float(scores[ch.index]),
            scores=scores,
            propensity=min(float(ch.propensity), 1.0),
            tie_set=tuple(ch.tie_set),
            mode=ch.mode,
            rng_state_before=before,
            rng_state_after=after,
            reason=ch.reason,
            details=dict(ch.details),
        )

    # -- extension points ---------------------------------------------------
    def needs_u_param(self, protocol: RunProtocol) -> bool:
        """Whether this policy reads ``ScoredCandidates.u_param`` under ``protocol``
        (see :func:`probe.select.scoring.requires_u_param`)."""
        return False

    def config(self) -> dict[str, Any]:
        """JSON-able constructor settings, for the run record."""
        return {}

    def _on_reset(self, init: PolicyInit) -> None:
        return None

    def _choose(self, ctx: DecisionContext, rng: np.random.Generator) -> _Choice:
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------
    @property
    def _cost_unit(self) -> str:
        return self._init.metadata.cost_unit if self._init is not None else "cost unit"

    @property
    def _outcome_unit(self) -> str:
        unit = self._init.metadata.outcome.unit if self._init is not None else None
        return f"{unit}²" if unit else "(outcome units)²"


# --------------------------------------------------------------------------- P1, P3


class RandomPolicy(PolicyBase):
    """P1: uniform over the feasible candidates."""

    id = "P1"
    name = "random"

    def _choose(self, ctx: DecisionContext, rng: np.random.Generator) -> _Choice:
        n = ctx.scored.n
        idx = int(rng.integers(n))
        p = 1.0 / n
        return _Choice(
            index=idx,
            scores=np.full(n, p),
            propensity=p,
            mode="random",
            reason=f"uniform random draw among {n} feasible candidates (p = 1/{n} = {_g(p)})",
        )


class StratifiedPolicy(PolicyBase):
    """P3: least-sampled stratum first (ties uniform), uniform within it."""

    id = "P3"
    name = "stratified"

    def _choose(self, ctx: DecisionContext, rng: np.random.Generator) -> _Choice:
        draw, least = least_sampled_draw(ctx.scored, ctx.stratum_counts, rng)
        p = float(draw.probs[draw.index])
        return _Choice(
            index=draw.index,
            scores=draw.probs,
            propensity=p,
            mode="random",
            reason=(
                f"least-sampled stratum ({least} obs; {draw.n_strata} strata tied) "
                f"{draw.stratum} drawn uniformly, then uniformly among its "
                f"{draw.n_in_stratum} feasible candidates "
                f"(p = 1/{draw.n_strata} × 1/{draw.n_in_stratum} = {_g(p)})"
            ),
            details={
                "stratum": draw.stratum,
                "n_strata_tied": draw.n_strata,
                "n_in_stratum": draw.n_in_stratum,
                "min_count": least,
            },
        )


# --------------------------------------------------------------------------- P4-P7


class _ArgmaxPolicy(PolicyBase):
    """Deterministic argmax of a per-candidate score (ties: stable key)."""

    def _scores(self, ctx: DecisionContext) -> np.ndarray:
        raise NotImplementedError

    def _describe(self, ctx: DecisionContext, idx: int, scores: np.ndarray) -> str:
        raise NotImplementedError

    def _choose(self, ctx: DecisionContext, rng: np.random.Generator) -> _Choice:
        scores = self._scores(ctx)
        # +inf (an infinite variance) must be able to win: rank it as the largest float
        ranked = np.where(np.isposinf(scores), np.finfo(np.float64).max, scores)
        tol = ctx.protocol.tie_rel_tol
        ids = ctx.scored.action_ids()
        idx, ties = argmax_with_ties(ranked, ids, tol)
        reason = self._describe(ctx, idx, scores) + _tie_text(ties, tol)
        return _Choice(
            index=idx, scores=scores, propensity=1.0, mode="model", reason=reason, tie_set=ties
        )


class MaxPredictiveVariancePolicy(_ArgmaxPolicy):
    """P4: argmax variance of the model-averaged predictive."""

    id = "P4"
    name = "max predictive variance"

    def _scores(self, ctx: DecisionContext) -> np.ndarray:
        return np.array(ctx.scored.pred_var, dtype=np.float64)

    def _describe(self, ctx: DecisionContext, idx: int, scores: np.ndarray) -> str:
        ids = ctx.scored.action_ids()
        u = self._outcome_unit
        return (
            f"highest model-averaged predictive variance: {_g(scores[idx])} {u} at {ids[idx]}; "
            + _runner_up(scores, idx, ids, "{} " + u)
        )


class ParameterBALDPolicy(_ArgmaxPolicy):
    """P5: argmax posterior-weighted parameter EIG Σ_m π_m U_θ|m."""

    id = "P5"
    name = "parameter-BALD"

    def needs_u_param(self, protocol: RunProtocol) -> bool:
        return True

    def _scores(self, ctx: DecisionContext) -> np.ndarray:
        if not ctx.scored.u_param_available:
            raise ValueError("P5 needs u_param: score with need_u_param=True")
        return np.array(ctx.scored.u_param, dtype=np.float64)

    def _describe(self, ctx: DecisionContext, idx: int, scores: np.ndarray) -> str:
        ids = ctx.scored.action_ids()
        return (
            f"highest parameter EIG Σπ·U_θ|m: {_g(scores[idx])} nats at {ids[idx]}; "
            + _runner_up(scores, idx, ids, "{} nats")
        )


class ModelEIGPolicy(_ArgmaxPolicy):
    """P6: argmax U_M (unguarded and cost-blind)."""

    id = "P6"
    name = "model-indicator EIG (unguarded)"

    def _scores(self, ctx: DecisionContext) -> np.ndarray:
        return np.array(ctx.scored.u_model, dtype=np.float64)

    def _describe(self, ctx: DecisionContext, idx: int, scores: np.ndarray) -> str:
        sc = ctx.scored
        ids = sc.action_ids()
        return (
            f"highest U_M: {_g(scores[idx])} ± {_g(sc.u_model_err[idx])} nats "
            f"(H(π) = {_g(sc.u_model_upper)}) at {ids[idx]}; "
            + _runner_up(scores, idx, ids, "{} nats")
        )


class EIGPerCostPolicy(_ArgmaxPolicy):
    """P7: argmax U_M / cost."""

    id = "P7"
    name = "EIG/cost"

    def _scores(self, ctx: DecisionContext) -> np.ndarray:
        return np.asarray(ctx.scored.u_model / ctx.scored.cost, dtype=np.float64)

    def _describe(self, ctx: DecisionContext, idx: int, scores: np.ndarray) -> str:
        sc = ctx.scored
        ids = sc.action_ids()
        unit = self._cost_unit
        return (
            f"highest U_M/cost: {_g(sc.u_model[idx])} nats / {_g(sc.cost[idx])} {unit} = "
            f"{_g(scores[idx])} nats/{unit} at {ids[idx]}; "
            + _runner_up(scores, idx, ids, "{} nats/" + unit)
        )


# --------------------------------------------------------------------------- P8


class GuardedEIGPolicy(PolicyBase):
    """P8: ε-mixture of the audit design and argmax U/cost (module docstring)."""

    id = "P8"
    name = "guarded EIG/cost"

    def __init__(self, alarm_audit_eps: float = 1.0) -> None:
        super().__init__()
        if not 0.0 <= alarm_audit_eps <= 1.0:
            raise ValueError("alarm_audit_eps must lie in [0, 1]")
        self.alarm_audit_eps = float(alarm_audit_eps)

    def config(self) -> dict[str, Any]:
        return {"alarm_audit_eps": self.alarm_audit_eps}

    def needs_u_param(self, protocol: RunProtocol) -> bool:
        return protocol.lambda_param > 0.0

    def effective_eps(self, protocol: RunProtocol, alarm: bool) -> float:
        """ε in force: the protocol's audit fraction, raised to ``alarm_audit_eps``
        while the misspecification alarm is up (never lowered)."""
        eps = float(protocol.audit_eps)
        return max(eps, self.alarm_audit_eps) if alarm else eps

    def _choose(self, ctx: DecisionContext, rng: np.random.Generator) -> _Choice:
        sc = ctx.scored
        proto = ctx.protocol
        lam = float(proto.lambda_param)
        eps = self.effective_eps(proto, ctx.alarm)
        u = sc.objective(lam)
        scores = np.asarray(u / sc.cost, dtype=np.float64)
        ids = sc.action_ids()
        star, ties = argmax_with_ties(scores, ids, proto.tie_rel_tol)
        q = audit_probabilities(sc)
        branch_u: float | None = None
        if eps >= 1.0:
            audit = True
        elif eps <= 0.0:
            audit = False
        else:
            branch_u = float(rng.random())
            audit = branch_u < eps
        unit = self._cost_unit
        u_name = "U_M" if lam == 0.0 else f"U_M + {_g(lam)}·U_θ"
        model_text = (
            f"{u_name}/cost {_g(u[star])} nats / {_g(sc.cost[star])} {unit} = "
            f"{_g(scores[star])} nats/{unit} at {ids[star]}"
        )
        guard = f"ε = {_g(eps)}" + (" (alarm raised)" if ctx.alarm else "")
        if branch_u is not None:
            guard += f", branch draw {_g(branch_u)} {'<' if audit else '≥'} ε"
        if audit:
            draw = audit_draw(sc, rng)
            idx = draw.index
            tie_set: tuple[str, ...] = ()
            mode: DecisionMode = "audit"
            what = (
                f"audit design: stratum {draw.stratum} uniformly from {draw.n_strata} feasible "
                f"strata, then uniformly among its {draw.n_in_stratum} candidates "
                f"(q = {_g(q[idx])}); the model branch would take {model_text}"
            )
        else:
            idx, tie_set, mode = star, ties, "model"
            what = (
                f"model branch: highest {model_text}; "
                + _runner_up(scores, star, ids, "{} nats/" + unit)
                + _tie_text(ties, proto.tie_rel_tol)
            )
        model_prob = 1.0 if idx == star else 0.0
        propensity = eps * float(q[idx]) + (1.0 - eps) * model_prob
        reason = (
            f"guarded ({guard}) → {what}; propensity ε·q + (1 − ε)·1[d = d*] = {_g(propensity)}"
        )
        return _Choice(
            index=idx,
            scores=scores,
            propensity=propensity,
            mode=mode,
            reason=reason,
            tie_set=tie_set,
            details={
                "eps": eps,
                "eps_protocol": float(proto.audit_eps),
                "alarm": bool(ctx.alarm),
                "lambda": lam,
                "branch_draw": branch_u,
                "audit_prob": float(q[idx]),
                "model_prob": model_prob,
                "model_choice": ids[star],
                "model_tie_set": list(ties),
            },
        )


# --------------------------------------------------------------------------- P2, P9


class _ScheduledPolicy(PolicyBase):
    """Executes a fixed sequence of action ids, cycling, with nearest-feasible
    fallback. One :meth:`decide` per executed step advances the schedule."""

    _needs_reset = True
    _label: ClassVar[str] = "schedule"

    def __init__(self) -> None:
        super().__init__()
        self._schedule: tuple[str, ...] = ()
        self._targets: dict[str, tuple[Mapping[str, Any], str]] = {}
        self._metric: RecordMetric | None = None
        self._k = 0

    def _set_schedule(self, init: PolicyInit, schedule: Sequence[str]) -> None:
        cands = [c for c in init.initial_candidates if c.feasible] or list(init.initial_candidates)
        if not schedule:
            raise ValueError(f"{self.id}: empty schedule")
        self._schedule = tuple(schedule)
        self._targets = {c.action.id: (c.record, c.stratum) for c in cands}
        self._metric = RecordMetric.fit(init.metadata.covariates, [c.record for c in cands])
        self._k = 0

    @property
    def schedule(self) -> tuple[str, ...]:
        return self._schedule

    def _slot_text(self, slot: int) -> str:
        return f"slot {slot % len(self._schedule) + 1}/{len(self._schedule)}"

    def _choose(self, ctx: DecisionContext, rng: np.random.Generator) -> _Choice:
        assert self._metric is not None
        slot = self._k
        self._k += 1
        target = self._schedule[slot % len(self._schedule)]
        record, stratum = self._targets[target]
        sc = ctx.scored
        ids = sc.action_ids()
        near = self._metric.nearness(record, stratum, sc)
        cycle = slot // len(self._schedule)
        head = f"{self._label} {self._slot_text(slot)} (cycle {cycle + 1}): planned {target}"
        details: dict[str, Any] = {"slot": slot, "planned": target, "cycle": cycle}
        if target in ids:
            idx = ids.index(target)
            return _Choice(
                index=idx,
                scores=near,
                propensity=1.0,
                mode="schedule",
                reason=f"{head}, feasible: taken",
                tie_set=(target,),
                details={**details, "fallback": False},
            )
        idx, ties = argmax_with_ties(near, ids, ctx.protocol.tie_rel_tol)
        return _Choice(
            index=idx,
            scores=near,
            propensity=1.0,
            mode="schedule",
            reason=(
                f"{head}, not feasible now; nearest feasible {ids[idx]} "
                f"(record distance {_g(-near[idx])})" + _tie_text(ties, ctx.protocol.tie_rel_tol)
            ),
            tie_set=ties,
            details={**details, "fallback": True, "distance": float(-near[idx])},
        )


class FixedGridPolicy(_ScheduledPolicy):
    """P2: stratum round-robin schedule planned at reset."""

    id = "P2"
    name = "fixed grid"
    _label = "fixed grid"

    def _on_reset(self, init: PolicyInit) -> None:
        self._set_schedule(init, fixed_grid_schedule(init.initial_candidates))


class StaticMaximinPolicy(_ScheduledPolicy):
    """P9: static maximin-JS design planned at reset (H.12 tiny budgets)."""

    id = "P9"
    name = "static maximin design"
    _label = "static maximin plan"

    def __init__(self, max_plan_length: int = 16) -> None:
        super().__init__()
        if max_plan_length < 1:
            raise ValueError("max_plan_length must be >= 1")
        self.max_plan_length = int(max_plan_length)
        self._plan: MaximinPlan | None = None

    def config(self) -> dict[str, Any]:
        return {"max_plan_length": self.max_plan_length}

    @property
    def plan(self) -> MaximinPlan | None:
        return self._plan

    def _on_reset(self, init: PolicyInit) -> None:
        proto = init.protocol
        plan = maximin_plan(
            init.hypothesis_set,
            init.initial_candidates,
            budget=proto.budget,
            max_length=min(self.max_plan_length, proto.max_steps),
            rel_tol=proto.tie_rel_tol,
        )
        self._plan = plan
        # nothing affordable at reset: fall back to the fixed-grid sweep
        schedule = plan.action_ids or fixed_grid_schedule(init.initial_candidates)
        self._set_schedule(init, schedule)

    def _choose(self, ctx: DecisionContext, rng: np.random.Generator) -> _Choice:
        ch = super()._choose(ctx, rng)
        plan = self._plan
        slot = int(ch.details["slot"])
        if plan is None or not plan.action_ids:
            return ch
        j = slot % len(plan)
        a, b = plan.weakest_pair[j]
        value = _g(plan.min_divergence[j])
        if plan.divergence == "gaussian_joint_kl":
            what = f"weakest ordered pair KL({a} ‖ {b}) of the joint design = {value} nats"
        else:
            what = f"weakest pair {a}–{b} at cumulative JS {value} nats"
        extra = f"; plan value after this slot: {what}"
        return _Choice(
            index=ch.index,
            scores=ch.scores,
            propensity=ch.propensity,
            mode=ch.mode,
            reason=ch.reason + extra,
            tie_set=ch.tie_set,
            details={
                **ch.details,
                "plan_divergence": plan.divergence,
                "plan_min_divergence": plan.min_divergence[j],
            },
        )


# --------------------------------------------------------------------------- P10


class OraclePolicy(PolicyBase):
    """P10: knows the true hypothesis (evaluator code only) and maximises the
    expected log-odds gain of the truth per unit cost
    (:func:`probe.select.oracle.truth_log_odds_gain`).

    Build it from an opened truth, ``OraclePolicy(truth.open(key))`` or
    :meth:`from_sealed`; out-of-set truths (``true_hypothesis_id`` None) target the
    residual hypothesis. The reason string never names the target."""

    id = "P10"
    name = "oracle"
    _needs_reset = True

    def __init__(self, opened_truth: Mapping[str, Any]) -> None:
        super().__init__()
        if not isinstance(opened_truth, Mapping) or "true_hypothesis_id" not in opened_truth:
            raise TypeError("OraclePolicy needs an opened truth (SealedTruth.open(key))")
        tid = opened_truth["true_hypothesis_id"]
        if tid is not None and not isinstance(tid, str):
            raise TypeError("true_hypothesis_id must be a string or None")
        self.__true_id: str | None = tid
        self._target: int | None = None

    @classmethod
    def from_sealed(cls, truth: SealedTruth, key: EvaluatorKey) -> OraclePolicy:
        opened = truth.open(key)
        if not truth.verify(opened):
            raise ValueError("opened truth does not match its commitment")
        return cls(opened)

    def __repr__(self) -> str:  # never print the truth
        return "OraclePolicy(id='P10')"

    def _on_reset(self, init: PolicyInit) -> None:
        hs = init.hypothesis_set
        target = self.__true_id if self.__true_id is not None else hs.residual.id
        hids = [h.id for h in hs.hypotheses]
        if target not in hids:
            raise ValueError("the true hypothesis is not registered in this hypothesis set")
        self._target = hids.index(target)

    def _choose(self, ctx: DecisionContext, rng: np.random.Generator) -> _Choice:
        assert self._target is not None
        sc = ctx.scored
        gain = truth_log_odds_gain(sc.predictives, ctx.weights, self._target)
        scores = np.asarray(gain / sc.cost, dtype=np.float64)
        ids = sc.action_ids()
        tol = ctx.protocol.tie_rel_tol
        idx, ties = argmax_with_ties(scores, ids, tol)
        unit = self._cost_unit
        reason = (
            f"oracle: highest expected log-odds gain of the true hypothesis per cost: "
            f"{_g(gain[idx])} nats / {_g(sc.cost[idx])} {unit} = {_g(scores[idx])} "
            f"nats/{unit} at {ids[idx]}; "
            + _runner_up(scores, idx, ids, "{} nats/" + unit)
            + _tie_text(ties, tol)
        )
        return _Choice(
            index=idx,
            scores=scores,
            propensity=1.0,
            mode="oracle",
            reason=reason,
            tie_set=ties,
            details={"gain": float(gain[idx])},
        )
