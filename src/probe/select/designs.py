"""Non-adaptive designs used by the baseline and guarded policies (handover §12 H.5,
H.7, H.12, Part I).

Everything here is a deterministic function of its inputs and, for the randomized
designs, of the generator passed in. Every randomized draw comes with the exact
probability of every candidate, so propensities are known, not estimated.

**Stratum groups.** The feasible candidates of a step are grouped by
``Candidate.stratum`` (strata sorted by key, members in candidate order, i.e. by
action id). Only strata with at least one feasible candidate take part.

**Audit design** (the guarded policy's ε-branch, H.5/H.7): a stratum uniformly
among the S feasible strata, then a candidate uniformly within it, so
q(d) = 1 / (S · n_{s(d)}). Every feasible unit of the declared population has a
known, positive inclusion probability at every audit draw, which is what the
design-based (Horvitz–Thompson/IPW) estimates of H.7 need.

**Least-sampled stratified design** (P3): among the feasible strata, those with
the fewest observations so far (warm-up included) tie; one of them uniformly,
then a candidate uniformly within it: q(d) = 1 / (T · n_{s(d)}) on the T tied
strata, 0 elsewhere.

**Fixed-grid schedule** (P2): planned once from the initial candidates. Strata
are visited round-robin in key order; within a stratum, candidates follow a fixed
hash order (:func:`~probe.select.base.stable_key`), so successive sweeps spread
over sites/settings instead of repeating the first one. The schedule cycles.

**Nearest feasible** (P2 and P9 fallback): when a planned action is not feasible
(or not affordable) now, the feasible candidate closest to it in *design-record*
space is taken. The distance is a sum over the declared covariates: numeric
covariates contribute |a − b| / range (range over the initial candidates),
categorical and binary ones 1 when they differ; a different stratum adds 1.
Ties go to the smallest stable key.

**Static maximin design** (P9, H.12 "very small budgets"): planned once at reset
from the *prior* at the initial candidates' records, by greedy maximin over
hypothesis pairs of the divergence of the whole planned design:

* ``linear_gaussian`` sets: the exact KL between the moment-matched Gaussian
  *joint* prior predictives of all planned observations, for every *ordered*
  pair (see :class:`_GaussianJointKL`): the expected log Bayes factor for m
  against m′ when m is true. Being joint, it accounts for the correlation between
  planned observations through shared parameters, so redundant points add little
  and repeats are allowed. (Per-observation divergences fail here: with zero prior
  means, H2 "shade + surface" and H3 "shade × hour" have *identical* prior
  predictive variance at a tree/grass site, so their pointwise divergence is 0
  although two observations separate them.)
* ``finite_grid`` sets: the sum over planned observations of the Jensen–Shannon
  divergence JS_mm′(d) = I(M; Y_d) under a uniform prior on {m, m′}, computed by
  the exact U_M kernel (:func:`probe.engine.eig.model_indicator_eig`;
  0 ≤ JS ≤ log 2), unordered pairs, from the reset-time state and without
  repeats. Grid states evolve with the clock, which a static plan cannot project,
  so this is an independence approximation.

Each slot takes, among candidates whose reset-time cost fits the budget not yet
planned, the largest gain of the *minimum over pairs* per unit cost; ties (within
the protocol's δ, including the all-zero case of an unidentifiable pair) go to the
largest gain of the pairwise sum per unit cost, then the smallest stable key.
Stated limitations: records and costs are the reset-time ones (time-dependent
covariates such as the hour, and travel between planned points, are not
projected), and a static plan ignores the warm-up data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from probe.engine.eig import QuadratureSettings, model_indicator_eig
from probe.engine.factory import build_models
from probe.engine.ir import CovariateSpec, HypothesisSet, Record
from probe.engine.models import HypothesisModel
from probe.engine.nig import NIGModel
from probe.engine.predictive import DiscretePredictive, Predictive, StudentTPredictive
from probe.envs.base import Candidate
from probe.select.base import ScoredCandidates, argmax_with_ties, stable_key

__all__ = [
    "MaximinPlan",
    "RecordMetric",
    "StratumDraw",
    "audit_draw",
    "audit_probabilities",
    "fixed_grid_schedule",
    "least_sampled_draw",
    "least_sampled_probabilities",
    "lexicographic_argmax",
    "maximin_plan",
    "pairwise_js",
    "stratum_groups",
]


# --------------------------------------------------------------------------- strata


def stratum_groups(scored: ScoredCandidates) -> list[tuple[str, list[int]]]:
    """``[(stratum, member indices)]`` with strata in key order and members in
    candidate (action-id) order."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(scored.candidates):
        groups.setdefault(c.stratum, []).append(i)
    return sorted(groups.items())


@dataclass(frozen=True)
class StratumDraw:
    """One draw from a stratified design: the chosen index, its stratum, how many
    strata were eligible and how many candidates the stratum had, and the exact
    selection probability of every candidate."""

    index: int
    stratum: str
    n_strata: int
    n_in_stratum: int
    probs: np.ndarray


def _probs_over(scored: ScoredCandidates, groups: Sequence[tuple[str, list[int]]]) -> np.ndarray:
    p = np.zeros(scored.n)
    for _, members in groups:
        p[members] = 1.0 / (len(groups) * len(members))
    return p


def _draw(
    scored: ScoredCandidates, groups: Sequence[tuple[str, list[int]]], rng: np.random.Generator
) -> StratumDraw:
    if not groups:
        raise ValueError("no feasible candidates to draw from")
    s = int(rng.integers(len(groups)))
    stratum, members = groups[s]
    j = int(rng.integers(len(members)))
    return StratumDraw(
        index=members[j],
        stratum=stratum,
        n_strata=len(groups),
        n_in_stratum=len(members),
        probs=_probs_over(scored, groups),
    )


def audit_probabilities(scored: ScoredCandidates) -> np.ndarray:
    """q(d) = 1 / (S · n_{s(d)}) of the audit design (sums to 1)."""
    return _probs_over(scored, stratum_groups(scored))


def audit_draw(scored: ScoredCandidates, rng: np.random.Generator) -> StratumDraw:
    """Stratum uniformly among the feasible strata, then a candidate uniformly in it
    (two ``rng.integers`` draws)."""
    return _draw(scored, stratum_groups(scored), rng)


def _least_sampled(
    scored: ScoredCandidates, counts: Mapping[str, int]
) -> tuple[list[tuple[str, list[int]]], int]:
    groups = stratum_groups(scored)
    if not groups:
        return [], 0
    least = min(int(counts.get(s, 0)) for s, _ in groups)
    return [(s, m) for s, m in groups if int(counts.get(s, 0)) == least], least


def least_sampled_probabilities(scored: ScoredCandidates, counts: Mapping[str, int]) -> np.ndarray:
    """q(d) of the least-sampled stratified design (P3)."""
    return _probs_over(scored, _least_sampled(scored, counts)[0])


def least_sampled_draw(
    scored: ScoredCandidates, counts: Mapping[str, int], rng: np.random.Generator
) -> tuple[StratumDraw, int]:
    """A draw of the least-sampled stratified design and the tied minimum count."""
    groups, least = _least_sampled(scored, counts)
    return _draw(scored, groups, rng), least


# --------------------------------------------------------------------------- distances


@dataclass(frozen=True)
class RecordMetric:
    """L1 distance in design-record space (see the module docstring)."""

    covariates: tuple[CovariateSpec, ...]
    scales: Mapping[str, float]
    """Range of each numeric covariate over the reference records (1 if constant)."""

    @staticmethod
    def fit(covariates: Sequence[CovariateSpec], records: Sequence[Record]) -> RecordMetric:
        scales: dict[str, float] = {}
        for cov in covariates:
            if cov.kind != "numeric":
                continue
            vals = [float(r[cov.name]) for r in records if cov.name in r]
            span = (max(vals) - min(vals)) if vals else 0.0
            scales[cov.name] = span if span > 0.0 and math.isfinite(span) else 1.0
        return RecordMetric(tuple(covariates), scales)

    def distance(self, a: Record, b: Record) -> float:
        total = 0.0
        for cov in self.covariates:
            name = cov.name
            if name not in a or name not in b:
                continue
            va, vb = a[name], b[name]
            if cov.kind == "numeric":
                total += abs(float(va) - float(vb)) / self.scales[name]
            elif va != vb:
                total += 1.0
        return total

    def nearness(self, target: Record, target_stratum: str, scored: ScoredCandidates) -> np.ndarray:
        """−distance from ``target`` to every scored candidate (higher is nearer)."""
        return np.asarray(
            [
                -(self.distance(target, c.record) + (c.stratum != target_stratum))
                for c in scored.candidates
            ],
            dtype=np.float64,
        )


# --------------------------------------------------------------------------- schedules


def _plan_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """The initial candidates a plan is built from: the feasible ones (all of them if
    none is feasible), in action-id order."""
    feas = [c for c in candidates if c.feasible] or list(candidates)
    return sorted(feas, key=lambda c: c.action.id)


def fixed_grid_schedule(candidates: Sequence[Candidate]) -> tuple[str, ...]:
    """Round-robin over strata (key order); within a stratum, stable-key order."""
    groups: dict[str, list[str]] = {}
    for c in _plan_candidates(candidates):
        groups.setdefault(c.stratum, []).append(c.action.id)
    queues = [sorted(ids, key=stable_key) for _, ids in sorted(groups.items())]
    out: list[str] = []
    for r in range(max((len(q) for q in queues), default=0)):
        out.extend(q[r] for q in queues if r < len(q))
    return tuple(out)


def lexicographic_argmax(
    primary: np.ndarray, secondary: np.ndarray, action_ids: Sequence[str], rel_tol: float
) -> tuple[int, tuple[str, ...]]:
    """Argmax of ``primary`` (ties within ``rel_tol``), then of ``secondary`` among the
    tied, then the smallest stable key (all via :func:`argmax_with_ties`). NaN
    entries of ``primary`` are never chosen."""
    i, ties = argmax_with_ties(primary, action_ids, rel_tol)
    if len(ties) == 1:
        return i, ties
    tied = set(ties)
    sec = np.where([a in tied for a in action_ids], np.asarray(secondary, dtype=np.float64), np.nan)
    if not np.any(np.isfinite(sec)):
        return i, ties
    return argmax_with_ties(sec, action_ids, rel_tol)


# --------------------------------------------------------------------------- maximin


def pairwise_js(
    predictives: Sequence[Predictive],
    pairs: Sequence[tuple[int, int]],
    quadrature: QuadratureSettings | None = None,
) -> np.ndarray:
    """JS_mm′(d) for every pair (rows) and candidate (columns), in one batched call
    of the exact U_M kernel (a candidate's value does not depend on its batch)."""
    n = predictives[0].n if predictives else 0
    if not pairs or n == 0:
        return np.zeros((len(pairs), n))
    a = [predictives[i] for i, _ in pairs]
    b = [predictives[j] for _, j in pairs]
    res = model_indicator_eig([_concat(a), _concat(b)], [0.5, 0.5], quadrature)
    return np.asarray(res.value).reshape(len(pairs), n)


def _concat(preds: Sequence[Predictive]) -> Predictive:
    if all(isinstance(p, StudentTPredictive) for p in preds):
        ts = [p for p in preds if isinstance(p, StudentTPredictive)]
        return StudentTPredictive(
            df=np.concatenate([t.df for t in ts]),
            loc=np.concatenate([t.loc for t in ts]),
            scale=np.concatenate([t.scale for t in ts]),
        )
    if all(isinstance(p, DiscretePredictive) for p in preds):
        ds = [p for p in preds if isinstance(p, DiscretePredictive)]
        return DiscretePredictive(np.concatenate([d.probs for d in ds], axis=0))
    raise TypeError("all predictives must be of one family")


@dataclass(frozen=True)
class MaximinPlan:
    """A static maximin design. Per planned slot: the action, its reset-time cost,
    the divergence of the weakest pair of the plan so far, and that pair."""

    action_ids: tuple[str, ...]
    costs: tuple[float, ...]
    min_divergence: tuple[float, ...]
    weakest_pair: tuple[tuple[str, str], ...]
    pairs: tuple[tuple[str, str], ...]
    divergence: str
    """``"gaussian_joint_kl"`` (ordered pairs) or ``"pointwise_js"`` (unordered)."""

    def __len__(self) -> int:
        return len(self.action_ids)


class _PairObjective(Protocol):
    kind: str
    pairs: list[tuple[int, int]]
    repeats: bool

    def values(self, plan: Sequence[int], idx: np.ndarray) -> np.ndarray:
        """Divergence of every pair (rows) for the design ``plan + [d]``, for each
        candidate d in ``idx`` (columns)."""
        ...


class _GaussianJointKL:
    """KL(N_m ‖ N_m′) of the joint prior predictive of a whole design, ordered pairs.

    Under linear-Gaussian hypothesis m the observations y_S of a design S are
    jointly t-distributed (σ² is shared); moment-matched with σ² at its prior mean
    s² = b/(a − 1) (b/a when a ≤ 1) they are N(X_S μ, s²(I + X_S V X_Sᵀ)). The KL of
    two such Gaussians is closed form,
    ½[tr(B⁻¹A) + (μ_B − μ_A)ᵀB⁻¹(μ_B − μ_A) − k + log det B − log det A],
    and accounts for the correlation between planned observations (a repeated or
    redundant point adds little), which no per-observation divergence can. It is the
    expected log Bayes factor for m against m′ when m is true, so the maximin over
    ordered pairs bounds the evidence for whichever hypothesis is true."""

    kind = "gaussian_joint_kl"
    repeats = True

    def __init__(self, models: Sequence[NIGModel], records: Sequence[Record]) -> None:
        self.means: list[np.ndarray] = []
        self.grams: list[np.ndarray] = []
        self.s2: list[float] = []
        for m in models:
            post = m.posterior
            x = m.features(records)
            s2 = post.b / (post.a - 1.0) if post.a > 1.0 else post.b / post.a
            self.means.append(np.sum(x * post.mu, axis=1))
            g = x @ post.cov @ x.T
            self.grams.append(s2 * 0.5 * (g + g.T))
            self.s2.append(s2)
        k = len(models)
        self.pairs = [(i, j) for i in range(k) for j in range(k) if i != j]

    def values(self, plan: Sequence[int], idx: np.ndarray) -> np.ndarray:
        n = idx.size
        s = np.asarray(plan, dtype=np.int64)
        rows = np.concatenate([np.broadcast_to(s, (n, s.size)), idx[:, None]], axis=1)
        k = rows.shape[1]
        eye = np.eye(k)
        covs = [
            s2 * eye + g[rows[:, :, None], rows[:, None, :]]
            for s2, g in zip(self.s2, self.grams, strict=True)
        ]
        mus = [mu[rows] for mu in self.means]
        logdet = [np.linalg.slogdet(c)[1] for c in covs]
        out = np.empty((len(self.pairs), n))
        for p, (i, j) in enumerate(self.pairs):
            a, b = covs[i], covs[j]
            tr = np.trace(np.linalg.solve(b, a), axis1=1, axis2=2)
            d = mus[j] - mus[i]
            quad = np.sum(d * np.linalg.solve(b, d[:, :, None])[:, :, 0], axis=1)
            out[p] = 0.5 * (tr + quad - k + logdet[j] - logdet[i])
        return np.maximum(out, 0.0)  # KL ≥ 0; only rounding can go below


class _PointwiseJS:
    """Σ over planned observations of JS_mm′(d) of the prior predictives from the
    reset-time state (independence approximation), unordered pairs, no repeats."""

    kind = "pointwise_js"
    repeats = False

    def __init__(
        self,
        models: Sequence[HypothesisModel],
        records: Sequence[Record],
        quadrature: QuadratureSettings | None,
    ) -> None:
        k = len(models)
        self.pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        preds = [m.predictive(records) for m in models]
        self.js = pairwise_js(preds, self.pairs, quadrature)  # (P, n), computed once

    def values(self, plan: Sequence[int], idx: np.ndarray) -> np.ndarray:
        base = np.sum(self.js[:, list(plan)], axis=1) if len(plan) else np.zeros(len(self.pairs))
        return base[:, None] + self.js[:, idx]


def maximin_plan(
    hypothesis_set: HypothesisSet,
    candidates: Sequence[Candidate],
    *,
    budget: float,
    max_length: int,
    rel_tol: float,
    quadrature: QuadratureSettings | None = None,
) -> MaximinPlan:
    """Greedy static maximin design from the prior (module docstring)."""
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    cands = _plan_candidates(candidates)
    ids = [c.action.id for c in cands]
    cost = np.asarray([c.cost for c in cands], dtype=np.float64)
    records = [c.record for c in cands]
    models = build_models(hypothesis_set)
    hids = [h.id for h in hypothesis_set.hypotheses]
    nig = [m for m in models if isinstance(m, NIGModel)]
    obj: _PairObjective
    if len(nig) == len(models):
        obj = _GaussianJointKL(nig, records)
    else:
        obj = _PointwiseJS(models, records, quadrature)
    labels = tuple((hids[i], hids[j]) for i, j in obj.pairs)
    cum = np.zeros(len(obj.pairs))
    avail = np.ones(len(cands), dtype=bool)
    left = float(budget)
    plan: list[int] = []
    mins: list[float] = []
    weakest: list[tuple[str, str]] = []
    for _ in range(max_length):
        ok = avail & (cost <= left + 1e-12 * max(1.0, abs(left)))
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            break
        c = cost[idx]
        if obj.pairs:
            vals = obj.values(plan, idx)  # (P, n_eligible)
            gain_min = (np.min(vals, axis=0) - float(np.min(cum))) / c
            gain_sum = (np.sum(vals, axis=0) - float(np.sum(cum))) / c
        else:  # a single hypothesis: nothing to discriminate; stable-key order
            vals = np.zeros((0, idx.size))
            gain_min = np.zeros(idx.size)
            gain_sum = np.zeros(idx.size)
        j, _ = lexicographic_argmax(gain_min, gain_sum, [ids[int(i)] for i in idx], rel_tol)
        k = int(idx[j])
        plan.append(k)
        left -= float(cost[k])
        if not obj.repeats:
            avail[k] = False
        if obj.pairs:
            cum = vals[:, j]
            w = int(np.argmin(cum))
            mins.append(float(cum[w]))
            weakest.append(labels[w])
        else:
            mins.append(0.0)
            weakest.append(("", ""))
    return MaximinPlan(
        action_ids=tuple(ids[k] for k in plan),
        costs=tuple(float(cost[k]) for k in plan),
        min_divergence=tuple(mins),
        weakest_pair=tuple(weakest),
        pairs=labels,
        divergence=obj.kind,
    )
