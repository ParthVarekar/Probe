"""Information-gain kernels (handover §12 H.4–H.5).

Model-indicator EIG (the primary selection objective) for a batch of candidate
designs d, given hypothesis weights π and one-step predictives p_m(· | d):

    U_M(d) = I(M; Y_d | D) = H[Σ_m π_m p_m] − Σ_m π_m H[p_m]
           = Σ_m π_m KL(p_m ‖ f),   f = Σ_m π_m p_m,

with 0 ≤ U_M ≤ H(π) ≤ log K. It is computed in the KL form, whose integrand is a
sum of log-ratios, so a tiny U_M is not the difference of two large entropies.

* **Discrete outcomes** (:class:`DiscretePredictive`): the sum over the support is
  exact; the reported error is 0.
* **Student-t outcomes** (:class:`StudentTPredictive`): one 1-D integral per
  candidate, by vectorized *adaptive Gauss–Kronrod* (G7/K15 pairs, QUADPACK error
  estimate); see :func:`_integrate_line`. In brief:

  - *Range.* Each side is cut where every component's own tail probability is
    at most ``tail_mass`` (1e−18; the Student-t quantile, ≈ ±8.8 scale units in
    the Gaussian limit and far beyond for heavy tails), instead of a fixed
    "±12 scale units". The integrand is f(y)·KL(r(y) ‖ π) with responsibilities
    r(y), so it is non-negative and at most f(y)·(−log min π); the cut-off mass is
    therefore rigorously bounded by (−log min π)·Σ_m π_m P_m(outside) (exact
    Student-t CDFs), which is added to the error.
  - *Breakpoints* sit at loc_m + {0, ±5, ±25, ±125, ...}·scale_m (geometric out
    to the span of all components), plus the midpoints between neighbouring
    locations, so narrow components inside wide ones and tail mass between
    far-apart components are always seen.
  - *Coordinates.* Every panel is anchored at the component location A nearest to
    it (constant over the panel thanks to the midpoint breakpoints) and mapped by
    y = A + h·sinh(v), h = that component's scale: linear near A, logarithmic in
    the polynomial tails (where a Student-t integrand decays exponentially in v),
    and every standardized offset (y − loc_m)/scale_m is formed from the single-
    rounding differences A − loc_m and the local coordinate, so components ~10⁹ of their
    own scales apart are resolved to full precision.
  - Panels are bisected until each meets its share of the tolerance (error
    budgets halve on bisection), every candidate in one array pass per round.
    Log densities are combined with log-sum-exp. The per-candidate error estimate
    is the sum of the accepted panels' QUADPACK estimates plus the truncation
    bound (conservative: on the reference suite the true error is orders of
    magnitude below it).

Bounds are *asserted* before clipping: a raw value outside [0, H(π)] by more than
its error estimate plus rounding raises :class:`NumericalIntegrityError` instead
of being silently clipped.

Also here: tempered posterior weights over hypotheses, mixture predictive
moments, goal-oriented EIG for a binary query, and two **test oracles** (plain
Monte Carlo on the closed-form predictives, and brute-force nested Monte Carlo
on posterior parameter draws). The oracles use explicit seeded generators and
are never used for selection.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy import special

from probe.engine.grid import GridModel, logsumexp
from probe.engine.ir import HypothesisSet, Record
from probe.engine.models import HypothesisModel
from probe.engine.nig import LinearQueryKernel, NIGModel
from probe.engine.predictive import (
    DiscretePredictive,
    Predictive,
    StudentTPredictive,
    t_log_normalizer,
)
from probe.engine.queries import ContrastQuery, GridQuery, Query

__all__ = [
    "DegeneratePosteriorError",
    "EIGResult",
    "GoalEIGResult",
    "NumericalIntegrityError",
    "QuadratureSettings",
    "SupportsNestedMC",
    "binary_entropy",
    "entropy",
    "goal_eig",
    "log_posterior_weights",
    "log_prior_weights",
    "mc_model_eig",
    "mixture_moments",
    "model_indicator_eig",
    "nested_mc_model_eig",
    "posterior_weights",
    "weighted_parameter_eig",
]


class NumericalIntegrityError(ArithmeticError):
    """An information quantity violated its mathematical bounds by more than its
    numerical error estimate. Never clipped away silently."""


class DegeneratePosteriorError(ValueError):
    """Every hypothesis has zero posterior weight (all evidences are −inf)."""


# --------------------------------------------------------------------------- basics


def entropy(weights: np.ndarray | Sequence[float]) -> float:
    """Shannon entropy (nats) of a probability vector, 0·log 0 = 0. A point mass gives
    +0.0, never −0.0 (canonical JSON, and so every logged hash, tells them apart)."""
    w = np.asarray(weights, dtype=np.float64)
    return float(0.0 - np.sum(special.xlogy(w, w)))


def binary_entropy(p: np.ndarray | float) -> np.ndarray:
    """H_b(p) in nats (+0.0 at p ∈ {0, 1})."""
    q = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return np.asarray(0.0 - (special.xlogy(q, q) + special.xlogy(1.0 - q, 1.0 - q)))


def _rowdot(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Σ_j a[..., j]·v[j] by a plain reduction over the last axis. Unlike a BLAS
    matrix-vector product, the value for one row does not depend on which other rows
    share the array, so a candidate's information gain is a function of that
    candidate alone (bitwise), whatever batch it is scored in."""
    return np.sum(a * v, axis=-1)


def _active(w: np.ndarray) -> list[int]:
    """Indices of positively weighted hypotheses (plain ints, fixed order)."""
    return [int(i) for i in np.flatnonzero(w > 0.0)]


def _check_weights(weights: np.ndarray | Sequence[float], k: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (k,):
        raise ValueError(f"weights must have shape ({k},), got {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    s = float(w.sum())
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1 (got {s!r})")
    return w / s


# --------------------------------------------------------------------------- posterior weights


def log_prior_weights(hypothesis_set: HypothesisSet) -> np.ndarray:
    """log π⁰ from the hypotheses' ``prior_weight`` (normalised), set order."""
    w = np.asarray([h.prior_weight for h in hypothesis_set.hypotheses], dtype=np.float64)
    return np.log(w / w.sum())


def log_posterior_weights(
    log_prior: np.ndarray | Sequence[float],
    log_evidences: np.ndarray | Sequence[float],
    eta: float = 1.0,
) -> np.ndarray:
    """log π_m ∝ log π⁰_m + η·log p(y_{≥r_m} | m, y_{<r_m}), normalised by log-sum-exp.

    ``log_prior`` may be unnormalised. η ∈ (0, 1] tempers the likelihood (a
    documented robustness setting, handover §12 H.4); η = 1 is exact Bayes.
    Evidence −inf (a refuted hypothesis) gives weight exactly 0. NaN or +inf
    inputs are rejected, and :class:`DegeneratePosteriorError` is raised when no
    hypothesis keeps positive weight.
    """
    lp = np.asarray(log_prior, dtype=np.float64)
    le = np.asarray(log_evidences, dtype=np.float64)
    if lp.ndim != 1 or lp.shape != le.shape or lp.size == 0:
        raise ValueError("log_prior and log_evidences must be 1-D arrays of equal length")
    if not 0.0 < eta <= 1.0:
        raise ValueError("tempering eta must lie in (0, 1]")
    if np.any(np.isnan(lp)) or np.any(np.isnan(le)):
        raise ValueError("NaN in log prior or log evidence")
    if np.any(np.isposinf(lp)) or np.any(np.isposinf(le)):
        raise ValueError("+inf in log prior or log evidence")
    s = lp + eta * le
    norm = float(logsumexp(s))
    if norm == -math.inf:
        raise DegeneratePosteriorError("every hypothesis has zero posterior weight")
    return s - norm


def posterior_weights(
    log_prior: np.ndarray | Sequence[float],
    log_evidences: np.ndarray | Sequence[float],
    eta: float = 1.0,
) -> np.ndarray:
    """π (sums to 1); see :func:`log_posterior_weights`."""
    w = np.exp(log_posterior_weights(log_prior, log_evidences, eta))
    return w / w.sum()


# --------------------------------------------------------------------------- mixture moments


def mixture_moments(
    predictives: Sequence[Predictive], weights: np.ndarray | Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the model-averaged predictive Σ_m π_m p_m per candidate:
    E = Σ π_m μ_m, Var = Σ π_m [v_m + (μ_m − E)²]. Hypotheses with π_m = 0 are
    ignored (so an infinite-variance component with zero weight is harmless);
    a positively weighted Student-t with ν ≤ 2 makes the variance +inf."""
    w = _check_weights(weights, len(predictives))
    _family(predictives)
    act = _active(w)
    means = np.stack([predictives[i].mean() for i in act])
    var = np.stack([predictives[i].variance() for i in act])
    wa = w[act][:, None]
    mean = np.sum(wa * means, axis=0)
    with np.errstate(invalid="ignore"):
        total = np.sum(wa * (var + (means - mean) ** 2), axis=0)
    return mean, total


def weighted_parameter_eig(
    models: Sequence[HypothesisModel],
    weights: np.ndarray | Sequence[float],
    records: Sequence[Record],
) -> np.ndarray:
    """Σ_m π_m U_θ|m(d), the secondary objective (handover §12 H.5)."""
    w = _check_weights(weights, len(models))
    out = np.zeros(len(records))
    for wi, m in zip(w, models, strict=True):
        if wi > 0.0:
            out = out + wi * np.asarray(m.parameter_eig(records), dtype=np.float64)
    return out


# --------------------------------------------------------------------------- quadrature

# Gauss–Kronrod 7/15 abscissae and weights (QUADPACK qk15), on [-1, 1].
_XGK_POS = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
    ]
)
_WGK_POS = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
    ]
)
_WGK_0 = 0.209482141084727828012999174891714
_WG_POS = np.array(  # Gauss weights at _XGK_POS[1], [3], [5]
    [
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
    ]
)
_WG_0 = 0.417959183673469387755102040816327

_GK_NODES = np.concatenate([-_XGK_POS, [0.0], _XGK_POS[::-1]])
_GK_WK = np.concatenate([_WGK_POS, [_WGK_0], _WGK_POS[::-1]])
_GK_WG = np.zeros(15)
_GK_WG[[1, 3, 5]] = _WG_POS
_GK_WG[7] = _WG_0
_GK_WG[[13, 11, 9]] = _WG_POS
_EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class QuadratureSettings:
    """Adaptive Gauss–Kronrod controls (see :func:`_integrate_line`).

    The tolerance per candidate is max(atol, rtol·|initial estimate|); it is split
    evenly over the initial panels and halved on every bisection. Breakpoints sit
    at loc + ``marks``·scale per component, extended by factors of
    ``geometric_ratio`` beyond the outermost mark out to the span of all
    components. Each side of the line is cut where every component's tail
    probability is at most ``tail_mass``, with ``tail_splits`` initial panels per
    tail; the cut-off mass enters the error estimate as a rigorous bound."""

    atol: float = 1e-9
    rtol: float = 1e-7
    max_rounds: int = 40
    max_panels: int = 2_000_000
    marks: tuple[float, ...] = field(default=(-5.0, 0.0, 5.0))
    geometric_ratio: float = 5.0
    tail_mass: float = 1e-18
    tail_splits: int = 4

    def __post_init__(self) -> None:
        if not (self.atol > 0.0 and self.rtol >= 0.0):
            raise ValueError("atol must be positive and rtol non-negative")
        if not 0.0 < self.tail_mass < 1e-3:
            raise ValueError("tail_mass must lie in (0, 1e-3)")
        if self.max_rounds < 1 or self.max_panels < 1 or self.tail_splits < 1:
            raise ValueError("max_rounds, max_panels and tail_splits must be positive")
        if not self.geometric_ratio > 1.0:
            raise ValueError("geometric_ratio must exceed 1")


_DEFAULT_QUAD = QuadratureSettings()
_Y_CAP = 1e100
"""The range never extends more than 1e100 × the narrowest scale beyond the
outermost location, so standardized values stay far from float overflow (only
heavier-than-Cauchy tails, ν ≲ 0.2, ever reach this cap; their cut-off mass is
then larger, but it is still bounded exactly and reported)."""
_MAX_SPAN = 1e100
"""Components further apart than this many of the narrowest scale are rejected as
numerically meaningless."""
_DF_GAUSS = 1e300
"""Finite stand-in for ν = +inf in the log-density kernel."""


def _nearest_location(y: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Index of the component location nearest to each y (ties: smaller scale,
    then lower index). ``y`` (P,), ``loc``/``scale`` (P, K)."""
    dist = np.abs(y[:, None] - loc)
    best = np.min(dist, axis=1, keepdims=True)
    return np.argmin(np.where(dist == best, scale, np.inf), axis=1)


def _integrate_line(
    breaks: np.ndarray,
    loc: np.ndarray,
    scale: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    integrand: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    settings: QuadratureSettings,
) -> tuple[np.ndarray, np.ndarray, int]:
    """∫ g_i(y) dy over [y_lo_i, y_hi_i] for candidates i = 0..n−1, vectorized
    adaptive G7/K15 (QUADPACK qk15 rule and error estimate).

    ``loc``/``scale`` (n, K) are the mixture components, ``breaks`` (n, B) extra
    initial breakpoints in y (NaN = unused). Returns (values, error estimates,
    number of integrand evaluations); the caller adds its bound on the mass
    outside [y_lo, y_hi].

    **Anchored coordinates.** The component locations and the midpoints between
    neighbouring locations are breakpoints, so the location nearest to a panel is
    the same for all of it. Each panel is anchored there: with A = loc_a and
    h = scale_a of that component, y = A + h·sinh(v), and the integrand is called
    as ``integrand(t, off, cand)`` with the *local* coordinate t = y − A (P, 15),
    the offsets off[:, m] = A − loc_m (P, K; one rounding each) and the candidate of each
    panel row; y − loc_m = off_m + t. Because A is the nearest location,
    |t| ≤ |y − loc_m| for every m, so each standardized value (y − loc_m)/scale_m
    carries a relative rounding error of a few ε however far apart the components
    are. (In one global coordinate, a component 10⁹ of its scales away from the
    frame's origin is blurred to ~10⁻⁷ of its width, and the adaptive loop chases
    that noise until the panel cap.) Near A the map is linear; far from it a
    Student-t integrand ∝ |y|^−(ν+1) becomes ∝ exp(−ν|v|), so heavy tails out to
    the quantile range have no endpoint singularity.

    The error estimate is the sum of the accepted panels' QUADPACK estimates
    (each at least 50ε·∫|g| over the panel, QUADPACK's rounding floor)."""
    n = loc.shape[0]
    rows = np.arange(n)
    srt = np.sort(loc, axis=1)
    mids = 0.5 * (srt[:, 1:] + srt[:, :-1])
    pts = np.concatenate([breaks, loc, mids], axis=1)
    pts = np.clip(pts, y_lo[:, None], y_hi[:, None])  # NaN stays NaN
    finite = np.isfinite(pts)
    inner_lo = np.min(np.where(finite, pts, np.inf), axis=1)
    inner_hi = np.max(np.where(finite, pts, -np.inf), axis=1)
    tails = []
    steps = np.arange(1, settings.tail_splits, dtype=np.float64) / settings.tail_splits
    for outer, inner in ((y_lo, inner_lo), (y_hi, inner_hi)):
        ia = _nearest_location(outer, loc, scale)
        anchor, h_t = loc[rows, ia], scale[rows, ia]
        v0 = np.arcsinh((outer - anchor) / h_t)
        v1 = np.arcsinh((inner - anchor) / h_t)
        v = v0[:, None] + (v1 - v0)[:, None] * steps
        tails.append(anchor[:, None] + h_t[:, None] * np.sinh(v))
    pts = np.sort(np.concatenate([y_lo[:, None], pts, *tails, y_hi[:, None]], axis=1), axis=1)
    lo_y, hi_y = pts[:, :-1], pts[:, 1:]
    valid = hi_y > lo_y  # False wherever NaN is involved
    cand = np.broadcast_to(rows[:, None], lo_y.shape)[valid]
    y_a, y_b = lo_y[valid], hi_y[valid]
    a = _nearest_location(0.5 * (y_a + y_b), loc[cand], scale[cand])
    anchor, h = loc[cand, a], scale[cand, a]
    lo = np.arcsinh((y_a - anchor) / h)
    hi = np.arcsinh((y_b - anchor) / h)
    keep = hi > lo
    lo, hi, cand, a, h = lo[keep], hi[keep], cand[keep], a[keep], h[keep]
    n_init = np.bincount(cand, minlength=n).astype(np.float64)
    offsets = loc[:, :, None] - loc[:, None, :]  # [i, a, m] = loc_a − loc_m

    val = np.zeros(n)
    err = np.zeros(n)
    budget: np.ndarray | None = None
    n_evals = 0
    for rnd in range(settings.max_rounds):
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        vv = mid[:, None] + half[:, None] * _GK_NODES
        hh = h[:, None]
        g = integrand(hh * np.sinh(vv), offsets[cand, a], cand) * (hh * np.cosh(vv))
        n_evals += g.size
        resk = _rowdot(g, _GK_WK)
        resg = _rowdot(g, _GK_WG)
        mean = 0.5 * resk
        resabs = _rowdot(np.abs(g), _GK_WK)
        resasc = _rowdot(np.abs(g - mean[:, None]), _GK_WK)
        raw = np.abs(resk - resg)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = resasc * np.minimum(1.0, (200.0 * raw / resasc) ** 1.5)
        e = np.where((resasc > 0.0) & (raw > 0.0), scaled, raw)
        e = np.maximum(e, 50.0 * _EPS * resabs) * half
        part = resk * half
        if budget is None:
            est = np.bincount(cand, weights=part, minlength=n)
            tol = np.maximum(settings.atol, settings.rtol * np.abs(est))
            budget = (tol / np.maximum(n_init, 1.0))[cand]
        last = rnd == settings.max_rounds - 1 or 2 * lo.size > settings.max_panels
        tiny = half <= 4.0 * _EPS * np.maximum(1.0, np.abs(mid))  # nodes no longer distinct
        ok = (e <= budget) | tiny | last
        val += np.bincount(cand[ok], weights=part[ok], minlength=n)
        err += np.bincount(cand[ok], weights=e[ok], minlength=n)
        if ok.all():
            break
        bad = ~ok
        lo_b, hi_b, mid_b, b_b = lo[bad], hi[bad], mid[bad], 0.5 * budget[bad]
        lo = np.concatenate([lo_b, mid_b])
        hi = np.concatenate([mid_b, hi_b])
        cand = np.concatenate([cand[bad], cand[bad]])
        a = np.concatenate([a[bad], a[bad]])
        h = np.concatenate([h[bad], h[bad]])
        budget = np.concatenate([b_b, b_b])
    if not (np.all(np.isfinite(val)) and np.all(np.isfinite(err))):
        raise NumericalIntegrityError("non-finite value in information-gain quadrature")
    return val, err, n_evals


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class EIGResult:
    """Model-indicator EIG per candidate (nats)."""

    value: np.ndarray
    """U_M(d), clipped to [0, H(π)] after the bound check."""
    error: np.ndarray
    """Numerical error estimate (0 for discrete outcomes and exact shortcuts)."""
    raw: np.ndarray
    """Unclipped value as computed."""
    mixture_entropy: np.ndarray
    """H[Σ π_m p_m] (differential entropy for continuous outcomes)."""
    expected_entropy: np.ndarray
    """Σ π_m H[p_m] (closed form)."""
    upper_bound: float
    """H(π)."""
    n_evals: int = 0
    """Integrand evaluations used by the quadrature (0 for discrete)."""


@dataclass(frozen=True)
class GoalEIGResult:
    """Goal-oriented EIG U_q(d) = I(1[q]; Y_d | D) per candidate (nats)."""

    value: np.ndarray
    error: np.ndarray
    raw: np.ndarray
    query_prob: float
    """Model-averaged P(q | D)."""
    per_hypothesis_prob: np.ndarray
    """P(q | m, D) per hypothesis (set order)."""
    upper_bound: float
    """H_b(P(q | D)) ≤ log 2."""
    n_evals: int = 0


def _bounded(raw: np.ndarray, err: np.ndarray, upper: float, what: str) -> np.ndarray:
    slack = err + 1e-12 + 1e-12 * upper
    low = raw < -slack
    high = raw > upper + slack
    if np.any(low) or np.any(high):
        i = int(np.flatnonzero(low | high)[0])
        raise NumericalIntegrityError(
            f"{what} = {raw[i]!r} at candidate {i} is outside [0, {upper!r}] by more than "
            f"its error estimate {err[i]!r}"
        )
    return np.clip(raw, 0.0, upper)


def _family(predictives: Sequence[Predictive]) -> type:
    if len(predictives) == 0:
        raise ValueError("need at least one predictive")
    kinds = {type(p) for p in predictives}
    if len(kinds) != 1:
        raise TypeError("all predictives must be of one family")
    n = {p.n for p in predictives}
    if len(n) != 1:
        raise ValueError("all predictives must cover the same candidates")
    kind = kinds.pop()
    if kind is DiscretePredictive:
        s = {p.probs.shape[1] for p in predictives if isinstance(p, DiscretePredictive)}
        if len(s) != 1:
            raise ValueError("discrete predictives must share one support")
    return kind


# --------------------------------------------------------------------------- U_M


def model_indicator_eig(
    predictives: Sequence[Predictive],
    weights: np.ndarray | Sequence[float],
    settings: QuadratureSettings | None = None,
) -> EIGResult:
    """U_M(d) for every candidate covered by the per-hypothesis ``predictives``
    (hypothesis-set order, aligned with ``weights``). Exact zero is returned when
    π is a point mass or when all positively weighted predictives are identical at
    a candidate."""
    kind = _family(predictives)
    w = _check_weights(weights, len(predictives))
    act = _active(w)
    wa = w[act]
    h_pi = entropy(wa)
    n = predictives[0].n
    expected = np.zeros(n)
    for i in act:
        expected = expected + w[i] * predictives[i].entropy()
    if n == 0:
        z = np.zeros(0)
        return EIGResult(z, z, z, z, z, h_pi)
    if len(act) == 1:
        zero = np.zeros(n)
        return EIGResult(zero, zero, zero, expected.copy(), expected, h_pi)
    if kind is DiscretePredictive:
        probs = np.stack([_as_discrete(predictives[i]).probs for i in act])  # (K, n, S)
        raw = _discrete_mi(wa, probs)
        same = np.all(probs == probs[:1], axis=(0, 2))
        raw = np.where(same, 0.0, raw)
        err = np.zeros(n)
        slack_err = np.full(n, 1e-12 * probs.shape[2])
        value = _bounded(raw, slack_err, h_pi, "U_M")
        return EIGResult(value, err, raw, raw + expected, expected, h_pi)
    ts = [_as_t(predictives[i]) for i in act]
    raw, err, n_evals = _student_t_mi(wa, ts, settings or _DEFAULT_QUAD)
    value = _bounded(raw, err, h_pi, "U_M")
    return EIGResult(value, err, raw, raw + expected, expected, h_pi, n_evals)


def _as_discrete(p: Predictive) -> DiscretePredictive:
    assert isinstance(p, DiscretePredictive)
    return p


def _as_t(p: Predictive) -> StudentTPredictive:
    assert isinstance(p, StudentTPredictive)
    return p


def _discrete_mi(w: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Σ_k w_k Σ_y P_k(y) log(P_k(y) / Σ_j w_j P_j(y)) for probs (K, n, S)."""
    mix = np.einsum("k,kns->ns", w, probs)
    pos = probs > 0.0
    with np.errstate(divide="ignore"):
        logp = np.log(np.where(pos, probs, 1.0))
        logm = np.log(np.where(mix > 0.0, mix, 1.0))
    terms = np.where(pos, probs * (logp - logm[None]), 0.0)
    return np.einsum("k,kn->n", w, terms.sum(axis=2))


@dataclass(frozen=True)
class _TParts:
    """Per-(candidate, hypothesis) Student-t parameters, shapes (n, K), with the
    log-density written as const + coef·log1p(((y − loc)·inv_scale)²·inv_df)."""

    loc: np.ndarray
    scale: np.ndarray
    df: np.ndarray
    const: np.ndarray
    coef: np.ndarray
    inv_scale: np.ndarray
    inv_df: np.ndarray

    @staticmethod
    def build(ts: Sequence[StudentTPredictive]) -> _TParts:
        loc = np.stack([t.loc for t in ts], axis=1).astype(np.float64)
        scale = np.stack([t.scale for t in ts], axis=1).astype(np.float64)
        df = np.stack([t.df for t in ts], axis=1).astype(np.float64)
        if not (np.all(np.isfinite(loc)) and np.all(np.isfinite(scale))):
            raise ValueError("Student-t predictives need finite loc and scale")
        if np.any(np.isnan(df)):
            raise ValueError("Student-t predictives need df > 0")
        const = t_log_normalizer(df) - np.log(scale)
        # ν = +inf (Gaussian): −½(ν+1)·log1p(z²/ν) → −z²/2 is evaluated at ν = 1e300,
        # identical to double precision (scipy's CDF/quantile take +inf directly)
        df_k = np.minimum(df, _DF_GAUSS)
        return _TParts(loc, scale, df, const, -0.5 * (df_k + 1.0), 1.0 / scale, 1.0 / df_k)

    def take(self, idx: np.ndarray) -> _TParts:
        return _TParts(
            self.loc[idx],
            self.scale[idx],
            self.df[idx],
            self.const[idx],
            self.coef[idx],
            self.inv_scale[idx],
            self.inv_df[idx],
        )

    def logpdf(self, t: np.ndarray, off: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """log p_k(y) at y = A + t, for local coordinates t (P, Q) and offsets
        off[:, k] = A − loc_k (P, K) (see :func:`_integrate_line`): shape (P, Q, K)."""
        z = t[:, :, None] + off[:, None, :]
        z *= self.inv_scale[cand][:, None, :]
        z *= z
        z *= self.inv_df[cand][:, None, :]
        np.log1p(z, out=z)
        z *= self.coef[cand][:, None, :]
        z += self.const[cand][:, None, :]
        return z

    def check_span(self) -> None:
        span = np.max(self.loc, axis=1) - np.min(self.loc, axis=1)
        if np.any(span > _MAX_SPAN * np.min(self.scale, axis=1)):
            raise NumericalIntegrityError(
                "predictive components are more than 1e100 of the narrowest scale apart"
            )

    def breaks(self, settings: QuadratureSettings, width: np.ndarray) -> np.ndarray:
        """Initial breakpoints loc + scale·marks, extended geometrically
        (± r^j·scale beyond the largest mark, r = ``settings.geometric_ratio``) out to
        twice the frame width. No initial panel near a component is then wider than
        r times its distance to that component, so tail mass at a panel edge is
        always seen by the Kronrod nodes (a single panel spanning thousands of
        scale units between separated components would otherwise miss it)."""
        m = np.asarray(settings.marks, dtype=np.float64)
        base = self.loc[:, :, None] + self.scale[:, :, None] * m[None, None, :]
        n = self.loc.shape[0]
        out = [base.reshape(n, -1)]
        top = float(np.max(np.abs(m))) if m.size else 1.0
        top = max(top, 1.0)
        reach = 2.0 * width[:, None] / self.scale  # (n, K), in scale units
        r = settings.geometric_ratio
        n_ext = int(np.ceil(np.log(max(float(np.max(reach)) / top, 1.0)) / np.log(r)))
        n_ext = min(n_ext, 300)
        mult = top * r ** np.arange(1, n_ext + 1, dtype=np.float64)
        if mult.size:
            keep = mult[None, None, :] <= reach[:, :, None] * r
            d = np.where(keep, self.scale[:, :, None] * mult[None, None, :], np.nan)
            out += [(self.loc[:, :, None] - d).reshape(n, -1)]
            out += [(self.loc[:, :, None] + d).reshape(n, -1)]
        return np.concatenate(out, axis=1)

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.min(self.loc, axis=1)
        hi = np.max(self.loc, axis=1)
        center = 0.5 * (lo + hi)
        width = np.maximum(np.max(self.scale, axis=1), 0.5 * (hi - lo))
        return center, width

    def limits(self, w: np.ndarray, tail_mass: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Integration range [y_lo, y_hi] per candidate, chosen so that every
        component has tail probability ≤ ``tail_mass`` beyond each end (Student-t
        quantile; capped at ``_Y_CAP`` narrowest scales beyond the outermost
        location), and the mixture mass outside it, Σ_k w_k P_k(Y ∉ [y_lo, y_hi]),
        from exact Student-t CDFs at the final (possibly capped) ends."""
        uq, inv = np.unique(self.df, return_inverse=True)
        zq = -special.stdtrit(uq, tail_mass)[inv.ravel()].reshape(self.df.shape)
        zq = np.where(np.isfinite(zq) & (zq > 0.0), zq, _Y_CAP)
        s_min = np.min(self.scale, axis=1)
        cap_lo = np.min(self.loc, axis=1) - _Y_CAP * s_min
        cap_hi = np.max(self.loc, axis=1) + _Y_CAP * s_min
        y_lo = np.maximum(np.min(self.loc - zq * self.scale, axis=1), cap_lo)
        y_hi = np.minimum(np.max(self.loc + zq * self.scale, axis=1), cap_hi)
        below = special.stdtr(self.df, (y_lo[:, None] - self.loc) * self.inv_scale)
        above = special.stdtr(self.df, (self.loc - y_hi[:, None]) * self.inv_scale)
        outside = _rowdot(below + above, w)
        if not np.all(np.isfinite(outside)):
            raise NumericalIntegrityError("non-finite tail mass in information-gain quadrature")
        return y_lo, y_hi, outside


def _student_t_mi(
    w: np.ndarray, ts: Sequence[StudentTPredictive], settings: QuadratureSettings
) -> tuple[np.ndarray, np.ndarray, int]:
    parts = _TParts.build(ts)
    n = parts.loc.shape[0]
    same = (
        np.all(parts.loc == parts.loc[:, :1], axis=1)
        & np.all(parts.scale == parts.scale[:, :1], axis=1)
        & np.all(parts.df == parts.df[:, :1], axis=1)
    )
    raw = np.zeros(n)
    err = np.zeros(n)
    todo = np.flatnonzero(~same)
    if todo.size == 0:
        return raw, err, 0
    sub = parts.take(todo)
    sub.check_span()
    logw = np.log(w)

    def integrand(t: np.ndarray, off: np.ndarray, cand: np.ndarray) -> np.ndarray:
        # Σ_k π_k p_k (log p_k − log f) with d_k = log π_k + log p_k − m (m = max_k),
        # e_k = exp(d_k), s = Σ_k e_k:  log p_k − log f = d_k − log π_k − log s, so
        # the integrand is e^m [Σ_k e_k (d_k − log π_k) − s log s] = f·KL(r ‖ π) ≥ 0
        # with responsibilities r_k = e_k / s. The dominant component contributes
        # −log π_k − log s with no large cancelling terms.
        d = sub.logpdf(t, off, cand)
        d += logw
        m = np.max(d, axis=2)
        d -= m[:, :, None]
        e = np.exp(d)
        s = np.sum(e, axis=2)
        inner = np.einsum("pqk,pqk->pq", e, d) - _rowdot(e, logw) - s * np.log(s)
        return np.exp(m) * inner

    _, width = sub.frame()
    y_lo, y_hi, outside = sub.limits(w, settings.tail_mass)
    val, er, n_evals = _integrate_line(
        sub.breaks(settings, width), sub.loc, sub.scale, y_lo, y_hi, integrand, settings
    )
    raw[todo] = val
    # outside [y_lo, y_hi]: 0 ≤ f·KL(r ‖ π) ≤ f·(−log min π), so the cut-off part of
    # the integral lies in [0, (−log min π)·Σ_k π_k P_k(outside)]
    err[todo] = er + float(-np.min(logw)) * outside
    return raw, err, n_evals


# --------------------------------------------------------------------------- goal EIG


def goal_eig(
    models: Sequence[HypothesisModel],
    weights: np.ndarray | Sequence[float],
    records: Sequence[Record],
    query: Query,
    settings: QuadratureSettings | None = None,
) -> GoalEIGResult:
    """U_q(d) = I(1[q]; Y_d | D) for a binary query under the model average.

    Writing r(y) = P(q | D, Y_d = y) and p = P(q | D),
    U_q = ∫ f(y) KL(Bern(r(y)) ‖ Bern(p)) dy (a sum for counts), with
    r(y) = Σ_m π_m p_m(y) P(q | m, D, y) / f(y) and 0 ≤ U_q ≤ H_b(p) ≤ log 2.

    * ``linear_gaussian`` + :class:`ContrastQuery`: P(q | m, D, y) is the exact
      Student-t CDF of the updated NIG posterior of cₘᵀβ
      (:class:`~probe.engine.nig.LinearQueryKernel`). The only approximation is
      the adaptive quadrature over y (same scheme and error estimate as U_M, with
      extra breakpoints where each hypothesis's posterior mean crosses τ).
    * ``finite_grid`` + :class:`GridQuery`: exact finite sums over grid points and
      outcomes; error 0.

    Approximation stated plainly: the query is interpreted *within* each
    hypothesis (its own parameterization) and averaged with π. No hypothesis is
    assumed true, and the model-indicator uncertainty itself is not counted as
    information about q except through how it moves r(y).
    """
    if len(models) == 0:
        raise ValueError("need at least one model")
    w = _check_weights(weights, len(models))
    act = _active(w)
    wa = w[act]
    n = len(records)
    if isinstance(query, ContrastQuery):
        if not all(isinstance(m, NIGModel) for m in models):
            raise TypeError("ContrastQuery needs linear_gaussian (NIG) models")
        nig = [m for m in models if isinstance(m, NIGModel)]
        kernels = [nig[i].linear_query(records, query) for i in act]
        probs = np.array([k.prior_prob for k in kernels])
        per_h = np.zeros(len(models))
        per_h[act] = probs
        p = float(np.clip(wa @ probs, 0.0, 1.0))
        upper = float(binary_entropy(p))
        if n == 0 or upper <= 0.0:
            z = np.zeros(n)
            return GoalEIGResult(z, z, z, p, per_h, upper)
        ts = [nig[i].predictive(records) for i in act]
        raw, err, n_evals = _contrast_query_mi(wa, ts, kernels, p, settings or _DEFAULT_QUAD)
        value = _bounded(raw, err, upper, "U_q")
        return GoalEIGResult(value, err, raw, p, per_h, upper, n_evals)
    if isinstance(query, GridQuery):
        if not all(isinstance(m, GridModel) for m in models):
            raise TypeError("GridQuery needs finite_grid models")
        grid = [m for m in models if isinstance(m, GridModel)]
        per_h = np.zeros(len(models))
        per_h[act] = [grid[i].query_prob(query) for i in act]
        p = float(np.clip(w @ per_h, 0.0, 1.0))
        upper = float(binary_entropy(p))
        if n == 0 or upper <= 0.0:
            z = np.zeros(n)
            return GoalEIGResult(z, z, z, p, per_h, upper)
        joint = np.zeros((n, 2, grid[int(act[0])].support_size))
        for j, i in enumerate(act):
            joint = joint + wa[j] * grid[i].query_joint(records, query)
        raw = _joint_mi(joint)
        slack = np.full(n, 1e-12 * joint.shape[2])
        value = _bounded(raw, slack, upper, "U_q")
        return GoalEIGResult(value, np.zeros(n), raw, p, per_h, upper)
    raise TypeError(f"unknown query type {type(query).__name__}")


def _joint_mi(joint: np.ndarray) -> np.ndarray:
    """I(A; Y) per candidate for joint tables (n, A, S)."""
    pa = joint.sum(axis=2, keepdims=True)
    py = joint.sum(axis=1, keepdims=True)
    denom = pa * py
    pos = joint > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pos, joint / np.where(denom > 0.0, denom, 1.0), 1.0)
    return np.sum(np.where(pos, joint * np.log(ratio), 0.0), axis=(1, 2))


def _contrast_query_mi(
    w: np.ndarray,
    ts: Sequence[StudentTPredictive],
    kernels: Sequence[LinearQueryKernel],
    p: float,
    settings: QuadratureSettings,
) -> tuple[np.ndarray, np.ndarray, int]:
    parts = _TParts.build(ts)
    for j, kern in enumerate(kernels):
        # the kernel's residual y − xᵀμ is formed from the same anchored offsets
        if not np.array_equal(kern.x_mu, parts.loc[:, j]):
            raise ValueError("query kernels and predictives disagree on the predictive mean")
    parts.check_span()
    logw = np.log(w)
    log_p = math.log(p)
    log_1mp = math.log1p(-p)

    def integrand(t: np.ndarray, off: np.ndarray, cand: np.ndarray) -> np.ndarray:
        # f·KL(Bern(r) ‖ Bern(p)) with f = Σ_k π_k p_k and r = Σ_k resp_k·P(q | k, D, y)
        lp = parts.logpdf(t, off, cand)
        a = lp + logw
        m = np.max(a, axis=2)
        e = np.exp(a - m[:, :, None])
        s = np.sum(e, axis=2)
        resp = e / s[:, :, None]
        rq = np.stack(
            [kern.prob_after_resid(t + off[:, j, None], cand) for j, kern in enumerate(kernels)],
            axis=2,
        )
        r = np.clip(np.sum(resp * rq, axis=2), 0.0, 1.0)
        kl = special.xlogy(r, r) - r * log_p + special.xlogy(1.0 - r, 1.0 - r) - (1.0 - r) * log_1mp
        return np.exp(m) * s * kl

    trans = np.stack([kern.transition_points() for kern in kernels], axis=1)
    _, width = parts.frame()
    breaks = np.concatenate([parts.breaks(settings, width), trans], axis=1)
    y_lo, y_hi, outside = parts.limits(w, settings.tail_mass)
    val, err, n_evals = _integrate_line(
        breaks, parts.loc, parts.scale, y_lo, y_hi, integrand, settings
    )
    # outside [y_lo, y_hi]: 0 ≤ f·KL(Bern(r) ‖ Bern(p)) ≤ f·max(−log p, −log(1 − p))
    return val, err + max(-log_p, -log_1mp) * outside, n_evals


# --------------------------------------------------------------------------- test oracles


class SupportsNestedMC(Protocol):
    """Parameter-level access used by the brute-force nested Monte Carlo oracle."""

    def sample_parameters(self, rng: np.random.Generator, size: int) -> Any: ...

    def simulate(self, params: Any, record: Record, rng: np.random.Generator) -> np.ndarray: ...

    def conditional_logpdf(self, params: Any, record: Record, y: np.ndarray) -> np.ndarray: ...


def _sample_predictive(p: Predictive, i: int, rng: np.random.Generator, size: int) -> np.ndarray:
    if isinstance(p, StudentTPredictive):
        nu = float(p.df[i])
        z = rng.standard_normal(size) if math.isinf(nu) else rng.standard_t(nu, size=size)
        return p.loc[i] + p.scale[i] * z
    cdf = np.cumsum(p.probs[i])
    u = rng.random(size) * cdf[-1]
    return np.minimum(np.searchsorted(cdf, u, side="right"), cdf.size - 1)


def _logdens(p: Predictive, i: int, y: np.ndarray) -> np.ndarray:
    if isinstance(p, StudentTPredictive):
        sub = p.take([i])
        return sub.logpdf(y[:, None])[:, 0]
    with np.errstate(divide="ignore"):
        return np.log(p.probs[i][np.asarray(y, dtype=np.int64)])


def mc_model_eig(
    predictives: Sequence[Predictive],
    weights: np.ndarray | Sequence[float],
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """**Test oracle.** Stratified Monte Carlo of U_M = Σ_m π_m E_{y~p_m}[log p_m(y) −
    log f(y)] on the closed-form predictives (``n_samples`` draws per hypothesis
    and candidate). Returns (estimate, standard error) per candidate."""
    _family(predictives)
    w = _check_weights(weights, len(predictives))
    act = _active(w)
    logw = np.log(w[act])
    n = predictives[0].n
    est = np.zeros(n)
    var = np.zeros(n)
    for i in range(n):
        for j, m in enumerate(act):
            y = _sample_predictive(predictives[m], i, rng, n_samples)
            lps = np.stack([_logdens(predictives[k], i, y) for k in act], axis=1)
            lf = logsumexp(lps + logw, axis=1)
            t = lps[:, j] - lf
            est[i] += w[m] * float(np.mean(t))
            var[i] += w[m] ** 2 * float(np.var(t, ddof=1)) / n_samples
    return est, np.sqrt(var)


def nested_mc_model_eig(
    models: Sequence[SupportsNestedMC],
    weights: np.ndarray | Sequence[float],
    record: Record,
    rng: np.random.Generator,
    n_outer: int,
    n_inner: int,
    chunk: int = 256,
) -> tuple[float, float]:
    """**Test oracle.** Brute-force nested Monte Carlo of U_M at one design, using
    only posterior parameter draws and the conditional likelihood (no closed-form
    predictive): outer θ ~ p(θ | m, D), y ~ p(y | θ, m, d) stratified over m;
    inner p̂(y | m') = mean_l p(y | θ'_l, m') over ``n_inner`` fresh draws per m'.
    Biased by O(1/n_inner) like every NMC estimator. Returns (estimate, standard
    error of the outer average)."""
    w = _check_weights(weights, len(models))
    act = _active(w)
    if len(act) == 1:
        return 0.0, 0.0
    inner = {int(m): models[m].sample_parameters(rng, n_inner) for m in act}
    logw = np.log(w[act])
    log_l = math.log(n_inner)
    est = 0.0
    var = 0.0
    for j, m in enumerate(act):
        theta = models[m].sample_parameters(rng, n_outer)
        y = np.asarray(models[m].simulate(theta, record, rng))
        terms = np.empty(n_outer)
        for s in range(0, n_outer, chunk):
            yc = y[s : s + chunk]
            lp = np.stack(
                [
                    logsumexp(models[k].conditional_logpdf(inner[int(k)], record, yc), axis=1)
                    - log_l
                    for k in act
                ],
                axis=1,
            )
            lf = logsumexp(lp + logw, axis=1)
            terms[s : s + chunk] = lp[:, j] - lf
        est += float(w[m]) * float(np.mean(terms))
        var += float(w[m]) ** 2 * float(np.var(terms, ddof=1)) / n_outer
    return est, math.sqrt(var)
