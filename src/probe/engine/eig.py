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
  estimate) over the real line in the coordinate y = c + w·sinh(v), under which
  the polynomial Student-t tails decay exponentially; the range extends until the
  heaviest tail has decayed by e^−46 (far beyond the plan's "±12 scale units"),
  and an estimate of the cut-off mass is added to the error.
  Initial breakpoints sit at loc_m + {0, ±5, ±25, ±125, ...}·scale_m for every
  component (geometric out to the span of all components), so narrow components
  inside wide ones and tail mass between far-apart components are always seen.
  Panels are bisected until each meets its share of the tolerance (error budgets
  halve on bisection), every candidate in one array pass per round. Log densities
  are combined with log-sum-exp. The per-candidate error estimate is the sum of
  the accepted panels' QUADPACK estimates (conservative: on the reference suite
  the true error is orders of magnitude below it).

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
    """Shannon entropy (nats) of a probability vector, 0·log 0 = 0."""
    w = np.asarray(weights, dtype=np.float64)
    return float(-np.sum(special.xlogy(w, w)))


def binary_entropy(p: np.ndarray | float) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return np.asarray(-special.xlogy(q, q) - special.xlogy(1.0 - q, 1.0 - q))


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
    """Adaptive Gauss–Kronrod controls. The tolerance per candidate is
    max(atol, rtol·|initial estimate|); it is split evenly over the initial panels
    and halved on every bisection. The tails are integrated out to where the
    heaviest component has decayed by exp(−``tail_decay``) beyond its outermost
    breakpoint, in ``tail_splits`` initial panels per side."""

    atol: float = 1e-9
    rtol: float = 1e-7
    max_rounds: int = 40
    max_panels: int = 2_000_000
    marks: tuple[float, ...] = field(default=(-5.0, 0.0, 5.0))
    geometric_ratio: float = 5.0
    tail_decay: float = 46.0
    tail_splits: int = 4


_DEFAULT_QUAD = QuadratureSettings()
_Y_CAP = 1e100
"""Integration never extends beyond |y − c| = 1e100 × the narrowest feature, so
standardized values stay far from float overflow."""


def _integrate_line(
    breaks: np.ndarray,
    center: np.ndarray,
    width: np.ndarray,
    feature: np.ndarray,
    tail_rate: np.ndarray,
    integrand: Callable[[np.ndarray, np.ndarray], np.ndarray],
    settings: QuadratureSettings,
) -> tuple[np.ndarray, np.ndarray, int]:
    """∫_ℝ g_i(y) dy for candidates i = 0..n−1, vectorized adaptive G7/K15.

    ``breaks`` (n, B) are initial breakpoints in y (NaN = unused); ``integrand(y,
    cand)`` receives node values y of shape (P, 15) and the candidate index of each
    panel row, and returns g of the same shape. Returns (values, error estimates,
    number of integrand evaluations).

    Coordinates: y = c + w·sinh(v). Near the components (|y − c| ≲ w) the map is
    almost linear; in the tails a Student-t integrand ∝ |y|^−(ν+1) becomes
    ∝ exp(−ν|v|), smooth and exponentially decaying, so there is no endpoint
    singularity even for ν < 1 and no loss of coordinate precision. ``tail_rate``
    (n,) is that decay rate (the smallest ν). The range is cut where the
    integrand has decayed by exp(−tail_decay) past the outermost breakpoint (or
    at the overflow guard); the cut-off mass is estimated as 2|g(v_end)|/rate and
    added to the error.

    The error estimate is the sum of (i) the accepted panels' QUADPACK estimates,
    (ii) that truncation estimate and (iii) a *coordinate-resolution* term: nodes
    are represented to about ε·(|c| + 10w) in y, which perturbs features of width
    ``feature`` (n,) by that relative amount, so 4ε(|c| + 10w)/feature · ∫|g| is
    added. (iii) is negligible unless components span ≳ 10⁶ of their scales."""
    n = breaks.shape[0]
    with np.errstate(invalid="ignore"):
        v = np.arcsinh((breaks - center[:, None]) / width[:, None])
    finite = np.isfinite(v)
    v_min = np.min(np.where(finite, v, np.inf), axis=1)
    v_max = np.max(np.where(finite, v, -np.inf), axis=1)
    v_min = np.where(np.isfinite(v_min), v_min, 0.0)
    v_max = np.where(np.isfinite(v_max), v_max, 0.0)
    v_cap = np.arcsinh(_Y_CAP * feature / width)
    reach = settings.tail_decay / tail_rate
    v_lo = np.maximum(v_min - reach, -v_cap)
    v_hi = np.minimum(v_max + reach, v_cap)
    steps = np.arange(1, settings.tail_splits, dtype=np.float64) / settings.tail_splits
    left = v_lo[:, None] + (v_min - v_lo)[:, None] * steps
    right = v_max[:, None] + (v_hi - v_max)[:, None] * steps
    inner = np.where(finite, np.clip(v, v_lo[:, None], v_hi[:, None]), v_hi[:, None])
    edges = np.sort(
        np.concatenate([v_lo[:, None], left, inner, right, v_hi[:, None]], axis=1), axis=1
    )
    lo = edges[:, :-1].ravel()
    hi = edges[:, 1:].ravel()
    cand = np.repeat(np.arange(n), edges.shape[1] - 1)
    keep = hi > lo
    lo, hi, cand = lo[keep], hi[keep], cand[keep]
    n_init = np.bincount(cand, minlength=n).astype(np.float64)

    val = np.zeros(n)
    err = np.zeros(n)
    absint = np.zeros(n)
    budget: np.ndarray | None = None
    n_evals = 0
    for rnd in range(settings.max_rounds):
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        vv = mid[:, None] + half[:, None] * _GK_NODES
        wc = width[cand][:, None]
        y = center[cand][:, None] + wc * np.sinh(vv)
        g = integrand(y, cand) * (wc * np.cosh(vv))
        n_evals += g.size
        resk = g @ _GK_WK
        resg = g @ _GK_WG
        mean = 0.5 * resk
        resabs = np.abs(g) @ _GK_WK
        resasc = np.abs(g - mean[:, None]) @ _GK_WK
        raw = np.abs(resk - resg)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = resasc * np.minimum(1.0, (200.0 * raw / resasc) ** 1.5)
        e = np.where((resasc > 0.0) & (raw > 0.0), scaled, raw)
        e = np.maximum(e, 50.0 * _EPS * resabs) * half
        k = resk * half
        if budget is None:
            est = np.bincount(cand, weights=k, minlength=n)
            tol = np.maximum(settings.atol, settings.rtol * np.abs(est))
            budget = (tol / np.maximum(n_init, 1.0))[cand]
        last = rnd == settings.max_rounds - 1 or 2 * lo.size > settings.max_panels
        tiny = half <= 4.0 * _EPS * np.maximum(1.0, np.abs(mid))  # nodes no longer distinct
        ok = (e <= budget) | tiny | last
        val += np.bincount(cand[ok], weights=k[ok], minlength=n)
        err += np.bincount(cand[ok], weights=e[ok], minlength=n)
        absint += np.bincount(cand[ok], weights=resabs[ok] * half[ok], minlength=n)
        if ok.all():
            break
        bad = ~ok
        lo_b, hi_b, mid_b, c_b, b_b = lo[bad], hi[bad], mid[bad], cand[bad], budget[bad]
        lo = np.concatenate([lo_b, mid_b])
        hi = np.concatenate([mid_b, hi_b])
        cand = np.concatenate([c_b, c_b])
        budget = np.concatenate([0.5 * b_b, 0.5 * b_b])
    ends = np.stack([v_lo, v_hi], axis=1)
    g_end = integrand(center[:, None] + width[:, None] * np.sinh(ends), np.arange(n))
    g_end = g_end * width[:, None] * np.cosh(ends)
    err += 2.0 * np.sum(np.abs(g_end), axis=1) / tail_rate
    err += 4.0 * _EPS * (np.abs(center) + 10.0 * width) / feature * absint
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
        const = t_log_normalizer(df) - np.log(scale)
        return _TParts(loc, scale, df, const, -0.5 * (df + 1.0), 1.0 / scale, 1.0 / df)

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

    def logpdf(self, y: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """log p_k(y) for node values y (P, Q): shape (P, Q, K)."""
        z = y[:, :, None] - self.loc[cand][:, None, :]
        z *= self.inv_scale[cand][:, None, :]
        z *= z
        z *= self.inv_df[cand][:, None, :]
        np.log1p(z, out=z)
        z *= self.coef[cand][:, None, :]
        z += self.const[cand][:, None, :]
        return z

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
        n_ext = min(n_ext, 60)
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
    logw = np.log(w)

    def integrand(y: np.ndarray, cand: np.ndarray) -> np.ndarray:
        # Σ_k π_k p_k (log p_k − log f) with d_k = log π_k + log p_k − m (m = max_k),
        # e_k = exp(d_k), s = Σ_k e_k:  log p_k − log f = d_k − log π_k − log s, so
        # the integrand is e^m [Σ_k e_k (d_k − log π_k) − s log s]. The dominant
        # component contributes −log π_k − log s with no large cancelling terms.
        d = sub.logpdf(y, cand)
        d += logw
        m = np.max(d, axis=2)
        d -= m[:, :, None]
        e = np.exp(d)
        s = np.sum(e, axis=2)
        inner = np.einsum("pqk,pqk->pq", e, d) - e @ logw - s * np.log(s)
        return np.exp(m) * inner

    center, width = sub.frame()
    feature = np.min(sub.scale, axis=1)
    rate = np.min(sub.df, axis=1)
    val, er, n_evals = _integrate_line(
        sub.breaks(settings, width), center, width, feature, rate, integrand, settings
    )
    raw[todo] = val
    err[todo] = er
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
    logw = np.log(w)
    log_p = math.log(p)
    log_1mp = math.log1p(-p)

    def integrand(y: np.ndarray, cand: np.ndarray) -> np.ndarray:
        lp = parts.logpdf(y, cand)
        a = lp + logw
        m = np.max(a, axis=2)
        e = np.exp(a - m[:, :, None])
        s = np.sum(e, axis=2)
        resp = e / s[:, :, None]
        rq = np.stack([k.prob_after(y, cand) for k in kernels], axis=2)
        r = np.clip(np.sum(resp * rq, axis=2), 0.0, 1.0)
        kl = special.xlogy(r, r) - r * log_p + special.xlogy(1.0 - r, 1.0 - r) - (1.0 - r) * log_1mp
        return np.exp(m) * s * kl

    trans = np.stack([k.transition_points() for k in kernels], axis=1)
    center, width = parts.frame()
    breaks = np.concatenate([parts.breaks(settings, width), trans], axis=1)
    widths = np.concatenate([parts.scale, *(k.transition_widths()[:, None] for k in kernels)], 1)
    feature = np.nanmin(np.where(widths > 0.0, widths, np.nan), axis=1)
    feature = np.where(np.isfinite(feature), feature, np.min(parts.scale, axis=1))
    rate = np.min(parts.df, axis=1)
    return _integrate_line(breaks, center, width, feature, rate, integrand, settings)


# --------------------------------------------------------------------------- test oracles


class SupportsNestedMC(Protocol):
    """Parameter-level access used by the brute-force nested Monte Carlo oracle."""

    def sample_parameters(self, rng: np.random.Generator, size: int) -> Any: ...

    def simulate(self, params: Any, record: Record, rng: np.random.Generator) -> np.ndarray: ...

    def conditional_logpdf(self, params: Any, record: Record, y: np.ndarray) -> np.ndarray: ...


def _sample_predictive(p: Predictive, i: int, rng: np.random.Generator, size: int) -> np.ndarray:
    if isinstance(p, StudentTPredictive):
        return p.loc[i] + p.scale[i] * rng.standard_t(p.df[i], size=size)
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
