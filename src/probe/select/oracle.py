"""Truth-targeted information for the oracle reference policy P10 (handover §12 Part I).

The oracle knows *which* registered hypothesis T is true (the residual when the
truth lies outside the set), but not its parameters: true parameters would need
each family's hidden simulator, while the hypothesis identity is family-agnostic.
It is therefore an upper-bound reference for the selection problem given the
true structure, not a clairvoyant of the next outcome.

**Objective.** After observing y at design d, the posterior log-odds of T
against the rest change by log p_T(y | d) − log f_{−T}(y | d), where
f_{−T} = Σ_{m≠T} π_m p_m / Σ_{m≠T} π_m is the posterior mixture of the rivals.
When the data come from T, the expected change is

    G(d) = E_{y ~ p_T(·|d)}[log p_T(y | d) − log f_{−T}(y | d)] = KL(p_T ‖ f_{−T}) ≥ 0,

so the oracle takes argmax G(d)/c(d): the cheapest expected progress of the truth's
posterior odds. (Unlike U_M it weights outcomes by the truth's own predictive.)
Each log-ratio is capped at ``LOG_RATIO_CAP`` = 30 nats before averaging: an
outcome that no rival can produce has an infinite ratio, and the stopping rule
needs only log(1/α) ≈ 3 nats, so with the cap decisive designs compare by how
likely they are to be decisive.

**Numerics.**

* Discrete outcomes: an exact finite sum over the support.
* Student-t: E_{Z~t_ν}[h(loc_T + scale_T·Z)] by fixed composite 16-point
  Gauss–Legendre quadrature in the coordinate z = sinh(v), v over ±asinh(z*) with
  z* the t_ν quantile of tail mass 1e−14. Panels: 16 uniform ones in v, split
  further at every rival's location ± {0, 1, 3, 8} of its scale, so narrow or
  distant rival components are resolved. The map is linear near the centre and
  logarithmic in the polynomial tails, so the weighted integrand is smooth; the
  weights are renormalised to sum to one. Deterministic and vectorized over
  candidates; accuracy against QUADPACK on reference cases (narrow, distant,
  heavy-tailed and Gaussian components) is tested. The oracle is a reference, not
  a claimed policy, so a fixed rule is preferred to the adaptive U_M kernel.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import special

from probe.engine.grid import logsumexp
from probe.engine.predictive import DiscretePredictive, Predictive, StudentTPredictive

__all__ = ["LOG_RATIO_CAP", "truth_log_odds_gain"]

LOG_RATIO_CAP = 30.0
_PANELS = 16
_ORDER = 16
_TAIL = 1e-14
_RIVAL_MARKS = (-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0)
"""Extra breakpoints at each rival's location ± these multiples of its scale."""

_GL_X, _GL_W = leggauss(_ORDER)


def truth_log_odds_gain(
    predictives: Sequence[Predictive],
    weights: np.ndarray | Sequence[float],
    target: int,
    cap: float = LOG_RATIO_CAP,
) -> np.ndarray:
    """G(d) per candidate (nats; see the module docstring). ``predictives`` are per
    hypothesis (set order) over the same candidates, ``weights`` the current
    posterior, ``target`` the index of the true hypothesis. Zero when no rival has
    positive weight."""
    w = np.asarray(weights, dtype=np.float64)
    k = len(predictives)
    if w.shape != (k,) or not 0 <= target < k:
        raise ValueError("weights and target must match the predictives")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    n = predictives[0].n
    rivals = [i for i in range(k) if i != target and w[i] > 0.0]
    if n == 0 or not rivals:
        return np.zeros(n)
    wr = w[rivals] / float(np.sum(w[rivals]))
    p_t = predictives[target]
    if isinstance(p_t, DiscretePredictive):
        return _discrete_gain(p_t, [predictives[i] for i in rivals], wr, cap)
    if isinstance(p_t, StudentTPredictive):
        return _student_t_gain(p_t, [predictives[i] for i in rivals], wr, cap)
    raise TypeError(f"unsupported predictive {type(p_t).__name__}")


def _discrete_gain(
    p_t: DiscretePredictive, rivals: Sequence[Predictive], wr: np.ndarray, cap: float
) -> np.ndarray:
    probs = []
    for r in rivals:
        if not isinstance(r, DiscretePredictive) or r.probs.shape != p_t.probs.shape:
            raise TypeError("all predictives must be discrete over one support")
        probs.append(r.probs)
    f = wr[0] * probs[0]
    for j in range(1, len(probs)):
        f = f + wr[j] * probs[j]
    p = p_t.probs
    pos = p > 0.0
    with np.errstate(divide="ignore"):
        lp = np.log(np.where(pos, p, 1.0))
        lf = np.where(f > 0.0, np.log(np.where(f > 0.0, f, 1.0)), -np.inf)
    ratio = np.minimum(lp - lf, cap)
    terms = np.where(pos, p * ratio, 0.0)
    return np.maximum(np.sum(terms, axis=1), 0.0)


def _student_t_gain(
    p_t: StudentTPredictive, rivals: Sequence[Predictive], wr: np.ndarray, cap: float
) -> np.ndarray:
    ts = []
    for r in rivals:
        if not isinstance(r, StudentTPredictive):
            raise TypeError("all predictives must be Student-t")
        ts.append(r)
    n = p_t.n
    v_max = np.arcsinh(-special.stdtrit(p_t.df, _TAIL))  # (n,)
    # panel edges in v per candidate: a uniform grid over [−v_max, v_max] plus each
    # rival's location ± marks·scale (in the truth's standardized coordinate)
    grid = np.linspace(-1.0, 1.0, _PANELS + 1)[None, :] * v_max[:, None]
    marks = np.asarray(_RIVAL_MARKS)
    extra = [
        np.arcsinh(
            ((t.loc - p_t.loc)[:, None] + marks[None, :] * t.scale[:, None]) / p_t.scale[:, None]
        )
        for t in ts
    ]
    edges = np.concatenate([grid, *extra], axis=1)
    edges = np.sort(np.clip(edges, -v_max[:, None], v_max[:, None]), axis=1)  # (n, E)
    half = 0.5 * (edges[:, 1:] - edges[:, :-1])  # (n, E−1); zero-width panels add 0
    mid = 0.5 * (edges[:, 1:] + edges[:, :-1])
    v = (mid[:, :, None] + half[:, :, None] * _GL_X).reshape(n, -1).T  # (Q, n)
    wq = (half[:, :, None] * _GL_W).reshape(n, -1).T
    z = np.sinh(v)
    std = StudentTPredictive(p_t.df, np.zeros_like(p_t.loc), np.ones_like(p_t.scale))
    dens = np.exp(std.logpdf(z)) * np.cosh(v) * wq
    y = p_t.loc[None, :] + p_t.scale[None, :] * z
    log_t = p_t.logpdf(y)
    log_r = np.stack([np.log(wr[j]) + ts[j].logpdf(y) for j in range(len(ts))])  # (R, Q, n)
    log_f = logsumexp(log_r, axis=0)
    h = np.minimum(log_t - log_f, cap)
    gain = np.sum(dens * h, axis=0) / np.sum(dens, axis=0)
    return np.maximum(gain, 0.0)
