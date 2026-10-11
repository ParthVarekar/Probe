"""Per-observation predictive checks (handover §12 H.7): PIT and realized vs
expected surprise under the model-averaged predictive.

At step t the run loop predicted the outcome at the executed design with the
model-averaged predictive f_t = Σ_m π_m p_m(· | d_t, D_{<t}) (the posterior
weights and per-hypothesis predictives *before* the update). For the observed
y_t this module computes:

* **PIT** u_t = F_t(y_t), with F_t = Σ_m π_m F_m the mixture CDF (Student-t
  CDFs, ``scipy.special.stdtr``; Gaussian when ν = ∞). For discrete outcomes the
  **randomized PIT** u_t = F_t(y_t − 1) + V_t·f_t(y_t) is used, V_t ~ U(0, 1)
  drawn from ``make_rng(seed, "pit", step)``: a fixed, replayable stream keyed by
  the step, independent of everything else.
* **Surprise** s_t = −log f_t(y_t) (density or mass), its expectation under f_t,
  the predictive entropy H_t = E_f[−log f_t(Y)], and its variance (the
  "varentropy") V_t = Var_f[−log f_t(Y)]; the **standardized surprise** is
  z_t = (s_t − H_t)/√V_t.

**What "well specified" means here.** If the data really come from the Bayesian
model (a hypothesis drawn from the prior, parameters from its prior, then the
observations: the *prior predictive*) and the weights are the exact posterior
(tempering η = 1), then given the past u_t ~ U(0, 1) exactly (so the u_t are
i.i.d. uniform) and z_t has conditional mean 0 and variance 1. For a *fixed* true
parameter inside a hypothesis this holds only approximately (the predictive is
wider than the truth early on), which is why the alarms built on these
statistics are *calibrated by simulation*, not exact tests (see
:mod:`probe.diagnostics.alarms`).

**Numerics.** For discrete predictives everything is an exact finite sum. For a
Student-t mixture, H and V are expectations Σ_m π_m E_{p_m}[g(Y)] with
g = −log f or (−log f)²; each component's expectation is computed in the
logistic-quantile coordinate y = loc_m + scale_m·Q_ν(expit(s)), where the
integrand is smooth and decays like e^{−|s|}, by fixed composite Gauss–Legendre
(10 nodes per panel; panels of a quarter unit of s over |s| ≤ 36, split further
at every component's loc + {0, ±1, ±2, ..., ±64}·scale mapped into s, so a
narrow component's spike in −log f is resolved at its own scale; for very heavy
tails, ν < 0.18, the range shrinks to |s| ≤ 200ν so the quantiles stay finite).
It is deterministic and agrees with adaptive quadrature to ~1e−10 on the test
suite, including a 1000:1 scale ratio.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy import special, stats

from probe.determinism import make_rng
from probe.engine.grid import logsumexp
from probe.engine.predictive import (
    DiscretePredictive,
    Predictive,
    StudentTPredictive,
    t_log_normalizer,
)

__all__ = [
    "ObservationDiagnostics",
    "RunningDiagnostics",
    "ks_distance",
    "ks_pvalue",
    "mixture_cdf",
    "mixture_log_density",
    "observation_diagnostics",
    "pit_value",
    "surprise_moments",
]

_WEIGHT_SUM_TOL = 1e-9
_S_MAX = 36.0
_PANEL = 0.25
_GL_X, _GL_W = np.polynomial.legendre.leggauss(10)
_BREAK_OFFSETS = np.concatenate([[0.0], *[[-(2.0**k), 2.0**k] for k in range(7)]])


def _check(
    predictives: Sequence[Predictive], weights: np.ndarray | Sequence[float], index: int
) -> tuple[np.ndarray, list[int], bool]:
    """(normalised weights, positively weighted indices, discrete?) after validation."""
    w = np.asarray(weights, dtype=np.float64)
    k = len(predictives)
    if k == 0 or w.shape != (k,):
        raise ValueError(f"weights must have shape ({k},) with at least one predictive")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or abs(float(w.sum()) - 1.0) > _WEIGHT_SUM_TOL:
        raise ValueError("weights must be finite, non-negative and sum to 1")
    discrete = all(isinstance(p, DiscretePredictive) for p in predictives)
    continuous = all(isinstance(p, StudentTPredictive) for p in predictives)
    if not (discrete or continuous):
        raise TypeError("predictives must all be Student-t or all discrete")
    for p in predictives:
        if not 0 <= index < p.n:
            raise IndexError(f"design index {index} outside 0..{p.n - 1}")
    act = [i for i in range(k) if w[i] > 0.0]
    return w / w.sum(), act, discrete


def _t_params(
    predictives: Sequence[Predictive], act: list[int], index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df, loc, scale = [], [], []
    for i in act:
        p = predictives[i]
        assert isinstance(p, StudentTPredictive)
        df.append(float(p.df[index]))
        loc.append(float(p.loc[index]))
        scale.append(float(p.scale[index]))
    return np.asarray(df), np.asarray(loc), np.asarray(scale)


def _mix_probs(
    predictives: Sequence[Predictive], w: np.ndarray, act: list[int], index: int
) -> np.ndarray:
    """Σ_m π_m p_m(· | d) over the support, accumulated in hypothesis order."""
    total = np.zeros(0)
    for n_done, i in enumerate(act):
        p = predictives[i]
        assert isinstance(p, DiscretePredictive)
        row = w[i] * np.clip(p.probs[index], 0.0, None)
        total = row if n_done == 0 else total + row
    return total / float(np.sum(total))


def _log1p_sq(w: np.ndarray) -> np.ndarray:
    """log(1 + w²) without overflow for huge |w|."""
    aw = np.abs(w)
    big = aw > 1e150
    safe = np.where(big, 1.0, aw)
    return np.where(big, 2.0 * np.log(np.where(big, aw, 1.0)), np.log1p(safe * safe))


def _t_logpdf(y: np.ndarray, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """log t_ν(y; loc, scale) for y (..., 1) against components (K,) → (..., K)."""
    z = (y - loc) / scale
    gauss = np.isinf(df)
    nu = np.where(gauss, 1.0, df)
    kernel = np.where(gauss, -0.5 * z * z, -0.5 * (nu + 1.0) * _log1p_sq(z / np.sqrt(nu)))
    return t_log_normalizer(df) - np.log(scale) + kernel


def _t_cdf(y: float, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (y - loc) / scale
    gauss = np.isinf(df)
    return np.where(gauss, special.ndtr(z), special.stdtr(np.where(gauss, 1.0, df), z))


def _logit_cdf(y: np.ndarray, df: float, loc: float, scale: float) -> np.ndarray:
    """s = log F(y) − log(1 − F(y)) for one Student-t component, from both tails
    (±inf where a tail underflows; the caller clips)."""
    z = (y - loc) / scale
    if math.isinf(df):
        lo, hi = special.log_ndtr(z), special.log_ndtr(-z)
    else:
        with np.errstate(divide="ignore"):
            lo, hi = np.log(special.stdtr(df, z)), np.log(special.stdtr(df, -z))
    with np.errstate(invalid="ignore"):
        out = lo - hi
    return np.where(np.isnan(out), 0.0, out)


def _t_quantile(s: np.ndarray, df: float) -> np.ndarray:
    """Standard Student-t quantile at u = expit(s), formed from the smaller tail
    probability expit(−|s|) so it stays accurate for u near 1."""
    tail = special.expit(-np.abs(s))
    q = special.ndtri(tail) if math.isinf(df) else special.stdtrit(df, tail)
    return np.where(s > 0.0, -q, q)


def mixture_log_density(
    predictives: Sequence[Predictive],
    weights: np.ndarray | Sequence[float],
    y: float | int,
    index: int = 0,
) -> float:
    """log f(y) of the model-averaged predictive at design ``index`` (log mass for
    discrete outcomes; −inf when y is impossible under every weighted hypothesis)."""
    w, act, discrete = _check(predictives, weights, index)
    if discrete:
        p = _mix_probs(predictives, w, act, index)
        k = _count_index(y, p.shape[0])
        return float(np.log(p[k])) if p[k] > 0.0 else -math.inf
    df, loc, scale = _t_params(predictives, act, index)
    lp = _t_logpdf(np.asarray([float(y)]), df, loc, scale) + np.log(w[act])
    return float(logsumexp(lp))


def _count_index(y: float | int, support: int) -> int:
    yf = float(y)
    if not math.isfinite(yf) or yf != math.floor(yf) or not 0 <= yf < support:
        raise ValueError(f"discrete outcome must be an integer in 0..{support - 1}, got {y!r}")
    return int(yf)


def mixture_cdf(
    predictives: Sequence[Predictive],
    weights: np.ndarray | Sequence[float],
    y: float | int,
    index: int = 0,
) -> float:
    """F(y) = Σ π_m F_m(y) of the model-averaged predictive (P(Y ≤ y) for counts)."""
    w, act, discrete = _check(predictives, weights, index)
    if discrete:
        p = _mix_probs(predictives, w, act, index)
        k = _count_index(y, p.shape[0])
        return float(min(1.0, np.sum(p[: k + 1])))
    df, loc, scale = _t_params(predictives, act, index)
    return float(np.clip(np.sum(w[act] * _t_cdf(float(y), df, loc, scale)), 0.0, 1.0))


def pit_value(
    predictives: Sequence[Predictive],
    weights: np.ndarray | Sequence[float],
    y: float | int,
    *,
    seed: int,
    step: int,
    index: int = 0,
) -> float:
    """PIT of y under the model-averaged predictive; randomized for discrete
    outcomes with V ~ U(0, 1) from ``make_rng(seed, "pit", step)``."""
    w, act, discrete = _check(predictives, weights, index)
    if not discrete:
        return mixture_cdf(predictives, w, y, index)
    p = _mix_probs(predictives, w, act, index)
    k = _count_index(y, p.shape[0])
    v = float(make_rng(seed, "pit", step).random())
    below = float(np.sum(p[:k]))
    return float(np.clip(below + v * float(p[k]), 0.0, 1.0))


def surprise_moments(
    predictives: Sequence[Predictive],
    weights: np.ndarray | Sequence[float],
    index: int = 0,
) -> tuple[float, float]:
    """(H, V): mean and variance of −log f(Y) for Y ~ f, the model-averaged
    predictive at design ``index`` (exact sums for counts, quadrature otherwise;
    see the module docstring)."""
    w, act, discrete = _check(predictives, weights, index)
    if discrete:
        p = _mix_probs(predictives, w, act, index)
        pos = p > 0.0
        nl = -np.log(p[pos])
        h = float(np.sum(p[pos] * nl))
        return h, float(np.sum(p[pos] * (nl - h) ** 2))
    df, loc, scale = _t_params(predictives, act, index)
    wa = w[act]
    log_wa = np.log(wa)
    m1 = 0.0
    m2 = 0.0
    for c in range(df.shape[0]):
        smax = min(_S_MAX, 200.0 * float(df[c]))
        n_panels = max(8, math.ceil(2.0 * smax / _PANEL))
        # uniform panels plus the other components' loc + {0, ±1, ±2, ..., ±64}·scale
        # mapped into this component's coordinate s = logit F_c(y), so a narrow
        # component's spike in −log f is resolved at its own scale
        ys = (loc[:, None] + scale[:, None] * _BREAK_OFFSETS[None, :]).ravel()
        edges = np.unique(
            np.concatenate(
                [np.linspace(-smax, smax, n_panels + 1), _logit_cdf(ys, df[c], loc[c], scale[c])]
            ).clip(-smax, smax)
        )
        half = 0.5 * (edges[1:] - edges[:-1])
        mid = 0.5 * (edges[1:] + edges[:-1])
        s = (mid[:, None] + half[:, None] * _GL_X[None, :]).ravel()
        ws = (half[:, None] * _GL_W[None, :]).ravel() * special.expit(s) * special.expit(-s)
        y = loc[c] + scale[c] * _t_quantile(s, float(df[c]))
        nl = -logsumexp(_t_logpdf(y[:, None], df, loc, scale) + log_wa, axis=1)
        m1 += float(wa[c]) * float(np.sum(ws * nl))
        m2 += float(wa[c]) * float(np.sum(ws * nl * nl))
    return m1, max(m2 - m1 * m1, 0.0)


@dataclass(frozen=True)
class ObservationDiagnostics:
    """Diagnostics of one observed outcome (JSON-able via :meth:`to_dict`)."""

    step: int
    y: float
    kind: Literal["continuous", "discrete"]
    pit: float
    log_density: float
    """log f_t(y_t) (−inf if impossible under the model average)."""
    surprise: float
    """−log f_t(y_t)."""
    entropy: float
    """H_t = E[−log f_t(Y)]: the expected surprise."""
    varentropy: float
    """Var[−log f_t(Y)]."""
    z: float
    """(surprise − entropy)/√varentropy; 0 for a certain outcome that occurred,
    +inf for an impossible one."""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for k, v in out.items():
            if isinstance(v, float) and not math.isfinite(v):
                out[k] = "inf" if v > 0 else ("-inf" if v < 0 else "nan")
        return out


def observation_diagnostics(
    predictives: Sequence[Predictive],
    weights: np.ndarray | Sequence[float],
    y: float | int,
    *,
    seed: int,
    step: int,
    index: int = 0,
) -> ObservationDiagnostics:
    """PIT, surprise, entropy, varentropy and standardized surprise of one observed
    outcome, from the pre-update ``weights`` and per-hypothesis ``predictives`` (one
    batch per hypothesis, as in :class:`~probe.select.base.ScoredCandidates`)."""
    _, _, discrete = _check(predictives, weights, index)
    pit = pit_value(predictives, weights, y, seed=seed, step=step, index=index)
    logf = mixture_log_density(predictives, weights, y, index)
    h, v = surprise_moments(predictives, weights, index)
    s = -logf
    if s == math.inf:
        z = math.inf
    elif v > 0.0:
        z = (s - h) / math.sqrt(v)
    else:  # a point-mass predictive: the only possible outcome occurred
        z = 0.0
    return ObservationDiagnostics(
        step=int(step),
        y=float(y),
        kind="discrete" if discrete else "continuous",
        pit=pit,
        log_density=logf,
        surprise=s,
        entropy=h,
        varentropy=v,
        z=z,
    )


def ks_distance(u: np.ndarray | Sequence[float]) -> float:
    """Kolmogorov–Smirnov distance sup_x |F̂_n(x) − x| of values in [0, 1] from
    the uniform distribution (0 for no values)."""
    x = np.sort(np.asarray(u, dtype=np.float64))
    n = x.shape[0]
    if n == 0:
        return 0.0
    if np.any(np.isnan(x)) or x[0] < 0.0 or x[-1] > 1.0:
        raise ValueError("PIT values must lie in [0, 1]")
    i = np.arange(1, n + 1, dtype=np.float64)
    return float(max(np.max(i / n - x), np.max(x - (i - 1.0) / n)))


def ks_pvalue(d: float, n: int) -> float:
    """P(D_n ≥ d) for n i.i.d. uniforms (exact one-sample two-sided distribution,
    ``scipy.stats.kstwo``). A fixed-n reference only: monitored after every step,
    it is not a valid sequential test."""
    if n <= 0:
        return 1.0
    return float(stats.kstwo.sf(d, n))


class RunningDiagnostics:
    """Running PIT KS distance and running mean of the standardized surprise over
    the observed outcomes so far."""

    def __init__(self) -> None:
        self._pits: list[float] = []
        self._z_sum = 0.0

    @property
    def n(self) -> int:
        return len(self._pits)

    def update(self, d: ObservationDiagnostics) -> None:
        if not 0.0 <= d.pit <= 1.0:
            raise ValueError("PIT outside [0, 1]")
        if math.isnan(d.z):
            raise ValueError("standardized surprise is NaN")
        self._pits.append(float(d.pit))
        self._z_sum += float(d.z)

    def pits(self) -> np.ndarray:
        return np.asarray(self._pits, dtype=np.float64)

    def pit_ks(self) -> float:
        return ks_distance(self._pits)

    def pit_ks_pvalue(self) -> float:
        return ks_pvalue(self.pit_ks(), self.n)

    def surprise_mean(self) -> float:
        """Running mean of z (0 before the first observation; +inf after an
        impossible one)."""
        return self._z_sum / self.n if self._pits else 0.0
