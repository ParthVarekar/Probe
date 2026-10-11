"""Anytime-valid confidence sequences for a mean (handover §12 H.8 (ii)).

A (1 − α) confidence sequence (CS) is a sequence of intervals C_1, C_2, ... with

    P(∀t ≥ 1: μ ∈ C_t) ≥ 1 − α,

so it may be monitored after every observation and the run stopped at any
data-dependent time without inflating the error (naive fixed-n intervals are
forbidden on adaptively collected data, H.8). Implemented in-house because the
reference package (``confseq``) has no Windows or Python ≥ 3.11 wheels.

Assumption for every method: the observations X_1, X_2, ... have a *common
conditional mean* μ given the past, E[X_t | X_{<t}] = μ. Independence is not
needed, so observations whose noise or selection depends on earlier data (an
adaptive design that keeps the target mean fixed) are covered. Conditional on the
past, each X_t must also satisfy the method's tail condition:

**Sub-Gaussian, normal mixture** (:func:`subgaussian_cs`; Robbins 1970; Howard,
Ramdas, McAuliffe & Sekhon 2021, "Time-uniform, nonparametric, nonasymptotic
confidence sequences", Ann. Statist. 49(2), §3). If E[exp(λ(X_t − μ)) | past] ≤
exp(λ²σ²/2) for all λ (σ known, e.g. Gaussian noise with known sd), then with
S_t = Σ(X_i − μ) and V_t = tσ² the mixture over λ ~ N(0, 1/ρ),

    M_t = sqrt(ρ/(V_t + ρ)) · exp(S_t² / (2(V_t + ρ))),

is a non-negative supermartingale with M_0 = 1, and Ville's inequality
P(∃t: M_t ≥ 1/α) ≤ α gives the two-sided boundary

    |S_t| < sqrt((V_t + ρ) · log((V_t + ρ) / (ρ α²)))      for all t, w.p. ≥ 1 − α.

ρ > 0 tunes where the CS is tightest: :func:`optimal_rho` minimises the boundary
at a planned intrinsic time V = σ²·t_opt (the boundary is valid for every ρ).

**Bounded, predictable plug-in empirical Bernstein** (:func:`bounded_cs`,
``method="eb"``; Waudby-Smith & Ramdas 2023, "Estimating means of bounded random
variables by betting", JRSS-B 86(1), Theorem 2). For X_t ∈ [0, 1] (rescaled from
[lo, hi]) and predictable λ_t ∈ [0, 1):

    C_t = Σλ_i X_i / Σλ_i  ±  (log(2/α) + Σ v_i ψ_E(λ_i)) / Σλ_i,
    v_i = 4(X_i − μ̂_{i−1})²,  ψ_E(λ) = (−log(1 − λ) − λ)/4,
    λ_t = min(c, sqrt(2 log(2/α) / (σ̂²_{t−1} · t · log(1 + t)))),
    μ̂_t = (½ + Σ_{i≤t} X_i)/(t + 1),  σ̂²_t = (¼ + Σ_{i≤t}(X_i − μ̂_i)²)/(t + 1),

with c = ½ by default. It adapts to the (unknown) variance, so it is much
tighter than Hoeffding for low-variance data. ``method="hoeffding"`` instead uses
the normal mixture with σ = (hi − lo)/2 (Hoeffding's lemma).

**Running intersection.** Since every C_t covers μ simultaneously with
probability ≥ 1 − α, the running intersection ∩_{s ≤ t} C_s is also a (1 − α)-CS
(the default). An empty intersection (lower > upper) can only happen on the
miscoverage event and signals that the assumptions are violated; it is returned
as is, never repaired.

All computations are closed form and deterministic; the batch functions work on
arrays whose *last* axis is time, so many replications can be evaluated at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize

__all__ = [
    "BoundedMeanCS",
    "ConfidenceSequence",
    "SubGaussianMeanCS",
    "bounded_cs",
    "normal_mixture_boundary",
    "optimal_rho",
    "subgaussian_cs",
]

DEFAULT_T_OPT = 100
"""Default planned sample size at which the normal-mixture CS is tightest."""


@dataclass(frozen=True)
class ConfidenceSequence:
    """Bounds after each of t = 1..T observations (arrays of shape (..., T))."""

    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    method: str

    def covers(self, mu: float | np.ndarray) -> np.ndarray:
        """Whether μ lies in every interval up to each time (shape (..., T))."""
        m = np.asarray(mu, dtype=np.float64)
        if m.ndim:
            m = m[..., None]  # one μ per replication, broadcast over time
        inside = (self.lower <= m) & (m <= self.upper)
        return np.asarray(np.logical_and.accumulate(inside, axis=-1))


def _check_alpha(alpha: float) -> float:
    a = float(alpha)
    if not 0.0 < a < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return a


def normal_mixture_boundary(v: float | np.ndarray, alpha: float, rho: float) -> np.ndarray:
    """Two-sided normal-mixture boundary on |S_t| at intrinsic time v = Σσ²:
    sqrt((v + ρ)·log((v + ρ)/(ρα²))) (module docstring)."""
    a = _check_alpha(alpha)
    if not rho > 0.0:
        raise ValueError("rho must be positive")
    vv = np.asarray(v, dtype=np.float64)
    if np.any(vv < 0.0):
        raise ValueError("intrinsic time must be non-negative")
    return np.sqrt((vv + rho) * np.log((vv + rho) / (rho * a * a)))


def optimal_rho(v_opt: float, alpha: float) -> float:
    """ρ minimising the normal-mixture boundary at intrinsic time ``v_opt``.

    With u = ρ/v the boundary² is v(1 + u)·log((1 + u)/(uα²)), whose derivative in
    u vanishes where g(u) = log((1 + u)/(uα²)) − 1/u = 0; g is strictly
    increasing (g′ = 1/(u²(1 + u))), so the root is unique and is found by
    bracketing (deterministic)."""
    a = _check_alpha(alpha)
    if not v_opt > 0.0:
        raise ValueError("v_opt must be positive")
    log_a2 = 2.0 * math.log(a)

    def g(u: float) -> float:
        return math.log1p(1.0 / u) - log_a2 - 1.0 / u

    u = optimize.brentq(g, 1e-12, 1e12, xtol=1e-15, rtol=1e-13, maxiter=500)
    return float(u) * float(v_opt)


def _running(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum.accumulate(lower, axis=-1), np.minimum.accumulate(upper, axis=-1)


def subgaussian_cs(
    x: np.ndarray,
    sigma: float,
    alpha: float = 0.05,
    *,
    t_opt: int = DEFAULT_T_OPT,
    rho: float | None = None,
    running_intersection: bool = True,
) -> ConfidenceSequence:
    """Normal-mixture CS for the common conditional mean of conditionally
    σ-sub-Gaussian observations ``x`` (time on the last axis). ``rho`` defaults to
    :func:`optimal_rho` at intrinsic time σ²·``t_opt``."""
    a = _check_alpha(alpha)
    if not sigma > 0.0:
        raise ValueError("sigma must be positive")
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 0 or xx.shape[-1] == 0:
        raise ValueError("need at least one observation")
    if not np.all(np.isfinite(xx)):
        raise ValueError("observations must be finite")
    if rho is None:
        if t_opt < 1:
            raise ValueError("t_opt must be at least 1")
        rho = optimal_rho(sigma * sigma * t_opt, a)
    t = np.arange(1, xx.shape[-1] + 1, dtype=np.float64)
    est = np.cumsum(xx, axis=-1) / t
    radius = normal_mixture_boundary(sigma * sigma * t, a, rho) / t
    lower, upper = est - radius, est + radius
    if running_intersection:
        lower, upper = _running(lower, upper)
    return ConfidenceSequence(est, lower, upper, a, "subgaussian_normal_mixture")


def _eb_unit(z: np.ndarray, alpha: float, c: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PrPl-EB on [0, 1] data (time last): (centre, lower, upper) per time."""
    t = np.arange(1, z.shape[-1] + 1, dtype=np.float64)
    log_term = math.log(2.0 / alpha)
    mu_hat = (0.5 + np.cumsum(z, axis=-1)) / (t + 1.0)  # μ̂_t, t = 1..T
    mu_prev = np.concatenate([np.full((*z.shape[:-1], 1), 0.5), mu_hat[..., :-1]], axis=-1)
    sq = (z - mu_hat) ** 2
    var_hat = (0.25 + np.cumsum(sq, axis=-1)) / (t + 1.0)  # σ̂²_t
    var_prev = np.concatenate([np.full((*z.shape[:-1], 1), 0.25), var_hat[..., :-1]], axis=-1)
    lam = np.minimum(c, np.sqrt(2.0 * log_term / (var_prev * t * np.log1p(t))))
    v = 4.0 * (z - mu_prev) ** 2
    psi = (-np.log1p(-lam) - lam) / 4.0
    sum_lam = np.cumsum(lam, axis=-1)
    centre = np.cumsum(lam * z, axis=-1) / sum_lam
    radius = (log_term + np.cumsum(v * psi, axis=-1)) / sum_lam
    return centre, np.maximum(centre - radius, 0.0), np.minimum(centre + radius, 1.0)


def bounded_cs(
    x: np.ndarray,
    lo: float,
    hi: float,
    alpha: float = 0.05,
    *,
    method: Literal["eb", "hoeffding"] = "eb",
    c: float = 0.5,
    t_opt: int = DEFAULT_T_OPT,
    running_intersection: bool = True,
) -> ConfidenceSequence:
    """CS for the common conditional mean of observations in [lo, hi] (time on the
    last axis): predictable plug-in empirical Bernstein (default) or Hoeffding via
    the normal mixture. Values outside [lo, hi] are rejected."""
    a = _check_alpha(alpha)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError("need finite lo < hi")
    if not 0.0 < c < 1.0:
        raise ValueError("c must lie in (0, 1)")
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 0 or xx.shape[-1] == 0:
        raise ValueError("need at least one observation")
    if not np.all(np.isfinite(xx)) or np.any(xx < lo) or np.any(xx > hi):
        raise ValueError(f"observations must lie in [{lo}, {hi}]")
    width = hi - lo
    z = (xx - lo) / width
    if method == "eb":
        centre, lower, upper = _eb_unit(z, a, c)
        name = "bounded_predictable_plugin_eb"
    elif method == "hoeffding":
        cs = subgaussian_cs(z, 0.5, a, t_opt=t_opt, running_intersection=False)
        centre = cs.estimate
        lower, upper = np.maximum(cs.lower, 0.0), np.minimum(cs.upper, 1.0)
        name = "bounded_hoeffding_normal_mixture"
    else:
        raise ValueError(f"unknown method {method!r}")
    if running_intersection:
        lower, upper = _running(lower, upper)
    return ConfidenceSequence(lo + width * centre, lo + width * lower, lo + width * upper, a, name)


# --------------------------------------------------------------------------- incremental


class SubGaussianMeanCS:
    """Incremental form of :func:`subgaussian_cs` for the run loop: O(1) per update,
    the same numbers as the batch function on the same prefix."""

    def __init__(
        self,
        sigma: float,
        alpha: float = 0.05,
        *,
        t_opt: int = DEFAULT_T_OPT,
        rho: float | None = None,
        running_intersection: bool = True,
    ) -> None:
        self.alpha = _check_alpha(alpha)
        if not sigma > 0.0:
            raise ValueError("sigma must be positive")
        if rho is None and t_opt < 1:
            raise ValueError("t_opt must be at least 1")
        self.sigma = float(sigma)
        self.rho = optimal_rho(sigma * sigma * t_opt, self.alpha) if rho is None else float(rho)
        self.running_intersection = running_intersection
        self._n = 0
        self._sum = 0.0
        self._lo = -math.inf
        self._hi = math.inf

    @property
    def n(self) -> int:
        return self._n

    def update(self, x: float) -> None:
        xf = float(x)
        if not math.isfinite(xf):
            raise ValueError("observation must be finite")
        self._n += 1
        self._sum += xf
        est = self._sum / self._n
        v = self.sigma * self.sigma * self._n
        r = float(normal_mixture_boundary(v, self.alpha, self.rho)) / self._n
        lo, hi = est - r, est + r
        if self.running_intersection:
            lo, hi = max(self._lo, lo), min(self._hi, hi)
        self._lo, self._hi = lo, hi

    def estimate(self) -> float:
        return self._sum / self._n if self._n else math.nan

    def interval(self) -> tuple[float, float]:
        """(lower, upper); (−inf, +inf) before the first observation."""
        return self._lo, self._hi


class BoundedMeanCS:
    """Incremental predictable plug-in empirical-Bernstein CS (:func:`bounded_cs`
    with ``method="eb"``) for observations in [lo, hi]."""

    def __init__(
        self,
        lo: float,
        hi: float,
        alpha: float = 0.05,
        *,
        c: float = 0.5,
        running_intersection: bool = True,
    ) -> None:
        self.alpha = _check_alpha(alpha)
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            raise ValueError("need finite lo < hi")
        if not 0.0 < c < 1.0:
            raise ValueError("c must lie in (0, 1)")
        self.lo, self.hi, self.c = float(lo), float(hi), float(c)
        self.running_intersection = running_intersection
        self._log_term = math.log(2.0 / self.alpha)
        self._n = 0
        self._sum_z = 0.0
        self._sum_sq = 0.0
        self._mu_prev = 0.5
        self._var_prev = 0.25
        self._sum_lam = 0.0
        self._sum_lam_z = 0.0
        self._sum_vpsi = 0.0
        self._centre = math.nan
        self._lo_u = 0.0
        self._hi_u = 1.0

    @property
    def n(self) -> int:
        return self._n

    def update(self, x: float) -> None:
        xf = float(x)
        if not (math.isfinite(xf) and self.lo <= xf <= self.hi):
            raise ValueError(f"observation must lie in [{self.lo}, {self.hi}]")
        z = (xf - self.lo) / (self.hi - self.lo)
        self._n += 1
        t = float(self._n)
        lam = min(self.c, math.sqrt(2.0 * self._log_term / (self._var_prev * t * math.log1p(t))))
        v = 4.0 * (z - self._mu_prev) ** 2
        psi = (-math.log1p(-lam) - lam) / 4.0
        self._sum_lam += lam
        self._sum_lam_z += lam * z
        self._sum_vpsi += v * psi
        centre = self._sum_lam_z / self._sum_lam
        radius = (self._log_term + self._sum_vpsi) / self._sum_lam
        lo_u, hi_u = max(centre - radius, 0.0), min(centre + radius, 1.0)
        if self.running_intersection:
            lo_u, hi_u = max(self._lo_u, lo_u), min(self._hi_u, hi_u)
        self._centre, self._lo_u, self._hi_u = centre, lo_u, hi_u
        # plug-in estimates for the next step
        self._sum_z += z
        mu_hat = (0.5 + self._sum_z) / (t + 1.0)
        self._sum_sq += (z - mu_hat) ** 2
        self._var_prev = (0.25 + self._sum_sq) / (t + 1.0)
        self._mu_prev = mu_hat

    def estimate(self) -> float:
        return self.lo + (self.hi - self.lo) * self._centre if self._n else math.nan

    def interval(self) -> tuple[float, float]:
        """(lower, upper); (lo, hi) before the first observation."""
        w = self.hi - self.lo
        return self.lo + w * self._lo_u, self.lo + w * self._hi_u
