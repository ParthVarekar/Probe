"""One-step-ahead predictive distributions, batched over candidate designs.

``linear_gaussian`` hypotheses yield :class:`StudentTPredictive` (closed-form NIG
posterior predictive); ``finite_grid`` hypotheses yield :class:`DiscretePredictive`
over the count support 0..max_count. These are the only objects the EIG and
stopping code consume, so engines stay interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

__all__ = [
    "DiscretePredictive",
    "Predictive",
    "StudentTPredictive",
    "student_t_entropy",
    "t_log_normalizer",
]

_SERIES_DF = 50.0
_HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))
_ROW_SUM_TOL = 1e-9


def t_log_normalizer(df: np.ndarray) -> np.ndarray:
    """L(ν) = log Γ((ν+1)/2) − log Γ(ν/2) − ½ log(νπ), the log density of the
    standard t_ν at 0.

    For ν < 50 the log-gamma difference is used directly. For ν ≥ 50 it would lose
    up to ~1e-8 to cancellation (at ν = 1e8), so the asymptotic series
    L(ν) = −½ log 2π − 1/(4ν) + 1/(24ν³) − 1/(20ν⁵) + 17/(112ν⁷) + O(ν⁻⁹)
    (from the Bernoulli expansion of log Γ(x + ½) − log Γ(x), x = ν/2) is used;
    its truncation error is below 1e-15 there."""
    df = np.asarray(df, dtype=np.float64)
    big = df >= _SERIES_DF
    small = np.where(big, 1.0, df)
    direct = (
        special.gammaln(0.5 * (small + 1.0))
        - special.gammaln(0.5 * small)
        - 0.5 * np.log(small * np.pi)
    )
    inv = 1.0 / np.where(big, df, 1.0)
    inv2 = inv * inv
    series = -_HALF_LOG_2PI + inv * (
        -0.25 + inv2 * (1.0 / 24.0 + inv2 * (-1.0 / 20.0 + inv2 * (17.0 / 112.0)))
    )
    return np.where(big, series, direct)


def student_t_entropy(df: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Differential entropy (nats) of t_ν(μ, s):
    log s + ((ν+1)/2)[ψ((ν+1)/2) − ψ(ν/2)] + log(√ν · B(ν/2, 1/2)).

    The last term equals −L(ν) (:func:`t_log_normalizer`). For ν ≥ 50 the
    digamma difference cancels catastrophically (error ∝ ν·ε), so
    ((ν+1)/2)Δψ = ½ + 3/(4ν) + 1/(4ν²) − 1/(8ν³) − 1/(8ν⁴) + 1/(4ν⁵) + 1/(4ν⁶)
    − 17/(16ν⁷) + O(ν⁻⁸) is used instead; it tends to the Gaussian ½ log(2πe s²)."""
    df = np.asarray(df, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    big = df >= _SERIES_DF
    small = np.where(big, 1.0, df)
    direct = (
        0.5 * (small + 1.0) * (special.digamma(0.5 * (small + 1.0)) - special.digamma(0.5 * small))
    )
    v = 1.0 / np.where(big, df, 1.0)
    series = 0.5 + v * (
        0.75
        + v * (0.25 + v * (-0.125 + v * (-0.125 + v * (0.25 + v * (0.25 + v * (-17.0 / 16.0))))))
    )
    return np.log(scale) + np.where(big, series, direct) - t_log_normalizer(df)


@dataclass(frozen=True)
class StudentTPredictive:
    """Location-scale Student-t per design: arrays of shape (n,). ``df`` may be +inf
    (the Gaussian limit); NaN and non-finite loc/scale are rejected."""

    df: np.ndarray
    loc: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        n = self.loc.shape
        if self.df.shape != n or self.scale.shape != n or self.loc.ndim != 1:
            raise ValueError("df, loc and scale must be 1-D arrays of equal length")
        # NaN compares False with everything, so validate positively: finite loc and
        # scale, scale > 0, and df > 0 or +inf (the Gaussian limit).
        if not (np.all(np.isfinite(self.loc)) and np.all(np.isfinite(self.scale))):
            raise ValueError("Student-t predictive needs finite loc and scale")
        if not (np.all(self.scale > 0) and np.all(self.df > 0)):
            raise ValueError("Student-t predictive needs positive df and scale (no NaN)")

    @property
    def n(self) -> int:
        return int(self.loc.shape[0])

    def logpdf(self, y: np.ndarray) -> np.ndarray:
        """log density; ``y`` broadcasts against (n,), e.g. shape (n,) or (m, n).
        ν = +inf is the Gaussian N(loc, scale²)."""
        z = (np.asarray(y, dtype=np.float64) - self.loc) / self.scale
        nu = self.df
        gauss = np.isinf(nu)
        nu_f = np.where(gauss, 1.0, nu)
        kernel = np.where(gauss, -0.5 * z * z, -0.5 * (nu_f + 1.0) * np.log1p(z * z / nu_f))
        return t_log_normalizer(nu) - np.log(self.scale) + kernel

    def entropy(self) -> np.ndarray:
        return student_t_entropy(self.df, self.scale)

    def mean(self) -> np.ndarray:
        """The location (the mean wherever it exists, ν > 1)."""
        return self.loc.copy()

    def variance(self) -> np.ndarray:
        """scale²·ν/(ν − 2); infinite where ν ≤ 2, scale² in the Gaussian limit ν = ∞."""
        nu = self.df
        finite = np.isfinite(nu)
        nu_f = np.where(finite & (nu > 2.0), nu, 3.0)  # no inf/inf (a NaN plus a warning)
        ratio = np.where(finite, nu_f / (nu_f - 2.0), 1.0)
        return np.where(nu > 2.0, self.scale**2 * ratio, np.inf)

    def take(self, idx: np.ndarray | list[int]) -> StudentTPredictive:
        i = np.asarray(idx)
        return StudentTPredictive(self.df[i], self.loc[i], self.scale[i])


@dataclass(frozen=True)
class DiscretePredictive:
    """PMF per design over support 0..S-1: ``probs`` has shape (n, S), rows sum to 1
    within 1e-9 (absolute; the information kernels treat each row as exact)."""

    probs: np.ndarray

    def __post_init__(self) -> None:
        if self.probs.ndim != 2:
            raise ValueError("probs must be (n, S)")
        if not np.all(np.isfinite(self.probs)) or np.any(self.probs < -1e-15):
            raise ValueError("each row of probs must be a probability vector")
        if np.any(np.abs(self.probs.sum(axis=1) - 1.0) > _ROW_SUM_TOL):
            raise ValueError("each row of probs must sum to 1 (within 1e-9)")

    @property
    def n(self) -> int:
        return int(self.probs.shape[0])

    def logpmf(self, y: np.ndarray) -> np.ndarray:
        """log P(y) per design; ``y`` is an int array of shape (n,)."""
        y = np.asarray(y, dtype=np.int64)
        p = self.probs[np.arange(self.n), y]
        with np.errstate(divide="ignore"):
            return np.log(p)

    def entropy(self) -> np.ndarray:
        p = np.clip(self.probs, 0.0, 1.0)
        return 0.0 - np.sum(special.xlogy(p, p), axis=1)  # 0.0 − (+0) is +0, never −0

    def mean(self) -> np.ndarray:
        # row sums rather than a BLAS matrix-vector product, whose result for a row can
        # depend on its position in the batch
        return np.sum(self.probs * np.arange(self.probs.shape[1], dtype=np.float64), axis=1)

    def variance(self) -> np.ndarray:
        """Σ_y p(y)(y − mean)², centred so it never cancels below zero."""
        s = np.arange(self.probs.shape[1], dtype=np.float64)
        dev = s[None, :] - self.mean()[:, None]
        return np.sum(self.probs * dev * dev, axis=1)

    def take(self, idx: np.ndarray | list[int]) -> DiscretePredictive:
        return DiscretePredictive(self.probs[np.asarray(idx)])


Predictive = StudentTPredictive | DiscretePredictive
