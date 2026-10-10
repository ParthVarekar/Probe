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

__all__ = ["DiscretePredictive", "Predictive", "StudentTPredictive", "student_t_entropy"]


def student_t_entropy(df: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Differential entropy (nats) of t_ν(μ, s):
    log s + ((ν+1)/2)[ψ((ν+1)/2) − ψ(ν/2)] + log(√ν · B(ν/2, 1/2))."""
    df = np.asarray(df, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    return (
        np.log(scale)
        + 0.5 * (df + 1.0) * (special.digamma(0.5 * (df + 1.0)) - special.digamma(0.5 * df))
        + 0.5 * np.log(df)
        + special.betaln(0.5 * df, 0.5)
    )


@dataclass(frozen=True)
class StudentTPredictive:
    """Location-scale Student-t per design: arrays of shape (n,)."""

    df: np.ndarray
    loc: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        n = self.loc.shape
        if self.df.shape != n or self.scale.shape != n or self.loc.ndim != 1:
            raise ValueError("df, loc and scale must be 1-D arrays of equal length")
        if np.any(self.scale <= 0) or np.any(self.df <= 0):
            raise ValueError("Student-t predictive needs positive df and scale")

    @property
    def n(self) -> int:
        return int(self.loc.shape[0])

    def logpdf(self, y: np.ndarray) -> np.ndarray:
        """log density; ``y`` broadcasts against (n,), e.g. shape (n,) or (m, n)."""
        z = (np.asarray(y, dtype=np.float64) - self.loc) / self.scale
        nu = self.df
        return (
            special.gammaln(0.5 * (nu + 1.0))
            - special.gammaln(0.5 * nu)
            - 0.5 * np.log(nu * np.pi)
            - np.log(self.scale)
            - 0.5 * (nu + 1.0) * np.log1p(z * z / nu)
        )

    def entropy(self) -> np.ndarray:
        return student_t_entropy(self.df, self.scale)

    def mean(self) -> np.ndarray:
        return self.loc.copy()

    def variance(self) -> np.ndarray:
        """Infinite where ν ≤ 2."""
        with np.errstate(divide="ignore"):
            v = np.where(self.df > 2, self.scale**2 * self.df / (self.df - 2.0), np.inf)
        return v

    def take(self, idx: np.ndarray | list[int]) -> StudentTPredictive:
        i = np.asarray(idx)
        return StudentTPredictive(self.df[i], self.loc[i], self.scale[i])


@dataclass(frozen=True)
class DiscretePredictive:
    """PMF per design over support 0..S-1: ``probs`` has shape (n, S), rows sum to 1."""

    probs: np.ndarray

    def __post_init__(self) -> None:
        if self.probs.ndim != 2:
            raise ValueError("probs must be (n, S)")
        if np.any(self.probs < -1e-15) or not np.allclose(self.probs.sum(axis=1), 1.0, atol=1e-9):
            raise ValueError("each row of probs must be a probability vector")

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
        return -np.sum(special.xlogy(p, p), axis=1)

    def mean(self) -> np.ndarray:
        return self.probs @ np.arange(self.probs.shape[1], dtype=np.float64)

    def variance(self) -> np.ndarray:
        s = np.arange(self.probs.shape[1], dtype=np.float64)
        m = self.probs @ s
        return self.probs @ (s * s) - m * m

    def take(self, idx: np.ndarray | list[int]) -> DiscretePredictive:
        return DiscretePredictive(self.probs[np.asarray(idx)])


Predictive = StudentTPredictive | DiscretePredictive
