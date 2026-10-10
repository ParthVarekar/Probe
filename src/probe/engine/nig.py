"""Exact Normal-Inverse-Gamma engine for ``linear_gaussian`` hypotheses.

Model: y = φ(d)ᵀβ + ε, ε ~ N(0, σ²), β | σ² ~ N(μ₀, σ² V₀), σ² ~ InvGamma(a, b).

State is kept as (μ, Σ = Λ⁻¹, a, b) and updated one observation at a time with
the Sherman–Morrison rank-1 identity, so each step is O(p²) and the prequential
decomposition log p(y₁..ₙ) = Σ log p(yᵢ | y<ᵢ) holds by construction.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import special

from probe.engine.ir import (
    CovariateSpec,
    LinearGaussianHypothesis,
    Record,
    column_names,
    design_matrix,
    prior_arrays,
)
from probe.engine.predictive import StudentTPredictive

__all__ = ["NIGModel"]

_LOG_2PI_E = math.log(2.0 * math.pi * math.e)


class NIGModel:
    def __init__(self, h: LinearGaussianHypothesis, covariates: Sequence[CovariateSpec]) -> None:
        self.hypothesis_id = h.id
        self.is_residual = h.is_residual
        self._h = h
        self._cov = tuple(covariates)
        self._columns = column_names(h)
        mu0, v0 = prior_arrays(h)
        self._mu = mu0.copy()
        self._sigma = np.diag(v0)  # Σ = Λ⁻¹ (unscaled by σ²)
        self._a = float(h.prior.a0)
        self._b = float(h.prior.b0)
        self._log_evidence = 0.0
        self._xs: list[np.ndarray] = []
        self._ys: list[float] = []

    # -- helpers -------------------------------------------------------------
    def _phi(self, records: Sequence[Record]) -> np.ndarray:
        return design_matrix(self._h, records, self._cov)

    @property
    def n_obs(self) -> int:
        return len(self._ys)

    # -- predictive ----------------------------------------------------------
    def predictive(self, records: Sequence[Record]) -> StudentTPredictive:
        x = self._phi(records)
        quad = np.einsum("ij,jk,ik->i", x, self._sigma, x)
        n = x.shape[0]
        return StudentTPredictive(
            df=np.full(n, 2.0 * self._a),
            loc=x @ self._mu,
            scale=np.sqrt(self._b / self._a * (1.0 + quad)),
        )

    def log_predictive(self, record: Record, y: float | int) -> float:
        return float(self.predictive([record]).logpdf(np.asarray([float(y)]))[0])

    # -- update --------------------------------------------------------------
    def update(self, record: Record, y: float | int | None) -> None:
        if y is None:
            return
        y = float(y)
        x = self._phi([record])[0]
        self._log_evidence += self.log_predictive(record, y)
        sx = self._sigma @ x
        c = 1.0 + float(x @ sx)
        resid = y - float(x @ self._mu)
        self._mu = self._mu + sx * (resid / c)
        self._sigma = self._sigma - np.outer(sx, sx) / c
        self._sigma = 0.5 * (self._sigma + self._sigma.T)  # keep symmetric
        self._a += 0.5
        self._b += 0.5 * resid * resid / c
        self._xs.append(x)
        self._ys.append(y)

    # -- evidence ------------------------------------------------------------
    def log_evidence(self) -> float:
        return self._log_evidence

    def max_log_likelihood(self) -> float:
        """sup over (β, σ²) of Σ log N(yᵢ | xᵢᵀβ, σ²): the Gaussian MLE."""
        n = self.n_obs
        if n == 0:
            return 0.0
        x = np.vstack(self._xs)
        y = np.asarray(self._ys)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        rss = float(np.sum((y - x @ beta) ** 2))
        rank = int(np.linalg.matrix_rank(x))
        scale = max(float(np.sum(y * y)), 1.0)
        if n <= rank or rss <= 1e-12 * scale:
            return math.inf  # model interpolates the data: σ̂² → 0, likelihood unbounded
        return -0.5 * n * (math.log(2.0 * math.pi * rss / n) + 1.0)

    # -- information gain ----------------------------------------------------
    def parameter_eig(self, records: Sequence[Record]) -> np.ndarray:
        """I((β, σ²); Y_d) = H[Y_d] − E[H(Y_d | β, σ²)], exact:
        the second term is ½(log 2πe + E[log σ²]) with E[log σ²] = log b − ψ(a)."""
        pred = self.predictive(records)
        e_log_sigma2 = math.log(self._b) - float(special.digamma(self._a))
        return pred.entropy() - 0.5 * (_LOG_2PI_E + e_log_sigma2)

    # -- misc ----------------------------------------------------------------
    def clone(self) -> NIGModel:
        return copy.deepcopy(self)

    def summary(self) -> dict[str, Any]:
        sd = np.sqrt(np.diag(self._sigma) * self._b / max(self._a - 1.0, 1e-12))
        return {
            "hypothesis_id": self.hypothesis_id,
            "kind": "linear_gaussian",
            "n": self.n_obs,
            "columns": self._columns,
            "beta_mean": self._mu.tolist(),
            "beta_sd": sd.tolist(),
            "a": self._a,
            "b": self._b,
            "log_evidence": self._log_evidence,
        }
