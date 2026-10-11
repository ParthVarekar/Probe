"""Exact Normal-Inverse-Gamma engine for ``linear_gaussian`` hypotheses.

Model: y = φ(d)ᵀβ + ε, ε ~ N(0, σ²), β | σ² ~ N(μ₀, σ² V₀), σ² ~ InvGamma(a, b).

State is kept as (μ, Σ = Λ⁻¹, a, b) and updated one observation at a time with
the Sherman–Morrison rank-1 identity, so each step is O(p²) and the prequential
decomposition log p(y₁..ₙ) = Σ log p(yᵢ | y<ᵢ) holds by construction:

* c = 1 + xᵀΣx, r = y − xᵀμ
* μ ← μ + Σx·r/c,  Σ ← Σ − (Σx)(Σx)ᵀ/c,  a ← a + ½,  b ← b + ½·r²/c

which equals the batch closed form of handover §12 H.4 (V_n, μ_n, a_n, b_n and
the log evidence; checked in ``tests/engine/test_nig.py``). The one-step
predictive is t_{2a}(xᵀμ, (b/a)(1 + xᵀΣx)).

Besides the :class:`~probe.engine.models.HypothesisModel` contract this module
offers exact helpers used by the EIG kernels: :meth:`NIGModel.linear_query`
(posterior probability of a linear-contrast event before and after one more
observation, for goal-oriented EIG) and the parameter sampler/likelihood trio
used by the brute-force nested Monte Carlo oracle.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
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
from probe.engine.queries import ContrastQuery

__all__ = ["LinearQueryKernel", "NIGModel", "NIGPosterior", "NIGSamples"]

_LOG_2PI_E = math.log(2.0 * math.pi * math.e)
_LOG_2PI = math.log(2.0 * math.pi)
_EPS = float(np.finfo(np.float64).eps)


def _rows_dot(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """x @ v as a plain row reduction: a BLAS matrix-vector product may round a row
    differently depending on its position in ``x``, which would make a candidate's
    predictive (and hence its EIG) depend on the other candidates in the batch."""
    return np.sum(x * v, axis=1)


@dataclass(frozen=True)
class NIGPosterior:
    """Current posterior β | σ² ~ N(mu, σ²·cov), σ² ~ InvGamma(a, b) (copies)."""

    mu: np.ndarray
    cov: np.ndarray
    a: float
    b: float


@dataclass(frozen=True)
class NIGSamples:
    """Joint posterior draws: ``beta`` (S, p), ``sigma2`` (S,)."""

    beta: np.ndarray
    sigma2: np.ndarray

    @property
    def size(self) -> int:
        return int(self.sigma2.shape[0])


@dataclass(frozen=True)
class LinearQueryKernel:
    """Exact posterior probability of q = 1[cᵀβ < τ] (``upper=False``: cᵀβ > τ)
    now, and after one more observation y at candidate i.

    After observing (x, y) the NIG posterior is updated in closed form, so with
    c' = 1 + xᵀΣx and r = y − xᵀμ:

    * cᵀμ' = cᵀμ + (cᵀΣx)·r/c'
    * cᵀΣ'c = cᵀΣc − (cᵀΣx)²/c'
    * a' = a + ½,  b' = b + ½·r²/c'
    * cᵀβ | D' ~ t_{2a'}(cᵀμ', sqrt((b'/a')·cᵀΣ'c))

    and P(q | D, y) is a Student-t CDF. No approximation is involved; the only
    numerical step in goal EIG is the 1-D integral over y.
    """

    prior_prob: float
    x_mu: np.ndarray
    cc: np.ndarray
    """1 + xᵀΣx per candidate."""
    c_sx: np.ndarray
    """cᵀΣx per candidate."""
    c_mu: float
    c_sc: float
    a: float
    b: float
    threshold: float
    below: bool
    """True for the event cᵀβ < τ, False for cᵀβ > τ."""

    @property
    def deterministic(self) -> bool:
        """True when c = 0: the hypothesis answers the query without uncertainty."""
        return self.c_sc <= 0.0

    def prob_after(self, y: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """P(q | D ∪ {(x_cand, y)}); ``y`` has shape (P, Q) and ``cand`` (P,)
        indexes the candidate of each row."""
        y = np.asarray(y, dtype=np.float64)
        return self.prob_after_resid(y - self.x_mu[np.asarray(cand)][:, None], cand)

    def prob_after_resid(self, r: np.ndarray, cand: np.ndarray) -> np.ndarray:
        """:meth:`prob_after` in terms of the residual r = y − xᵀμ (P, Q), which the
        quadrature forms exactly from anchored coordinates."""
        r = np.asarray(r, dtype=np.float64)
        if self.deterministic:
            return np.full(r.shape, self.prior_prob)
        idx = np.asarray(cand)[:, None]
        cc = self.cc[idx]
        csx = self.c_sx[idx]
        mean = self.c_mu + csx * r / cc
        var_unit = np.maximum(self.c_sc - csx * csx / cc, 0.0)
        a1 = self.a + 0.5
        b1 = self.b + 0.5 * r * r / cc
        scale = np.sqrt(b1 / a1 * var_unit)
        diff = self.threshold - mean if self.below else mean - self.threshold
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(scale > 0.0, diff / np.where(scale > 0.0, scale, 1.0), 0.0)
        p = special.stdtr(2.0 * a1, t)
        # zero posterior spread: the answer is a step function of y
        return np.where(scale > 0.0, p, (diff > 0.0).astype(np.float64))

    def transition_points(self) -> np.ndarray:
        """y per candidate at which the post-update mean cᵀμ' crosses τ (NaN when
        the candidate does not move cᵀμ). Used as quadrature breakpoints."""
        with np.errstate(divide="ignore", invalid="ignore"):
            yt = self.x_mu + (self.threshold - self.c_mu) * self.cc / self.c_sx
        return np.where(np.isfinite(yt), yt, np.nan)


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

    def features(self, records: Sequence[Record]) -> np.ndarray:
        """φ_m at each record, shape (n, p)."""
        return self._phi(records)

    @property
    def n_obs(self) -> int:
        return len(self._ys)

    @property
    def n_params(self) -> int:
        return len(self._columns)

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    @property
    def posterior(self) -> NIGPosterior:
        return NIGPosterior(self._mu.copy(), self._sigma.copy(), self._a, self._b)

    def _quad(self, x: np.ndarray) -> np.ndarray:
        """xᵢᵀΣxᵢ per row, clamped at 0 against rounding."""
        return np.maximum(np.einsum("ij,jk,ik->i", x, self._sigma, x), 0.0)

    def _predictive_x(self, x: np.ndarray) -> StudentTPredictive:
        n = x.shape[0]
        return StudentTPredictive(
            df=np.full(n, 2.0 * self._a),
            loc=_rows_dot(x, self._mu),
            scale=np.sqrt(self._b / self._a * (1.0 + self._quad(x))),
        )

    # -- predictive ----------------------------------------------------------
    def predictive(self, records: Sequence[Record]) -> StudentTPredictive:
        return self._predictive_x(self._phi(records))

    def log_predictive(self, record: Record, y: float | int) -> float:
        yf = _finite_y(y)
        return float(self.predictive([record]).logpdf(np.asarray([yf]))[0])

    # -- update --------------------------------------------------------------
    def update(self, record: Record, y: float | int | None) -> None:
        if y is None:
            return
        yf = _finite_y(y)
        x = self._phi([record])
        self._log_evidence += float(self._predictive_x(x).logpdf(np.asarray([yf]))[0])
        x = x[0]
        sx = self._sigma @ x
        c = 1.0 + max(float(x @ sx), 0.0)
        resid = yf - float(x @ self._mu)
        self._mu = self._mu + sx * (resid / c)
        self._sigma = self._sigma - np.outer(sx, sx) / c
        self._sigma = 0.5 * (self._sigma + self._sigma.T)  # keep symmetric
        self._a += 0.5
        self._b += 0.5 * resid * resid / c
        self._xs.append(x)
        self._ys.append(yf)

    # -- evidence ------------------------------------------------------------
    def log_evidence(self) -> float:
        return self._log_evidence

    def max_log_likelihood(self) -> float:
        """sup over (β, σ²) of Σ log N(yᵢ | xᵢᵀβ, σ²) = −(n/2)(log(2π·RSS/n) + 1),
        the Gaussian MLE, with RSS the least-squares residual sum of squares.

        It is +inf when the model interpolates the data (σ̂² → 0): n ≤ rank, or a
        residual within the rounding floor of a backward-stable least-squares solve,
        ρ = 32(n + p)·ε·(1 + κ)·(‖y‖ + ‖X‖_F‖β̂‖) with κ the condition number of the
        retained singular values. Beyond the floor, RSS is replaced by its lower bound
        (‖r̂‖ − ρ)², so the value never falls below the true supremum (the conservative
        side for universal-inference denominators). The floor scales with the data's
        own rounding error, so an ordinary fit to data with a large mean and small
        noise is never mistaken for an exact one."""
        n = self.n_obs
        if n == 0:
            return 0.0
        x = np.vstack(self._xs)
        y = np.asarray(self._ys)
        p = x.shape[1]
        sv = np.linalg.svd(x, compute_uv=False)
        tol = (float(sv[0]) if sv.size else 0.0) * max(n, p) * _EPS  # numpy's rank rule
        kept = sv[sv > tol]
        rank = int(kept.size)
        if n <= rank:
            return math.inf
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = float(np.linalg.norm(y - x @ beta))
        kappa = float(kept[0] / kept[-1]) if rank else 1.0
        floor = (
            32.0
            * (n + p)
            * _EPS
            * (1.0 + kappa)
            * (float(np.linalg.norm(y)) + float(np.linalg.norm(x)) * float(np.linalg.norm(beta)))
        )
        if resid <= floor:
            return math.inf  # model interpolates the data: σ̂² → 0, likelihood unbounded
        rss_lo = (resid - floor) ** 2
        if rss_lo <= 0.0 or not math.isfinite(rss_lo):
            return math.inf
        return -0.5 * n * (math.log(2.0 * math.pi * rss_lo / n) + 1.0)

    # -- information gain ----------------------------------------------------
    def parameter_eig(self, records: Sequence[Record]) -> np.ndarray:
        """I((β, σ²); Y_d) = H[Y_d] − E[H(Y_d | β, σ²)], exact:
        the second term is ½(log 2πe + E[log σ²]) with E[log σ²] = log b − ψ(a)."""
        pred = self.predictive(records)
        e_log_sigma2 = math.log(self._b) - float(special.digamma(self._a))
        return pred.entropy() - 0.5 * (_LOG_2PI_E + e_log_sigma2)

    # -- goal-oriented queries -----------------------------------------------
    def contrast_vector(self, query: ContrastQuery) -> np.ndarray:
        """cₘ = Σ_j c_j φ_m(r_j): the query contrast in this hypothesis's β-space."""
        phi = self._phi(query.records)
        return np.asarray(query.coefs, dtype=np.float64) @ phi

    def linear_query(self, records: Sequence[Record], query: ContrastQuery) -> LinearQueryKernel:
        """Exact query-probability kernel for candidates ``records``."""
        c = self.contrast_vector(query)
        x = self._phi(records)
        below = query.direction == "<"
        tau = float(query.threshold)
        c_mu = float(c @ self._mu)
        sc = self._sigma @ c
        c_sc = max(float(c @ sc), 0.0) if np.any(c != 0.0) else 0.0
        if c_sc > 0.0:
            t = (tau - c_mu) / math.sqrt(self._b / self._a * c_sc)
            prior = float(special.stdtr(2.0 * self._a, t if below else -t))
        else:  # cᵀβ ≡ cᵀμ = 0 under this hypothesis
            prior = float((c_mu < tau) if below else (c_mu > tau))
        return LinearQueryKernel(
            prior_prob=prior,
            x_mu=_rows_dot(x, self._mu),
            cc=1.0 + self._quad(x),
            c_sx=_rows_dot(x, sc),
            c_mu=c_mu,
            c_sc=c_sc,
            a=self._a,
            b=self._b,
            threshold=tau,
            below=below,
        )

    # -- oracle helpers (brute-force nested Monte Carlo; tests and diagnostics) ---
    def sample_parameters(self, rng: np.random.Generator, size: int) -> NIGSamples:
        """Draw (β, σ²) from the current joint posterior."""
        sigma2 = self._b / rng.gamma(self._a, 1.0, size=size)
        try:
            root = np.linalg.cholesky(self._sigma)
        except np.linalg.LinAlgError:  # rounding made Σ semi-definite: use a clipped eigen-root
            evals, evecs = np.linalg.eigh(self._sigma)
            root = evecs * np.sqrt(np.maximum(evals, 0.0))
        z = rng.standard_normal((size, self._mu.shape[0]))
        beta = self._mu + np.sqrt(sigma2)[:, None] * (z @ root.T)
        return NIGSamples(beta=beta, sigma2=sigma2)

    def simulate(self, params: NIGSamples, record: Record, rng: np.random.Generator) -> np.ndarray:
        """One y ~ N(xᵀβ_s, σ²_s) per parameter draw s."""
        x = self._phi([record])[0]
        return params.beta @ x + np.sqrt(params.sigma2) * rng.standard_normal(params.size)

    def conditional_logpdf(self, params: NIGSamples, record: Record, y: np.ndarray) -> np.ndarray:
        """log N(y_i | xᵀβ_s, σ²_s), shape (len(y), S)."""
        x = self._phi([record])[0]
        m = params.beta @ x
        yy = np.asarray(y, dtype=np.float64)[:, None]
        return -0.5 * (_LOG_2PI + np.log(params.sigma2) + (yy - m) ** 2 / params.sigma2)

    # -- misc ----------------------------------------------------------------
    def clone(self) -> NIGModel:
        return copy.deepcopy(self)

    def summary(self) -> dict[str, Any]:
        """``beta_scale`` is the marginal Student-t scale of each coefficient (always
        defined); ``beta_sd`` is its standard deviation, ``None`` while a ≤ 1
        (the marginal variance is then infinite)."""
        diag = np.diag(self._sigma)
        scale = np.sqrt(diag * self._b / self._a)
        sd = np.sqrt(diag * self._b / (self._a - 1.0)).tolist() if self._a > 1.0 else None
        return {
            "hypothesis_id": self.hypothesis_id,
            "kind": "linear_gaussian",
            "n": self.n_obs,
            "columns": self._columns,
            "beta_mean": self._mu.tolist(),
            "beta_sd": sd,
            "beta_scale": scale.tolist(),
            "df": 2.0 * self._a,
            "a": self._a,
            "b": self._b,
            "log_evidence": self._log_evidence,
        }


def _finite_y(y: float | int) -> float:
    yf = float(y)
    if not math.isfinite(yf):
        raise ValueError(f"observation must be finite (use None for missing), got {y!r}")
    return yf
