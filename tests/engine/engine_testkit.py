"""Shared helpers for engine tests (imported by test modules in this directory).

Registers two *test-only* grid mechanisms (real mechanisms come from the system
lab) and provides an independent batch implementation of the NIG closed form
(handover §12 H.4) to check the rank-1 engine against.

Mechanisms
----------
``test_binomial`` (stateless)
    Y ~ Binomial(record["n"], p), grid parameter ``p``. p ∈ {0, 1} gives
    zero-probability outcomes.
``test_bucket`` (stateful token bucket)
    Parameters ``capacity`` (tokens), ``refill`` (tokens per unit gap) and ``eps``
    (observation noise). State: tokens per grid point, starting full. An action
    ``{"burst": b, "gap": g}`` first refills min(capacity, tokens + refill·g), then
    accepts a = min(b, ⌊tokens⌋). The observed count is a with probability 1 − eps
    and uniform on 0..b otherwise (eps = 0 makes it deterministic). ``advance``
    removes the a accepted tokens, whatever was observed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from hypothesis import settings
from scipy import special, stats

from probe.engine.ir import (
    CovariateSpec,
    GridHypothesis,
    HypothesisSet,
    InterceptTerm,
    LinearGaussianHypothesis,
    NIGPrior,
    NumericTerm,
    OutcomeSpec,
    Record,
)
from probe.engine.mechanisms import register_mechanism, registered_mechanisms

# Deterministic property testing: fixed derivation, no example database on disk.
PROPS = settings(max_examples=60, deadline=None, derandomize=True, database=None)
PROPS_FEW = settings(max_examples=20, deadline=None, derandomize=True, database=None)


# --------------------------------------------------------------------------- mechanisms


class BinomialMechanism:
    name = "test_binomial"

    def __init__(self, support_size: int) -> None:
        self.support_size = support_size

    def initial_state(self, params: Mapping[str, np.ndarray]) -> dict[str, Any]:
        return {}

    def predict(
        self, state: dict[str, Any], params: Mapping[str, np.ndarray], record: Record
    ) -> np.ndarray:
        n = int(record["n"])
        if not 0 <= n < self.support_size:
            raise ValueError("trials outside support")
        p = np.asarray(params["p"], dtype=np.float64)
        k = np.arange(self.support_size)
        return stats.binom.pmf(k[None, :], n, p[:, None])

    def advance(
        self, state: dict[str, Any], params: Mapping[str, np.ndarray], record: Record
    ) -> dict[str, Any]:
        return state


class BucketMechanism:
    name = "test_bucket"

    def __init__(self, support_size: int) -> None:
        self.support_size = support_size

    def initial_state(self, params: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(params["capacity"], dtype=np.float64).copy()

    @staticmethod
    def _accepted(
        tokens: np.ndarray, params: Mapping[str, np.ndarray], record: Record
    ) -> tuple[np.ndarray, np.ndarray]:
        cap = np.asarray(params["capacity"], dtype=np.float64)
        refill = np.asarray(params["refill"], dtype=np.float64)
        avail = np.minimum(cap, tokens + refill * float(record["gap"]))
        acc = np.minimum(float(record["burst"]), np.floor(avail))
        return avail, acc

    def predict(
        self, state: np.ndarray, params: Mapping[str, np.ndarray], record: Record
    ) -> np.ndarray:
        b = int(record["burst"])
        if not 0 <= b < self.support_size:
            raise ValueError("burst outside support")
        _, acc = self._accepted(state, params, record)
        eps = np.asarray(params["eps"], dtype=np.float64)
        g = acc.shape[0]
        out = np.zeros((g, self.support_size))
        out[:, : b + 1] = (eps / (b + 1))[:, None]
        out[np.arange(g), acc.astype(np.int64)] += 1.0 - eps
        return out

    def advance(
        self, state: np.ndarray, params: Mapping[str, np.ndarray], record: Record
    ) -> np.ndarray:
        avail, acc = self._accepted(state, params, record)
        return avail - acc


if "test_binomial" not in registered_mechanisms():
    register_mechanism("test_binomial")(BinomialMechanism)
if "test_bucket" not in registered_mechanisms():
    register_mechanism("test_bucket")(BucketMechanism)


# --------------------------------------------------------------------------- IR builders

COUNT_OUT = OutcomeSpec(name="count", kind="count", max_count=12)
GRID_COVS = (
    CovariateSpec(name="n", kind="numeric"),
    CovariateSpec(name="burst", kind="numeric"),
    CovariateSpec(name="gap", kind="numeric"),
)


def grid_h(
    hid: str,
    mechanism: str,
    params: Mapping[str, Sequence[float | int | str]],
    fixed: Mapping[str, float | int | str] | None = None,
    residual: bool = False,
) -> GridHypothesis:
    return GridHypothesis(
        id=hid,
        label=hid,
        statement=f"{hid} statement",
        mechanism=mechanism,
        params={k: tuple(v) for k, v in params.items()},
        fixed=dict(fixed or {}),
        is_residual=residual,
    )


def grid_set(*hyps: GridHypothesis) -> HypothesisSet:
    return HypothesisSet(outcome=COUNT_OUT, covariates=GRID_COVS, hypotheses=hyps)


def numeric_covs(p: int) -> tuple[CovariateSpec, ...]:
    return tuple(CovariateSpec(name=f"x{i}", kind="numeric") for i in range(p))


def linear_h(
    hid: str,
    n_slopes: int,
    *,
    prior_sd: float = 1.5,
    intercept_sd: float = 3.0,
    prior_means: Sequence[float] | None = None,
    a0: float = 2.0,
    b0: float = 1.0,
    residual: bool = False,
) -> LinearGaussianHypothesis:
    """Intercept + identity slopes on x0..x{n_slopes-1}, so φ(r) = (1, x0, x1, ...)."""
    means = list(prior_means) if prior_means is not None else [0.0] * (n_slopes + 1)
    terms: list[Any] = [InterceptTerm(prior_sd=intercept_sd, prior_mean=means[0])]
    for i in range(n_slopes):
        terms.append(NumericTerm(covariate=f"x{i}", prior_sd=prior_sd, prior_mean=means[i + 1]))
    return LinearGaussianHypothesis(
        id=hid,
        label=hid,
        statement=f"{hid} statement",
        terms=tuple(terms),
        prior=NIGPrior(a0=a0, b0=b0),
        is_residual=residual,
    )


def records_from(x: np.ndarray) -> list[dict[str, float]]:
    """Rows of an (n, p) array as design records {x0: .., x1: ..}."""
    return [{f"x{j}": float(v) for j, v in enumerate(row)} for row in np.atleast_2d(x)]


# --------------------------------------------------------------------------- batch NIG (H.4)


def batch_nig(
    x: np.ndarray, y: np.ndarray, mu0: np.ndarray, v0: np.ndarray, a0: float, b0: float
) -> dict[str, Any]:
    """Batch closed form (handover §12 H.4), written independently of the engine."""
    n = x.shape[0]
    v0inv = np.diag(1.0 / v0)
    prec = v0inv + x.T @ x
    vn = np.linalg.inv(prec)
    mun = vn @ (v0inv @ mu0 + x.T @ y)
    an = a0 + 0.5 * n
    bn = b0 + 0.5 * (y @ y + mu0 @ v0inv @ mu0 - mun @ prec @ mun)
    logdet_vn = -np.linalg.slogdet(prec)[1]
    logdet_v0 = float(np.sum(np.log(v0)))
    log_ev = (
        -0.5 * n * math.log(2.0 * math.pi)
        + 0.5 * (logdet_vn - logdet_v0)
        + a0 * math.log(b0)
        - an * math.log(bn)
        + special.gammaln(an)
        - special.gammaln(a0)
    )
    return {"mu": mun, "cov": vn, "a": an, "b": bn, "log_evidence": float(log_ev)}
