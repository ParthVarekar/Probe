"""Shared helpers for the stop tests: small hypothesis sets and adaptive simulation
loops for e-process validity and power.

Linear world (``linear_gaussian``): covariates x0 and x1 = x0², designs
x0 ∈ {−2, −1.5, ..., 2}. Hypotheses: ``L`` (intercept + x0), ``Q`` (intercept +
x1) and the residual ``R`` (intercept + x0 + x1, which nests both).

Grid world (``finite_grid``): Y ~ Binomial(n, expit(a + b·x)), designs
x ∈ {−2, ..., 2}, n ∈ {2, 4, 8}. Hypotheses: ``flat`` (b = 0, a on a grid),
``slope`` (a, b on grids with b ≠ 0) and the residual ``both`` (a, b with b = 0
allowed, nesting ``flat``).

Every loop is driven by a seeded generator; the design is *adaptive*: a function
of the data so far only (predictable), which is exactly the condition under which
the universal-inference e-process is valid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import special

from probe.determinism import make_rng
from probe.engine.eig import log_prior_weights, model_indicator_eig, posterior_weights
from probe.engine.factory import build_models
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
from probe.stop.eprocess import EProcessTracker

# --------------------------------------------------------------------------- linear world

LIN_COVS = (CovariateSpec(name="x0", kind="numeric"), CovariateSpec(name="x1", kind="numeric"))
LIN_OUT = OutcomeSpec(name="y", kind="continuous")
LIN_X = np.linspace(-2.0, 2.0, 9)
LIN_CANDS: list[dict[str, float]] = [{"x0": float(x), "x1": float(x * x)} for x in LIN_X]


def lin_h(
    hid: str,
    cols: tuple[str, ...],
    *,
    residual: bool = False,
    prior_sd: float = 2.0,
    a0: float = 2.0,
    b0: float = 1.0,
) -> LinearGaussianHypothesis:
    terms: list[Any] = [InterceptTerm(prior_sd=3.0)]
    terms += [NumericTerm(covariate=c, prior_sd=prior_sd) for c in cols]
    return LinearGaussianHypothesis(
        id=hid,
        label=hid,
        statement=f"{hid} statement",
        terms=tuple(terms),
        prior=NIGPrior(a0=a0, b0=b0),
        is_residual=residual,
    )


def linear_set() -> HypothesisSet:
    return HypothesisSet(
        outcome=LIN_OUT,
        covariates=LIN_COVS,
        hypotheses=(
            lin_h("L", ("x0",)),
            lin_h("Q", ("x1",)),
            lin_h("R", ("x0", "x1"), residual=True),
        ),
    )


# --------------------------------------------------------------------------- grid world


def binom_pmf(n: int, p: np.ndarray, support: int) -> np.ndarray:
    """Binomial(n, p) PMF over 0..support−1 for a column of p (G, 1), with exact
    zeros at p ∈ {0, 1} and beyond n (faster than ``scipy.stats.binom.pmf``)."""
    k = np.arange(support, dtype=np.float64)[None, :]
    ok = k <= n
    kk = np.where(ok, k, 0.0)
    log_c = special.gammaln(n + 1.0) - special.gammaln(kk + 1.0) - special.gammaln(n - kk + 1.0)
    logp = log_c + special.xlogy(kk, p) + special.xlog1py(n - kk, -p)
    return np.where(ok, np.exp(logp), 0.0)


class LogisticBinomialMechanism:
    """Y ~ Binomial(n, expit(a + b·x)); stateless. Test-only."""

    name = "stoptest_logistic"

    def __init__(self, support_size: int) -> None:
        self.support_size = support_size

    def initial_state(self, params: Mapping[str, np.ndarray]) -> None:
        return None

    def predict(self, state: None, params: Mapping[str, np.ndarray], record: Record) -> np.ndarray:
        n = int(record["n"])
        if not 0 <= n < self.support_size:
            raise ValueError("trials outside support")
        a = np.asarray(params["a"], dtype=np.float64)
        b = np.asarray(params["b"], dtype=np.float64)
        p = special.expit(a + b * float(record["x"]))[:, None]
        return binom_pmf(n, p, self.support_size)

    def advance(self, state: None, params: Mapping[str, np.ndarray], record: Record) -> None:
        return state


if "stoptest_logistic" not in registered_mechanisms():
    register_mechanism("stoptest_logistic")(LogisticBinomialMechanism)

GRID_COVS = (CovariateSpec(name="x", kind="numeric"), CovariateSpec(name="n", kind="numeric"))
GRID_OUT = OutcomeSpec(name="count", kind="count", max_count=8)
GRID_CANDS: list[dict[str, float]] = [
    {"x": float(x), "n": float(n)} for x in (-2, -1, 0, 1, 2) for n in (2, 4, 8)
]
A_GRID = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)


def grid_h(
    hid: str,
    params: Mapping[str, tuple[float, ...]],
    fixed: Mapping[str, float] | None = None,
    residual: bool = False,
    mechanism: str = "stoptest_logistic",
) -> GridHypothesis:
    return GridHypothesis(
        id=hid,
        label=hid,
        statement=f"{hid} statement",
        mechanism=mechanism,
        params=dict(params),
        fixed=dict(fixed or {}),
        is_residual=residual,
    )


def grid_set() -> HypothesisSet:
    return HypothesisSet(
        outcome=GRID_OUT,
        covariates=GRID_COVS,
        hypotheses=(
            grid_h("flat", {"a": A_GRID}, {"b": 0.0}),
            grid_h("slope", {"a": (-1.0, 0.0, 1.0), "b": (-1.0, -0.5, 0.5, 1.0)}),
            grid_h("both", {"a": A_GRID, "b": (-1.0, 0.0, 1.0)}, residual=True),
        ),
    )


# --------------------------------------------------------------------------- simulation

Design = Literal["eig", "contrast", "cycle"]


@dataclass
class SimResult:
    ids: tuple[str, ...]
    running_max: np.ndarray
    """(K, K) sup_t log E_t(i : j)."""
    final: np.ndarray
    """(K, K) log E_T(i : j)."""
    tracker: EProcessTracker
    models: list[Any]


def _choose(models: list[Any], hs: HypothesisSet, cands: list[Any], design: Design, t: int) -> int:
    """A predictable (past-data-only) design rule."""
    if design == "cycle":
        return t % len(cands)
    if design == "contrast":  # largest standardized disagreement between the first two
        a, b = models[0].predictive(cands), models[1].predictive(cands)
        score = (a.mean() - b.mean()) ** 2 / (a.variance() + b.variance())
        return int(np.argmax(score))
    preds = [m.predictive(cands) for m in models]
    w = posterior_weights(log_prior_weights(hs), [m.log_evidence() for m in models])
    return int(np.argmax(model_indicator_eig(preds, w).value))


def simulate(
    hs: HypothesisSet,
    cands: list[Any],
    draw: Any,
    *,
    seed: int,
    n_steps: int,
    design: Design,
    tag: str,
) -> SimResult:
    """One adaptive run: ``draw(record, rng)`` generates an outcome from the fixed
    truth; every model is fed (log predictive before update, then max likelihood)."""
    rng = make_rng(seed, "stop-test", tag)
    models = build_models(hs)
    tracker = EProcessTracker.for_models(models)
    for t in range(n_steps):
        rec = cands[_choose(models, hs, cands, design, t)]
        y = draw(rec, rng)
        lp = [m.log_predictive(rec, y) for m in models]
        for m in models:
            m.update(rec, y)
        tracker.observe_models(lp, models)
    return SimResult(
        tracker.hypothesis_ids, tracker.running_max_log_e(), tracker.log_e(), tracker, models
    )


def linear_truth(beta: tuple[float, float, float], sigma: float) -> Any:
    """y = β0 + β1·x0 + β2·x1 + σ·ε (fixed parameters)."""

    def draw(rec: Mapping[str, float], rng: np.random.Generator) -> float:
        mu = beta[0] + beta[1] * rec["x0"] + beta[2] * rec["x1"]
        return float(mu + sigma * rng.standard_normal())

    return draw


def grid_truth(a: float, b: float) -> Any:
    def draw(rec: Mapping[str, float], rng: np.random.Generator) -> int:
        p = float(special.expit(a + b * rec["x"]))
        return int(rng.binomial(int(rec["n"]), p))

    return draw


def type_one_bound(alpha: float, n_runs: int) -> float:
    """α + 2 binomial standard errors (handover §12 Part P acceptance criterion)."""
    return alpha + 2.0 * float(np.sqrt(alpha * (1.0 - alpha) / n_runs))
