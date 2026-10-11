"""Shared helpers for the diagnostics tests: small hypothesis sets and simulation
loops that record per-observation diagnostics and alarm traces.

* **Prior-predictive runs** (well specified by construction): draw a hypothesis
  from the prior weights, its parameters from its prior (a fresh model's
  ``sample_parameters``), then every outcome from that fixed parameter. Under the
  exact Bayesian predictive (η = 1) the PIT values are i.i.d. U(0, 1) and the
  standardized surprise has mean 0 and variance 1.
* **Misspecified runs**: a fixed truth outside every hypothesis.

Linear world: x0 ∈ {−2, −1.5, ..., 2}, x1 = x0²; hypotheses L (x0), Q (x1) and
the residual R (x0 + x1). Grid world: Y ~ Binomial(n, expit(a + b·x)).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import special

from probe.determinism import make_rng
from probe.diagnostics.alarms import AlarmMonitor, AlarmThresholds, DiagnosticTrace
from probe.diagnostics.pit import ObservationDiagnostics, observation_diagnostics
from probe.engine.eig import log_prior_weights, posterior_weights
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

LIN_COVS = (CovariateSpec(name="x0", kind="numeric"), CovariateSpec(name="x1", kind="numeric"))
LIN_OUT = OutcomeSpec(name="y", kind="continuous")
LIN_CANDS: list[dict[str, float]] = [
    {"x0": float(x), "x1": float(x * x)} for x in np.linspace(-2.0, 2.0, 9)
]


def lin_h(
    hid: str,
    cols: tuple[str, ...],
    *,
    residual: bool = False,
    prior_sd: float = 2.0,
    intercept_sd: float = 3.0,
    a0: float = 3.0,
    b0: float = 1.0,
) -> LinearGaussianHypothesis:
    terms: list[Any] = [InterceptTerm(prior_sd=intercept_sd)]
    terms += [NumericTerm(covariate=c, prior_sd=prior_sd) for c in cols]
    return LinearGaussianHypothesis(
        id=hid,
        label=hid,
        statement=f"{hid} statement",
        terms=tuple(terms),
        prior=NIGPrior(a0=a0, b0=b0),
        is_residual=residual,
    )


def linear_set(*hyps: LinearGaussianHypothesis) -> HypothesisSet:
    if not hyps:
        hyps = (lin_h("L", ("x0",)), lin_h("Q", ("x1",)), lin_h("R", ("x0", "x1"), residual=True))
    return HypothesisSet(outcome=LIN_OUT, covariates=LIN_COVS, hypotheses=hyps)


# --------------------------------------------------------------------------- grid world


def binom_pmf(n: int, p: np.ndarray, support: int) -> np.ndarray:
    k = np.arange(support, dtype=np.float64)[None, :]
    ok = k <= n
    kk = np.where(ok, k, 0.0)
    log_c = special.gammaln(n + 1.0) - special.gammaln(kk + 1.0) - special.gammaln(n - kk + 1.0)
    return np.where(ok, np.exp(log_c + special.xlogy(kk, p) + special.xlog1py(n - kk, -p)), 0.0)


class DiagLogisticMechanism:
    """Y ~ Binomial(n, expit(a + b·x)); stateless. Test-only."""

    name = "diagtest_logistic"

    def __init__(self, support_size: int) -> None:
        self.support_size = support_size

    def initial_state(self, params: Mapping[str, np.ndarray]) -> None:
        return None

    def predict(self, state: None, params: Mapping[str, np.ndarray], record: Record) -> np.ndarray:
        p = special.expit(
            np.asarray(params["a"], dtype=np.float64)
            + np.asarray(params["b"], dtype=np.float64) * float(record["x"])
        )
        return binom_pmf(int(record["n"]), p[:, None], self.support_size)

    def advance(self, state: None, params: Mapping[str, np.ndarray], record: Record) -> None:
        return state


if "diagtest_logistic" not in registered_mechanisms():
    register_mechanism("diagtest_logistic")(DiagLogisticMechanism)

GRID_COVS = (CovariateSpec(name="x", kind="numeric"), CovariateSpec(name="n", kind="numeric"))
GRID_OUT = OutcomeSpec(name="count", kind="count", max_count=6)
GRID_CANDS: list[dict[str, float]] = [
    {"x": float(x), "n": float(n)} for x in (-2, -1, 0, 1, 2) for n in (3, 6)
]


def grid_h(
    hid: str,
    params: Mapping[str, tuple[Any, ...]],
    fixed: Mapping[str, Any] | None = None,
    residual: bool = False,
    mechanism: str = "diagtest_logistic",
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


def grid_set(*hyps: GridHypothesis) -> HypothesisSet:
    if not hyps:
        hyps = (
            grid_h("flat", {"a": (-1.0, 0.0, 1.0)}, {"b": 0.0}),
            grid_h("slope", {"a": (-1.0, 0.0, 1.0), "b": (-1.0, 1.0)}),
            grid_h("both", {"a": (-1.5, 0.0, 1.5), "b": (-0.5, 0.0, 0.5)}, residual=True),
        )
    return HypothesisSet(outcome=GRID_OUT, covariates=GRID_COVS, hypotheses=hyps)


# --------------------------------------------------------------------------- runs


@dataclass
class DiagRun:
    diagnostics: list[ObservationDiagnostics]
    trace: DiagnosticTrace
    monitor: AlarmMonitor
    true_hypothesis: str | None


def prior_draw(hs: HypothesisSet, rng: np.random.Generator) -> tuple[str, Callable[..., Any]]:
    """A hypothesis from the prior weights and a fixed parameter from its prior."""
    w = np.exp(log_prior_weights(hs))
    k = int(rng.choice(len(w), p=w / w.sum()))
    model = build_models(hs)[k]
    params = model.sample_parameters(rng, 1)

    def draw(rec: Record, r: np.random.Generator) -> Any:
        y = model.simulate(params, rec, r)[0]
        return int(y) if isinstance(y, np.integer) else float(y)

    return hs.hypotheses[k].id, draw


def run_diagnostics(
    hs: HypothesisSet,
    cands: list[Any],
    *,
    seed: int,
    n_steps: int,
    tag: str,
    truth: Callable[..., Any] | None = None,
    thresholds: AlarmThresholds | None = None,
) -> DiagRun:
    """One run with a data-dependent design (largest model-averaged predictive
    variance, ties to the first candidate); diagnostics from the pre-update
    predictive, alarm fed with the post-update residual weight."""
    rng = make_rng(seed, "diag-test", tag)
    true_id: str | None = None
    if truth is None:
        true_id, truth = prior_draw(hs, rng)
    models = build_models(hs)
    log_prior = log_prior_weights(hs)
    res = next(i for i, h in enumerate(hs.hypotheses) if h.is_residual)
    monitor = AlarmMonitor(thresholds)
    out: list[ObservationDiagnostics] = []
    for t in range(n_steps):
        w = posterior_weights(log_prior, [m.log_evidence() for m in models])
        preds_all = [m.predictive(cands) for m in models]
        mean = sum(wi * p.mean() for wi, p in zip(w, preds_all, strict=True))
        var = sum(
            wi * (p.variance() + (p.mean() - mean) ** 2) for wi, p in zip(w, preds_all, strict=True)
        )
        idx = int(np.argmax(var))
        rec = cands[idx]
        y = truth(rec, rng)
        d = observation_diagnostics(preds_all, w, y, seed=seed, step=t, index=idx)
        out.append(d)
        for m in models:
            m.update(rec, y)
        w_post = posterior_weights(log_prior, [m.log_evidence() for m in models])
        monitor.update(d, float(w_post[res]))
    return DiagRun(out, monitor.trace(), monitor, true_id)


def sin_truth(amplitude: float = 2.5, freq: float = 1.5, sigma: float = 0.3, df: float = 3.0):
    """A fixed truth outside the linear set: smooth non-polynomial mean, t-noise."""

    def draw(rec: Mapping[str, float], rng: np.random.Generator) -> float:
        return float(amplitude * np.sin(freq * rec["x0"]) + sigma * rng.standard_t(df))

    return draw
