"""U_M against the brute-force nested Monte Carlo oracle (handover §12 Part P:
"EIG agrees with brute-force nested Monte Carlo within 3 Monte Carlo standard
errors on ≥ 20 toy configurations").

The oracle never touches the closed-form predictives or the quadrature: it draws
parameters from each hypothesis's posterior, simulates outcomes from the
likelihood and estimates every marginal by an inner average over fresh parameter
draws. A handful of configurations run in the fast suite; the full 2 × 25 sweep
is marked slow.
"""

import numpy as np
import pytest
from engine_testkit import COUNT_OUT, grid_h, linear_h, numeric_covs, records_from

from probe.determinism import make_rng
from probe.engine.eig import model_indicator_eig, nested_mc_model_eig, posterior_weights
from probe.engine.grid import GridModel
from probe.engine.nig import NIGModel


def nig_config(seed: int):
    """2-4 linear hypotheses (different feature subsets and NIG priors), fitted on
    0-10 shared observations; weights = ½ posterior + ½ uniform."""
    rng = make_rng(seed, "nmc-nig")
    covs = numeric_covs(2)
    k = int(rng.integers(2, 5))
    hs = [
        linear_h(
            f"H{i}",
            int(rng.integers(0, 3)),
            prior_sd=float(rng.uniform(0.3, 2.0)),
            intercept_sd=float(rng.uniform(0.5, 3.0)),
            a0=float(rng.uniform(1.5, 4.0)),
            b0=float(rng.uniform(0.3, 2.0)),
            residual=i == k - 1,
        )
        for i in range(k)
    ]
    n_fit = int(rng.choice([0, 2, 5, 10]))
    x = rng.normal(size=(n_fit, 2))
    y = 0.3 + x @ np.array([0.8, -0.5]) + 0.6 * rng.normal(size=n_fit)
    models = [NIGModel(h, covs) for h in hs]
    for m in models:
        for r, v in zip(records_from(x), y, strict=True):
            m.update(r, float(v))
    w = posterior_weights(np.zeros(k), [m.log_evidence() for m in models])
    w = 0.5 * w + 0.5 / k
    cand = records_from(rng.uniform(-3, 3, size=(1, 2)))[0]
    return models, w, cand


def grid_config(seed: int):
    """2-4 binomial hypotheses with different p-grids, fitted on 0-2 observations."""
    rng = make_rng(seed, "nmc-grid")
    k = int(rng.integers(2, 5))
    hs = []
    for i in range(k):
        g = np.unique(np.round(rng.uniform(0.02, 0.98, size=int(rng.integers(1, 6))), 3))
        hs.append(
            grid_h(f"G{i}", "test_binomial", {"p": tuple(float(v) for v in g)}, None, i == k - 1)
        )
    models = [GridModel(h, COUNT_OUT) for h in hs]
    for _ in range(int(rng.integers(0, 3))):
        nt = int(rng.integers(1, 6))
        yv = int(rng.integers(0, nt + 1))
        for m in models:
            m.update({"n": nt}, yv)
    w = posterior_weights(np.zeros(k), [m.log_evidence() for m in models])
    w = 0.5 * w + 0.5 / k
    return models, w, {"n": int(rng.integers(1, 13))}


def _check(family, seed, n_outer, n_inner):
    models, w, cand = (nig_config if family == "nig" else grid_config)(seed)
    u = model_indicator_eig([m.predictive([cand]) for m in models], w)
    est, se = nested_mc_model_eig(
        models, w, cand, make_rng(seed, "nmc-oracle", family), n_outer, n_inner
    )
    assert se > 0
    assert abs(est - u.value[0]) <= 3.0 * se + u.error[0], (family, seed, u.value[0], est, se)


@pytest.mark.parametrize("family", ["nig", "grid"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_u_m_agrees_with_nested_monte_carlo_fast(family, seed):
    _check(family, seed, 1500, 1500)


@pytest.mark.slow
@pytest.mark.parametrize("family", ["nig", "grid"])
@pytest.mark.parametrize("seed", list(range(3, 28)))
def test_u_m_agrees_with_nested_monte_carlo_sweep(family, seed):
    _check(family, seed, 2000, 2000)


def test_nested_mc_point_mass_is_zero():
    models, _, cand = grid_config(0)
    w = np.zeros(len(models))
    w[0] = 1.0
    assert nested_mc_model_eig(models, w, cand, make_rng(0, "pm"), 10, 10) == (0.0, 0.0)
