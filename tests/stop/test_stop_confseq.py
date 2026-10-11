"""Confidence sequences: time-uniform coverage by simulation (including adaptive,
martingale-dependent data), shrinkage, incremental = batch, and input checks."""

import math

import numpy as np
import pytest
from scipy import optimize

from probe.determinism import make_rng
from probe.stop.confseq import (
    BoundedMeanCS,
    SubGaussianMeanCS,
    bounded_cs,
    normal_mixture_boundary,
    optimal_rho,
    subgaussian_cs,
)

ALPHA = 0.05


def _bound(n: int, alpha: float = ALPHA) -> float:
    return alpha + 2.0 * math.sqrt(alpha * (1.0 - alpha) / n)


def _miscoverage(cs, mu) -> float:
    """Fraction of replications whose CS missed μ at *any* time."""
    return float(1.0 - np.mean(cs.covers(mu)[..., -1]))


# --------------------------------------------------------------------------- sub-Gaussian


def test_subgaussian_time_uniform_coverage_gaussian():
    rng = make_rng(1, "cs", "gauss")
    mu, sigma = 0.3, 1.7
    x = mu + sigma * rng.standard_normal((3000, 1000))
    cs = subgaussian_cs(x, sigma, ALPHA)
    assert _miscoverage(cs, mu) <= _bound(3000)
    # without the running intersection the intervals still cover uniformly
    raw = subgaussian_cs(x, sigma, ALPHA, running_intersection=False)
    assert _miscoverage(raw, mu) <= _bound(3000)
    assert np.all(cs.upper - cs.lower <= raw.upper - raw.lower + 1e-15)


def test_subgaussian_coverage_with_adaptive_noise():
    """Martingale data: the noise level of X_t depends on the past (but the
    conditional mean is μ and every step is 1-sub-Gaussian): still covered."""
    rng = make_rng(2, "cs", "adaptive")
    n, t_max, mu = 2000, 600, -0.4
    x = np.empty((n, t_max))
    s = np.zeros(n)
    for t in range(t_max):
        scale = np.where(s > mu * t, 1.0, 0.25)  # louder when running above μ
        x[:, t] = mu + scale * rng.standard_normal(n)
        s += x[:, t]
    assert _miscoverage(subgaussian_cs(x, 1.0, ALPHA), mu) <= _bound(n)


def test_subgaussian_boundary_is_tightest_near_t_opt_and_shrinks():
    for t_opt in (20, 200):
        rho = optimal_rho(float(t_opt), ALPHA)
        here = float(normal_mixture_boundary(float(t_opt), ALPHA, rho))
        for r in (0.5 * rho, 0.9 * rho, 1.1 * rho, 2.0 * rho):
            assert here <= float(normal_mixture_boundary(float(t_opt), ALPHA, r)) + 1e-12
    # closed-form optimum condition log((1+u)/(u α²)) = 1/u
    u = optimal_rho(1.0, ALPHA)
    assert math.log((1 + u) / (u * ALPHA**2)) == pytest.approx(1.0 / u, rel=1e-10)
    # numerical minimiser agrees
    res = optimize.minimize_scalar(
        lambda r: float(normal_mixture_boundary(50.0, ALPHA, r)), bounds=(1e-3, 1e4),
        method="bounded", options={"xatol": 1e-10},
    )
    assert optimal_rho(50.0, ALPHA) == pytest.approx(res.x, rel=1e-4)
    # radius/t → 0 like sqrt(log t / t)
    t = np.array([10.0, 100.0, 1000.0, 10000.0])
    rad = normal_mixture_boundary(t, ALPHA, optimal_rho(100.0, ALPHA)) / t
    assert np.all(np.diff(rad) < 0.0)
    assert rad[-1] < 0.05


def test_subgaussian_incremental_equals_batch():
    rng = make_rng(3, "cs", "inc")
    x = 2.0 + rng.standard_normal(200)
    batch = subgaussian_cs(x, 1.0, 0.1, t_opt=30)
    inc = SubGaussianMeanCS(1.0, 0.1, t_opt=30)
    assert inc.interval() == (-math.inf, math.inf)
    for t, v in enumerate(x):
        inc.update(v)
        lo, hi = inc.interval()
        assert lo == pytest.approx(batch.lower[t], rel=1e-12, abs=1e-12)
        assert hi == pytest.approx(batch.upper[t], rel=1e-12, abs=1e-12)
    assert inc.estimate() == pytest.approx(float(np.mean(x)))
    assert inc.n == 200


# --------------------------------------------------------------------------- bounded


@pytest.mark.parametrize(
    ("name", "lo", "hi", "mu"),
    [
        ("bernoulli_rare", 0.0, 1.0, 0.05),
        ("bernoulli_half", 0.0, 1.0, 0.5),
        ("beta_scaled", -2.0, 3.0, -2.0 + 5.0 * 2.0 / 7.0),
        ("two_point_extreme", 10.0, 11.0, 10.9),
    ],
)
@pytest.mark.parametrize("method", ["eb", "hoeffding"])
def test_bounded_time_uniform_coverage(name, lo, hi, mu, method):
    rng = make_rng(4, "cs", name)
    n, t_max = 1500, 500
    if name.startswith("bernoulli"):
        x = rng.binomial(1, mu, size=(n, t_max)).astype(np.float64)
    elif name == "beta_scaled":
        x = lo + (hi - lo) * rng.beta(2.0, 5.0, size=(n, t_max))
    else:
        x = lo + (hi - lo) * rng.binomial(1, 0.9, size=(n, t_max))
    cs = bounded_cs(x, lo, hi, ALPHA, method=method)
    assert _miscoverage(cs, mu) <= _bound(n)
    assert np.all(cs.lower >= lo) and np.all(cs.upper <= hi)


def test_bounded_eb_coverage_with_adaptive_distribution():
    """The distribution switches with the past (Bernoulli(μ) vs the constant μ), the
    conditional mean stays μ: the empirical-Bernstein CS still covers."""
    rng = make_rng(5, "cs", "switch")
    n, t_max, mu = 1500, 400, 0.3
    x = np.empty((n, t_max))
    s = np.zeros(n)
    for t in range(t_max):
        coin = rng.binomial(1, mu, size=n).astype(np.float64)
        x[:, t] = np.where(s > mu * t, coin, mu)
        s += x[:, t]
    assert _miscoverage(bounded_cs(x, 0.0, 1.0, ALPHA), mu) <= _bound(n)


def test_bounded_eb_adapts_to_low_variance():
    rng = make_rng(6, "cs", "lowvar")
    x = rng.beta(200.0, 200.0, size=2000)  # sd ≈ 0.025
    eb = bounded_cs(x, 0.0, 1.0, ALPHA)
    hoe = bounded_cs(x, 0.0, 1.0, ALPHA, method="hoeffding")
    w_eb = eb.upper[-1] - eb.lower[-1]
    w_h = hoe.upper[-1] - hoe.lower[-1]
    assert w_eb < 0.3 * w_h
    assert eb.lower[-1] < 0.5 < eb.upper[-1]


def test_bounded_incremental_equals_batch():
    rng = make_rng(7, "cs", "inc-b")
    x = 5.0 + 3.0 * rng.beta(2.0, 3.0, size=300)
    batch = bounded_cs(x, 5.0, 8.0, 0.05, c=0.75)
    inc = BoundedMeanCS(5.0, 8.0, 0.05, c=0.75)
    assert inc.interval() == (5.0, 8.0)
    for t, v in enumerate(x):
        inc.update(v)
        lo, hi = inc.interval()
        assert lo == pytest.approx(batch.lower[t], rel=1e-12, abs=1e-12)
        assert hi == pytest.approx(batch.upper[t], rel=1e-12, abs=1e-12)
    assert inc.estimate() == pytest.approx(float(batch.estimate[-1]), rel=1e-12)


def test_confseq_input_validation():
    with pytest.raises(ValueError):
        subgaussian_cs(np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        subgaussian_cs(np.zeros(0), 1.0)
    with pytest.raises(ValueError):
        subgaussian_cs(np.array([1.0, math.nan]), 1.0)
    with pytest.raises(ValueError):
        bounded_cs(np.array([0.5, 1.5]), 0.0, 1.0)
    with pytest.raises(ValueError):
        bounded_cs(np.array([0.5]), 1.0, 0.0)
    with pytest.raises(ValueError):
        bounded_cs(np.array([0.5]), 0.0, 1.0, method="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        normal_mixture_boundary(1.0, 1.5, 1.0)
    with pytest.raises(ValueError):
        BoundedMeanCS(0.0, 1.0).update(2.0)
    with pytest.raises(ValueError):
        SubGaussianMeanCS(1.0).update(math.inf)
