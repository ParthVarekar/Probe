"""Predictive distributions: input validation, the Gaussian limit ν = ∞, exact
normalisation of PMFs and numerically safe moments."""

import math

import numpy as np
import pytest
from scipy import stats

from probe.engine.predictive import DiscretePredictive, StudentTPredictive


def _t(df, loc, scale):
    return StudentTPredictive(
        np.asarray(df, dtype=float), np.asarray(loc, dtype=float), np.asarray(scale, dtype=float)
    )


@pytest.mark.parametrize(
    ("df", "loc", "scale"),
    [
        ([math.nan], [0.0], [1.0]),
        ([4.0], [0.0], [math.nan]),
        ([4.0], [math.nan], [1.0]),
        ([4.0], [math.inf], [1.0]),
        ([4.0], [0.0], [math.inf]),
        ([0.0], [0.0], [1.0]),
        ([4.0], [0.0], [0.0]),
        ([-math.inf], [0.0], [1.0]),
    ],
)
def test_student_t_rejects_nan_and_non_finite_parameters(df, loc, scale):
    """NaN compares False with everything, so a `scale <= 0` check alone let NaN
    through and every downstream density silently became NaN."""
    with pytest.raises(ValueError):
        _t(df, loc, scale)


def test_student_t_gaussian_limit():
    """ν = +inf is the Gaussian N(loc, scale²): density, entropy and variance."""
    p = _t([math.inf, math.inf], [1.0, -2.0], [0.5, 3.0])
    y = np.array([[0.3, 4.0], [1.0, -2.0], [9.0, -30.0]])
    np.testing.assert_allclose(
        p.logpdf(y), stats.norm.logpdf(y, [1.0, -2.0], [0.5, 3.0]), rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(p.entropy(), stats.norm.entropy(scale=[0.5, 3.0]), rtol=1e-14)
    np.testing.assert_array_equal(p.variance(), [0.25, 9.0])
    # and it is the limit of large finite ν
    big = _t([1e12, 1e12], [1.0, -2.0], [0.5, 3.0])
    np.testing.assert_allclose(big.logpdf(y), p.logpdf(y), rtol=1e-10)


def test_student_t_variance():
    p = _t([1.0, 2.0, 2.5, 30.0], [0.0] * 4, [2.0] * 4)
    np.testing.assert_array_equal(p.variance()[:2], [math.inf, math.inf])
    np.testing.assert_allclose(p.variance()[2:], [4.0 * 2.5 / 0.5, 4.0 * 30 / 28], rtol=1e-15)


def test_discrete_rows_must_sum_to_one_within_1e9():
    """np.allclose's default rtol (1e-5) used to accept rows summing to 1 + 9e-6,
    which the 'exact' discrete EIG then treated as probabilities."""
    with pytest.raises(ValueError):
        DiscretePredictive(np.array([[0.5, 0.500009]]))
    with pytest.raises(ValueError):
        DiscretePredictive(np.array([[0.5, 0.5 - 2e-9]]))
    with pytest.raises(ValueError):
        DiscretePredictive(np.array([[np.nan, 1.0]]))
    DiscretePredictive(np.array([[0.5, 0.5 + 5e-10]]))  # within tolerance


def test_discrete_moments_and_entropy():
    probs = np.array([[0.0, 0.0, 1.0], [0.25, 0.5, 0.25], [1.0, 0.0, 0.0]])
    d = DiscretePredictive(probs)
    np.testing.assert_allclose(d.mean(), [2.0, 1.0, 0.0])
    np.testing.assert_allclose(d.variance(), [0.0, 0.5, 0.0], atol=0)
    ent = d.entropy()
    assert ent[0] == 0.0 and math.copysign(1.0, ent[0]) == 1.0  # +0.0, not −0.0
    # centred variance never cancels below zero at a large-count near point mass
    s = 10_001
    p = np.zeros((1, s))
    p[0, -1] = 1.0 - 1e-15
    p[0, -2] = 1e-15
    assert DiscretePredictive(p).variance()[0] >= 0.0


def test_discrete_mean_is_independent_of_batch_position():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(40), size=301)
    d = DiscretePredictive(probs)
    full = d.mean()
    alone = np.array([DiscretePredictive(probs[i : i + 1]).mean()[0] for i in range(301)])
    np.testing.assert_array_equal(full, alone)
