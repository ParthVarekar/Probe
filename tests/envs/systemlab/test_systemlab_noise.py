import numpy as np
import pytest
from scipy import stats

from probe.determinism import make_rng
from probe.envs.systemlab.noise import (
    ReportNoise,
    flip_channel,
    report_probs,
    report_probs_point,
    sample_report,
)


@pytest.mark.parametrize("n", [0, 1, 4, 12])
@pytest.mark.parametrize("eps", [0.0, 0.03, 0.2])
def test_flip_channel_rows_are_distributions(n, eps):
    c = flip_channel(n, eps)
    assert c.shape == (n + 1, n + 1)
    assert np.all(c >= 0)
    np.testing.assert_allclose(c.sum(axis=1), 1.0, atol=1e-12)
    if eps == 0.0:
        np.testing.assert_array_equal(c, np.eye(n + 1))
    else:
        assert np.all(c > 0)  # every count reachable: no exact zeros
    assert not c.flags.writeable


def test_flip_channel_matches_closed_form():
    n, eps = 5, 0.1
    c = flip_channel(n, eps)
    for k in range(n + 1):
        for y in range(n + 1):
            want = sum(
                stats.binom.pmf(j, k, 1 - eps) * stats.binom.pmf(y - j, n - k, eps)
                for j in range(0, min(k, y) + 1)
            )
            assert c[k, y] == pytest.approx(want, abs=1e-14)


def test_report_probs_glitch_and_padding():
    k = np.array([0, 3, 5])
    p = report_probs_point(k, 5, 0.05, 0.2, 9)
    assert p.shape == (3, 9)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(p[:, 6:] == 0.0)
    assert np.all(p[:, :6] >= 0.2 / 6 - 1e-15)
    # per-grid-point noise parameters
    p2 = report_probs_point(k, 5, np.array([0.0, 0.05, 0.3]), np.array([0.0, 0.2, 0.0]), 6)
    assert p2[0, 0] == 1.0
    np.testing.assert_allclose(p2[1], p[1, :6])


def test_report_probs_mixture_is_linear():
    n = 4
    pk = np.array([[0.2, 0.3, 0.0, 0.5, 0.0]])
    mixed = report_probs(pk, n, 0.07, 0.1, 5)
    parts = sum(pk[0, k] * report_probs_point(np.array([k]), n, 0.07, 0.1, 5) for k in range(n + 1))
    np.testing.assert_allclose(mixed, parts, atol=1e-14)


def test_report_probs_rejects_bad_shapes():
    with pytest.raises(ValueError):
        report_probs_point(np.array([6]), 5, 0.0, 0.0, 6)
    with pytest.raises(ValueError):
        report_probs_point(np.array([1]), 5, 0.0, 0.0, 5)  # support too small
    with pytest.raises(ValueError):
        ReportNoise(eps=0.6)
    with pytest.raises(ValueError):
        ReportNoise(glitch=1.0)


@pytest.mark.parametrize(
    ("k", "n", "noise"),
    [(3, 8, ReportNoise(0.05, 0.0)), (0, 6, ReportNoise(0.1, 0.25)), (12, 12, ReportNoise(0.03))],
)
def test_sampler_matches_pmf(k, n, noise):
    """Counter-based draws reproduce the channel law (chi-square, fixed seeds)."""
    draws = np.array(
        [sample_report(k, n, noise, make_rng(11, "noise-test", i)) for i in range(20_000)]
    )
    p = report_probs_point(np.array([k]), n, noise.eps, noise.glitch, n + 1)[0]
    obs = np.bincount(draws, minlength=n + 1)
    keep = p * draws.size >= 5
    exp = p[keep] * draws.size
    o = obs[keep]
    tail_o, tail_e = draws.size - o.sum(), draws.size - exp.sum()
    if tail_e > 0:
        o, exp = np.append(o, tail_o), np.append(exp, tail_e)
    chi2 = float(np.sum((o - exp) ** 2 / exp))
    assert stats.chi2.sf(chi2, len(o) - 1) > 1e-3
    assert np.all(obs[p == 0] == 0)


def test_sampler_consumes_a_fixed_stream():
    """Same key ⇒ same draw; the number of uniforms consumed depends only on n."""
    a = sample_report(2, 5, ReportNoise(0.1, 0.2), make_rng(1, "x"))
    b = sample_report(2, 5, ReportNoise(0.1, 0.2), make_rng(1, "x"))
    assert a == b
    r1, r2 = make_rng(1, "y"), make_rng(1, "y")
    sample_report(0, 5, ReportNoise(0.0, 0.0), r1)
    sample_report(5, 5, ReportNoise(0.3, 0.5), r2)
    assert r1.random() == r2.random()
