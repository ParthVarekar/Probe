"""EIG kernels: quadrature rule, accuracy against independent references, posterior
weights, mixture moments, goal-oriented EIG and the per-step timing record."""

import functools
import itertools
import math
import time

import numpy as np
import pytest
from engine_testkit import COUNT_OUT, grid_h, linear_h, numeric_covs, records_from
from numpy.polynomial.legendre import leggauss
from scipy import integrate, special, stats

from probe.determinism import make_rng
from probe.engine import eig as eig_mod
from probe.engine.eig import (
    DegeneratePosteriorError,
    NumericalIntegrityError,
    QuadratureSettings,
    entropy,
    goal_eig,
    log_posterior_weights,
    log_prior_weights,
    mc_model_eig,
    mixture_moments,
    model_indicator_eig,
    posterior_weights,
    weighted_parameter_eig,
)
from probe.engine.grid import GridModel
from probe.engine.ir import HypothesisSet, OutcomeSpec
from probe.engine.nig import NIGModel
from probe.engine.predictive import DiscretePredictive, StudentTPredictive, t_log_normalizer
from probe.engine.queries import ContrastQuery, GridQuery


def _t(df, loc, scale):
    return StudentTPredictive(
        np.atleast_1d(np.asarray(df, float)),
        np.atleast_1d(np.asarray(loc, float)),
        np.atleast_1d(np.asarray(scale, float)),
    )


@functools.lru_cache(maxsize=4096)
def _t_log_normalizer_by_quadrature(nu: float) -> float:
    """−log ∫ (1 + z²/ν)^(−(ν+1)/2) dz by QUADPACK, independent of the engine's
    closed form. (The textbook gammaln((ν+1)/2) − gammaln(ν/2) − ½ log νπ loses
    ~10⁻¹⁰ to cancellation at ν ≈ 10⁵–10⁶, and scipy's betaln does no better, so
    neither can serve as an oracle at the 10⁻¹² level for large ν.)"""
    kw = {"epsabs": 1e-300, "epsrel": 1e-13, "limit": 400}

    log100 = math.log(100.0)

    def log_kernel(log_z: float) -> float:
        return -0.5 * (nu + 1.0) * float(np.logaddexp(0.0, 2.0 * log_z - math.log(nu)))

    def kernel(z: float) -> float:
        return math.exp(log_kernel(math.log(z))) if z > 0.0 else 1.0

    def tail(v: float) -> float:  # z = 100·e^v, dz = z dv
        return math.exp(log_kernel(log100 + v) + log100 + v)

    total = 0.0
    for a, b in itertools.pairwise((0.0, 1.0, 3.0, 10.0, 30.0, 100.0)):
        total += integrate.quad(kernel, a, b, **kw)[0]
    top = min(700.0, 80.0 / nu + 20.0)  # the tail integrand decays like e^(−νv)
    total += integrate.quad(tail, 0.0, top, **kw)[0]
    return -math.log(2.0 * total)


def test_reference_normalizer_matches_closed_form():
    nus = [0.5, 1.0, 4.0, 30.0, 49.0, 50.0, 304.0, 2151.6, 1e5, 6.6e5, 1e7, 1e9]
    ref = np.array([_t_log_normalizer_by_quadrature(nu) for nu in nus])
    # both sides are good to ~1e-14 (gammaln cancellation below ν = 50, quad epsrel)
    np.testing.assert_allclose(t_log_normalizer(np.array(nus)), ref, rtol=0, atol=2e-14)


def _mixture_logpdfs(w, df, loc, scale):
    """Arrays for the positively weighted components and a function y -> (log p_k(y),
    log π_k), with normalizers from :func:`_t_log_normalizer_by_quadrature`."""
    w, df, loc, scale = (np.asarray(v, dtype=float) for v in (w, df, loc, scale))
    const = np.array([_t_log_normalizer_by_quadrature(float(nu)) for nu in df]) - np.log(scale)
    keep = w > 0
    w, df, loc, scale, const = w[keep], df[keep], loc[keep], scale[keep], const[keep]

    def logpdfs(y):
        return const - 0.5 * (df + 1) * np.log1p(((y - loc) / scale) ** 2 / df)

    return logpdfs, np.log(w), loc, scale


def _reference_line(h, loc, scale):
    """∫ h(y) dy by scipy adaptive quad (QUADPACK): the line is cut at
    loc_k ± scale_k·{0, 1, 3, 10, 30, ...} (out to the span of all components) and
    every piece, plus both infinite tails, is integrated separately."""
    span = float(loc.max() - loc.min() + 10 * scale.max())
    mults = [0.0]
    m = 1.0
    while m * scale.min() <= 2 * span:
        mults += [m, 3 * m]
        m *= 10
    mults = np.asarray(mults)
    pts = np.unique(np.concatenate([loc + sgn * scale * k for k in mults for sgn in (-1, 1)]))
    kw = {"epsabs": 1e-14, "epsrel": 1e-12, "limit": 500}
    total = 0.0
    for a, b in itertools.pairwise(pts):
        total += integrate.quad(h, a, b, **kw)[0]
    # tails: y = pts[0] − e^v and y = pts[-1] + e^v decay exponentially in v even for
    # very heavy tails (QUADPACK's own infinite-range map struggles with ν < 1)
    for v0, v1 in ((-40.0, 0.0), (0.0, 10.0), (10.0, 30.0), (30.0, 80.0), (80.0, 300.0)):
        total += integrate.quad(lambda v: h(pts[0] - math.exp(v)) * math.exp(v), v0, v1, **kw)[0]
        total += integrate.quad(lambda v: h(pts[-1] + math.exp(v)) * math.exp(v), v0, v1, **kw)[0]
    return total


def _reference_u(w, df, loc, scale):
    """U_M = ∫ Σ_k π_k p_k (log p_k − log f) dy by :func:`_reference_line`,
    independent of the engine."""
    logpdfs, logw, loc, scale = _mixture_logpdfs(w, df, loc, scale)

    def h(y):
        lp = logpdfs(y)
        a = lp + logw
        lf = special.logsumexp(a)
        return float(np.sum(np.exp(a) * (lp - lf)))

    return _reference_line(h, loc, scale)


def _reference_deficit(w, df, loc, scale):
    """H(π) − U_M = ∫ f(y)·H(r(y)) dy (expected entropy of the responsibilities),
    integrated directly: it is ≥ 0 and tiny when the components barely overlap, so
    it is computed to full relative precision where H(π) − U_M would cancel."""
    logpdfs, logw, loc, scale = _mixture_logpdfs(w, df, loc, scale)

    def h(y):
        a = logpdfs(y) + logw
        lf = special.logsumexp(a)
        r = np.exp(a - lf)
        return math.exp(lf) * float(-np.sum(special.xlogy(r, r)))

    return _reference_line(h, loc, scale)


# --------------------------------------------------------------------------- the rule


def test_gauss_kronrod_constants():
    xg, wg = leggauss(7)
    nodes, wk, wgk = eig_mod._GK_NODES, eig_mod._GK_WK, eig_mod._GK_WG
    np.testing.assert_allclose(nodes[[1, 3, 5, 7, 9, 11, 13]], xg, atol=1e-15)
    np.testing.assert_allclose(wgk[[1, 3, 5, 7, 9, 11, 13]], wg, atol=1e-15)
    assert wk.sum() == pytest.approx(2.0, abs=1e-15)
    for deg in range(23):  # K15 integrates polynomials of degree ≤ 22 exactly
        exact = 0.0 if deg % 2 else 2.0 / (deg + 1)
        assert float(wk @ nodes**deg) == pytest.approx(exact, abs=1e-14)
    assert float(wk @ nodes**24) != pytest.approx(2.0 / 25, abs=1e-12)


# --------------------------------------------------------------------------- U_M accuracy

CONFIGS = [
    # (weights, df, loc, scale)
    ([0.5, 0.5], [4, 4], [0, 1], [1, 1]),
    ([0.3, 0.7], [10, 3], [0, 0], [1, 2]),
    ([0.2, 0.3, 0.5], [5, 8, 30], [-1, 0, 2], [0.5, 1, 2]),
    ([0.5, 0.5], [6, 6], [0, 500], [1, 1]),  # well separated: U ≈ H(π)
    ([0.6, 0.4], [3, 3], [0, 0.001], [1, 1]),  # nearly identical: tiny U
    ([0.5, 0.5], [5, 5], [0, 0], [100.0, 0.05]),  # narrow inside wide
    ([0.25, 0.25, 0.25, 0.25], [1, 1, 2, 0.5], [0, 3, -2, 1], [1, 0.3, 2, 1]),  # heavy tails
    ([0.1, 0.2, 0.3, 0.2, 0.2], [4, 9, 16, 25, 60], [-3, -1, 0, 1, 3], [1, 1.5, 0.5, 2, 1]),
    # Regression: NIG predictives after ~300 observations (ν = 304, overlapping).
    # A tail cut-off that assumed exp(−ν|v|) decay right past loc ± 5·scale dropped
    # ~4e-9 of mixture mass here while reporting an error of 8e-10.
    (
        [0.1, 0.1, 0.6, 0.1, 0.1],
        [304, 304, 304, 304, 304],
        [1.6078, 2.9012, 2.4868, 2.425, 0.4356],
        [0.7336, 0.5274, 0.5033, 0.5465, 1.254],
    ),
    ([0.3, 0.3, 0.4], [1e6, 1e6, 1e6], [0, 0.7, 2.0], [1, 0.5, 1.5]),  # Gaussian limit
]


@pytest.mark.parametrize(("w", "df", "loc", "scale"), CONFIGS)
def test_student_t_eig_matches_scipy_reference(w, df, loc, scale):
    preds = [_t(d, lo, s) for d, lo, s in zip(df, loc, scale, strict=True)]
    res = model_indicator_eig(preds, w)
    ref = _reference_u(w, df, loc, scale)
    assert abs(res.raw[0] - ref) <= max(res.error[0], 1e-12) + 1e-12, (res.raw[0], ref)
    assert abs(res.raw[0] - ref) < 1e-8 and res.error[0] < 1e-6
    assert 0.0 <= res.value[0] <= res.upper_bound == pytest.approx(entropy(w))
    h_t = np.array([p.entropy()[0] for p in preds])
    assert res.expected_entropy[0] == pytest.approx(float(np.dot(w, h_t)))
    assert res.mixture_entropy[0] == pytest.approx(res.raw[0] + res.expected_entropy[0])


@pytest.mark.parametrize("df", [1.0, 4.0, 30.0, 1e4])
@pytest.mark.parametrize("spacing", [50.0, 1e4, 1e8])
def test_student_t_eig_separated_components_reach_entropy_of_weights(df, spacing):
    """U_M = H(π) − E_f[H(r)]. As the components separate, the deficit (the expected
    entropy of the responsibilities, integrated directly by scipy) vanishes and U_M
    reaches H(π). The deficit itself is checked to within the error estimate,
    which is stricter than comparing U_M with H(π): for ν = 30 at spacing 50 the
    true deficit is 6.3e-11, larger than the engine's error estimate."""
    w = np.array([0.2, 0.3, 0.5])
    loc, scale = [0.0, spacing, 2 * spacing], [1.0, 2.0, 3.0]
    preds = [_t(df, lo, s) for lo, s in zip(loc, scale, strict=True)]
    res = model_indicator_eig(preds, w)
    deficit = _reference_deficit(w, [df] * 3, loc, scale)
    assert deficit >= 0.0
    got = entropy(w) - res.raw[0]
    assert abs(got - deficit) <= res.error[0] + 1e-15, (got, deficit, res.error[0])
    if df >= 30.0:  # light tails: the overlap is negligible
        assert deficit < 1e-10
        assert res.value[0] == pytest.approx(entropy(w), abs=1e-9)
    ref = _reference_u(w, [df] * 3, loc, scale)
    assert abs(res.raw[0] - ref) <= res.error[0] + 1e-9, (res.raw[0], ref, res.error[0])


def test_extreme_separation_is_resolved_without_chasing_rounding_noise():
    """Two components ~1.5e9 of the narrow one's scales apart, far from the origin.
    In a single global coordinate the narrow component is only resolvable to
    ~1e-7 of its width, and the adaptive loop used to bisect that noise up to the
    panel cap (26 million integrand nodes for this one candidate). Anchored
    coordinates resolve it exactly."""
    w = [0.94624732, 0.05375268]
    preds = [_t(89.5594457, -247299.72063605, 0.00901049), _t(82088.629, 302190.12486249, 3.69e-4)]
    res = model_indicator_eig(preds, w)
    assert res.n_evals < 20_000, res.n_evals
    assert res.error[0] < 1e-9
    assert abs(res.raw[0] - entropy(w)) <= res.error[0] + 1e-15  # no overlap at all
    # A narrow component inside a wide one, translated far from the origin: U_M is
    # translation invariant. (scipy's reference, integrating in global y, is itself
    # blurred at offset 1e6 and is only consulted at offset 0.)
    ref = _reference_u([0.5, 0.5], [5.0, 40.0], [0.0, 7.0], [1e3, 1e-3])
    for offset in (0.0, 1e6, 1e9, -1e12):
        preds = [_t(5.0, offset, 1e3), _t(40.0, offset + 7.0, 1e-3)]
        res = model_indicator_eig(preds, [0.5, 0.5])
        assert abs(res.raw[0] - ref) <= res.error[0] + 1e-12, (offset, res.raw[0], ref)
        assert res.n_evals < 5_000, res.n_evals


def test_discrete_eig_matches_entropy_formula():
    rng = make_rng(0, "disc")
    k, n, s = 4, 6, 9
    probs = rng.dirichlet(np.ones(s) * 0.4, size=(k, n))
    probs[0, 0, :4] = 0.0  # zero-probability outcomes
    probs[0, 0] /= probs[0, 0].sum()
    w = rng.dirichlet(np.ones(k))
    res = model_indicator_eig([DiscretePredictive(p) for p in probs], w)
    mix = np.einsum("k,kns->ns", w, probs)
    h_mix = -np.sum(special.xlogy(mix, mix), axis=1)
    h_each = -np.sum(special.xlogy(probs, probs), axis=2)
    np.testing.assert_allclose(res.raw, h_mix - w @ h_each, atol=1e-13)
    np.testing.assert_array_equal(res.error, 0.0)
    assert np.all(res.value <= entropy(w))


def test_zero_weight_hypotheses_are_ignored_exactly():
    a, b = _t([4, 4], [0, 1], [1, 2]), _t([9, 3], [1, -1], [0.5, 1])
    wild = _t([0.3, 0.3], [1e6, -1e6], [1e-6, 1e5])
    r2 = model_indicator_eig([a, b], [0.4, 0.6])
    r3 = model_indicator_eig([a, wild, b], [0.4, 0.0, 0.6])
    np.testing.assert_array_equal(r2.raw, r3.raw)
    np.testing.assert_array_equal(r2.error, r3.error)


def test_monte_carlo_oracle_agrees_with_quadrature():
    rng = make_rng(1, "mc")
    preds = [_t([5, 8], [0.0, 1.0], [1.0, 0.5]), _t([3, 12], [0.5, 1.2], [2.0, 0.4])]
    preds.append(_t([20, 4], [-1.0, 0.8], [0.7, 1.0]))
    w = np.array([0.5, 0.3, 0.2])
    res = model_indicator_eig(preds, w)
    est, se = mc_model_eig(preds, w, rng, 20_000)
    assert np.all(np.abs(est - res.value) < 4 * se), (est, res.value, se)
    dprobs = rng.dirichlet(np.ones(5), size=(3, 2))
    dpreds = [DiscretePredictive(p) for p in dprobs]
    dres = model_indicator_eig(dpreds, w)
    est, se = mc_model_eig(dpreds, w, rng, 20_000)
    assert np.all(np.abs(est - dres.value) < 4 * se)


def test_bound_violation_raises_instead_of_clipping():
    raw = np.array([0.1, -0.5])
    with pytest.raises(NumericalIntegrityError):
        eig_mod._bounded(raw, np.zeros(2), 1.0, "U_M")
    with pytest.raises(NumericalIntegrityError):
        eig_mod._bounded(np.array([1.2]), np.array([0.01]), 1.0, "U_M")
    out = eig_mod._bounded(np.array([-1e-13, 1.0 + 5e-3]), np.array([0.0, 1e-2]), 1.0, "U_M")
    np.testing.assert_array_equal(out, [0.0, 1.0])


def test_input_validation():
    a = _t([4], [0], [1])
    with pytest.raises(ValueError):
        model_indicator_eig([a, a], [0.5, 0.6])
    with pytest.raises(ValueError):
        model_indicator_eig([a, a], [1.5, -0.5])
    with pytest.raises(TypeError):
        model_indicator_eig([a, DiscretePredictive(np.array([[1.0]]))], [0.5, 0.5])
    with pytest.raises(ValueError):
        model_indicator_eig([a, _t([4, 4], [0, 0], [1, 1])], [0.5, 0.5])


def test_tight_tolerance_settings_agree():
    preds = [_t([4, 7], [0, 2], [1, 3]), _t([5, 30], [1, 2.5], [0.3, 0.5])]
    loose = model_indicator_eig(preds, [0.5, 0.5], QuadratureSettings(atol=1e-6, rtol=1e-4))
    tight = model_indicator_eig(preds, [0.5, 0.5], QuadratureSettings(atol=1e-13, rtol=1e-12))
    np.testing.assert_allclose(loose.raw, tight.raw, atol=1e-6)
    assert tight.n_evals >= loose.n_evals


# --------------------------------------------------------------------------- weights, moments


def test_posterior_weights_and_tempering():
    lp = np.log([0.5, 0.25, 0.25])
    le = np.array([-10.0, -12.0, -np.inf])
    w = posterior_weights(lp, le)
    e2 = math.exp(-2.0)
    np.testing.assert_allclose(w, [1 / (1 + 0.5 * e2), 0.5 * e2 / (1 + 0.5 * e2), 0.0])
    assert w[2] == 0.0
    wt = posterior_weights(lp, le, eta=0.5)
    e1 = math.exp(-1.0)
    np.testing.assert_allclose(wt, [1 / (1 + 0.5 * e1), 0.5 * e1 / (1 + 0.5 * e1), 0.0])
    big = posterior_weights([0.0, 0.0], [-1e6, -1e6 - 1.0])  # log-sum-exp stability
    np.testing.assert_allclose(big, [1 / (1 + e1), e1 / (1 + e1)])
    np.testing.assert_allclose(posterior_weights([3.0, 3.0], [0.0, 0.0]), [0.5, 0.5])
    lw = log_posterior_weights(lp, le)
    assert lw[2] == -np.inf and special.logsumexp(lw) == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(DegeneratePosteriorError):
        posterior_weights([0.0, 0.0], [-np.inf, -np.inf])
    for bad in ([np.nan, 0.0], [np.inf, 0.0]):
        with pytest.raises(ValueError):
            posterior_weights([0.0, 0.0], bad)
    for eta in (0.0, 1.5):
        with pytest.raises(ValueError):
            posterior_weights([0.0, 0.0], [0.0, 0.0], eta=eta)


def test_log_prior_weights_from_hypothesis_set():
    covs = numeric_covs(1)
    h1 = linear_h("A", 1).model_copy(update={"prior_weight": 3.0})
    h0 = linear_h("Z", 0, residual=True)
    hs = HypothesisSet(
        outcome=OutcomeSpec(name="y", kind="continuous"), covariates=covs, hypotheses=(h1, h0)
    )
    np.testing.assert_allclose(np.exp(log_prior_weights(hs)), [0.75, 0.25])


def test_mixture_moments():
    a = _t([5, 1.5], [0.0, 1.0], [1.0, 2.0])
    b = _t([10, 10], [2.0, -1.0], [0.5, 1.0])
    mean, var = mixture_moments([a, b], [0.25, 0.75])
    np.testing.assert_allclose(mean, [1.5, -0.5])
    va = 1.0 * 5 / 3
    vb = 0.25 * 10 / 8
    assert var[0] == pytest.approx(0.25 * va + 0.75 * vb + 0.25 * 1.5**2 + 0.75 * 0.5**2)
    assert var[1] == np.inf  # df 1.5 component has infinite variance
    _, var0 = mixture_moments([a, b], [0.0, 1.0])
    assert np.all(np.isfinite(var0))
    d = DiscretePredictive(np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]))
    mean, var = mixture_moments([d, d], [0.5, 0.5])
    np.testing.assert_allclose(mean, [0.5, 2.0])
    np.testing.assert_allclose(var, [0.25, 0.0], atol=1e-15)


def test_weighted_parameter_eig():
    covs = numeric_covs(1)
    ms = [NIGModel(linear_h("A", 1), covs), NIGModel(linear_h("B", 0), covs)]
    recs = [{"x0": 0.0}, {"x0": 2.0}]
    got = weighted_parameter_eig(ms, [0.3, 0.7], recs)
    np.testing.assert_allclose(
        got, 0.3 * ms[0].parameter_eig(recs) + 0.7 * ms[1].parameter_eig(recs)
    )


# --------------------------------------------------------------------------- goal EIG


def _fitted_nig_models(seed=0):
    covs = numeric_covs(2)
    hs = [
        linear_h("Hfull", 2),
        linear_h("Hx0", 1),
        linear_h("Hnull", 0, residual=True),
    ]
    rng = make_rng(seed, "goal-data")
    x = rng.normal(size=(6, 2))
    y = 0.4 + 0.9 * x[:, 0] - 0.3 * x[:, 1] + 0.5 * rng.normal(size=6)
    models = [NIGModel(h, covs) for h in hs]
    for m in models:
        for r, v in zip(records_from(x), y, strict=True):
            m.update(r, float(v))
    w = posterior_weights(np.zeros(3), [m.log_evidence() for m in models])
    return models, w, x, y


def test_goal_eig_contrast_query_matches_explicit_updates():
    """Independent check: r(y) from explicitly updated clones, trapezoid over y."""
    models, w, _, _ = _fitted_nig_models()
    q = ContrastQuery(
        records=({"x0": 1.0, "x1": 0.0}, {"x0": 0.0, "x1": 0.0}), coefs=(1.0, -1.0), threshold=0.6
    )
    cands = records_from(np.array([[2.0, 0.0], [0.0, 2.0]]))
    res = goal_eig(models, w, cands, q)
    assert res.upper_bound == pytest.approx(
        float(
            -special.xlogy(res.query_prob, res.query_prob)
            - special.xlogy(1 - res.query_prob, 1 - res.query_prob)
        )
    )

    def p_query(model):
        c = model.contrast_vector(q)
        post = model.posterior
        var = float(c @ post.cov @ c)
        if var == 0.0:
            return float(0.0 < 0.6)
        return float(
            stats.t.cdf((0.6 - c @ post.mu) / math.sqrt(post.b / post.a * var), 2 * post.a)
        )

    p = float(np.dot(w, [p_query(m) for m in models]))
    assert res.query_prob == pytest.approx(p, rel=1e-12)
    for i, r in enumerate(cands):
        preds = [m.predictive([r]) for m in models]
        lo = min(pr.loc[0] - 40 * pr.scale[0] for pr in preds)
        hi = max(pr.loc[0] + 40 * pr.scale[0] for pr in preds)
        ys = np.linspace(lo, hi, 2001)
        dens = np.stack([pr.logpdf(ys[:, None])[:, 0] for pr in preds], axis=1)
        resp = np.exp(dens + np.log(w))
        f = resp.sum(axis=1)
        rq = np.empty((ys.size, len(models)))
        for k, m in enumerate(models):
            for j, yv in enumerate(ys):
                mc = m.clone()
                mc.update(r, float(yv))
                rq[j, k] = p_query(mc)
        ry = np.clip((resp * rq).sum(axis=1) / f, 0, 1)
        kl = (
            special.xlogy(ry, ry)
            - ry * math.log(p)
            + special.xlogy(1 - ry, 1 - ry)
            - (1 - ry) * math.log1p(-p)
        )
        ref = integrate.trapezoid(f * kl, ys)
        assert res.value[i] == pytest.approx(ref, rel=1e-4, abs=1e-6), (i, res.value[i], ref)
        assert 0.0 <= res.value[i] <= res.upper_bound


@pytest.mark.parametrize("n_obs", [0, 300])
def test_goal_eig_matches_scipy_reference_heavy_and_light_tails(n_obs):
    """U_q against scipy QUADPACK on the same integrand f·KL(Bern(r) ‖ Bern(p)), with
    r(y) from the (separately validated) exact query kernels. n_obs = 0 gives ν = 4
    (heavy tails); n_obs = 300 gives ν = 304, where the integrand lives out to ~±9
    scale units, and extreme y moves the query posterior the most."""
    covs = numeric_covs(2)
    hs = [linear_h("Hfull", 2), linear_h("Hx0", 1), linear_h("Hnull", 0, residual=True)]
    rng = make_rng(3, "goal-late")
    x = rng.normal(size=(n_obs, 2))
    y = 0.4 + 0.08 * x[:, 0] - 0.03 * x[:, 1] + 0.5 * rng.normal(size=n_obs)
    models = [NIGModel(h, covs) for h in hs]
    for m in models:
        for r, v in zip(records_from(x), y, strict=True):
            m.update(r, float(v))
    w = posterior_weights(np.zeros(3), [m.log_evidence() for m in models])
    w = 0.5 * w + 0.5 / 3
    q = ContrastQuery(
        records=({"x0": 1.0, "x1": 0.0}, {"x0": 0.0, "x1": 0.0}), coefs=(1.0, -1.0), threshold=0.05
    )
    cands = records_from(np.array([[3.0, 0.0], [0.0, 3.0], [-2.0, 1.0]]))
    res = goal_eig(models, w, cands, q)
    p = res.query_prob
    assert 0.0 < p < 1.0
    for i, r in enumerate(cands):
        preds = [m.predictive([r]) for m in models]
        kernels = [m.linear_query([r], q) for m in models]

        def h(yv, preds=preds, kernels=kernels):
            lp = np.array([pr.logpdf(np.array([yv]))[0] for pr in preds]) + np.log(w)
            lf = special.logsumexp(lp)
            resp = np.exp(lp - lf)
            rq = np.array([k.prob_after(np.array([[yv]]), np.array([0]))[0, 0] for k in kernels])
            rr = float(np.clip(resp @ rq, 0.0, 1.0))
            kl = (
                special.xlogy(rr, rr)
                - rr * math.log(p)
                + special.xlogy(1.0 - rr, 1.0 - rr)
                - (1.0 - rr) * math.log1p(-p)
            )
            return math.exp(lf) * kl

        locs = np.array([pr.loc[0] for pr in preds])
        scales = np.array([pr.scale[0] for pr in preds])
        ref = _reference_line(h, locs, scales)
        assert abs(res.raw[i] - ref) <= res.error[i] + 1e-12, (i, res.raw[i], ref, res.error[i])
        assert res.error[i] < 1e-7


def test_goal_eig_is_zero_for_a_determined_query_and_bounded():
    models, w, _, _ = _fitted_nig_models(1)
    cands = records_from(np.array([[1.0, 1.0], [3.0, -1.0], [0.0, 0.0]]))
    q0 = ContrastQuery(records=({"x0": 1.0, "x1": 1.0},) * 2, coefs=(1.0, -1.0), threshold=1.0)
    res0 = goal_eig(models, w, cands, q0)
    assert res0.query_prob == 1.0 and np.all(res0.value == 0.0)
    q = ContrastQuery(records=({"x0": 1.0, "x1": 0.0},), coefs=(1.0,), threshold=1.0, direction=">")
    res = goal_eig(models, w, cands, q)
    assert res.upper_bound <= math.log(2)
    assert np.all(res.value >= 0) and np.all(res.value <= res.upper_bound)
    assert np.all(res.error < 1e-6)


def test_goal_eig_grid_query_matches_enumeration():
    hs = [
        grid_h("B1", "test_binomial", {"p": (0.1, 0.3, 0.5, 0.7, 0.9)}),
        grid_h("B2", "test_binomial", {"p": (0.2, 0.4, 0.6)}),
        grid_h("B3", "test_binomial", {}, fixed={"p": 0.5}, residual=True),
    ]
    models = [GridModel(h, COUNT_OUT) for h in hs]
    for m in models:
        m.update({"n": 4}, 3)
    w = posterior_weights(np.zeros(3), [m.log_evidence() for m in models])
    q = GridQuery(param="p", op=">", value=0.45)
    cands = [{"n": 0}, {"n": 2}, {"n": 10}]
    res = goal_eig(models, w, cands, q)
    for i, r in enumerate(cands):
        joint = np.zeros((2, 13))
        for wm, m in zip(w, models, strict=True):
            gw = m.grid_weights()
            pv = m.grid_table()["p"]
            pmf = stats.binom.pmf(np.arange(13)[None, :], r["n"], pv[:, None])
            joint[0] += wm * ((gw * (pv > 0.45)) @ pmf)
            joint[1] += wm * ((gw * (pv <= 0.45)) @ pmf)
        pa, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        mask = joint > 0
        ref = float(np.sum(joint[mask] * np.log(joint[mask] / (pa * py)[mask])))
        assert res.value[i] == pytest.approx(ref, abs=1e-13)
    assert res.value[0] == 0.0 and res.value[2] > res.value[1] > 0
    assert res.per_hypothesis_prob[2] == 1.0  # fixed p = 0.5 > 0.45
    missing = GridQuery(param="rate", op="<", value=1.0, missing=True)
    assert goal_eig(models, w, cands, missing).query_prob == 1.0
    with pytest.raises(TypeError):
        goal_eig(models, w, cands, ContrastQuery(records=({"n": 1},), coefs=(1.0,), threshold=0))


# --------------------------------------------------------------------------- timing


@pytest.mark.parametrize("n_fit", [12, 300])
def test_per_step_eig_timing_5x200(capsys, n_fit):
    """Records the per-step cost for 5 hypotheses × 200 candidates (handover §12 H.5
    estimate: milliseconds to tens of milliseconds on a laptop), early in a run
    (12 observations, ν ≈ 16) and late (300 observations, ν ≈ 304). Weights are
    kept away from a point mass so every hypothesis is integrated."""
    covs = numeric_covs(3)
    hs = [
        linear_h("H1", 1),
        linear_h("H2", 2),
        linear_h("H3", 3),
        linear_h("H4", 3, prior_sd=0.3),
        linear_h("H0", 0, residual=True),
    ]
    rng = make_rng(0, "timing")
    x = rng.normal(size=(n_fit, 3))
    y = 0.5 + x @ np.array([1.0, -0.5, 0.2]) + 0.5 * rng.normal(size=n_fit)
    models = [NIGModel(h, covs) for h in hs]
    for m in models:
        for r, v in zip(records_from(x), y, strict=True):
            m.update(r, float(v))
    w = 0.5 * posterior_weights(np.zeros(5), [m.log_evidence() for m in models]) + 0.1
    cands = records_from(rng.uniform(-3, 3, size=(200, 3)))

    t0 = time.perf_counter()
    preds = [m.predictive(cands) for m in models]
    t_pred = time.perf_counter() - t0
    model_indicator_eig(preds, w)  # warm-up
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = model_indicator_eig(preds, w)
        runs.append(time.perf_counter() - t0)
    t_eig = float(np.median(runs))
    with capsys.disabled():
        print(
            f"\n[M1 timing] U_M for 5 hypotheses x 200 candidates after {n_fit} observations: "
            f"{1e3 * t_eig:.1f} ms (median of 5; {res.n_evals} quadrature nodes, max error "
            f"estimate {res.error.max():.1e}); predictives: {1e3 * t_pred:.1f} ms"
        )
    assert t_eig < 0.5
    assert np.all(res.value >= 0) and np.all(res.value <= entropy(w) + 1e-15)
    assert np.all(res.error < 1e-7)
