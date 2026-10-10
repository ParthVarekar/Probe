"""NIG engine: rank-1 updates vs the batch closed form, prequential evidence,
missing data, MLE, parameter EIG, query kernels and clone independence."""

import math

import numpy as np
import pytest
from engine_testkit import batch_nig, linear_h, numeric_covs, records_from
from scipy import integrate, special, stats

from probe.determinism import canonical_json, make_rng
from probe.engine.ir import prior_arrays
from probe.engine.models import HypothesisModel
from probe.engine.nig import NIGModel
from probe.engine.queries import ContrastQuery


def _data(seed: int, n: int, p: int, noise: float = 0.7):
    rng = make_rng(seed, "nig-data")
    x = rng.normal(size=(n, p))
    beta = rng.normal(size=p + 1)
    y = beta[0] + x @ beta[1:] + noise * rng.standard_normal(n)
    return x, y


def _fit(h, covs, x, y):
    m = NIGModel(h, covs)
    for r, yi in zip(records_from(x), y, strict=True):
        m.update(r, float(yi))
    return m


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_rank1_posterior_matches_batch_closed_form(seed):
    p = 3
    covs = numeric_covs(p)
    h = linear_h("H", p, prior_means=[0.5, -0.2, 0.0, 1.0], a0=1.5 + seed, b0=0.5 + seed)
    x, y = _data(seed, 25, p)
    m = _fit(h, covs, x, y)
    mu0, v0 = prior_arrays(h)
    phi = np.column_stack([np.ones(len(y)), x])
    ref = batch_nig(phi, y, mu0, v0, h.prior.a0, h.prior.b0)
    post = m.posterior
    np.testing.assert_allclose(post.mu, ref["mu"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(post.cov, ref["cov"], rtol=1e-9, atol=1e-13)
    assert post.a == pytest.approx(ref["a"], rel=1e-14)
    assert post.b == pytest.approx(ref["b"], rel=1e-10)
    assert m.log_evidence() == pytest.approx(ref["log_evidence"], rel=1e-10, abs=1e-9)
    assert m.n_obs == 25

    # predictive t_{2a}(xᵀμ, (b/a)(1 + xᵀVx)) from the batch posterior
    xq = make_rng(seed, "query").normal(size=(7, p))
    pred = m.predictive(records_from(xq))
    phq = np.column_stack([np.ones(7), xq])
    quad = np.einsum("ij,jk,ik->i", phq, ref["cov"], phq)
    np.testing.assert_allclose(pred.loc, phq @ ref["mu"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(pred.scale, np.sqrt(ref["b"] / ref["a"] * (1 + quad)), rtol=1e-10)
    np.testing.assert_allclose(pred.df, 2 * ref["a"])


def test_prequential_sum_equals_log_evidence_and_closed_form():
    p = 2
    covs = numeric_covs(p)
    h = linear_h("H", p)
    x, y = _data(11, 40, p)
    m = NIGModel(h, covs)
    total = 0.0
    for r, yi in zip(records_from(x), y, strict=True):
        total += m.log_predictive(r, float(yi))  # before the update
        m.update(r, float(yi))
    phi = np.column_stack([np.ones(len(y)), x])
    mu0, v0 = prior_arrays(h)
    closed = batch_nig(phi, y, mu0, v0, h.prior.a0, h.prior.b0)["log_evidence"]
    assert total == pytest.approx(m.log_evidence(), rel=1e-13, abs=1e-12)
    assert m.log_evidence() == pytest.approx(closed, rel=1e-10)


def test_long_run_stays_on_the_batch_solution():
    """2000 rank-1 steps with correlated, uncentred features stay within 1e-8 of
    the batch posterior (no visible drift from the covariance downdates)."""
    p = 4
    covs = numeric_covs(p)
    h = linear_h("H", p, prior_sd=10.0, intercept_sd=30.0)
    rng = make_rng(5, "drift")
    base = rng.normal(size=(2000, 1))
    x = 3.0 + base + 0.05 * rng.normal(size=(2000, p))
    y = 1.0 + x @ np.arange(1, p + 1) + rng.normal(size=2000)
    m = _fit(h, covs, x, y)
    phi = np.column_stack([np.ones(len(y)), x])
    mu0, v0 = prior_arrays(h)
    ref = batch_nig(phi, y, mu0, v0, h.prior.a0, h.prior.b0)
    np.testing.assert_allclose(m.posterior.mu, ref["mu"], rtol=1e-8)
    np.testing.assert_allclose(m.posterior.cov, ref["cov"], rtol=1e-6, atol=1e-12)
    assert m.posterior.b == pytest.approx(ref["b"], rel=1e-8)
    assert m.log_evidence() == pytest.approx(ref["log_evidence"], rel=1e-9)


def test_missing_y_is_ignored_and_nonfinite_rejected():
    covs = numeric_covs(1)
    m = NIGModel(linear_h("H", 1), covs)
    m.update({"x0": 0.3}, 1.2)
    before = m.posterior
    ev = m.log_evidence()
    m.update({"x0": 5.0}, None)
    after = m.posterior
    assert m.n_obs == 1
    assert m.log_evidence() == ev
    np.testing.assert_array_equal(before.mu, after.mu)
    np.testing.assert_array_equal(before.cov, after.cov)
    assert (before.a, before.b) == (after.a, after.b)
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            m.update({"x0": 0.0}, bad)
        with pytest.raises(ValueError):
            m.log_predictive({"x0": 0.0}, bad)


def test_max_log_likelihood():
    covs = numeric_covs(2)
    h = linear_h("H", 2)
    m = NIGModel(h, covs)
    assert m.max_log_likelihood() == 0.0
    x, y = _data(3, 6, 2)
    recs = records_from(x)
    for i in range(3):  # n = 3 = rank: interpolates
        m.update(recs[i], float(y[i]))
        assert m.max_log_likelihood() == math.inf
    for i in range(3, 6):
        m.update(recs[i], float(y[i]))
    phi = np.column_stack([np.ones(6), x])
    beta, *_ = np.linalg.lstsq(phi, y, rcond=None)
    s2 = float(np.sum((y - phi @ beta) ** 2)) / 6
    expected = float(np.sum(stats.norm.logpdf(y, phi @ beta, math.sqrt(s2))))
    assert m.max_log_likelihood() == pytest.approx(expected, rel=1e-10)
    # it is the supremum: perturbing (β, σ²) never increases the likelihood
    rng = make_rng(0, "mle")
    for _ in range(20):
        b2 = beta + 0.05 * rng.normal(size=3)
        sd2 = math.sqrt(s2) * math.exp(0.1 * rng.normal())
        assert np.sum(stats.norm.logpdf(y, phi @ b2, sd2)) <= m.max_log_likelihood() + 1e-12


def test_max_log_likelihood_uses_rank_and_detects_exact_fit():
    covs = numeric_covs(2)
    m = NIGModel(linear_h("H", 2), covs)
    for t in range(4):  # x1 = x0: rank 2 < p = 3; 4 points > rank, exact linear fit
        m.update({"x0": float(t), "x1": float(t)}, 2.0 + 3.0 * t)
    assert m.max_log_likelihood() == math.inf
    m.update({"x0": 4.0, "x1": 4.0}, 0.0)  # breaks the exact fit
    assert math.isfinite(m.max_log_likelihood())


def test_parameter_eig_is_nonnegative_and_matches_monte_carlo():
    p = 2
    covs = numeric_covs(p)
    h = linear_h("H", p)
    x, y = _data(8, 5, p)
    m = _fit(h, covs, x, y)
    xq = np.array([[0.0, 0.0], [1.5, -2.0], [4.0, 4.0]])
    recs = records_from(xq)
    eig = m.parameter_eig(recs)
    assert np.all(eig >= 0)
    pred = m.predictive(recs)
    rng = make_rng(8, "param-eig-mc")
    n = 40_000
    for i, r in enumerate(recs):
        theta = m.sample_parameters(rng, n)
        ys = m.simulate(theta, r, rng)
        mean = theta.beta @ m.features([r])[0]
        cond = stats.norm.logpdf(ys, mean, np.sqrt(theta.sigma2))  # log p(y_s | θ_s)
        marg = pred.take([i]).logpdf(ys[:, None])[:, 0]
        t = cond - marg
        se = t.std(ddof=1) / math.sqrt(n)
        assert abs(t.mean() - eig[i]) < 4 * se + 1e-3, (i, t.mean(), eig[i], se)
    # more data → less to learn at a fixed design
    m2 = m.clone()
    for r, yi in zip(records_from(x), y, strict=True):
        m2.update(r, float(yi))
    assert np.all(m2.parameter_eig(recs) < eig)


def test_clone_is_independent():
    covs = numeric_covs(1)
    m = NIGModel(linear_h("H", 1), covs)
    m.update({"x0": 1.0}, 0.5)
    c = m.clone()
    snap = (m.posterior.mu.copy(), m.log_evidence(), m.n_obs)
    c.update({"x0": -1.0}, 3.0)
    assert (m.n_obs, m.log_evidence()) == (snap[2], snap[1])
    np.testing.assert_array_equal(m.posterior.mu, snap[0])
    m.update({"x0": 2.0}, -1.0)
    assert c.n_obs == 2 and c.log_evidence() != m.log_evidence()


def test_conforms_to_protocol_and_summary_is_canonical_json():
    covs = numeric_covs(1)
    m = NIGModel(linear_h("H", 1, a0=0.5), covs)
    assert isinstance(m, HypothesisModel)
    s = m.summary()
    assert s["beta_sd"] is None  # a = 0.5: marginal variance infinite
    canonical_json(s)
    for t in range(4):
        m.update({"x0": float(t)}, float(t))
    s = m.summary()
    assert s["n"] == 4 and len(s["beta_sd"]) == 2
    np.testing.assert_allclose(
        s["beta_scale"], np.sqrt(np.diag(m.posterior.cov) * m.posterior.b / m.posterior.a)
    )
    canonical_json(s)


def test_linear_query_kernel_matches_explicit_update():
    """P(cᵀβ < τ | D, y) from the kernel equals the probability computed after
    actually updating a clone with (x, y)."""
    p = 2
    covs = numeric_covs(p)
    m = _fit(linear_h("H", p), covs, *_data(4, 6, p))
    q = ContrastQuery(
        records=({"x0": 1.0, "x1": 0.0}, {"x0": 0.0, "x1": 0.0}), coefs=(1.0, -1.0), threshold=0.2
    )
    cands = records_from(np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]]))
    kern = m.linear_query(cands, q)
    c = m.contrast_vector(q)
    np.testing.assert_allclose(c, [0.0, 1.0, 0.0])

    def prob(model):
        post = model.posterior
        sd = math.sqrt(post.b / post.a * float(c @ post.cov @ c))
        return float(stats.t.cdf((0.2 - c @ post.mu) / sd, 2 * post.a))

    assert kern.prior_prob == pytest.approx(prob(m), rel=1e-12)
    ys = np.array([-3.0, 0.0, 0.7, 5.0])
    for i, r in enumerate(cands):
        got = kern.prob_after(ys[None, :], np.array([i]))[0]
        for j, yv in enumerate(ys):
            mc = m.clone()
            mc.update(r, float(yv))
            assert got[j] == pytest.approx(prob(mc), rel=1e-9, abs=1e-14)
    gt = ContrastQuery(records=q.records, coefs=q.coefs, threshold=0.2, direction=">")
    assert m.linear_query(cands, gt).prior_prob == pytest.approx(1 - prob(m), rel=1e-12)


def test_contrast_that_vanishes_is_deterministic():
    covs = numeric_covs(1)
    m = NIGModel(linear_h("H", 1), covs)
    q = ContrastQuery(records=({"x0": 1.0}, {"x0": 1.0}), coefs=(1.0, -1.0), threshold=0.5)
    k = m.linear_query([{"x0": 0.0}], q)
    assert k.deterministic and k.prior_prob == 1.0  # 0 < 0.5
    assert np.all(k.prob_after(np.zeros((1, 3)), np.array([0])) == 1.0)


def test_evidence_matches_brute_force_double_integral():
    """Intercept-only model (β, σ²): integrate likelihood × prior numerically."""
    covs = numeric_covs(1)
    h = linear_h("H", 0, intercept_sd=2.0, prior_means=[0.3], a0=2.5, b0=1.5)
    y = np.array([0.9, 1.7, 0.4, 1.2, 2.2, 0.8])
    m = NIGModel(h, covs)
    for v in y:
        m.update({"x0": 0.0}, float(v))
    a0, b0, mu0, v0 = 2.5, 1.5, 0.3, 4.0
    c = m.log_evidence()

    n, sy, syy = len(y), float(y.sum()), float(y @ y)
    log_norm_s = a0 * math.log(b0) - math.lgamma(a0)

    def integrand(beta, log_s2):
        s2 = math.exp(log_s2)
        rss = syy - 2.0 * beta * sy + n * beta * beta
        ll = -0.5 * n * math.log(2.0 * math.pi * s2) - 0.5 * rss / s2
        lp_b = -0.5 * math.log(2.0 * math.pi * s2 * v0) - 0.5 * (beta - mu0) ** 2 / (s2 * v0)
        # InvGamma(a0, b0) density of σ², times the Jacobian σ² of the log σ² variable
        lp_s = log_norm_s - (a0 + 1.0) * log_s2 - b0 / s2 + log_s2
        return math.exp(ll + lp_b + lp_s - c)

    val, err = integrate.dblquad(
        integrand, -8.0, 6.0, lambda _: -4.0, lambda _: 8.0, epsabs=1e-10, epsrel=1e-9
    )
    assert math.log(val) == pytest.approx(0.0, abs=1e-6), (val, err)


@pytest.mark.slow
def test_evidence_matches_dynesty_two_coefficients():
    """Nested sampling (dynesty, seeded) on intercept + slope + σ²."""
    dynesty = pytest.importorskip("dynesty")
    covs = numeric_covs(1)
    h = linear_h("H", 1, prior_sd=1.0, intercept_sd=2.0, a0=3.0, b0=2.0)
    x, y = _data(21, 12, 1)
    m = _fit(h, covs, x, y)
    mu0, v0 = prior_arrays(h)
    a0, b0 = 3.0, 2.0

    def prior_transform(u):
        s2 = stats.invgamma.ppf(u[0], a0, scale=b0)
        beta = mu0 + np.sqrt(s2 * v0) * special.ndtri(u[1:])
        return np.concatenate([[s2], beta])

    def loglike(t):
        s2, b = t[0], t[1:]
        mean = b[0] + b[1] * x[:, 0]
        return float(np.sum(stats.norm.logpdf(y, mean, math.sqrt(s2))))

    sampler = dynesty.NestedSampler(
        loglike, prior_transform, 3, nlive=600, rstate=np.random.default_rng(1234)
    )
    sampler.run_nested(dlogz=0.01, print_progress=False)
    res = sampler.results
    logz, logz_err = float(res.logz[-1]), float(res.logzerr[-1])
    assert abs(logz - m.log_evidence()) < max(4 * logz_err, 0.05), (logz, logz_err)
