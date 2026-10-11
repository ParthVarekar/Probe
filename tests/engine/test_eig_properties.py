"""Property-based tests (hypothesis): information bounds, invariances and
fuzzing of both engines. Derandomized, so every run checks the same examples."""

import math

import numpy as np
from engine_testkit import COUNT_OUT, PROPS, PROPS_FEW, batch_nig, grid_h, linear_h, numeric_covs
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import special
from test_eig import _reference_u

from probe.engine.eig import entropy, model_indicator_eig
from probe.engine.grid import GridModel
from probe.engine.ir import prior_arrays
from probe.engine.nig import NIGModel
from probe.engine.predictive import DiscretePredictive, StudentTPredictive

_f = {"allow_nan": False, "allow_infinity": False}


@st.composite
def weight_vectors(draw, k):
    raw = draw(arrays(np.float64, k, elements=st.floats(0.0, 1.0, **_f)))
    assume(raw.sum() > 1e-6)
    return raw / raw.sum()


@st.composite
def t_mixtures(draw, max_k=5, max_n=3, span=1e3):
    """Student-t mixtures with ν log-uniform in [0.5, 1e6] (heavy tails through the
    Gaussian limit), locations within ±span and scales in [e^−6, e^6]."""
    k = draw(st.integers(2, max_k))
    n = draw(st.integers(1, max_n))
    w = draw(weight_vectors(k))
    log_df = draw(arrays(np.float64, (k, n), elements=st.floats(math.log(0.5), math.log(1e6))))
    df = np.exp(log_df)
    loc = draw(arrays(np.float64, (k, n), elements=st.floats(-span, span, **_f)))
    log_s = draw(arrays(np.float64, (k, n), elements=st.floats(-6.0, 6.0, **_f)))
    preds = [StudentTPredictive(df[i], loc[i], np.exp(log_s[i])) for i in range(k)]
    return w, preds


@st.composite
def pmf_mixtures(draw, max_k=5, max_n=3, max_s=8):
    k = draw(st.integers(2, max_k))
    n = draw(st.integers(1, max_n))
    s = draw(st.integers(1, max_s))
    w = draw(weight_vectors(k))
    raw = draw(arrays(np.float64, (k, n, s), elements=st.floats(0.0, 1.0, **_f)))
    raw[..., 0] += 1e-3  # every row has some mass; other entries may be exactly 0
    probs = raw / raw.sum(axis=2, keepdims=True)
    return w, [DiscretePredictive(p) for p in probs]


def _check_bounds(w, res):
    k = len(w)
    h_pi = entropy(w)
    assert h_pi <= math.log(k) + 1e-12
    assert res.upper_bound == h_pi or abs(res.upper_bound - h_pi) < 1e-15
    assert np.all(res.value >= 0.0) and np.all(res.value <= h_pi)
    assert np.all(np.isfinite(res.raw)) and np.all(np.isfinite(res.error))


@PROPS
@given(t_mixtures())
def test_student_t_eig_bounds(case):
    w, preds = case
    res = model_indicator_eig(preds, w)
    _check_bounds(w, res)
    assert np.all(res.error < 1e-6)


@PROPS
@given(pmf_mixtures())
def test_discrete_eig_bounds(case):
    w, preds = case
    _check_bounds(w, model_indicator_eig(preds, w))


@PROPS
@given(t_mixtures(), st.integers(2, 6), st.data())
def test_identical_student_t_predictives_give_exactly_zero(case, k, data):
    pred = case[1][0]
    w = data.draw(weight_vectors(k))
    res = model_indicator_eig([pred] * k, w)
    assert np.all(res.value == 0.0) and np.all(res.error == 0.0)


@PROPS
@given(pmf_mixtures(), st.data())
def test_identical_discrete_predictives_give_exactly_zero(case, data):
    _, preds = case
    k = data.draw(st.integers(2, 6))
    w = data.draw(weight_vectors(k))
    assert np.all(model_indicator_eig([preds[0]] * k, w).value == 0.0)


@PROPS
@given(st.one_of(t_mixtures(), pmf_mixtures()), st.data())
def test_point_mass_weights_give_exactly_zero(case, data):
    _, preds = case
    j = data.draw(st.integers(0, len(preds) - 1))
    w = np.zeros(len(preds))
    w[j] = 1.0
    res = model_indicator_eig(preds, w)
    assert np.all(res.value == 0.0) and res.upper_bound == 0.0


@PROPS
@given(t_mixtures(), st.randoms(use_true_random=False))
def test_student_t_eig_permutation_invariance(case, rnd):
    w, preds = case
    perm = list(range(len(w)))
    rnd.shuffle(perm)
    a = model_indicator_eig(preds, w)
    b = model_indicator_eig([preds[i] for i in perm], w[perm])
    assert np.all(np.abs(a.raw - b.raw) <= a.error + b.error + 1e-12)


@PROPS
@given(pmf_mixtures(), st.randoms(use_true_random=False))
def test_discrete_eig_permutation_invariance(case, rnd):
    w, preds = case
    perm = list(range(len(w)))
    rnd.shuffle(perm)
    a = model_indicator_eig(preds, w)
    b = model_indicator_eig([preds[i] for i in perm], w[perm])
    np.testing.assert_allclose(a.raw, b.raw, atol=1e-12)


@PROPS_FEW
@given(t_mixtures(max_k=4, max_n=1, span=50.0))
def test_student_t_eig_matches_independent_quadrature(case):
    w, preds = case
    res = model_indicator_eig(preds, w)
    ref = _reference_u(
        w,
        [p.df[0] for p in preds],
        [p.loc[0] for p in preds],
        [p.scale[0] for p in preds],
    )
    assert abs(res.raw[0] - ref) <= res.error[0] + 1e-12, (res.raw[0], ref, res.error[0])


# --------------------------------------------------------------------------- engine fuzzing

_actions = st.fixed_dictionaries(
    {"burst": st.integers(0, 12), "gap": st.sampled_from([0.0, 0.5, 1.0, 3.0])}
)


@PROPS
@given(
    st.lists(st.tuples(_actions, st.one_of(st.none(), st.integers(0, 12))), max_size=12),
    st.sampled_from([0.0, 0.05]),
)
def test_grid_engine_fuzz_prequential_and_no_nan(steps, eps):
    h = grid_h("TB", "test_bucket", {"capacity": (1, 3, 5), "refill": (0.5, 2.0)}, {"eps": eps})
    m = GridModel(h, COUNT_OUT)
    total = 0.0
    for action, y in steps:
        if y is not None and y > action["burst"] and eps > 0:
            y = action["burst"]  # keep noisy worlds mostly possible; eps = 0 hits zeros
        if y is not None:
            total += m.log_predictive(action, y)
        m.update(action, y)
        w = m.grid_weights()
        assert not np.any(np.isnan(w)) and abs(w.sum() - 1.0) < 1e-12
        pred = m.predictive([{"burst": 4, "gap": 1.0}])
        assert np.all(np.isfinite(pred.probs))
        assert np.all(np.isfinite(m.parameter_eig([{"burst": 4, "gap": 1.0}])))
    assert not math.isnan(m.log_evidence())
    if math.isinf(total):
        assert m.log_evidence() == -math.inf and m.refuted
    else:
        assert total == math.fsum([total]) and abs(total - m.log_evidence()) < 1e-9
    assert m.max_log_likelihood() >= m.log_evidence() - 1e-9 or m.n_obs == 0


@PROPS
@given(
    st.lists(
        st.tuples(
            st.floats(-3.0, 3.0, **_f),
            st.floats(-3.0, 3.0, **_f),
            st.one_of(st.none(), st.floats(-20.0, 20.0, **_f)),
        ),
        max_size=15,
    )
)
def test_nig_engine_fuzz_matches_batch(rows):
    covs = numeric_covs(2)
    h = linear_h("H", 2, prior_means=[0.2, -0.1, 0.4], a0=1.2, b0=0.7)
    m = NIGModel(h, covs)
    total = 0.0
    xs, ys = [], []
    for x0, x1, y in rows:
        r = {"x0": x0, "x1": x1}
        if y is not None:
            total += m.log_predictive(r, y)
            xs.append([1.0, x0, x1])
            ys.append(y)
        m.update(r, y)
    assert abs(total - m.log_evidence()) <= 1e-9 * max(1.0, abs(total))
    mu0, v0 = prior_arrays(h)
    ref = batch_nig(np.array(xs).reshape(-1, 3), np.array(ys), mu0, v0, 1.2, 0.7)
    assert m.log_evidence() == ref["log_evidence"] or abs(
        m.log_evidence() - ref["log_evidence"]
    ) <= 1e-8 * max(1.0, abs(ref["log_evidence"]))
    np.testing.assert_allclose(m.posterior.mu, ref["mu"], rtol=1e-7, atol=1e-9)
    assert np.all(m.parameter_eig([{"x0": 1.0, "x1": -1.0}]) >= 0)
    # max log-likelihood dominates the prequential evidence (it is a supremum)
    assert m.max_log_likelihood() >= m.log_evidence() - 1e-9


@PROPS
@given(arrays(np.float64, 6, elements=st.floats(0.5, 1e9, **_f)))
def test_t_normaliser_continuous_across_series_switch(dfs):
    """The log normaliser and entropy are smooth in ν, including across the
    switch to asymptotic series at ν = 50."""
    from probe.engine.predictive import student_t_entropy, t_log_normalizer

    d = np.concatenate([dfs, [50.0 - 1e-9, 50.0]])
    ln = t_log_normalizer(d)
    assert abs(ln[-1] - ln[-2]) < 1e-12
    ent = student_t_entropy(d, np.ones_like(d))
    assert abs(ent[-1] - ent[-2]) < 1e-12
    # entropy decreases towards the Gaussian limit as ν grows
    order = np.argsort(d)
    assert np.all(np.diff(ent[order]) <= 1e-13)
    assert np.all(ent >= 0.5 * np.log(2 * np.pi * np.e) - 1e-15)
    small = d < 50
    if small.any():
        direct = (
            special.gammaln(0.5 * (d[small] + 1)) - special.gammaln(0.5 * d[small])
        ) - 0.5 * np.log(d[small] * np.pi)
        np.testing.assert_allclose(ln[small], direct, rtol=0, atol=1e-13)
