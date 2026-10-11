"""Finite-grid engine: exact posterior, prequential evidence, stateful mechanisms
with missing outcomes, zero-probability outcomes, MLE and exact parameter EIG."""

import math

import numpy as np
import pytest
from engine_testkit import COUNT_OUT, grid_h
from scipy import special, stats

from probe.determinism import canonical_json, make_rng
from probe.engine.grid import GridModel, logsumexp
from probe.engine.mechanisms import register_mechanism, registered_mechanisms
from probe.engine.models import HypothesisModel

P_GRID = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
BINOM = grid_h("B", "test_binomial", {"p": P_GRID})
BUCKET = grid_h(
    "TB",
    "test_bucket",
    {"capacity": (2, 4, 6, 8), "refill": (0.5, 1.0, 2.0)},
    fixed={"eps": 0.1},
)
BUCKET_EXACT = grid_h(
    "TBX", "test_bucket", {"capacity": (2, 4, 6, 8), "refill": (0.5, 1.0, 2.0)}, fixed={"eps": 0.0}
)


def _binom_data(seed, n=15):
    rng = make_rng(seed, "binom")
    trials = rng.integers(1, 12, size=n)
    ys = rng.binomial(trials, 0.62)
    return [{"n": int(t)} for t in trials], [int(v) for v in ys]


def test_binomial_posterior_and_evidence_match_enumeration():
    recs, ys = _binom_data(0)
    m = GridModel(BINOM, COUNT_OUT)
    total = 0.0
    for r, y in zip(recs, ys, strict=True):
        total += m.log_predictive(r, y)
        m.update(r, y)
    p = np.asarray(P_GRID)
    ll = np.array(
        [sum(stats.binom.logpmf(y, r["n"], pi) for r, y in zip(recs, ys, strict=True)) for pi in p]
    )
    w = np.exp(ll - special.logsumexp(ll))
    np.testing.assert_allclose(m.grid_weights(), w, rtol=1e-10, atol=1e-15)
    expected_ev = float(special.logsumexp(ll) - math.log(len(p)))
    assert m.log_evidence() == pytest.approx(expected_ev, rel=1e-12)
    assert total == pytest.approx(m.log_evidence(), rel=1e-13)
    assert m.max_log_likelihood() == pytest.approx(float(ll.max()), rel=1e-12)
    np.testing.assert_allclose(m.log_likelihoods(), ll, rtol=1e-12)

    # predictive = posterior-weighted mixture of mechanism pmfs
    pred = m.predictive([{"n": 4}, {"n": 12}])
    for i, nt in enumerate((4, 12)):
        mix = w @ stats.binom.pmf(np.arange(13)[None, :], nt, p[:, None])
        np.testing.assert_allclose(pred.probs[i], mix, rtol=1e-10, atol=1e-15)


def test_parameter_eig_is_exact_mutual_information():
    recs, ys = _binom_data(1, n=4)
    m = GridModel(BINOM, COUNT_OUT)
    for r, y in zip(recs, ys, strict=True):
        m.update(r, y)
    w = m.grid_weights()
    cands = [{"n": 0}, {"n": 1}, {"n": 6}, {"n": 12}]
    eig = m.parameter_eig(cands)
    for i, r in enumerate(cands):
        pk = stats.binom.pmf(np.arange(13)[None, :], r["n"], np.asarray(P_GRID)[:, None])
        joint = w[:, None] * pk
        py = joint.sum(axis=0)
        h_y = -np.sum(special.xlogy(py, py))
        h_y_given = -np.sum(w * np.sum(special.xlogy(pk, pk), axis=1))
        assert eig[i] == pytest.approx(h_y - h_y_given, abs=1e-12)
    assert eig[0] == 0.0  # zero trials: nothing to learn
    assert np.all(np.diff(eig) > 0)  # more trials, more information
    assert np.all(eig <= -np.sum(special.xlogy(w, w)) + 1e-12)


def _simulate_bucket(cap, refill, actions):
    """Reference token bucket for one parameter point; returns accepted counts."""
    tokens = float(cap)
    acc = []
    for a in actions:
        avail = min(cap, tokens + refill * a["gap"])
        k = min(a["burst"], math.floor(avail))
        acc.append(k)
        tokens = avail - k
    return acc


def test_stateful_mechanism_advances_on_every_action_including_missing():
    actions = [
        {"burst": 5, "gap": 0.0},
        {"burst": 6, "gap": 1.0},
        {"burst": 3, "gap": 0.5},
        {"burst": 8, "gap": 2.0},
        {"burst": 4, "gap": 0.0},
    ]
    observed = [5, None, 2, None, 3]  # None: the action ran, its outcome is missing
    m = GridModel(BUCKET, COUNT_OUT)
    for a, y in zip(actions, observed, strict=True):
        m.update(a, y)
    assert m.n_obs == 3 and m.n_actions == 5

    # brute force: replay every grid point through all actions
    table = m.grid_table()
    eps = 0.1
    ll = np.zeros(m.grid_size)
    for g in range(m.grid_size):
        acc = _simulate_bucket(float(table["capacity"][g]), float(table["refill"][g]), actions)
        for a, y, k in zip(actions, observed, acc, strict=True):
            if y is None:
                continue
            pmf = np.full(a["burst"] + 1, eps / (a["burst"] + 1))
            pmf[k] += 1 - eps
            ll[g] += math.log(pmf[y])
    np.testing.assert_allclose(m.log_likelihoods(), ll, rtol=1e-12)
    np.testing.assert_allclose(m.grid_weights(), np.exp(ll - special.logsumexp(ll)), rtol=1e-10)

    # the next prediction uses the state after all five actions
    nxt = {"burst": 6, "gap": 1.0}
    pred = m.predictive([nxt]).probs[0]
    w = m.grid_weights()
    ref = np.zeros(13)
    for g in range(m.grid_size):
        k = _simulate_bucket(
            float(table["capacity"][g]), float(table["refill"][g]), [*actions, nxt]
        )[-1]
        pmf = np.zeros(13)
        pmf[:7] = eps / 7
        pmf[k] += 1 - eps
        ref += w[g] * pmf
    np.testing.assert_allclose(pred, ref, rtol=1e-10, atol=1e-15)

    # a model that skipped the missing actions would predict differently
    skip = GridModel(BUCKET, COUNT_OUT)
    for a, y in zip(actions, observed, strict=True):
        if y is not None:
            skip.update(a, y)
    assert not np.allclose(skip.predictive([nxt]).probs, pred)


def test_zero_probability_outcomes_are_handled_without_nan():
    m = GridModel(BUCKET_EXACT, COUNT_OUT)
    # first burst of 3 from a full bucket: capacity 2 accepts 2, others accept 3
    lp = m.log_predictive({"burst": 3, "gap": 0.0}, 3)
    assert math.isfinite(lp)
    m.update({"burst": 3, "gap": 0.0}, 3)
    w = m.grid_weights()
    cap = m.grid_table()["capacity"]
    assert np.all(w[cap == 2] == 0.0) and np.all(np.isfinite(m.log_likelihoods()[cap > 2]))
    assert np.all(np.isneginf(m.log_likelihoods()[cap == 2]))
    assert not m.refuted and math.isfinite(m.log_evidence())

    # an outcome impossible under every grid point: 9 accepted out of a burst of 1
    assert m.log_predictive({"burst": 1, "gap": 0.0}, 9) == -math.inf
    m.update({"burst": 1, "gap": 0.0}, 9)
    assert m.refuted
    assert m.log_evidence() == -math.inf
    assert m.max_log_likelihood() == -math.inf
    pred = m.predictive([{"burst": 4, "gap": 1.0}])
    assert np.all(np.isfinite(pred.probs)) and pred.probs.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(m.parameter_eig([{"burst": 4, "gap": 1.0}])))
    s = m.summary()
    assert s["refuted"] is True and s["log_evidence"] is None and s["max_log_likelihood"] is None
    canonical_json(s)
    # further updates keep the evidence at −inf and never produce NaN
    m.update({"burst": 2, "gap": 1.0}, 1)
    assert m.log_evidence() == -math.inf and not np.any(np.isnan(m.grid_weights()))


def test_point_masses_and_degenerate_grids():
    h = grid_h("B01", "test_binomial", {"p": (0.0, 1.0)})
    m = GridModel(h, COUNT_OUT)
    m.update({"n": 3}, 3)  # rules out p = 0
    np.testing.assert_array_equal(m.grid_weights(), [0.0, 1.0])
    assert m.log_evidence() == pytest.approx(math.log(0.5))
    assert m.parameter_eig([{"n": 3}])[0] == 0.0
    single = GridModel(grid_h("S", "test_binomial", {}, fixed={"p": 0.3}), COUNT_OUT)
    assert single.grid_size == 1
    assert single.parameter_eig([{"n": 5}])[0] == 0.0


def test_input_validation():
    m = GridModel(BINOM, COUNT_OUT)
    for bad in (-1, 13, 2.5, math.nan):
        with pytest.raises(ValueError):
            m.update({"n": 3}, bad)
    with pytest.raises(KeyError):
        GridModel(grid_h("X", "no_such_mechanism", {"p": (0.5,)}), COUNT_OUT)


def test_clone_independence_and_protocol():
    m = GridModel(BUCKET, COUNT_OUT)
    assert isinstance(m, HypothesisModel)
    m.update({"burst": 3, "gap": 0.0}, 3)
    c = m.clone()
    w0, s0 = m.grid_weights(), m.predictive([{"burst": 4, "gap": 0.0}]).probs
    c.update({"burst": 4, "gap": 0.0}, 0)
    c.update({"burst": 4, "gap": 0.0}, None)
    np.testing.assert_array_equal(m.grid_weights(), w0)
    np.testing.assert_array_equal(m.predictive([{"burst": 4, "gap": 0.0}]).probs, s0)
    assert m.n_actions == 1 and c.n_actions == 3
    canonical_json(m.summary())


def test_logsumexp_edge_cases():
    assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf
    assert float(logsumexp(np.array([0.0, 0.0]))) == pytest.approx(math.log(2))
    out = logsumexp(np.array([[-np.inf, -np.inf], [1000.0, 1000.0]]), axis=1)
    assert out[0] == -np.inf and out[1] == pytest.approx(1000 + math.log(2))


class _FragileMechanism:
    """Binomial outcomes; counts actions in its state; ``advance`` rejects a record
    with ``"jam": True`` (like a mechanism rejecting an out-of-order timestamp)."""

    name = "test_fragile"

    def __init__(self, support_size):
        self.support_size = support_size

    def initial_state(self, params):
        return np.zeros(len(params["p"]))

    def predict(self, state, params, record):
        k = np.arange(self.support_size)
        return stats.binom.pmf(k[None, :], int(record["n"]), np.asarray(params["p"])[:, None])

    def advance(self, state, params, record):
        if record.get("jam"):
            raise RuntimeError("mechanism cannot execute this action")
        return state + 1.0


if "test_fragile" not in registered_mechanisms():
    register_mechanism("test_fragile")(_FragileMechanism)


def test_failed_update_leaves_the_model_unchanged():
    """``update`` is atomic: when the mechanism (or the outcome) is rejected, neither
    the weights nor the evidence change. Previously the posterior and evidence were
    already updated when ``advance`` raised, so the model had conditioned on an
    action whose state transition never happened."""
    m = GridModel(grid_h("F", "test_fragile", {"p": (0.2, 0.5, 0.8)}), COUNT_OUT)
    m.update({"n": 4}, 3)
    snap = (m.grid_weights(), m.log_evidence(), m.log_likelihoods(), m.n_obs, m.n_actions)
    with pytest.raises(RuntimeError):
        m.update({"n": 4, "jam": True}, 4)
    with pytest.raises(RuntimeError):
        m.update({"n": 4, "jam": True}, None)
    with pytest.raises(ValueError):
        m.update({"n": 4}, 99)
    np.testing.assert_array_equal(m.grid_weights(), snap[0])
    assert m.log_evidence() == snap[1] and (m.n_obs, m.n_actions) == snap[3:]
    np.testing.assert_array_equal(m.log_likelihoods(), snap[2])
    np.testing.assert_array_equal(m._state, np.ones(3))


def test_summary_marginals_match_enumeration_for_mixed_type_parameters():
    """Marginals are sums of grid weights per parameter value. Matching values with
    ``col == v`` gave 0 for the numeric values of a mixed-type parameter such as
    (1, "lru", 2.5), since numpy stores that column as strings."""
    h = grid_h("M", "test_binomial", {"mode": (1, "lru", 2.5), "p": (0.2, 0.5, 0.9)})
    m = GridModel(h, COUNT_OUT)
    m.update({"n": 6}, 5)
    s = m.summary()
    w = m.grid_weights()
    table = m.grid_table()
    for name, values in h.params.items():
        marg = s["marginals"][name]
        assert sum(marg.values()) == pytest.approx(1.0, abs=1e-14)
        for v in values:
            mask = np.array([str(c) == str(v) for c in table[name]])
            assert marg[str(v)] == pytest.approx(float(w[mask].sum()), abs=1e-15)
    assert s["marginals"]["p"]["0.9"] > s["marginals"]["p"]["0.2"]
    canonical_json(s)
