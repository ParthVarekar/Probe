"""Designs: audit and stratified probabilities, the fixed-grid schedule, the record
metric, lexicographic argmax, the static maximin planner and the oracle's
truth-targeted information (against QUADPACK and exact sums)."""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
import pytest
from scipy import integrate, special, stats
from select_testkit import prepare, synthetic_scored

from probe.determinism import make_rng
from probe.engine.eig import model_indicator_eig
from probe.engine.factory import build_models
from probe.engine.ir import CovariateSpec
from probe.engine.nig import NIGModel
from probe.engine.predictive import DiscretePredictive, StudentTPredictive
from probe.envs.microclimate import FAMILY as MICROCLIMATE
from probe.envs.systemlab import FAMILY as SYSTEMLAB
from probe.select.base import stable_key
from probe.select.designs import (
    RecordMetric,
    _GaussianJointKL,
    audit_draw,
    audit_probabilities,
    fixed_grid_schedule,
    least_sampled_draw,
    least_sampled_probabilities,
    lexicographic_argmax,
    maximin_plan,
    pairwise_js,
)
from probe.select.oracle import LOG_RATIO_CAP, truth_log_odds_gain

STRATA = ["s0", "s0", "s0", "s1", "s2", "s2"]


# --------------------------------------------------------------------------- audit / P3


def test_audit_probabilities_are_stratified_uniform() -> None:
    sc = synthetic_scored([0.0] * 6, strata=STRATA)
    q = audit_probabilities(sc)
    np.testing.assert_allclose(q, [1 / 9, 1 / 9, 1 / 9, 1 / 3, 1 / 6, 1 / 6], rtol=0, atol=1e-15)
    assert math.fsum(q) == pytest.approx(1.0, abs=1e-15)


def test_audit_draw_frequencies_match_probabilities() -> None:
    sc = synthetic_scored([0.0] * 6, strata=STRATA)
    q = audit_probabilities(sc)
    n = 30_000
    counts = Counter(audit_draw(sc, make_rng(9, "audit", i)).index for i in range(n))
    freq = np.array([counts[i] / n for i in range(6)])
    sd = np.sqrt(q * (1 - q) / n)
    assert np.all(np.abs(freq - q) < 4.5 * sd)
    d = audit_draw(sc, make_rng(1, "x"))
    np.testing.assert_array_equal(d.probs, q)
    assert d.stratum == STRATA[d.index] and d.n_strata == 3


def test_least_sampled_design() -> None:
    sc = synthetic_scored([0.0] * 6, strata=STRATA)
    counts = {"s0": 2, "s1": 1, "s2": 1}
    q = least_sampled_probabilities(sc, counts)
    np.testing.assert_allclose(q, [0, 0, 0, 1 / 2, 1 / 4, 1 / 4], rtol=0, atol=1e-15)
    for i in range(50):
        d, least = least_sampled_draw(sc, counts, make_rng(3, "ls", i))
        assert least == 1 and d.stratum in {"s1", "s2"} and q[d.index] > 0
    # unseen strata count as 0 and come first
    q0 = least_sampled_probabilities(sc, {"s0": 1, "s1": 1})
    np.testing.assert_allclose(q0, [0, 0, 0, 0, 1 / 2, 1 / 2], rtol=0, atol=1e-15)


# --------------------------------------------------------------------------- schedules


def test_fixed_grid_schedule_round_robin() -> None:
    env = MICROCLIMATE.make_world("R1", 0).env
    cands = env.candidates()
    sched = fixed_grid_schedule(cands)
    assert sorted(sched) == sorted(c.action.id for c in cands if c.feasible)
    assert sched == fixed_grid_schedule(list(reversed(cands)))  # order-independent
    stratum = {c.action.id: c.stratum for c in cands}
    n_strata = len(set(stratum.values()))
    first = [stratum[a] for a in sched[:n_strata]]
    assert len(set(first)) == n_strata  # every stratum before any repeats
    assert first == sorted(first)
    # within a stratum: stable-key order
    for s in set(stratum.values()):
        members = [a for a in sched if stratum[a] == s]
        assert members == sorted(members, key=stable_key)


def test_record_metric() -> None:
    covs = (
        CovariateSpec(name="x", kind="numeric"),
        CovariateSpec(name="c", kind="categorical", levels=("a", "b")),
        CovariateSpec(name="u", kind="binary"),
    )
    recs = [{"x": 0.0, "c": "a", "u": True}, {"x": 4.0, "c": "b", "u": False}]
    m = RecordMetric.fit(covs, recs)
    assert m.scales == {"x": 4.0}
    assert m.distance(recs[0], recs[0]) == 0.0
    assert m.distance(recs[0], recs[1]) == pytest.approx(3.0)
    assert m.distance(recs[0], {"x": 1.0, "c": "a", "u": True}) == pytest.approx(0.25)
    const = RecordMetric.fit(covs, [recs[0], recs[0]])
    assert const.scales == {"x": 1.0}


def test_lexicographic_argmax() -> None:
    ids = ["a", "b", "c", "d"]
    i, ties = lexicographic_argmax(
        np.array([1.0, 2.0, 2.0, np.nan]), np.array([9.0, 1.0, 3.0, 99.0]), ids, 1e-9
    )
    assert (ids[i], ties) == ("c", ("c",))
    i, ties = lexicographic_argmax(np.zeros(3), np.ones(3), ids[:3], 1e-9)
    assert ties == ("a", "b", "c") and ids[i] == min(ids[:3], key=stable_key)


# --------------------------------------------------------------------------- maximin


def _gauss_kl(ma, ca, mb, cb):
    k = len(ma)
    inv = np.linalg.inv(cb)
    d = mb - ma
    return 0.5 * (
        np.trace(inv @ ca) + d @ inv @ d - k + np.linalg.slogdet(cb)[1] - np.linalg.slogdet(ca)[1]
    )


def test_gaussian_joint_kl_matches_direct_formula() -> None:
    hs = MICROCLIMATE.hypothesis_set("R1")
    env = MICROCLIMATE.make_world("R1", 1).env
    records = [c.record for c in env.candidates()][:20]
    models = [m for m in build_models(hs) if isinstance(m, NIGModel)]
    obj = _GaussianJointKL(models, records)
    plan = [3, 11, 3]
    vals = obj.values(plan, np.arange(len(records)))
    for p, (i, j) in enumerate(obj.pairs):
        for d in (0, 7, 19):
            design = [*plan, d]
            stats_ = []
            for m in (models[i], models[j]):
                post = m.posterior
                x = m.features([records[t] for t in design])
                s2 = post.b / (post.a - 1.0)
                stats_.append((x @ post.mu, s2 * (np.eye(len(design)) + x @ post.cov @ x.T)))
            ref = _gauss_kl(*stats_[0], *stats_[1])
            assert vals[p, d] == pytest.approx(max(ref, 0.0), rel=1e-8, abs=1e-12)
    # a design point repeated adds less than it did the first time (shared parameters)
    one = obj.values([], np.array([5]))[:, 0]
    two = obj.values([5], np.array([5]))[:, 0]
    assert np.all(two <= 2.0 * one + 1e-12) and np.all(two >= one - 1e-12)


def test_pairwise_js_matches_per_pair_kernel() -> None:
    env = SYSTEMLAB.make_world("R1", 0).env
    records = [c.record for c in env.candidates()]
    models = build_models(SYSTEMLAB.hypothesis_set("R1"))
    preds = [m.predictive(records) for m in models]
    pairs = list(itertools.combinations(range(len(models)), 2))
    js = pairwise_js(preds, pairs)
    for p, (i, j) in enumerate(pairs):
        ref = model_indicator_eig([preds[i], preds[j]], [0.5, 0.5]).value
        np.testing.assert_array_equal(js[p], ref)
    assert np.all(js >= 0.0) and np.all(js <= math.log(2.0) + 1e-15)


@pytest.mark.parametrize(
    ("family", "regime", "budget"),
    [
        ("microclimate", "R1", "medium"),
        ("microclimate", "R6", "large"),
        ("systemlab", "R1", "small"),
    ],
)
def test_maximin_plan_deterministic_diverse_and_within_budget(
    family: str, regime: str, budget: str
) -> None:
    fam = {"microclimate": MICROCLIMATE, "systemlab": SYSTEMLAB}[family]
    env = fam.make_world(regime, 2).env
    hs = fam.hypothesis_set(regime)
    b = fam.regimes()[regime].budgets[budget]
    cands = env.candidates()
    plan = maximin_plan(hs, cands, budget=b, max_length=12, rel_tol=1e-9)
    again = maximin_plan(hs, list(reversed(cands)), budget=b, max_length=12, rel_tol=1e-9)
    assert plan == again
    assert 1 <= len(plan) <= 12
    assert math.fsum(plan.costs) <= b + 1e-9
    cost = {c.action.id: c.cost for c in cands}
    assert list(plan.costs) == [cost[a] for a in plan.action_ids]
    assert all(a in cost for a in plan.action_ids)
    # the weakest pair's divergence never decreases as the plan grows
    assert all(y >= x - 1e-12 for x, y in itertools.pairwise(plan.min_divergence))
    expect = "gaussian_joint_kl" if family == "microclimate" else "pointwise_js"
    assert plan.divergence == expect
    if family == "systemlab":
        assert len(set(plan.action_ids)) == len(plan)  # no repeats in the pointwise mode
    if len(plan) >= 6:
        stratum = {c.action.id: c.stratum for c in cands}
        assert len(set(plan.action_ids)) >= 4  # diverse: not one point repeated
        assert len({stratum[a] for a in plan.action_ids}) >= 2


def test_maximin_plan_separates_pairs_no_single_point_can() -> None:
    """With zero prior means, H2 and H3 have identical prior predictives at a
    tree/grass site, so no single observation separates them there; the joint
    divergence of the plan still grows past every pointwise value of that pair."""
    env = MICROCLIMATE.make_world("R1", 3).env
    hs = MICROCLIMATE.hypothesis_set("R1")
    plan = maximin_plan(hs, env.candidates(), budget=240.0, max_length=8, rel_tol=1e-9)
    assert plan.min_divergence[-1] > 0.0
    assert plan.min_divergence[-1] > plan.min_divergence[0]


# --------------------------------------------------------------------------- oracle


def _t(df, loc, scale):
    return StudentTPredictive(
        np.asarray([df], dtype=float), np.asarray([loc], dtype=float), np.asarray([scale], float)
    )


def _quad_gain(df, loc, scale, w):
    wr = np.asarray(w[1:], dtype=float) / sum(w[1:])

    def lp(y: float, i: int) -> float:
        if math.isinf(df[i]):
            return float(stats.norm.logpdf(y, loc[i], scale[i]))
        return float(stats.t.logpdf(y, df[i], loc[i], scale[i]))

    def g(y: float) -> float:
        lt = lp(y, 0)
        lf = special.logsumexp([math.log(wr[j]) + lp(y, j + 1) for j in range(len(wr))])
        return math.exp(lt) * min(lt - lf, LOG_RATIO_CAP)

    edges = sorted(
        {-math.inf, math.inf}
        | {lo + k * s for lo, s in zip(loc, scale, strict=True) for k in (-20, -5, 0, 5, 20)}
    )
    kw = {"epsabs": 1e-14, "epsrel": 1e-12, "limit": 500}
    return sum(integrate.quad(g, a, b, **kw)[0] for a, b in itertools.pairwise(edges))


ORACLE_CASES = [
    ([4, 4, 4], [0, 0.5, -1], [1, 1, 2], [0.3, 0.5, 0.2]),
    ([10, 6, 30], [0, 0, 0], [1, 0.3, 3], [0.2, 0.4, 0.4]),
    ([math.inf, math.inf], [0, 3], [1, 1], [0.5, 0.5]),
    ([4.5, 4.5, 4.5], [1, 1, 40], [1, 1.5, 0.5], [0.1, 0.8, 0.1]),  # distant narrow rival
    ([2.5, 7, 100], [0, 0.2, -0.3], [1, 0.1, 1.2], [0.5, 0.25, 0.25]),  # heavy tail, narrow
]


@pytest.mark.parametrize(("df", "loc", "scale", "w"), ORACLE_CASES)
def test_oracle_gain_matches_quadpack(df, loc, scale, w) -> None:
    preds = [_t(*a) for a in zip(df, loc, scale, strict=True)]
    gain = truth_log_odds_gain(preds, np.asarray(w, dtype=float), 0)[0]
    ref = _quad_gain(df, loc, scale, w)
    assert gain == pytest.approx(ref, rel=1e-7, abs=1e-10)
    if len(df) == 2 and all(math.isinf(d) for d in df):
        assert gain == pytest.approx(0.5 * (loc[1] - loc[0]) ** 2, rel=1e-12)  # Gaussian KL


def test_oracle_gain_discrete_exact_and_capped() -> None:
    p_t = DiscretePredictive(np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]))
    r1 = DiscretePredictive(np.array([[0.25, 0.25, 0.5], [0.2, 0.3, 0.5]]))
    r2 = DiscretePredictive(np.array([[1.0, 0.0, 0.0], [0.6, 0.2, 0.2]]))
    w = np.array([0.4, 0.3, 0.3])
    gain = truth_log_odds_gain([p_t, r1, r2], w, 0)
    f0 = 0.5 * r1.probs[0] + 0.5 * r2.probs[0]
    f1 = 0.5 * r1.probs[1] + 0.5 * r2.probs[1]
    ref0 = sum(p * math.log(p / f) for p, f in zip(p_t.probs[0], f0, strict=True) if p > 0)
    ref1 = sum(p * math.log(p / f) for p, f in zip(p_t.probs[1], f1, strict=True) if p > 0)
    np.testing.assert_allclose(gain, [ref0, ref1], rtol=1e-13)
    # an outcome no rival can produce: the log-ratio is capped, not infinite
    only = truth_log_odds_gain([p_t, r2], np.array([0.5, 0.5]), 0)
    assert only[0] == pytest.approx(0.5 * math.log(0.5) + 0.5 * LOG_RATIO_CAP, rel=1e-13)
    # no rival with positive weight: nothing to gain
    np.testing.assert_array_equal(truth_log_odds_gain([p_t, r1], np.array([1.0, 0.0]), 0), 0.0)


def test_oracle_gain_on_a_world_is_non_negative_and_targets_the_truth() -> None:
    st = prepare("microclimate", "R10", 1)
    sc = st.score(need_u_param=False)
    w = st.weights()
    for t in range(len(st.models)):
        g = truth_log_odds_gain(sc.predictives, w, t)
        assert g.shape == (sc.n,) and np.all(np.isfinite(g)) and np.all(g >= 0.0)
    same = truth_log_odds_gain([sc.predictives[0], sc.predictives[0]], np.array([0.5, 0.5]), 0)
    assert np.all(same < 1e-12)
