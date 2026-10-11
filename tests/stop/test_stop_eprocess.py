"""Universal-inference e-processes: bookkeeping, conventions, e-BH, and validity
(type-I error under an adaptive design) and power by simulation."""

import math

import numpy as np
import pytest
from stop_testkit import (
    GRID_CANDS,
    GRID_COVS,
    GRID_OUT,
    LIN_CANDS,
    grid_h,
    grid_set,
    grid_truth,
    linear_set,
    linear_truth,
    simulate,
    type_one_bound,
)

from probe.determinism import canonical_json, make_rng
from probe.engine.factory import build_models
from probe.engine.ir import HypothesisSet
from probe.stop.eprocess import EProcessTracker, e_to_p, ebh, json_float, log_e_value

ALPHAS = (0.05, 0.1, 0.2, 0.5)


def _crossing_rates(sups: np.ndarray) -> dict[float, float]:
    """Fraction of runs with sup_t log E_t ≥ log(1/α), per α."""
    return {a: float(np.mean(sups >= -math.log(a))) for a in ALPHAS}


def _assert_valid(sups: np.ndarray) -> None:
    rates = _crossing_rates(sups)
    for a, r in rates.items():
        assert r <= type_one_bound(a, sups.size), (a, r, rates)


# --------------------------------------------------------------------------- conventions


def test_log_e_value_conventions():
    assert log_e_value(-3.0, -5.0) == pytest.approx(2.0)
    assert log_e_value(-3.0, math.inf) == -math.inf  # rival interpolates: cannot reject
    assert log_e_value(-3.0, -math.inf) == math.inf  # rival refuted
    assert log_e_value(-math.inf, -math.inf) == -math.inf  # refuted numerator never wins
    assert log_e_value(-math.inf, 2.0) == -math.inf
    with pytest.raises(ValueError):
        log_e_value(math.nan, 0.0)
    with pytest.raises(ValueError):
        log_e_value(math.inf, 0.0)


def test_json_float():
    assert json_float(1.5) == 1.5
    assert json_float(math.inf) == "inf"
    assert json_float(-math.inf) == "-inf"
    assert json_float(math.nan) == "nan"


# --------------------------------------------------------------------------- bookkeeping


def test_tracker_matches_prequential_evidence_and_max_likelihood():
    hs = linear_set()
    res = simulate(
        hs, LIN_CANDS, linear_truth((0.3, 0.7, 0.0), 0.5), seed=3, n_steps=25,
        design="cycle", tag="bookkeeping",
    )
    tr, models = res.tracker, res.models
    # the numerator is the prequential sum, i.e. each model's log evidence
    np.testing.assert_allclose(
        tr.log_numerators(), [m.log_evidence() for m in models], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(tr.max_log_likelihoods(), [m.max_log_likelihood() for m in models])
    le = tr.log_e()
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            want = 0.0 if i == j else mi.log_evidence() - mj.max_log_likelihood()
            assert le[i, j] == pytest.approx(want, rel=1e-12, abs=1e-12)
    assert tr.log_e_value("Q", "L") == le[1, 0]
    assert tr.versus_rivals("L") == {"Q": le[0, 1], "R": le[0, 2]}
    assert np.all(tr.running_max_log_e() >= np.maximum(le, 0.0) - 1e-15)
    assert tr.n_obs == tr.n_steps == 25
    canonical_json(tr.summary())  # JSON-able (non-finite values encoded)


def test_tracker_early_steps_cannot_reject_interpolating_rival():
    """With n ≤ p the NIG maximum likelihood is +inf, so every e-value against that
    rival is 0 (log −inf): 'cannot reject'."""
    hs = linear_set()
    models = build_models(hs)
    tr = EProcessTracker.for_models(models)
    rec = LIN_CANDS[2]
    lp = [m.log_predictive(rec, 1.0) for m in models]
    for m in models:
        m.update(rec, 1.0)
    tr.observe_models(lp, models)
    assert np.all(np.isneginf(tr.log_e()[~np.eye(3, dtype=bool)]))
    assert np.all(tr.running_max_log_e() == 0.0)
    assert tr.summary()["log_e"]["L"]["Q"] == "-inf"


def test_missing_outcomes_add_nothing_to_numerators():
    tr = EProcessTracker(["a", "b"])
    tr.observe([-1.0, -2.0], [-0.5, -1.5])
    before = tr.log_numerators()
    tr.observe(None, [-0.5, -1.5])
    np.testing.assert_array_equal(tr.log_numerators(), before)
    assert tr.n_steps == 2 and tr.n_obs == 1


def test_running_max_is_monotone_and_starts_at_one():
    tr = EProcessTracker(["a", "b"])
    np.testing.assert_array_equal(tr.running_max_log_e(), np.zeros((2, 2)))
    tr.observe([0.0, -10.0], [0.0, 0.0])  # log E(a:b) = 0, log E(b:a) = -10
    tr.observe([5.0, -10.0], [0.0, 0.0])  # log E(a:b) = 5
    tr.observe([-9.0, -10.0], [0.0, 0.0])  # log E(a:b) = -4
    assert tr.log_e()[0, 1] == pytest.approx(-4.0)
    assert tr.running_max_log_e()[0, 1] == pytest.approx(5.0)
    assert tr.running_max_log_e()[1, 0] == 0.0


def test_refuted_grid_rival_gives_infinite_evidence():
    """A count impossible under every grid point of the rival: its max likelihood is
    −inf and the e-value against it is +inf."""
    hs = HypothesisSet(
        outcome=GRID_OUT,
        covariates=GRID_COVS,
        hypotheses=(
            grid_h("never", {"a": (-800.0,)}, {"b": 0.0}),  # expit(-800) = 0: P(Y = n) = 0
            grid_h("free", {"a": (-1.0, 0.0, 1.0)}, {"b": 0.0}, residual=True),
        ),
    )
    models = build_models(hs)
    tr = EProcessTracker.for_models(models)
    rec = {"x": 0.0, "n": 8.0}
    lp = [m.log_predictive(rec, 8) for m in models]
    for m in models:
        m.update(rec, 8)
    tr.observe_models(lp, models)
    assert tr.log_e()[1, 0] == math.inf
    assert tr.log_e()[0, 1] == -math.inf


def test_tracker_input_validation_leaves_state_untouched():
    tr = EProcessTracker(["a", "b"])
    tr.observe([-1.0, -1.0], [0.0, 0.0])
    snap = tr.summary()
    with pytest.raises(ValueError):
        tr.observe([-1.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        tr.observe([-1.0, math.nan], [0.0, 0.0])
    with pytest.raises(ValueError):
        tr.observe([-1.0, math.inf], [0.0, 0.0])
    with pytest.raises(ValueError):
        tr.observe([-1.0, -1.0], [0.0, math.nan])
    assert tr.summary() == snap
    with pytest.raises(ValueError):
        EProcessTracker(["a", "a"])
    with pytest.raises(ValueError):
        EProcessTracker([])


def test_for_models_requires_fresh_models_and_matching_ids():
    hs = linear_set()
    models = build_models(hs)
    models[0].update(LIN_CANDS[0], 1.0)
    with pytest.raises(ValueError, match="registration"):
        EProcessTracker.for_models(models)
    fresh = build_models(hs)
    tr = EProcessTracker.for_models(fresh)
    with pytest.raises(ValueError, match="do not match"):
        tr.observe_models([0.0, 0.0, 0.0], list(reversed(fresh)))


# --------------------------------------------------------------------------- e-BH


def test_ebh_known_example_and_ties():
    # K = 4, α = 0.1: thresholds K/(αk) = 40, 20, 13.3, 10 → k* = 2
    np.testing.assert_array_equal(ebh([100, 30, 1, 0.5], 0.1), [True, True, False, False])
    np.testing.assert_array_equal(ebh([30, 100, 0.5, 1], 0.1), [True, True, False, False])
    np.testing.assert_array_equal(ebh([1, 2, 3], 0.05), [False, False, False])
    np.testing.assert_array_equal(ebh([math.inf, 0.0], 0.05), [True, False])
    np.testing.assert_array_equal(ebh([25.0, 25.0, 25.0], 0.2), [True, True, True])
    assert ebh([], 0.1).shape == (0,)
    with pytest.raises(ValueError):
        ebh([-1.0], 0.1)


def test_ebh_equals_bh_on_inverse_e_values():
    """e-BH is BH applied to p = 1/e (Wang & Ramdas 2022, Prop. 2)."""
    rng = make_rng(0, "ebh")
    for _ in range(200):
        k = int(rng.integers(1, 12))
        e = np.exp(rng.normal(1.5, 2.0, size=k))
        alpha = float(rng.uniform(0.01, 0.3))
        p = 1.0 / e
        order = np.argsort(p, kind="stable")
        ok = p[order] <= alpha * np.arange(1, k + 1) / k
        bh = np.zeros(k, dtype=bool)
        if ok.any():
            kk = int(np.max(np.flatnonzero(ok))) + 1
            bh = p <= alpha * kk / k
        np.testing.assert_array_equal(ebh(e, alpha), bh)


def test_e_to_p():
    np.testing.assert_allclose(e_to_p(np.log([0.5, 1.0, 20.0])), [1.0, 1.0, 0.05])
    assert e_to_p(math.inf) == 0.0
    assert e_to_p(-math.inf) == 1.0


# --------------------------------------------------------------------------- validity


def _tight_set() -> HypothesisSet:
    """A *simple* null (one grid point) against a mixture that contains it: the
    e-process is then an exact likelihood-ratio martingale, so P(sup E ≥ 1/α) is
    close to α and the validity check has teeth."""
    return HypothesisSet(
        outcome=GRID_OUT,
        covariates=GRID_COVS,
        hypotheses=(
            grid_h("point", {"a": (0.5,)}, {"b": 0.0}),
            grid_h("near", {"a": (-0.5, 0.0, 0.5, 1.0, 1.5), "b": (-0.5, 0.0, 0.5)}, residual=True),
        ),
    )


def _null_sups(hs, cands, draw, *, n_runs, n_steps, design, tag, pairs):
    out = []
    for s in range(n_runs):
        r = simulate(hs, cands, draw, seed=s, n_steps=n_steps, design=design, tag=tag)
        out.append([r.running_max[i, j] for i, j in pairs])
    return np.asarray(out)


def test_eprocess_validity_linear_fast():
    """Null: data from L (intercept + x0) with fixed parameters; adaptive design.
    E(Q : L) and E(R : L) must cross 1/α with probability ≤ α (+ 2 s.e.)."""
    sups = _null_sups(
        linear_set(), LIN_CANDS, linear_truth((0.5, 0.8, 0.0), 0.7),
        n_runs=150, n_steps=30, design="contrast", tag="lin-null-fast", pairs=[(1, 0), (2, 0)],
    )
    for col in range(sups.shape[1]):
        _assert_valid(sups[:, col])


def test_eprocess_validity_grid_fast():
    """Composite grid null 'flat' (b = 0, a on a grid; truth a = 0.5 on the grid)."""
    sups = _null_sups(
        grid_set(), GRID_CANDS, grid_truth(0.5, 0.0),
        n_runs=120, n_steps=25, design="contrast", tag="grid-null-fast", pairs=[(1, 0), (2, 0)],
    )
    for col in range(sups.shape[1]):
        _assert_valid(sups[:, col])


def test_eprocess_validity_simple_null_is_nearly_tight():
    sups = _null_sups(
        _tight_set(), GRID_CANDS, grid_truth(0.5, 0.0),
        n_runs=300, n_steps=30, design="cycle", tag="tight-fast", pairs=[(1, 0)],
    )[:, 0]
    _assert_valid(sups)
    # the simulation is sensitive: the crossing rate at α = 1/2 is far from 0
    assert _crossing_rates(sups)[0.5] > 0.1


def test_non_predictable_numerator_is_caught_by_the_simulation():
    """Negative control: evaluating the numerator *after* the update (not
    predictable) breaks validity, and the same simulation detects it."""
    hs = _tight_set()
    draw = grid_truth(0.5, 0.0)
    sups = []
    for s in range(150):
        rng = make_rng(s, "stop-test", "cheat")
        models = build_models(hs)
        tr = EProcessTracker.for_models(models)
        for t in range(30):
            rec = GRID_CANDS[t % len(GRID_CANDS)]
            y = draw(rec, rng)
            for m in models:
                m.update(rec, y)
            tr.observe_models([m.log_predictive(rec, y) for m in models], models)
        sups.append(tr.running_max_log_e()[1, 0])
    rates = _crossing_rates(np.asarray(sups))
    assert rates[0.05] > type_one_bound(0.05, 150)


@pytest.mark.slow
def test_eprocess_validity_linear_eig_design_1000_runs():
    """Acceptance criterion (handover §12 Part P): type-I error ≤ α + 2 s.e. over
    1,000 null runs, with the model-indicator-EIG adaptive design."""
    sups = _null_sups(
        linear_set(), LIN_CANDS, linear_truth((0.5, 0.8, 0.0), 0.7),
        n_runs=1000, n_steps=60, design="eig", tag="lin-null-slow", pairs=[(1, 0), (2, 0)],
    )
    for col in range(sups.shape[1]):
        _assert_valid(sups[:, col])


@pytest.mark.slow
def test_eprocess_validity_grid_eig_design_1000_runs():
    sups = _null_sups(
        grid_set(), GRID_CANDS, grid_truth(0.5, 0.0),
        n_runs=1000, n_steps=40, design="eig", tag="grid-null-slow", pairs=[(1, 0), (2, 0)],
    )
    for col in range(sups.shape[1]):
        _assert_valid(sups[:, col])


@pytest.mark.slow
def test_eprocess_validity_simple_null_1000_runs():
    sups = _null_sups(
        _tight_set(), GRID_CANDS, grid_truth(0.5, 0.0),
        n_runs=1000, n_steps=40, design="contrast", tag="tight-slow", pairs=[(1, 0)],
    )[:, 0]
    _assert_valid(sups)


# --------------------------------------------------------------------------- power


def test_eprocess_power_linear():
    """Truth Q (intercept + x0²): E(Q : L) grows past 1/α in (almost) every run,
    and the reverse e-process stays tiny."""
    finals = []
    for s in range(20):
        r = simulate(
            linear_set(), LIN_CANDS, linear_truth((0.2, 0.0, 0.9), 0.7), seed=s, n_steps=40,
            design="contrast", tag="lin-alt",
        )
        finals.append((r.running_max[1, 0], r.final[1, 0], r.final[0, 1]))
    f = np.asarray(finals)
    assert np.mean(f[:, 0] >= -math.log(0.05)) >= 0.9
    assert np.median(f[:, 1]) > 10.0
    assert np.all(f[:, 2] < 0.0)


def test_eprocess_power_grid_and_growth():
    r_short = simulate(
        grid_set(), GRID_CANDS, grid_truth(0.0, 1.0), seed=7, n_steps=10, design="eig",
        tag="grid-alt",
    )
    r_long = simulate(
        grid_set(), GRID_CANDS, grid_truth(0.0, 1.0), seed=7, n_steps=40, design="eig",
        tag="grid-alt",
    )
    assert r_long.final[1, 0] > r_short.final[1, 0]
    assert r_long.final[1, 0] > -math.log(0.05)
