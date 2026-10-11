"""score_candidates: agreement with direct engine calls (bitwise), filtering, bounds,
the lazy parameter-EIG rule, read-only tables and the per-step timing record."""

from __future__ import annotations

import math
import statistics
import time

import numpy as np
import pytest
from select_testkit import make_protocol, prepare

from probe.engine.eig import (
    entropy,
    mixture_moments,
    model_indicator_eig,
    weighted_parameter_eig,
)
from probe.engine.nig import NIGModel
from probe.engine.predictive import DiscretePredictive, StudentTPredictive
from probe.envs.base import Action, Candidate
from probe.select.registry import make_policy
from probe.select.scoring import nig_parameter_eig, requires_u_param, score_candidates


def _same_predictive(a: object, b: object) -> None:
    if isinstance(a, StudentTPredictive):
        assert isinstance(b, StudentTPredictive)
        for name in ("df", "loc", "scale"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    else:
        assert isinstance(a, DiscretePredictive) and isinstance(b, DiscretePredictive)
        np.testing.assert_array_equal(a.probs, b.probs)


@pytest.mark.parametrize(
    ("family", "regime", "seed"),
    [("microclimate", "R1", 0), ("microclimate", "R10", 2), ("systemlab", "R1", 1)],
)
def test_scoring_matches_direct_engine_calls(family: str, regime: str, seed: int) -> None:
    st = prepare(family, regime, seed)
    w = st.weights()
    scored = st.score(need_u_param=True)
    cands = sorted(
        (c for c in st.env.candidates() if c.feasible and c.cost <= st.budget_remaining()),
        key=lambda c: c.action.id,
    )
    assert scored.action_ids() == [c.action.id for c in cands]
    records = [c.record for c in cands]
    preds = [m.predictive(records) for m in st.models]
    assert len(scored.predictives) == len(st.models)
    for got, ref in zip(scored.predictives, preds, strict=True):
        _same_predictive(got, ref)
    res = model_indicator_eig(preds, w)
    np.testing.assert_array_equal(scored.u_model, res.value)
    np.testing.assert_array_equal(scored.u_model_err, res.error)
    assert scored.u_model_upper == res.upper_bound
    _, var = mixture_moments(preds, w)
    np.testing.assert_array_equal(scored.pred_var, var)
    np.testing.assert_array_equal(scored.cost, [c.cost for c in cands])
    # parameter EIG: bitwise the engine's (for NIG computed from the predictive in hand)
    assert scored.u_param_available
    np.testing.assert_array_equal(scored.u_param, weighted_parameter_eig(st.models, w, records))


def test_nig_parameter_eig_from_predictive_is_the_engines() -> None:
    st = prepare("microclimate", "R1", 4)
    records = [c.record for c in st.env.candidates() if c.feasible]
    for m in st.models:
        assert isinstance(m, NIGModel)
        pred = m.predictive(records)
        np.testing.assert_array_equal(nig_parameter_eig(m, pred), m.parameter_eig(records))


@pytest.mark.parametrize(
    ("family", "regime", "seed"), [("microclimate", "R1", 3), ("systemlab", "R1", 3)]
)
def test_u_model_bounds_and_secondary_scores(family: str, regime: str, seed: int) -> None:
    st = prepare(family, regime, seed)
    w = st.weights()
    scored = st.score(need_u_param=True)
    k = len(st.models)
    assert scored.n > 0
    assert scored.u_model_upper == pytest.approx(entropy(w), abs=1e-12)
    assert scored.u_model_upper <= math.log(k) + 1e-12
    assert np.all(np.isfinite(scored.u_model))
    assert np.all(scored.u_model >= 0.0) and np.all(scored.u_model <= scored.u_model_upper)
    assert np.all(scored.u_model_err >= 0.0) and np.all(np.isfinite(scored.u_model_err))
    assert np.all(scored.pred_var > 0.0)
    assert np.all(np.isfinite(scored.u_param)) and np.all(scored.u_param >= 0.0)
    assert np.all(scored.cost > 0.0)


def test_grid_parameter_eig_is_lazy() -> None:
    """Grid engines would re-run the mechanisms for U_θ|m, so it is computed only on
    request; NIG engines always get it (free)."""
    st = prepare("systemlab", "R1", 2)
    lazy = st.score(need_u_param=False)
    assert not lazy.u_param_available and np.all(np.isnan(lazy.u_param))
    full = st.score(need_u_param=True)
    np.testing.assert_array_equal(lazy.u_model, full.u_model)
    np.testing.assert_array_equal(lazy.pred_var, full.pred_var)
    # default: only when the protocol's combined objective needs it
    assert not st.score(need_u_param=None).u_param_available
    st.protocol = make_protocol("systemlab", "R1", lambda_param=0.5)
    auto = st.score(need_u_param=None)
    assert auto.u_param_available
    np.testing.assert_array_equal(auto.u_param, full.u_param)
    # NIG: free, so always available
    mc = prepare("microclimate", "R1", 2)
    assert mc.score(need_u_param=False).u_param_available


def test_objective_combines_u_model_and_u_param() -> None:
    st = prepare("systemlab", "R1", 5)
    full = st.score(need_u_param=True)
    np.testing.assert_array_equal(full.objective(0.0), full.u_model)
    np.testing.assert_array_equal(full.objective(0.3), full.u_model + 0.3 * full.u_param)
    lazy = st.score(need_u_param=False)
    np.testing.assert_array_equal(lazy.objective(0.0), lazy.u_model)  # λ = 0 never reads it
    with pytest.raises(ValueError, match="need_u_param"):
        lazy.objective(0.3)
    with pytest.raises(ValueError):
        full.objective(-1.0)


def test_requires_u_param_per_policy() -> None:
    p0 = make_protocol("microclimate", "R1")
    p1 = make_protocol("microclimate", "R1", lambda_param=0.2)
    needs = {pid: requires_u_param(make_policy(pid), p0) for pid in ("P1", "P4", "P5", "P7", "P8")}
    assert needs == {"P1": False, "P4": False, "P5": True, "P7": False, "P8": False}
    assert requires_u_param(make_policy("P8"), p1)
    assert requires_u_param(object(), p0)  # unknown policies: compute (safe default)


def test_feasibility_budget_filter_order_and_read_only() -> None:
    st = prepare("microclimate", "R1", 6)
    cands = st.env.candidates()
    # make a few infeasible and shuffle the order
    mixed = [
        Candidate(c.action, c.record, c.cost, c.stratum, feasible=(i % 3 != 0))
        for i, c in enumerate(cands)
    ][::-1]
    scored = score_candidates(st.models, st.weights(), mixed, st.protocol, need_u_param=False)
    expect = sorted(c.action.id for i, c in enumerate(cands) if i % 3 != 0)
    assert scored.action_ids() == expect
    # budget filter keeps exactly the affordable ones (a cost equal to the budget fits)
    limit = float(np.median([c.cost for c in cands]))
    by_budget = score_candidates(
        st.models, st.weights(), cands, st.protocol, need_u_param=False, budget_remaining=limit
    )
    assert by_budget.action_ids() == sorted(c.action.id for c in cands if c.cost <= limit)
    assert 0 < by_budget.n < len(cands)
    for arr in (scored.cost, scored.u_model, scored.u_model_err, scored.u_param, scored.pred_var):
        with pytest.raises(ValueError):
            arr[0] = 1.0


def test_candidate_score_does_not_depend_on_the_batch() -> None:
    st = prepare("microclimate", "R1", 7)
    full = st.score(need_u_param=False)
    keep = list(full.candidates[::3])
    part = score_candidates(st.models, st.weights(), keep, st.protocol, need_u_param=False)
    idx = [full.action_ids().index(a) for a in part.action_ids()]
    np.testing.assert_array_equal(part.u_model, full.u_model[idx])
    np.testing.assert_array_equal(part.u_param, full.u_param[idx])
    np.testing.assert_array_equal(part.pred_var, full.pred_var[idx])


def test_empty_and_invalid_inputs() -> None:
    st = prepare("systemlab", "R1", 0)
    w = st.weights()
    empty = score_candidates(st.models, w, st.env.candidates(), st.protocol, budget_remaining=0.5)
    assert empty.n == 0
    for arr in (empty.cost, empty.u_model, empty.u_model_err, empty.u_param, empty.pred_var):
        assert arr.shape == (0,)
    assert all(p.n == 0 for p in empty.predictives)
    cands = st.env.candidates()
    with pytest.raises(ValueError, match="weights"):
        score_candidates(st.models, w[:-1], cands, st.protocol)
    with pytest.raises(ValueError, match="duplicate"):
        score_candidates(st.models, w, [cands[0], cands[0]], st.protocol)
    bad = Candidate(Action("x", {}), cands[0].record, 0.0, cands[0].stratum)
    with pytest.raises(ValueError, match="cost"):
        score_candidates(st.models, w, [bad], st.protocol)
    with pytest.raises(ValueError):
        score_candidates([], [], cands, st.protocol)


def test_score_candidates_timing_microclimate() -> None:
    """Per-step cost of the shared score table on a microclimate world (96 candidates
    × 5 NIG engines): measured ~10 ms on the development laptop (Windows 11), almost
    all of it the U_M quadrature. Printed with ``pytest -s``; loosely bounded."""
    st = prepare("microclimate", "R1", 11)
    w = st.weights()
    cands = st.env.candidates()
    score_candidates(st.models, w, cands, st.protocol)  # warm caches
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        scored = score_candidates(st.models, w, cands, st.protocol)
        times.append(time.perf_counter() - t0)
    ms = [1e3 * t for t in times]
    print(
        f"\nscore_candidates microclimate R1: {scored.n} candidates x {len(st.models)} "
        f"hypotheses: median {statistics.median(ms):.1f} ms, min {min(ms):.1f} ms"
    )
    assert statistics.median(times) < 0.5
