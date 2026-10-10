"""Interface contract of the microclimate family: regimes, schema, candidates,
feasibility, records and hypothesis design matrices."""

import time

import numpy as np
import pytest

from probe.determinism import canonical_json
from probe.engine.ir import LinearGaussianHypothesis, column_names, design_matrix
from probe.engine.nig import NIGModel
from probe.envs.base import Action, InfeasibleActionError, check_records
from probe.envs.microclimate import (
    FAMILY,
    REGIME_CONFIGS,
    REGIME_IDS,
    SCENARIOS,
    build_hypothesis_set,
    scenario_manifest,
    scenario_manifest_hash,
)

ALL_REGIMES = [f"R{i}" for i in range(1, 11)]


def _run(env, n, seed=0):
    """Take up to ``n`` uniformly random feasible actions; return the observations."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        feas = [c for c in env.candidates() if c.feasible]
        if not feas:
            break
        out.append(env.execute(feas[int(rng.integers(len(feas)))].action))
    return out


def test_family_identity_and_regimes():
    assert FAMILY.name == "microclimate"
    assert isinstance(FAMILY.version, str) and FAMILY.version
    specs = FAMILY.regimes()
    assert list(specs) == ALL_REGIMES == list(REGIME_IDS)
    for rid, spec in specs.items():
        assert spec.id == rid
        assert spec.title and spec.description and spec.expected
        assert list(spec.budgets) == ["small", "medium", "large"]
        b = [spec.budgets[k] for k in ("small", "medium", "large")]
        assert 0 < b[0] < b[1] < b[2]
        assert isinstance(spec.warmup, int) and spec.warmup >= 0
        assert spec.in_set == (rid != "R3")
    canonical_json(dict(specs))  # JSON-able (goes into protocol hashes)


def test_unknown_regime_and_bad_seed_rejected():
    with pytest.raises(KeyError):
        FAMILY.make_world("R99", 0)
    with pytest.raises(KeyError):
        FAMILY.hypothesis_set("nope")
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError):
            FAMILY.make_world("R1", bad)


def test_scenario_manifest_marks_every_assumption_unverified():
    assert SCENARIOS
    for a in SCENARIOS:
        assert a.status == "[U]"
        assert a.basis.strip()
        if isinstance(a.value, list) and len(a.value) == 2:
            assert a.value[0] < a.value[1]
    man = scenario_manifest()
    assert set(man["regimes"]) == set(ALL_REGIMES)
    assert len(scenario_manifest_hash()) == 64
    world = FAMILY.make_world("R2", 0)
    assert world.env.metadata().extra["scenario_manifest_hash"] == scenario_manifest_hash()


def test_hypothesis_set_shape():
    hs = FAMILY.hypothesis_set("R1")
    assert [h.id for h in hs.hypotheses] == ["H1", "H2", "H3", "H4", "H0"]
    assert hs.residual.id == "H0"
    assert hs.outcome.kind == "continuous" and hs.outcome.unit == "degC"
    for rid in ALL_REGIMES:
        if rid != "R10":
            assert FAMILY.hypothesis_set(rid) == hs
    # The positive control registers the focused subset {H1, H4, H0} (same objects).
    h10 = FAMILY.hypothesis_set("R10")
    assert [h.id for h in h10.hypotheses] == ["H1", "H4", "H0"]
    assert all(h is hs.by_id(h.id) for h in h10.hypotheses)
    assert h10.covariates == hs.covariates and h10.residual.id == "H0"
    with pytest.raises(ValueError):
        build_hypothesis_set(("H1", "H9", "H0"))
    for h in hs.hypotheses:
        assert isinstance(h, LinearGaussianHypothesis)
        assert column_names(h)[0] == "1"  # intercept first (used by cloud modulation)
        # σ² prior mean b0/(a0-1) on the simulator's noise scale (σ ≈ 0.3 °C)
        assert 0.03 < h.prior.b0 / (h.prior.a0 - 1.0) < 0.3
    assert build_hypothesis_set() is build_hypothesis_set()


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_records_validate_and_every_design_matrix_builds(regime):
    hs = FAMILY.hypothesis_set(regime)
    for seed in (0, 1):
        env = FAMILY.make_world(regime, seed).env
        md = env.metadata()
        assert md.covariates == hs.covariates and md.outcome == hs.outcome
        cands = env.candidates()
        recs = [c.record for c in cands]
        check_records(recs, md.covariates)
        names = {c.name for c in md.covariates}
        assert all(set(r) == names for r in recs)  # only agent-knowable covariates
        obs = _run(env, 6, seed)
        check_records([o.record for o in obs], md.covariates)
        for h in hs.hypotheses:
            x = design_matrix(h, recs, hs.covariates)
            assert x.shape == (len(recs), len(column_names(h)))
            assert np.all(np.isfinite(x))
            model = NIGModel(h, hs.covariates)
            for o in obs:
                model.update(o.record, o.value)
            pred = model.predictive(recs[:5])
            assert np.all(np.isfinite(pred.loc)) and np.all(pred.scale > 0)


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_candidates_sorted_costed_and_stratified(regime):
    env = FAMILY.make_world(regime, 3).env
    md = env.metadata()
    cands = env.candidates()
    ids = [c.action.id for c in cands]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert 50 <= len(cands) <= 200
    expected_n = 104 if regime == "R10" else 192
    assert len(cands) == expected_n
    pop = set(md.population)
    settle = md.extra["settle_min"]
    walk = np.asarray(md.extra["campus"]["walk_minutes"])
    index = md.extra["campus"]["walk_index"]
    base = index.index("BASE")
    w0, w1 = md.extra["campaign_window_min"]
    for c in cands:
        p = c.action.params
        assert c.action.id == (
            f"site={p['site']}|shield={p['shield']}|dwell={p['dwell']:02d}|wait={p['wait']:02d}"
        )
        travel = walk[base, index.index(p["site"])]
        assert c.cost == pytest.approx(travel + settle[p["shield"]] + p["dwell"])
        assert c.cost > 0
        assert c.stratum in pop
        assert c.record["unshielded"] == (p["shield"] == "unshielded")
        hour = c.record["hour"]
        t_end = 60.0 * hour + p["dwell"] / 2.0
        assert c.feasible == (t_end <= w1 + 1e-3)
        assert hour * 60.0 >= w0
    assert any(c.feasible for c in cands)


def test_population_strata_cover_window_blocks():
    md = FAMILY.make_world("R1", 0).env.metadata()
    hours = {k.split("|")[2] for k in md.population}
    assert hours == {"hour=09", "hour=11", "hour=13", "hour=15"}
    md10 = FAMILY.make_world("R10", 0).env.metadata()
    assert {k.split("|")[2] for k in md10.population} == {"hour=11", "hour=13"}


def test_r10_shielded_readings_only_at_two_remote_open_masts():
    for seed in range(4):
        env = FAMILY.make_world("R10", seed).env
        md = env.metadata()
        masts = md.extra["mast_sites"]
        assert len(masts) == 2
        sites = {s["id"]: s for s in md.extra["campus"]["sites"]}
        assert all(sites[m]["shade"] == "open" for m in masts)
        shielded = {c.action.params["site"] for c in env.candidates() if not c.record["unshielded"]}
        assert shielded == set(masts)
        walk = md.extra["campus"]["walk_minutes"]
        idx = md.extra["campus"]["walk_index"]
        base = idx.index("BASE")
        open_walks = sorted(
            (walk[base][idx.index(s)] for s in sites if sites[s]["shade"] == "open"), reverse=True
        )
        assert min(walk[base][idx.index(m)] for m in masts) >= open_walks[1]


def test_r6_budget_allows_at_most_five_readings():
    spec = FAMILY.regimes()["R6"]
    for seed in range(5):
        env = FAMILY.make_world("R6", seed).env
        min_cost = min(c.cost for c in env.candidates())
        assert min_cost >= 10.0
        assert spec.budgets["large"] / min_cost < 6


def test_execute_rejects_unknown_tampered_and_infeasible_actions():
    env = FAMILY.make_world("R1", 0).env
    with pytest.raises(InfeasibleActionError):
        env.execute(Action(id="site=S99|shield=shielded|dwell=05|wait=00", params={}))
    good = next(c for c in env.candidates() if c.feasible)
    tampered = Action(id=good.action.id, params={**good.action.params, "dwell": 15})
    with pytest.raises(InfeasibleActionError):
        env.execute(tampered)
    with pytest.raises(TypeError):
        env.execute(good.action.id)  # type: ignore[arg-type]
    # Walk the clock to the end of the window, then everything is infeasible.
    _run(env, 500)
    assert env.exhausted
    assert not any(c.feasible for c in env.candidates())
    last = env.candidates()[0]
    with pytest.raises(InfeasibleActionError):
        env.execute(last.action)


def test_mutating_a_candidate_record_cannot_change_the_world():
    a = FAMILY.make_world("R1", 5).env
    b = FAMILY.make_world("R1", 5).env
    ca = next(c for c in a.candidates() if c.feasible and c.record["shade"] == "tree")
    cb = next(c for c in b.candidates() if c.action.id == ca.action.id)
    cb.record["shade"] = "open"  # type: ignore[index]
    cb.record["irr_proxy"] = 9.0  # type: ignore[index]
    oa, ob = a.execute(ca.action), b.execute(cb.action)
    assert oa.value == ob.value and oa.record == ob.record
    assert ob.record["shade"] == "tree"


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_clock_moves_forward_within_window_and_budget_accounting(regime):
    env = FAMILY.make_world(regime, 11).env
    w0, w1 = env.metadata().extra["campaign_window_min"]
    assert env.clock == w0 and env.spent == 0.0 and env.position == "BASE"
    last_clock, spent = env.clock, 0.0
    rng = np.random.default_rng(1)
    steps = 0
    while not env.exhausted:
        feas = [c for c in env.candidates() if c.feasible]
        c = feas[int(rng.integers(len(feas)))]
        o = env.execute(c.action)
        assert o.step == steps
        steps += 1
        assert o.clock > last_clock and o.clock == env.clock <= w1
        assert o.cost == c.cost
        spent += c.cost
        assert env.spent == pytest.approx(spent)
        assert env.position == c.action.params["site"]
        assert w0 / 60.0 < o.record["hour"] < w1 / 60.0
        assert o.origin == "simulated"
        assert (o.value is None) == (o.missingness == "missing")
        if o.value is not None:
            assert abs(o.value * 20 - round(o.value * 20)) < 1e-6  # 0.05 °C quantum
        last_clock = o.clock
    assert steps >= 3


def test_make_world_and_episode_are_fast():
    FAMILY.make_world("R1", 0)  # warm the pvlib cache (first call in a process is slow)
    times = []
    for rid in ALL_REGIMES:
        for seed in (1, 2):
            t0 = time.perf_counter()
            FAMILY.make_world(rid, seed)
            times.append(time.perf_counter() - t0)
    assert max(times) < 0.2, f"make_world too slow: {max(times):.3f}s"
    env = FAMILY.make_world("R1", 7).env
    t0 = time.perf_counter()
    n = len(_run(env, 1000))
    dt = time.perf_counter() - t0
    assert n >= 5 and dt < 1.0, (n, dt)


def test_regime_configs_match_specs():
    specs = FAMILY.regimes()
    for rid, cfg in REGIME_CONFIGS.items():
        assert specs[rid].budgets["large"] == cfg.budgets[2]
        assert cfg.in_set == bool(cfg.truth_pool)
        assert cfg.expected_verdict in ("hypothesis", "inconclusive", "none_fits")
    assert REGIME_CONFIGS["R3"].expected_verdict == "none_fits"
    assert REGIME_CONFIGS["R5"].expected_verdict == "inconclusive"
