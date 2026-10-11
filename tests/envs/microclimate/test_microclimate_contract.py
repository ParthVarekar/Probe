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
    expected_n = 52 if regime == "R10" else 96  # 24 sites x shields x dwell {5, 15}
    assert len(cands) == expected_n
    pop = set(md.population)
    settle = md.extra["settle_min"]
    walk = np.asarray(md.extra["campus"]["walk_minutes"])
    index = md.extra["campus"]["walk_index"]
    base = index.index("BASE")
    w0, w1 = md.extra["campaign_window_min"]
    for c in cands:
        p = c.action.params
        assert set(p) == {"site", "shield", "dwell"}  # (site, shield, dwell), handover H.1
        assert c.action.id == f"site={p['site']}|shield={p['shield']}|dwell={p['dwell']:02d}"
        travel = walk[base, index.index(p["site"])]
        assert c.cost == pytest.approx(travel + settle[p["shield"]] + p["dwell"])
        assert c.cost > 0
        assert c.stratum in pop
        assert c.record["unshielded"] == (p["shield"] == "unshielded")
        hour = c.record["hour"]
        t_end = 60.0 * hour + p["dwell"] / 2.0
        # no free waiting: the reading ends exactly `cost` minutes after the current clock
        assert t_end == pytest.approx(env.clock + c.cost, abs=1e-3)
        assert c.feasible == (t_end <= w1 + 1e-3)
        assert hour * 60.0 >= w0
    assert any(c.feasible for c in cands)


def test_population_strata_cover_window_blocks():
    md = FAMILY.make_world("R1", 0).env.metadata()
    hours = {k.split("|")[2] for k in md.population}
    assert hours == {"hour=09", "hour=11", "hour=13", "hour=15"}
    md10 = FAMILY.make_world("R10", 0).env.metadata()
    assert {k.split("|")[2] for k in md10.population} == {"hour=11", "hour=13"}


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_every_candidate_stratum_is_a_population_key_at_every_step(regime):
    """Late in R10 (window ends 15:00) readings that would end after the window used to
    get stratum hour=15, which is not a declared population cell; stratified/audit
    designs index counts by population key. Feasible candidates must also carry the
    block of their own reading time."""
    for seed in (0, 1):
        env = FAMILY.make_world(regime, seed).env
        pop = set(env.metadata().population)
        rng = np.random.default_rng(seed)
        while not env.exhausted:
            cands = env.candidates()
            for c in cands:
                assert c.stratum in pop, (env.clock, c.action.id, c.stratum)
                if c.feasible:
                    minute = 60.0 * c.record["hour"]
                    block = max(b for b in (9, 11, 13, 15) if minute >= 60.0 * b)
                    assert c.stratum.endswith(f"|hour={block:02d}")
                    assert c.stratum.startswith(
                        f"shade={c.record['shade']}|surface={c.record['surface']}|"
                    )
            feas = [c for c in cands if c.feasible]
            env.execute(feas[int(rng.integers(len(feas)))].action)


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


def test_r10_small_budget_affords_warmup_mast_and_follow_up():
    """The positive control must be able to separate policies in every budget cell:
    after a cheap shade-stratified warm-up, the small budget still pays for one
    shielded mast reading and two follow-up readings there."""
    spec = FAMILY.regimes()["R10"]
    for seed in range(24):
        env = FAMILY.make_world("R10", seed).env
        done: set[str] = set()
        for _ in range(spec.warmup):
            pool = [
                c
                for c in env.candidates()
                if c.feasible and c.record["unshielded"] and c.record["shade"] not in done
            ]
            c = min(pool, key=lambda c: (c.cost, c.action.id))
            done.add(str(c.record["shade"]))
            env.execute(c.action)
        mast = min(
            (c for c in env.candidates() if not c.record["unshielded"]), key=lambda c: c.cost
        )
        env.execute(mast.action)
        for _ in range(2):
            env.execute(min(env.candidates(), key=lambda c: (c.cost, c.action.id)).action)
        assert env.spent <= spec.budgets["small"], (seed, env.spent)


def _greedy_cheapest_count(env, budget):
    """Number of readings the cheapest-first policy fits in ``budget`` (staying put
    whenever that is cheapest, which maximises the count)."""
    spent, n = 0.0, 0
    while True:
        feas = [c for c in env.candidates() if c.feasible and spent + c.cost <= budget]
        if not feas:
            return n
        c = min(feas, key=lambda c: (c.cost, c.action.id))
        env.execute(c.action)
        spent += c.cost
        n += 1


def test_r6_budget_allows_at_most_three_four_five_readings():
    """Handover Part I: R6 is a tiny budget (<= 5 observations). Re-reading the site
    the team is standing at costs no travel, so the bound must hold for the cheapest
    *later* reading (settle + shortest dwell), not just the cheapest first move."""
    spec = FAMILY.regimes()["R6"]
    caps = {"small": 3, "medium": 4, "large": 5}
    for seed in range(8):
        env = FAMILY.make_world("R6", seed).env
        md = env.metadata()
        first = min(c.cost for c in env.candidates())
        later = min(md.extra["settle_min"].values()) + min(md.extra["dwell_options_min"])
        assert first >= 13.0 and later >= 10.0
        counts = []
        for level, cap in caps.items():
            budget = spec.budgets[level]
            assert 1 + (budget - first) // later <= cap  # analytic bound, any policy
            n = _greedy_cheapest_count(FAMILY.make_world("R6", seed).env, budget)
            assert 1 <= n <= cap
            counts.append(n)
        assert counts == sorted(counts) and counts[0] < counts[-1]  # levels are distinct


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_budget_binds_before_the_campaign_window(regime):
    """The clock advances by exactly the effort spent, and every budget fits in the
    window, so no policy can run out of window before running out of budget (with a
    free wait action, random baselines did, which made budget levels meaningless)."""
    spec = FAMILY.regimes()[regime]
    for seed in (0, 1):
        env = FAMILY.make_world(regime, seed).env
        w0, w1 = env.metadata().extra["campaign_window_min"]
        assert w0 + spec.budgets["large"] <= w1
        budget = spec.budgets["large"]
        rng = np.random.default_rng(seed)
        steps = 0
        while True:
            fits = [c for c in env.candidates() if env.spent + c.cost <= budget]
            assert all(c.feasible for c in fits)  # budget-affordable => feasible
            if not fits:
                break
            env.execute(fits[int(rng.integers(len(fits)))].action)
            steps += 1
            assert env.clock == pytest.approx(w0 + env.spent)
        assert steps >= 2
        assert env.clock <= w1


def test_execute_rejects_unknown_tampered_and_infeasible_actions():
    env = FAMILY.make_world("R1", 0).env
    with pytest.raises(InfeasibleActionError):
        env.execute(Action(id="site=S99|shield=shielded|dwell=05", params={}))
    good = next(c for c in env.candidates() if c.feasible)
    tampered = Action(id=good.action.id, params={**good.action.params, "dwell": 15})
    with pytest.raises(InfeasibleActionError):
        env.execute(tampered)
    with pytest.raises(TypeError):
        env.execute(good.action.id)  # type: ignore[arg-type]
    # there is no free "wait" action (it let random baselines burn the window)
    waiting = Action(id=good.action.id + "|wait=60", params={**good.action.params, "wait": 60})
    with pytest.raises(InfeasibleActionError):
        env.execute(waiting)
    assert env.step == 0 and env.spent == 0.0
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
    # the next caller (e.g. another policy component) must not see the mutation
    again = next(c for c in b.candidates() if c.action.id == ca.action.id)
    assert again.record == ca.record and again.record["shade"] == "tree"
    oa, ob = a.execute(ca.action), b.execute(cb.action)
    assert oa.value == ob.value and oa.record == ob.record
    assert ob.record["shade"] == "tree"


def test_candidate_params_and_metadata_are_fresh_copies():
    env = FAMILY.make_world("R4", 2).env
    before = canonical_json(env.candidates())
    md_before = canonical_json(env.metadata())
    c = env.candidates()[0]
    c.action.params["dwell"] = 99  # type: ignore[index]
    md = env.metadata()
    md.extra["settle_min"]["shielded"] = 0.0  # type: ignore[index]
    md.extra["campus"]["sites"][0]["shade"] = "open"  # type: ignore[index]
    assert canonical_json(env.candidates()) == before
    assert canonical_json(env.metadata()) == md_before
    # costs still use the real settling time
    shielded = [c for c in env.candidates() if c.action.params["shield"] == "shielded"]
    assert min(c.cost for c in shielded) >= 8.0 + 5.0


def test_observation_record_equals_the_executed_candidate_record():
    for regime in ("R1", "R3", "R10"):
        env = FAMILY.make_world(regime, 4).env
        rng = np.random.default_rng(4)
        for _ in range(8):
            feas = [c for c in env.candidates() if c.feasible]
            c = feas[int(rng.integers(len(feas)))]
            o = env.execute(c.action)
            assert o.record == c.record and o.action_id == c.action.id
            assert o.meta["t_end"] == o.clock == env.clock


def test_numpy_integer_seeds_give_the_same_world():
    a = FAMILY.make_world("R1", 7)
    b = FAMILY.make_world("R1", np.int64(7))  # type: ignore[arg-type]
    assert a.truth.commitment == b.truth.commitment
    assert canonical_json(a.env.candidates()) == canonical_json(b.env.candidates())
    with pytest.raises(ValueError):
        FAMILY.make_world("R1", np.int64(-1))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FAMILY.make_world("R1", np.bool_(True))  # type: ignore[arg-type]


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
    for rid, cfg in REGIME_CONFIGS.items():
        # the true hypothesis is always a registered member of the regime's set
        assert set(cfg.truth_pool) <= set(cfg.hypothesis_ids), rid
        assert "H0" in cfg.hypothesis_ids and "H0" not in cfg.truth_pool
        assert [specs[rid].budgets[k] for k in ("small", "medium", "large")] == list(cfg.budgets)


def test_cost_heterogeneity_matches_the_regime_design():
    """R1 (and R4, R10): spread campus, walks of 1 to ~20 min and longer settling for
    shielded readings. R2: compact campus and equal settling, so costs are near-uniform."""

    def walks(regime, seed):
        w = np.asarray(
            FAMILY.make_world(regime, seed).env.metadata().extra["campus"]["walk_minutes"]
        )
        return w[~np.eye(w.shape[0], dtype=bool)]

    for seed in range(6):
        assert walks("R1", seed).max() >= 2.0 * walks("R2", seed).max()
        assert walks("R1", seed).max() >= 10.0
        assert walks("R2", seed).max() <= 5.0 and walks("R2", seed).min() >= 1.0
    s1 = FAMILY.make_world("R1", 0).env.metadata().extra["settle_min"]
    s2 = FAMILY.make_world("R2", 0).env.metadata().extra["settle_min"]
    assert s1["shielded"] > s1["unshielded"] and s2["shielded"] == s2["unshielded"]
