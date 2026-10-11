"""Physical sanity of the simulator, the sensor model and the regime designs."""

import functools

import numpy as np
import pytest

from probe.engine.ir import column_names, design_matrix
from probe.envs.microclimate import FAMILY
from probe.envs.microclimate.solar import (
    NOMINAL_DIRECT_TRANSMISSION,
    irradiance_proxy,
    lagged_forcing,
    solar_day,
)
from probe.truth import EvaluatorKey

KEY = EvaluatorKey("physics tests")
HS = FAMILY.hypothesis_set("R1")


def _sites(env, shade=None):
    sites = env.metadata().extra["campus"]["sites"]
    return [s["id"] for s in sites if shade is None or s["shade"] in shade]


def _mean_structural(env, sites, shield, minute):
    return float(np.mean([env.oracle_structural_at(KEY, s, shield, minute) for s in sites]))


@functools.cache
def _trajectory(regime, seed, n=60):
    """(candidate, observation, oracle expected value) along a seeded random walk;
    cached so several tests share the same trajectories."""
    env = FAMILY.make_world(regime, seed).env
    rng = np.random.default_rng(seed + 1000)
    out = []
    for _ in range(n):
        feas = [c for c in env.candidates() if c.feasible]
        if not feas:
            break
        c = feas[int(rng.integers(len(feas)))]
        mu = env.oracle_expected(KEY, c.action)
        out.append((c, env.execute(c.action), mu))
    return tuple(out)


# ------------------------------------------------------------------ solar geometry


def test_solar_geometry_and_irradiance_proxy():
    sd = solar_day()
    assert sd is solar_day()  # cached once per process
    assert not sd.ghi.flags.writeable
    noon = sd.minutes[int(np.argmax(sd.ghi))]
    assert 12 * 60 <= noon <= 13 * 60  # solar noon ~12:35 MST at 111.93 W in mid-July
    assert 900 < sd.ghi.max() < 1100
    assert sd.ghi[0] < 5.0  # 05:00, before sunrise
    t = 12.5 * 60
    tr = NOMINAL_DIRECT_TRANSMISSION
    irr = {
        k: float(irradiance_proxy(sd, t, tr[k], svf))
        for k, svf in (("open", 0.9), ("tree", 0.3), ("built", 0.4))
    }
    assert irr["open"] > 0.8 > irr["tree"] > irr["built"] >= 0.0
    assert float(irradiance_proxy(sd, 300.0, 1.0, 1.0)) < 0.01
    g = lagged_forcing(60.0)
    assert g.max() == pytest.approx(1.0)
    assert sd.minutes[int(np.argmax(g))] > noon  # surface response lags the sun


# ------------------------------------------------------------------ shade and shield


@pytest.mark.parametrize("seed", range(6))
def test_physics_world_shaded_sites_cooler_and_unshielded_warmer_at_midday(seed):
    env = FAMILY.make_world("R3", seed).env
    for minute in (12.5 * 60, 14.0 * 60):
        open_sh = _mean_structural(env, _sites(env, {"open"}), "shielded", minute)
        shaded_sh = _mean_structural(env, _sites(env, {"tree", "built"}), "shielded", minute)
        assert shaded_sh < open_sh - 0.3
        open_un = _mean_structural(env, _sites(env, {"open"}), "unshielded", minute)
        assert open_un > open_sh + 0.3
    for s in _sites(env):
        for minute in (570.0, 780.0, 1050.0):
            gap = env.oracle_structural_at(KEY, s, "unshielded", minute) - (
                env.oracle_structural_at(KEY, s, "shielded", minute)
            )
            assert gap >= -1e-9  # an unshielded sensor never reads cooler


@pytest.mark.parametrize("regime", ["R1", "R2", "R4", "R6", "R7", "R8", "R9", "R10"])
def test_in_set_worlds_have_the_physical_sign_of_their_mechanism(regime):
    for seed in range(8):
        world = FAMILY.make_world(regime, seed)
        env, tid = world.env, world.truth.true_hypothesis_id(KEY)
        minute = 13.0 * 60
        open_sh = _mean_structural(env, _sites(env, {"open"}), "shielded", minute)
        shaded_sh = _mean_structural(env, _sites(env, {"tree", "built"}), "shielded", minute)
        open_un = _mean_structural(env, _sites(env, {"open"}), "unshielded", minute)
        if tid == "H4":
            assert shaded_sh == pytest.approx(open_sh, abs=1e-9)  # no true shade effect
            assert open_un > open_sh + 0.3  # artefact: unshielded reads warm in sun
        else:
            assert shaded_sh < open_sh - 0.1  # shade cools
            assert open_un == pytest.approx(open_sh, abs=1e-9)


def test_observed_readings_show_shade_and_shield_contrasts_in_physics_worlds():
    rows = [
        (c.record, o.value)
        for seed in range(30)
        for c, o, _ in _trajectory("R3", seed)
        if o.value is not None and 11.0 <= c.record["hour"] <= 16.0
    ]

    def mean(pred):
        vals = [v for r, v in rows if pred(r)]
        assert len(vals) >= 15
        return float(np.mean(vals))

    open_sh = mean(lambda r: r["shade"] == "open" and not r["unshielded"])
    open_un = mean(lambda r: r["shade"] == "open" and r["unshielded"])
    shaded_sh = mean(lambda r: r["shade"] != "open" and not r["unshielded"])
    assert open_un > open_sh + 0.3
    assert shaded_sh < open_sh - 0.3


# ------------------------------------------------------------------ structure


@pytest.mark.parametrize("regime", ["R1", "R2", "R4", "R5", "R6", "R10"])
def test_clean_in_set_worlds_are_exactly_well_specified(regime):
    for seed in range(4):
        world = FAMILY.make_world(regime, seed)
        payload = world.truth.open(KEY)["payload"]
        h = HS.by_id(payload["structure"]["hypothesis_id"])
        assert payload["structure"]["columns"] == column_names(h)
        beta = np.asarray(payload["structure"]["beta"])
        env = world.env
        cands = [c for c in env.candidates() if c.feasible][::7]
        x = design_matrix(h, [c.record for c in cands], HS.covariates)
        expected = np.asarray([env.oracle_expected(KEY, c.action) for c in cands])
        np.testing.assert_allclose(expected, x @ beta, atol=1e-9)


def test_r10_positive_control_mimicry():
    for seed in range(8):
        world = FAMILY.make_world("R10", seed)
        env = world.env
        payload = world.truth.open(KEY)["payload"]
        tid = world.truth.true_hypothesis_id(KEY)
        eff = payload["effects"]
        unsh = [c for c in env.candidates() if c.record["unshielded"] and c.feasible]
        h4_pattern = np.asarray(
            [eff["c0"] + eff["artefact_slope"] * c.record["irr_proxy"] for c in unsh]
        )
        world_unsh = np.asarray([env.oracle_expected(KEY, c.action) for c in unsh])
        # unshielded readings look (almost) the same under the true world and the H4 pattern
        assert np.sqrt(np.mean((world_unsh - h4_pattern) ** 2)) < 0.15
        masts = env.metadata().extra["mast_sites"]
        gaps = [
            env.oracle_expected_at(KEY, m, "unshielded", 780.0)
            - env.oracle_expected_at(KEY, m, "shielded", 780.0)
            for m in masts
        ]
        if tid == "H4":
            assert min(gaps) > 1.0  # one shielded mast reading separates H4 from H1
        else:
            assert payload["structure"]["mimicry"] is True
            assert max(abs(g) for g in gaps) < 1e-9


def test_r5_landscape_is_flat():
    for seed in range(4):
        env = FAMILY.make_world("R5", seed).env
        vals = [
            env.oracle_structural_at(KEY, s, sh, m)
            for s in _sites(env)
            for sh in ("shielded", "unshielded")
            for m in (600.0, 780.0, 960.0)
        ]
        assert max(vals) - min(vals) < 0.25


def test_r7_nuisance_is_nonstationary():
    for seed in range(4):
        world = FAMILY.make_world("R7", seed)
        env = world.env
        sites = env.metadata().extra["campus"]["sites"]
        far = max(
            sites, key=lambda s: abs(s["x_m"] - env.metadata().extra["campus"]["base"]["x_m"])
        )
        minutes = np.arange(560.0, 1070.0, 30.0)
        nuisance = [
            env.oracle_expected_at(KEY, far["id"], "shielded", m)
            - env.oracle_structural_at(KEY, far["id"], "shielded", m)
            for m in minutes
        ]
        assert np.std(nuisance) > 0.05


def test_r9_sensor_offsets_are_a_site_level_nuisance():
    for seed in range(4):
        env = FAMILY.make_world("R9", seed).env
        nuisance = [
            env.oracle_expected_at(KEY, s, "shielded", 780.0)
            - env.oracle_structural_at(KEY, s, "shielded", 780.0)
            for s in _sites(env)
        ]
        assert np.std(nuisance) > 0.15
    units = {o.meta["unit"] for _, o, _ in _trajectory("R9", 0, n=40)}
    assert len(units) > 3  # many units, fixed by site


# ------------------------------------------------------------------ sensor model


def test_clean_noise_is_centred_with_the_declared_scale_and_mar_rate():
    res, n, missing, cutouts = [], 0, 0, 0
    for seed in range(30):
        for _, o, mu in _trajectory("R1", seed):
            n += 1
            if o.value is None:
                missing += 1
                cutouts += "overheat_cutout" in o.qc_flags
            else:
                res.append(o.value - mu)
    res = np.asarray(res)
    assert abs(res.mean()) < 0.05
    assert 0.22 < res.std() < 0.34
    assert np.mean(np.abs(res) > 1.5) == 0.0
    assert 0.005 < missing / n < 0.07
    assert cutouts == 0


def test_heavy_tailed_regime_has_heavy_tails():
    res = np.asarray(
        [
            o.value - mu
            for seed in range(30)
            for _, o, mu in _trajectory("R8", seed)
            if o.value is not None
        ]
    )
    assert np.mean(np.abs(res) > 1.5) > 0.01


def test_overheating_cutouts_are_missing_not_at_random():
    cut, kept = [], []
    for seed in range(40):
        for c, o, _ in _trajectory("R3", seed):
            (cut if "overheat_cutout" in o.qc_flags else kept).append(c.record)
            if o.value is None:
                assert o.missingness == "missing" and o.qc_flags
    assert len(cut) >= 3
    # cut-outs concentrate in hot, sunny, unshielded readings
    assert np.mean([r["hour"] for r in cut]) > np.mean([r["hour"] for r in kept]) + 1.0
    assert np.mean([r["unshielded"] for r in cut]) > np.mean([r["unshielded"] for r in kept])


# ------------------------------------------------------------------ white-box sensor model


def _residuals(hidden, shield, dwell, n_starts):
    """Reading minus its noise-free expectation for many measurement identities
    (all sites in turn, start times every 0.5 min)."""
    from probe.envs.microclimate.records import design_records

    sites = hidden._campus.sites
    starts = 560.0 + 0.5 * np.arange(n_starts)
    items = [(sites[i % len(sites)], shield, t + dwell / 2.0) for i, t in enumerate(starts)]
    recs = design_records(items, hidden._solar)
    res, missing = [], 0
    for i, (t, rec) in enumerate(zip(starts, recs, strict=True)):
        k = i % len(sites)
        v, m, flags = hidden.measure(k, shield, dwell, float(t), rec)
        if v is None:
            assert m == "missing" and flags
            missing += 1
        else:
            res.append(v - hidden.expected(k, shield, float(t) + dwell / 2.0, rec))
    return np.asarray(res), missing / n_starts


def test_sensor_noise_variance_matches_the_declared_model_per_dwell(hidden_world):
    """Var(y - E y) = s_turb^2 + s_inst^2 * 5/D + s_ref^2 + 2 * q^2/12 (two quantised
    raw readings). The 5 vs 15 min contrast checks the instrument-averaging term."""
    cfg = FAMILY.regime_config("R1")
    hidden = hidden_world("R1", 5)
    q2 = 2.0 * 0.05**2 / 12.0
    var = {}
    for dwell in (5.0, 15.0):
        res, miss = _residuals(hidden, "shielded", dwell, 2400)
        expected = cfg.sd_turb**2 + cfg.sd_inst**2 * 5.0 / dwell + cfg.sd_ref**2 + q2
        assert abs(res.mean()) < 0.02
        # n ~ 2300: the sd of the sample variance is ~3 %; a missing 1/sqrt(D) averaging
        # term would put the 15-min variance 13 % too high.
        assert res.var() == pytest.approx(expected, rel=0.08), (dwell, res.var(), expected)
        assert 0.01 < miss < 0.06  # MAR rate 3 %
        var[dwell] = res.var()
    assert var[5.0] > var[15.0]


def test_heavy_tailed_regime_has_heavy_tails_beyond_the_spikes(hidden_world):
    """Spikes are positive only, so the negative tail isolates the Student-t
    turbulence of R8; the Gaussian regime has (almost) no mass there."""
    heavy, _ = _residuals(hidden_world("R8", 2), "shielded", 5.0, 1200)
    light, _ = _residuals(hidden_world("R1", 2), "shielded", 5.0, 1200)
    # Gaussian sd ~0.29 puts ~0.1 % below -0.9; 0.15 * t(2.5) about 1 %.
    assert np.sum(heavy < -0.9) >= 5
    assert np.sum(light < -0.9) <= 2
    assert heavy.mean() > light.mean() + 0.03  # positive spikes shift the mean


def test_cutout_probability_is_stable_at_extremes():
    from probe.envs.microclimate.scm import CUTOUT_MAX_P, cutout_probability

    for t_raw in (-1e6, -500.0, -400.0, 0.0, 40.0, 46.0, 52.0, 500.0, 1e6):
        p = cutout_probability(t_raw, 46.0)
        assert 0.0 <= p <= CUTOUT_MAX_P and np.isfinite(p)
    assert cutout_probability(46.0, 46.0) == pytest.approx(0.45)
    assert cutout_probability(-500.0, 46.0) == 0.0
    assert cutout_probability(1e6, 46.0) == pytest.approx(CUTOUT_MAX_P)
    grid = np.linspace(30.0, 60.0, 61)
    probs = [cutout_probability(float(t), 46.0) for t in grid]
    assert np.all(np.diff(probs) >= 0.0)


# ------------------------------------------------------------------ registered mechanisms


def _true_coefficients(world):
    s = world.truth.open(KEY)["payload"]["structure"]
    return dict(zip(s["columns"], s["beta"], strict=True))


def test_true_coefficients_have_the_registered_qualitative_signs():
    """Each in-set truth satisfies its hypothesis' qualitative prediction: shade
    cools (H1-H3), cooler surfaces than asphalt (H2), unshielded sensors read warm
    in proportion to irradiance (H4)."""
    seen = set()
    for regime in ("R1", "R2", "R4", "R7", "R8", "R9"):
        for seed in range(8):
            world = FAMILY.make_world(regime, seed)
            tid = world.truth.true_hypothesis_id(KEY)
            beta = _true_coefficients(world)
            seen.add(tid)
            if tid in ("H1", "H2", "H3"):
                assert beta["shade=tree"] < 0 and beta["shade=built"] < 0
            if tid == "H2":
                for lev in ("concrete", "grass", "cool_pavement"):
                    assert beta[f"surface={lev}"] < 0, (regime, seed, lev)
            if tid == "H4":
                intercept, artefact = column_names(HS.by_id("H4"))
                assert set(beta) == {intercept, artefact}
                assert beta[artefact] > 0.3  # degC per kW/m2, unshielded only
    assert seen == {"H1", "H2", "H3", "H4"}


def test_h3_truths_cool_most_in_the_afternoon():
    """H3's qualitative prediction: dT(tree) - dT(open) is more negative at 14-16 h
    than at 9-10 h (the shade effect follows lagged solar forcing)."""
    n = 0
    for seed in range(16):
        world = FAMILY.make_world("R1", seed)
        if world.truth.true_hypothesis_id(KEY) != "H3":
            continue
        env = world.env

        def contrast(minute, env=env):
            tree = _mean_structural(env, _sites(env, {"tree"}), "shielded", minute)
            return tree - _mean_structural(env, _sites(env, {"open"}), "shielded", minute)

        morning, afternoon = contrast(9.5 * 60), contrast(14.5 * 60)
        assert afternoon < morning - 0.1, (seed, morning, afternoon)
        n += 1
    assert n >= 3


def test_r10_mast_readings_are_the_discriminating_designs():
    """Positive-control validity (Part I R10): after a shade-stratified unshielded
    warm-up, the shielded mast readings separate H1 from H4 far better than any
    unshielded reading, also per minute of effort. Separation is
    (mu_H1 - mu_H4)^2 / (var_H1 + var_H4) of the exact NIG posterior predictives."""
    from probe.engine.nig import NIGModel

    hs = FAMILY.hypothesis_set("R10")
    ratios = []
    for seed in range(16):
        env = FAMILY.make_world("R10", seed).env
        models = {h.id: NIGModel(h, hs.covariates) for h in hs.hypotheses}
        done: set[str] = set()
        for _ in range(3):
            pool = [
                c
                for c in env.candidates()
                if c.feasible and c.record["unshielded"] and c.record["shade"] not in done
            ]
            c = min(pool, key=lambda c: (c.cost, c.action.id))
            done.add(str(c.record["shade"]))
            o = env.execute(c.action)
            for m in models.values():
                m.update(o.record, o.value)
        feas = [c for c in env.candidates() if c.feasible]
        recs = [c.record for c in feas]
        p1, p4 = models["H1"].predictive(recs), models["H4"].predictive(recs)
        sep = (p1.mean() - p4.mean()) ** 2 / (p1.variance() + p4.variance())
        shielded = np.asarray([not r["unshielded"] for r in recs])
        cost = np.asarray([c.cost for c in feas])
        assert shielded.any()
        assert sep[shielded].max() > 3.0 * sep[~shielded].max(), seed
        assert (sep / cost)[shielded].max() > 1.5 * (sep / cost)[~shielded].max(), seed
        ratios.append(sep[shielded].max() / sep[~shielded].max())
    assert np.median(ratios) > 20.0
