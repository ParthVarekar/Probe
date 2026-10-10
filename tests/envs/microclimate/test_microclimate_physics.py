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
