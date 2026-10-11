"""Simulator-vs-mechanism agreement: the environment's reported counts, sampled with
counter-based noise over many seeds at a fixed planted service, follow exactly the
PMF that the true hypothesis's mechanism predicts at the true grid point (so in-set
regimes are well specified)."""

import numpy as np
import pytest
from scipy import stats

from probe.engine.ir import GridHypothesis
from probe.engine.mechanisms import get_mechanism
from probe.envs.base import Action
from probe.envs.systemlab import FAMILY
from probe.envs.systemlab.env import SystemLabEnv
from probe.envs.systemlab.noise import ReportNoise
from probe.envs.systemlab.regimes import regime_config
from probe.envs.systemlab.services import ServiceSpec
from probe.truth import EvaluatorKey

KEY = EvaluatorKey("agreement test")

RATE_PATH = [
    (0, 12),
    (0, 6),
    (2, 4),
    (1, 8),
    (8, 12),
    (0, 2),
    (4, 6),
    (16, 12),
    (0, 1),
    (32, 8),
]
CACHE_PATH = ["abcabc", "cdacda", "aaabca", "abacad", "cdcdcd", "abcdab", "cdacbd", "ababab"]


def _chi2_pvalue(samples: np.ndarray, p: np.ndarray) -> float:
    n = samples.size
    obs = np.bincount(samples, minlength=p.size).astype(float)
    assert np.all(obs[p == 0] == 0), "simulator produced an outcome the mechanism calls impossible"
    keep = p * n >= 5
    o, e = obs[keep], p[keep] * n
    rest_o, rest_e = n - o.sum(), n - e.sum()
    if rest_e > 1e-9:
        o, e = np.append(o, rest_o), np.append(e, rest_e)
    if o.size < 2:
        return 1.0
    return float(stats.chi2.sf(np.sum((o - e) ** 2 / e), o.size - 1))


def _check(regime: str, service: ServiceSpec, noise: ReportNoise, hyp, row_params, path, n_seeds):
    cfg = regime_config(regime)
    s = cfg.hypotheses.outcome.max_count + 1
    mech = get_mechanism(hyp.mechanism, s)
    params = {k: np.asarray([v]) for k, v in row_params.items()}
    state = mech.initial_state(params)
    envs = [SystemLabEnv(cfg, seed, service, noise, "0" * 64) for seed in range(n_seeds)]
    pvals = []
    for step in path:
        if isinstance(step, str):
            action = Action(id=f"keys={step}", params={"keys": step})
        else:
            wait, burst = step
            action = Action(id=f"wait={wait:03d}|burst={burst:02d}", params=step_params(step))
        obs = [env.execute(action) for env in envs]
        rec = obs[0].record
        assert all(o.record == rec for o in obs)
        probs = mech.predict(state, params, rec)[0]
        ys = np.asarray([o.value for o in obs], dtype=np.int64)
        pvals.append(_chi2_pvalue(ys, probs))
        state = mech.advance(state, params, rec)
    return pvals


def step_params(step):
    return {"wait": step[0], "burst": step[1]}


def _assert_uniformish(pvals):
    # Each p-value is a valid test; with ~10 steps use a Bonferroni-style floor.
    assert min(pvals) > 1e-4, pvals


@pytest.mark.parametrize(
    ("regime", "seed"),
    [
        ("R1", 0),
        ("R1", 1),
        ("R1", 2),
        ("R1", 4),
        ("R2", 2),
        ("R5", 1),
        ("R6", 0),
        ("R8", 0),
        ("R8", 3),
        ("R10", 2),
    ],
)
def test_planted_world_matches_true_grid_point(regime, seed):
    world = FAMILY.make_world(regime, seed)
    payload = world.truth.open(KEY)["payload"]
    tid = world.truth.true_hypothesis_id(KEY)
    hyp = FAMILY.hypothesis_set(regime).by_id(tid)
    assert isinstance(hyp, GridHypothesis)
    table = hyp.grid_table()
    gi = payload["true_grid_index"]
    row = {k: table[k][gi].item() for k in table}
    assert row["eps"] == payload["noise"]["eps"]
    assert row["glitch"] == payload["noise"]["glitch"]
    spec = ServiceSpec(payload["service"]["kind"], payload["service"]["params"])
    noise = ReportNoise(**payload["noise"])
    _assert_uniformish(_check(regime, spec, noise, hyp, row, RATE_PATH, 2500))


@pytest.mark.parametrize(
    ("policy", "size", "eps", "glitch"),
    [("lru", 3, 0.05, 0.0), ("fifo", 2, 0.1, 0.2), ("lfu", 3, 0.02, 0.1)],
)
def test_cache_world_matches_mechanism(policy, size, eps, glitch):
    hyp = FAMILY.hypothesis_set("R11").by_id(policy)
    row = {"policy": policy, "size": size, "eps": eps, "glitch": glitch}
    spec = ServiceSpec("cache", {"policy": policy, "size": size})
    _assert_uniformish(_check("R11", spec, ReportNoise(eps, glitch), hyp, row, CACHE_PATH, 2500))


def test_allowance_residual_is_exact_for_its_own_law():
    """The residual mechanism's PMF is a proper distribution whose reported counts match
    Monte Carlo draws from the stated generative law (prefix of Bin(amax, q))."""
    mech = get_mechanism("systemlab.allowance", 13)
    params = {
        "alpha": np.array([0.5]),
        "beta": np.array([0.5]),
        "gamma": np.array([1.0]),
        "amax": np.array([8]),
        "memory": np.array([16]),
        "eps": np.array([0.05]),
        "glitch": np.array([0.1]),
    }
    st = mech.initial_state(params)
    st = mech.advance(st, params, {"wait": 0, "burst": 6, "clock": 0})
    rec = {"wait": 2, "burst": 6, "clock": 1}
    probs = mech.predict(st, params, rec)[0]
    from scipy.special import expit

    from probe.determinism import make_rng
    from probe.envs.systemlab.noise import sample_report

    q = expit(0.5 + 0.5 * np.log1p(2) - 1.0 * np.log1p(6))
    draws = []
    for i in range(20_000):
        rng = make_rng(5, "allowance", i)
        a = int(rng.binomial(8, q))
        draws.append(sample_report(min(6, a), 6, ReportNoise(0.05, 0.1), rng))
    assert _chi2_pvalue(np.asarray(draws), probs) > 1e-3
