"""Mechanism contract tests (probe.engine.mechanisms) and exact noise-free agreement
between the vectorized grid mechanisms and the independent scalar simulators."""

import copy

import numpy as np
import pytest

from probe.engine.ir import GridHypothesis
from probe.engine.mechanisms import get_mechanism, registered_mechanisms
from probe.envs.systemlab import FAMILY, MECHANISM_NAMES, REGIME_IDS
from probe.envs.systemlab.mechanisms import (
    AllowanceMechanism,
    CacheMechanism,
    ReuseMechanism,
)
from probe.envs.systemlab.services import Cache, RateLimiter, ServiceSpec, build_service


def _state_equal(a, b):
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def _random_records(regime, n_steps, seed):
    """A random feasible action path through a regime's action space (records only)."""
    env = FAMILY.make_world(regime, seed).env
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_steps):
        cands = [c for c in env.candidates() if c.feasible]
        c = cands[int(rng.integers(len(cands)))]
        out.append(c.record)
        env.execute(c.action)
    return out


def test_all_mechanisms_registered():
    for name in MECHANISM_NAMES:
        assert name in registered_mechanisms()
        m = get_mechanism(name, 13)
        assert m.name == name
        assert m.support_size == 13
    with pytest.raises(ValueError):
        get_mechanism("systemlab.token_bucket", 1)


@pytest.mark.parametrize("regime", REGIME_IDS)
def test_contract_on_random_paths(regime):
    hs = FAMILY.hypothesis_set(regime)
    s = hs.outcome.max_count + 1
    records = _random_records(regime, 12, seed=5)
    for h in hs.hypotheses:
        assert isinstance(h, GridHypothesis)
        mech = get_mechanism(h.mechanism, s)
        params = h.grid_table()
        g = h.grid_size()
        state = mech.initial_state(params)
        for arr in state.values():
            assert arr.shape[0] == g
        for rec in records:
            before = copy.deepcopy(state)
            probs = mech.predict(state, params, rec)
            _state_equal(state, before)  # predict does not mutate
            assert probs.shape == (g, s)
            assert np.all(probs >= 0)
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)
            n = int(rec["burst"]) if "burst" in rec else len(str(rec["keys"]))
            assert np.all(probs[:, n + 1 :] == 0.0)
            np.testing.assert_array_equal(probs, mech.predict(state, params, rec))
            new = mech.advance(state, params, rec)
            _state_equal(state, before)  # advance is pure
            _state_equal(new, mech.advance(state, params, rec))
            for arr in new.values():
                assert arr.shape[0] == g
            state = new


@pytest.mark.parametrize("regime", ["R1", "R3", "R8"])
def test_no_zero_likelihood_with_noise(regime):
    """With eps > 0 every count 0..n has positive probability under every grid point."""
    hs = FAMILY.hypothesis_set(regime)
    s = hs.outcome.max_count + 1
    rec = {"wait": 0, "burst": 12, "clock": 0}
    for h in hs.hypotheses:
        assert isinstance(h, GridHypothesis)
        mech = get_mechanism(h.mechanism, s)
        params = h.grid_table()
        probs = mech.predict(mech.initial_state(params), params, rec)
        assert np.all(probs[:, :13] > 0)


def test_time_cannot_go_backwards():
    h = FAMILY.hypothesis_set("R1").by_id("token_bucket")
    assert isinstance(h, GridHypothesis)
    mech = get_mechanism(h.mechanism, 13)
    params = h.grid_table()
    st = mech.advance(mech.initial_state(params), params, {"wait": 5, "burst": 3, "clock": 10})
    with pytest.raises(ValueError):
        mech.predict(st, params, {"wait": 0, "burst": 1, "clock": 3})
    with pytest.raises(ValueError):
        mech.predict(st, params, {"wait": 0, "burst": 13, "clock": 20})  # beyond support
    with pytest.raises(KeyError):
        mech.predict(st, params, {"wait": 0, "burst": 1})


def _rate_spec(h: GridHypothesis, params, i: int) -> ServiceSpec:
    kind = h.mechanism.split(".", 1)[1]
    names = {
        "token_bucket": ("capacity", "period"),
        "leaky_meter": ("capacity", "period"),
        "fixed_window": ("limit", "window", "phase"),
        "sliding_window": ("limit", "window"),
    }[kind]
    return ServiceSpec(kind, {k: int(params[k][i]) for k in names})


@pytest.mark.parametrize("kind", ["token_bucket", "leaky_meter", "fixed_window", "sliding_window"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rate_mechanisms_match_scalar_simulators(kind, seed):
    """Noise-free admitted counts agree exactly, grid point by grid point, along random
    action paths (vectorized mechanism vs. one-request-at-a-time simulator)."""
    regime = "R5" if kind == "leaky_meter" else "R1"
    h = FAMILY.hypothesis_set(regime).by_id(kind)
    assert isinstance(h, GridHypothesis)
    mech = get_mechanism(h.mechanism, 13)
    params = h.grid_table()
    services = [build_service(_rate_spec(h, params, i)) for i in range(h.grid_size())]
    state = mech.initial_state(params)
    rng = np.random.default_rng(seed)
    clock = 0
    for _ in range(60):
        wait = int(rng.choice([0, 0, 1, 2, 3, 4, 7, 8, 16, 31]))
        burst = int(rng.integers(0, 13))
        rec = {"wait": wait, "burst": burst, "clock": clock}
        k = mech.noiseless_counts(state, params, rec)
        want = []
        for svc in services:
            assert isinstance(svc, RateLimiter)
            want.append(svc.burst(clock + wait, burst))
        np.testing.assert_array_equal(k, np.asarray(want))
        probs = mech.predict(state, params, rec)
        assert np.all(probs[np.arange(len(k)), k] >= (1 - 0.03) ** burst - 1e-12)
        state = mech.advance(state, params, rec)
        clock += wait + 1


def test_leaky_meter_is_observationally_equivalent_to_token_bucket():
    hs = FAMILY.hypothesis_set("R5")
    tb, lm = hs.by_id("token_bucket"), hs.by_id("leaky_meter")
    assert isinstance(tb, GridHypothesis) and isinstance(lm, GridHypothesis)
    assert tb.params == lm.params
    m_tb, m_lm = get_mechanism(tb.mechanism, 13), get_mechanism(lm.mechanism, 13)
    p = tb.grid_table()
    s_tb, s_lm = m_tb.initial_state(p), m_lm.initial_state(p)
    for rec in _random_records("R5", 80, seed=9):
        np.testing.assert_array_equal(m_tb.predict(s_tb, p, rec), m_lm.predict(s_lm, p, rec))
        s_tb, s_lm = m_tb.advance(s_tb, p, rec), m_lm.advance(s_lm, p, rec)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cache_mechanism_matches_scalar_cache(seed):
    rng = np.random.default_rng(seed)
    pols = ["lru", "fifo", "lfu"] * 4
    sizes = [1, 2, 3, 5] * 3
    params = {
        "policy": np.asarray(pols),
        "size": np.asarray(sizes),
        "eps": np.zeros(12),
        "glitch": np.zeros(12),
    }
    mech = get_mechanism("systemlab.cache", 9)
    assert isinstance(mech, CacheMechanism)
    caches = [Cache(p, z) for p, z in zip(pols, sizes, strict=True)]
    state = mech.initial_state(params)
    for _ in range(40):
        keys = "".join(rng.choice(list("abcdefg"), size=int(rng.integers(1, 9))))
        rec = {"keys": keys, "clock": 0}
        k = mech.noiseless_counts(state, params, rec)
        np.testing.assert_array_equal(k, [c.run(keys) for c in caches])
        probs = mech.predict(state, params, rec)
        assert np.all(probs[np.arange(12), k] == 1.0)
        state = mech.advance(state, params, rec)
        for i, c in enumerate(caches):
            resident = {chr(ord("a") + int(x)) for x in state["keys"][i] if x >= 0}
            assert resident == set(c.resident())


def test_allowance_residual_shape_and_load_effect():
    mech = get_mechanism("systemlab.allowance", 13)
    assert isinstance(mech, AllowanceMechanism)
    params = {
        "alpha": np.array([1.0, 1.0]),
        "beta": np.array([0.5, 0.5]),
        "gamma": np.array([0.0, 2.0]),
        "amax": np.array([12, 12]),
        "memory": np.array([16, 16]),
        "eps": np.array([0.03, 0.03]),
        "glitch": np.array([0.0, 0.0]),
    }
    st = mech.initial_state(params)
    rec = {"wait": 0, "burst": 12, "clock": 0}
    fresh = mech.true_count_pmf(st, params, rec)
    np.testing.assert_allclose(fresh.sum(axis=1), 1.0)
    np.testing.assert_allclose(fresh[0], fresh[1])  # no load yet: gamma irrelevant
    st = mech.advance(st, params, rec)
    loaded = mech.true_count_pmf(st, params, {"wait": 0, "burst": 12, "clock": 1})
    mean = loaded @ np.arange(13)
    assert mean[1] < mean[0]  # gamma > 0: recent load lowers the allowance
    # load leaves the memory window
    later = mech.true_count_pmf(st, params, {"wait": 20, "burst": 12, "clock": 1})
    np.testing.assert_allclose(later[0], later[1])


def _brute_stack_distance(history: str, key: str) -> int:
    if key not in history:
        return -1
    last = history.rindex(key)
    return len(set(history[last + 1 :]) - {key})


def test_reuse_residual_stack_distances_and_pmf():
    mech = get_mechanism("systemlab.reuse", 9)
    assert isinstance(mech, ReuseMechanism)
    params = {
        "q_near": np.array([0.9, 0.5]),
        "q_mid": np.array([0.5, 0.5]),
        "q_far": np.array([0.1, 0.5]),
        "eps": np.zeros(2),
        "glitch": np.zeros(2),
    }
    st = mech.initial_state(params)
    rng = np.random.default_rng(3)
    history = ""
    for _ in range(25):
        keys = "".join(rng.choice(list("abcdefgh"), size=int(rng.integers(1, 9))))
        dists, _ = mech.stack_distances(st["last"][0], int(st["t"][0]), _codes(keys))
        h = history
        want = []
        for c in keys:
            want.append(_brute_stack_distance(h, c))
            h += c
        assert dists == want
        pmf = mech.true_count_pmf(st, params, {"keys": keys, "clock": 0})
        np.testing.assert_allclose(pmf.sum(axis=1), 1.0)
        n_comp = sum(d < 0 for d in dists)
        assert np.all(pmf[:, len(keys) - n_comp + 1 :] == 0)  # compulsory misses never hit
        st = mech.advance(st, params, {"keys": keys, "clock": 0})
        history += keys


def _codes(keys: str) -> np.ndarray:
    return np.frombuffer(keys.encode(), dtype=np.uint8).astype(np.int64) - ord("a")


def test_cache_record_validation():
    mech = get_mechanism("systemlab.cache", 7)
    params = {"policy": np.asarray(["lru"]), "size": np.asarray([2])}
    st = mech.initial_state(params)
    for bad in ["", "abC", "abcdefgh", 5]:
        with pytest.raises((ValueError, TypeError)):
            mech.predict(st, params, {"keys": bad, "clock": 0})
    with pytest.raises(ValueError):
        mech.initial_state({"policy": np.asarray(["random"]), "size": np.asarray([2])})
