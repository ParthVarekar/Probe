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
from probe.envs.systemlab.services import (
    Cache,
    DeployService,
    DualLimiter,
    FixedWindow,
    PenaltyBox,
    RateLimiter,
    ServiceSpec,
    SlidingWindow,
    TokenBucket,
    build_service,
)


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
            for k, arr in new.items():
                assert arr.shape[0] == g
                assert not np.shares_memory(arr, state[k])  # no aliasing of the old state
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


@pytest.mark.parametrize("kind", ["token_bucket", "leaky_meter", "fixed_window", "sliding_window"])
def test_empty_burst_advances_without_aliasing(kind):
    """Regression: the leaky meter returned the old state's ``tat`` array for an empty
    burst, so the new state aliased the old one."""
    regime = "R5" if kind == "leaky_meter" else "R1"
    h = FAMILY.hypothesis_set(regime).by_id(kind)
    assert isinstance(h, GridHypothesis)
    mech = get_mechanism(h.mechanism, 13)
    params = h.grid_table()
    st = mech.initial_state(params)
    new = mech.advance(st, params, {"wait": 3, "burst": 0, "clock": 0})
    for k, arr in new.items():
        assert not np.shares_memory(arr, st[k]), k
    np.testing.assert_array_equal(mech.noiseless_counts(new, params, {"wait": 0, "burst": 0, "clock": 4}), 0)


def _trace(svc: RateLimiter, steps):
    return [svc.burst(t, n) for t, n in steps]


def test_dual_limiter_admits_only_what_both_admit():
    # TB(3, 2) and FW(4 per 10): t=0 the bucket binds (3); t=4 the bucket has 2 tokens
    # but the window has 1 slot left; t=10 a new window, the bucket binds again (3).
    steps = [(0, 5), (4, 5), (10, 5)]
    dual = DualLimiter(TokenBucket(3, 2), FixedWindow(4, 10))
    assert _trace(dual, steps) == [3, 1, 3]
    assert _trace(TokenBucket(3, 2), steps) == [3, 2, 3]
    assert _trace(FixedWindow(4, 10), steps) == [4, 0, 4]
    spec = ServiceSpec(
        "dual",
        {},
        {
            "first": ServiceSpec("token_bucket", {"capacity": 3, "period": 2}),
            "second": ServiceSpec("fixed_window", {"limit": 4, "window": 10}),
        },
    )
    assert _trace(build_service(spec), steps) == [3, 1, 3]  # type: ignore[arg-type]


def test_penalty_box_blocks_after_an_inner_rejection_without_extending():
    # TB(2, 3) with a 5-tick penalty. t=0: 2 admitted, the 3rd is rejected -> blocked
    # until 5. t=4: blocked (the bucket alone would admit). t=5: free again. t=6: 1 admitted,
    # 1 rejected -> blocked until 11. t=10: blocked; t=11: admitted.
    steps = [(0, 3), (4, 1), (5, 1), (6, 2), (10, 1), (11, 1)]
    assert _trace(PenaltyBox(TokenBucket(2, 3), 5), steps) == [2, 0, 1, 1, 0, 1]
    assert _trace(TokenBucket(2, 3), steps) == [2, 1, 0, 1, 1, 0]  # the bare bucket differs
    # Requests during the penalty do not extend it.
    box = PenaltyBox(TokenBucket(1, 1), 3)
    assert _trace(box, [(0, 2), (1, 1), (2, 1), (3, 1)]) == [1, 0, 0, 1]


def test_deploy_switches_to_a_fresh_service_at_the_deploy_tick():
    # FW(2 per 8) until tick 10 (exclusive), then a fresh SW(3 per 4).
    steps = [(0, 3), (9, 3), (10, 5), (12, 5), (14, 5)]
    svc = DeployService(FixedWindow(2, 8), SlidingWindow(3, 4), at=10)
    assert _trace(svc, steps) == [2, 2, 3, 0, 3]
    # A burst one tick before the deploy is served by the old service.
    svc = DeployService(TokenBucket(1, 100), TokenBucket(5, 1), at=10)
    assert _trace(svc, [(0, 5), (9, 5), (10, 5)]) == [1, 0, 5]
    with pytest.raises(ValueError):
        DeployService(TokenBucket(1, 1), TokenBucket(1, 1), at=-1)
