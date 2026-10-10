"""Determinism and common random numbers (CRN) of the microclimate family."""

import numpy as np
import pytest

from probe.determinism import canonical_json, make_rng
from probe.envs.microclimate import FAMILY
from probe.truth import EvaluatorKey

KEY = EvaluatorKey("determinism tests")
ALL_REGIMES = [f"R{i}" for i in range(1, 11)]


def _trajectory(env, n, policy_seed):
    """A seeded random walk over feasible candidates; returns (candidate dumps, observations)."""
    rng = np.random.default_rng(policy_seed)
    cand_dumps, obs = [], []
    for _ in range(n):
        cands = env.candidates()
        cand_dumps.append(canonical_json(cands))
        feas = [c for c in cands if c.feasible]
        if not feas:
            break
        obs.append(env.execute(feas[int(rng.integers(len(feas)))].action))
    return cand_dumps, obs


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_same_seed_same_world_same_streams(regime):
    a, b = FAMILY.make_world(regime, 42), FAMILY.make_world(regime, 42)
    assert a.truth.commitment == b.truth.commitment == a.env.truth_commitment()
    assert canonical_json(a.env.metadata()) == canonical_json(b.env.metadata())
    ca, oa = _trajectory(a.env, 12, policy_seed=9)
    cb, ob = _trajectory(b.env, 12, policy_seed=9)
    assert ca == cb
    assert canonical_json(oa) == canonical_json(ob)
    assert a.truth.open(KEY) == b.truth.open(KEY)


def test_different_seeds_and_regimes_differ():
    commitments = {FAMILY.make_world(r, s).truth.commitment for r in ("R1", "R7") for s in range(5)}
    assert len(commitments) == 10
    _, o1 = _trajectory(FAMILY.make_world("R1", 1).env, 8, 0)
    _, o2 = _trajectory(FAMILY.make_world("R1", 2).env, 8, 0)
    assert canonical_json(o1) != canonical_json(o2)


def test_world_generation_independent_of_call_order():
    first = [FAMILY.make_world("R8", s).truth.commitment for s in (3, 4, 5)]
    FAMILY.make_world("R3", 99)
    again = [FAMILY.make_world("R8", s).truth.commitment for s in (5, 4, 3)][::-1]
    assert first == again


def test_salt_is_keyed_from_the_seed():
    for regime, seed in (("R1", 0), ("R3", 7), ("R10", 12)):
        world = FAMILY.make_world(regime, seed)
        opened = world.truth.open(KEY)
        assert opened["salt"] == make_rng(seed, "microclimate", regime, "salt").bytes(16).hex()
        assert opened["world_id"] == f"microclimate/{regime}/{seed}"
        assert world.truth.verify(opened)


def _pick(env, site, shield, dwell, wait):
    want = {"site": site, "shield": shield, "dwell": dwell, "wait": wait}
    return next(c for c in env.candidates() if c.feasible and c.action.params == want)


def _same(oa, ob):
    assert (oa.value, oa.missingness, oa.qc_flags) == (ob.value, ob.missingness, ob.qc_flags)
    assert oa.record == ob.record and oa.clock == ob.clock


def test_crn_same_measurement_same_value_regardless_of_history():
    """Two different histories that leave the team at the same site at the same clock
    must observe exactly the same values for the same next actions (R1 settling:
    unshielded 2 min, shielded 5 min, so 2+15 == (5+5) + (2+5))."""
    for seed in range(6):
        a = FAMILY.make_world("R1", seed).env
        b = FAMILY.make_world("R1", seed).env
        settle = a.metadata().extra["settle_min"]
        assert (settle["unshielded"], settle["shielded"]) == (2.0, 5.0)
        a.execute(_pick(a, "S05", "unshielded", 15, 0).action)  # history A: 17 minutes
        b.execute(_pick(b, "S05", "shielded", 5, 0).action)  # history B: 10 + 7 minutes
        b.execute(_pick(b, "S05", "unshielded", 5, 0).action)
        assert a.clock == b.clock and a.position == b.position == "S05"
        assert [c.action.id for c in a.candidates()] == [c.action.id for c in b.candidates()]
        for k in range(3):  # the next three identical actions observe identical values
            feas = [c for c in a.candidates() if c.feasible]
            action = feas[(37 * (k + seed)) % len(feas)].action
            _same(a.execute(action), b.execute(action))


def test_crn_value_independent_of_wait_used_to_reach_the_start_time():
    """History A waits 60 min at base, then reads S02. History B walks to S02 and spends
    exactly 60 min on three shielded 15-min readings (5 + 15 each), then reads S02 with
    no wait. Both readings have the same identity (site, shield, dwell, start time), so
    they must give the same value although the action ids differ (wait=60 vs wait=0)."""
    for seed in range(4):
        a = FAMILY.make_world("R1", seed).env
        b = FAMILY.make_world("R1", seed).env
        oa = a.execute(_pick(a, "S02", "unshielded", 5, 60).action)
        for _ in range(3):
            b.execute(_pick(b, "S02", "shielded", 15, 0).action)
        ob = b.execute(_pick(b, "S02", "unshielded", 5, 0).action)
        assert oa.meta["t_start"] == ob.meta["t_start"]
        assert oa.action_id != ob.action_id
        _same(oa, ob)


def test_no_shared_state_between_worlds():
    """Interleaving executions across worlds gives the same observations as running
    each world alone."""
    solo = {}
    for seed in (1, 2):
        _, obs = _trajectory(FAMILY.make_world("R9", seed).env, 10, policy_seed=seed)
        solo[seed] = canonical_json(obs)
    envs = {s: FAMILY.make_world("R9", s).env for s in (1, 2)}
    rngs = {s: np.random.default_rng(s) for s in (1, 2)}
    inter = {1: [], 2: []}
    for _ in range(10):
        for s in (2, 1):
            env = envs[s]
            feas = [c for c in env.candidates() if c.feasible]
            if feas:
                inter[s].append(env.execute(feas[int(rngs[s].integers(len(feas)))].action))
    for s in (1, 2):
        assert canonical_json(inter[s]) == solo[s]


def test_truth_block_randomisation_cycles_through_the_pool():
    def ids(regime, seeds):
        return [FAMILY.make_world(regime, s).truth.true_hypothesis_id(KEY) for s in seeds]

    r1 = ids("R1", range(8))
    for block in (r1[:4], r1[4:]):
        assert sorted(block) == ["H1", "H2", "H3", "H4"]
    r10 = ids("R10", range(6))
    for block in (r10[0:2], r10[2:4], r10[4:6]):
        assert sorted(block) == ["H1", "H4"]
    assert set(ids("R2", range(4))) == {"H1", "H2"}
    assert ids("R3", range(3)) == [None, None, None]
    # not a fixed pattern in the seed (blocks are permuted by a keyed RNG)
    r1_more = ids("R1", range(40))
    assert len({tuple(r1_more[i : i + 4]) for i in range(0, 40, 4)}) > 1
