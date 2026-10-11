"""Environment behaviour: determinism, counter-based CRN noise, records, costs,
feasibility and the EnvironmentFamily surface."""

import dataclasses
import os
import subprocess
import sys

import numpy as np
import pytest

from probe.determinism import canonical_json
from probe.envs.base import Action, InfeasibleActionError, check_records
from probe.envs.systemlab import FAMILY, REGIME_IDS, noise_rng
from probe.envs.systemlab.actions import RateActionSpace
from probe.envs.systemlab.env import SystemLabEnv
from probe.envs.systemlab.noise import ReportNoise, sample_report
from probe.envs.systemlab.regimes import regime_config
from probe.envs.systemlab.services import Cache, ServiceSpec, build_service
from probe.truth import EvaluatorKey

KEY = EvaluatorKey("env tests")


def _path(env, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        cands = [c for c in env.candidates() if c.feasible]
        out.append(cands[int(rng.integers(len(cands)))].action)
    return out


def _run(regime, seed, actions):
    env = FAMILY.make_world(regime, seed).env
    return [env.execute(a) for a in actions]


def test_family_surface():
    assert FAMILY.name == "systemlab"
    assert FAMILY.version
    assert tuple(FAMILY.regimes()) == REGIME_IDS
    assert set(REGIME_IDS) == {"R1", "R2", "R3", "R5", "R6", "R7", "R8", "R10", "R11"}
    for r, spec in FAMILY.regimes().items():
        assert spec.id == r
        assert FAMILY.hypothesis_set(r) is FAMILY.hypothesis_set(r)  # built once
    with pytest.raises(KeyError):
        FAMILY.make_world("R4", 0)
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            FAMILY.make_world("R1", bad)


@pytest.mark.parametrize("regime", REGIME_IDS)
def test_same_seed_same_stream(regime):
    env = FAMILY.make_world(regime, 21).env
    actions = _path(env, 15, seed=4)
    a = _run(regime, 21, actions)
    b = _run(regime, 21, actions)
    assert canonical_json([dataclasses.asdict(o) for o in a]) == canonical_json(
        [dataclasses.asdict(o) for o in b]
    )
    w1, w2 = FAMILY.make_world(regime, 21), FAMILY.make_world(regime, 21)
    assert w1.truth.commitment == w2.truth.commitment == w1.env.truth_commitment()
    assert w1.truth.open(KEY) == w2.truth.open(KEY)


def test_different_seeds_differ():
    commits = {FAMILY.make_world("R1", s).truth.commitment for s in range(30)}
    assert len(commits) == 30
    kinds = {FAMILY.make_world("R1", s).truth.true_hypothesis_id(KEY) for s in range(30)}
    assert kinds == {"token_bucket", "fixed_window", "sliding_window"}


def _true_service(world):
    p = world.truth.open(KEY)["payload"]

    def spec(d):
        return ServiceSpec(
            d["kind"], d["params"], {k: spec(v) for k, v in d.get("children", {}).items()}
        )

    return build_service(spec(p["service"])), ReportNoise(**p["noise"])


@pytest.mark.parametrize("regime", ["R1", "R3", "R7", "R8", "R11"])
def test_reports_are_counter_based_draws_from_the_true_count(regime):
    """Each report equals sample_report(k_true, n, noise, rng) with the rng keyed by
    (world seed, regime, action id, clock) alone: common random numbers by construction."""
    seed = 13
    world = FAMILY.make_world(regime, seed)
    svc, noise = _true_service(world)
    env = world.env
    for action in _path(FAMILY.make_world(regime, seed).env, 25, seed=2):
        clock = int(env.clock)
        obs = env.execute(action)
        if isinstance(svc, Cache):
            keys = action.params["keys"]
            k, n = svc.run(keys), len(keys)
        else:
            n = action.params["burst"]
            k = svc.burst(clock + action.params["wait"], n)
        want = sample_report(k, n, noise, noise_rng(seed, regime, action.id, clock))
        assert obs.value == want


def test_crn_same_action_same_clock_same_noise():
    """Two histories that reach the same clock and true count report the same value."""
    hits = 0
    for seed in range(40):
        a = FAMILY.make_world("R1", seed)
        b = FAMILY.make_world("R1", seed)
        a.env.execute(Action("wait=032|burst=01", {"wait": 32, "burst": 1}))
        b.env.execute(Action("wait=016|burst=01", {"wait": 16, "burst": 1}))
        b.env.execute(Action("wait=016|burst=01", {"wait": 16, "burst": 1}))
        b.env.execute(Action("wait=000|burst=01", {"wait": 0, "burst": 1}))
        assert a.env.clock == 33 and b.env.clock == 35
        a.env.execute(Action("wait=001|burst=01", {"wait": 1, "burst": 1}))
        assert a.env.clock == b.env.clock
        sa, _ = _true_service(a)
        sb, _ = _true_service(b)
        ka = [sa.burst(32, 1), sa.burst(34, 1), sa.burst(35, 12)]
        kb = [sb.burst(16, 1), sb.burst(33, 1), sb.burst(34, 1), sb.burst(35, 12)]
        last = Action("wait=000|burst=12", {"wait": 0, "burst": 12})
        ya, yb = a.env.execute(last).value, b.env.execute(last).value
        if ka[-1] == kb[-1]:
            hits += 1
            assert ya == yb
    assert hits >= 10


@pytest.mark.parametrize("regime", REGIME_IDS)
def test_candidates_and_records(regime):
    world = FAMILY.make_world(regime, 3)
    env = world.env
    md = env.metadata()
    hs = FAMILY.hypothesis_set(regime)
    assert md.family == "systemlab"
    assert md.origin == "simulated"
    assert md.outcome == hs.outcome
    assert md.covariates == hs.covariates
    for step in range(12):
        cands = env.candidates()
        ids = [c.action.id for c in cands]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
        check_records([c.record for c in cands], md.covariates)
        for c in cands:
            assert c.cost > 0
            assert c.stratum in md.population
            assert c.record["clock"] == env.clock
        c = cands[(7 * step + 3) % len(cands)]
        spent, clock = env.spent, env.clock
        obs = env.execute(c.action)
        check_records([obs.record], md.covariates)
        assert obs.step == step
        assert obs.missingness == "observed" and obs.origin == "simulated"
        assert isinstance(obs.value, int) and 0 <= obs.value <= md.outcome.max_count
        n = int(c.record["burst"]) if "burst" in c.record else len(str(c.record["keys"]))
        assert obs.value <= n
        assert obs.record == c.record
        assert obs.cost == c.cost
        assert env.spent == pytest.approx(spent + c.cost)
        assert env.clock > clock and obs.clock == env.clock


def test_costs_follow_the_cost_model():
    for regime, model in (("R1", "wall"), ("R2", "uniform"), ("R6", "uniform")):
        env = FAMILY.make_world(regime, 0).env
        for c in env.candidates():
            want = c.record["wait"] + 1 if model == "wall" else 1
            assert c.cost == want
        assert env.metadata().cost_unit == ("tick" if model == "wall" else "action")
    env = FAMILY.make_world("R11", 0).env
    assert {c.cost for c in env.candidates()} == {6.0}


def test_infeasible_and_unknown_actions():
    env = FAMILY.make_world("R1", 0).env
    with pytest.raises(InfeasibleActionError):
        env.execute(Action("wait=003|burst=01", {"wait": 3, "burst": 1}))
    with pytest.raises(InfeasibleActionError):
        env.execute(Action("wait=000|burst=01", {"wait": 1, "burst": 1}))
    assert env.spent == 0 and env.clock == 0


def test_horizon_makes_actions_infeasible_and_exhausts():
    cfg = regime_config("R1")
    short = dataclasses.replace(
        cfg, actions=RateActionSpace(waits=(0, 8), bursts=(1, 4), horizon=12)
    )
    env = SystemLabEnv(
        short,
        0,
        ServiceSpec("token_bucket", {"capacity": 4, "period": 2}),
        ReportNoise(0.03),
        "0" * 64,
    )
    env.execute(Action("wait=008|burst=04", {"wait": 8, "burst": 4}))
    assert env.clock == 9
    feas = {c.action.id: c.feasible for c in env.candidates()}
    assert feas == {
        "wait=000|burst=01": True,
        "wait=000|burst=04": True,
        "wait=008|burst=01": False,
        "wait=008|burst=04": False,
    }
    with pytest.raises(InfeasibleActionError):
        env.execute(Action("wait=008|burst=01", {"wait": 8, "burst": 1}))
    for _ in range(3):
        env.execute(Action("wait=000|burst=01", {"wait": 0, "burst": 1}))
    assert env.clock == 12
    assert env.exhausted


def test_env_rejects_mismatched_service():
    with pytest.raises(ValueError):
        SystemLabEnv(
            regime_config("R11"),
            0,
            ServiceSpec("token_bucket", {"capacity": 2, "period": 1}),
            ReportNoise(),
            "0" * 64,
        )


_DIGEST = """
import dataclasses
from probe.determinism import content_hash
from probe.envs.systemlab import FAMILY, REGIME_IDS
from probe.truth import EvaluatorKey


def digest():
    out = {}
    for regime in REGIME_IDS:
        world = FAMILY.make_world(regime, 5)
        env, obs = world.env, []
        for i in range(12):
            cands = [c for c in env.candidates() if c.feasible]
            obs.append(dataclasses.asdict(env.execute(cands[(7 * i + 3) % len(cands)].action)))
        opened = dict(world.truth.open(EvaluatorKey("digest")))
        out[regime] = [world.truth.commitment, content_hash(opened), content_hash(obs)]
    return content_hash(out)
"""


def test_worlds_are_identical_across_processes_and_hash_seeds():
    """Worlds, sealed payloads and observation streams are a pure function of
    (regime, seed): the same in a fresh interpreter with another PYTHONHASHSEED (so no
    dict/set ordering or hash() dependence) as in this process."""
    ns: dict = {}
    exec(_DIGEST, ns)
    here = ns["digest"]()
    hash_seed = "7" if os.environ.get("PYTHONHASHSEED") == "4242" else "4242"
    out = subprocess.run(
        [sys.executable, "-c", _DIGEST + "\nprint(digest())"],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": hash_seed},
    )
    assert out.stdout.strip() == here
