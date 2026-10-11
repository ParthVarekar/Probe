"""Hidden-truth sealing (handover §12 H.1, H.11): nothing reachable through the
agent-facing environment surface depends on, or contains, the planted truth."""

import dataclasses
import pickle

import numpy as np
import pytest

from probe.determinism import canonical_json
from probe.envs.systemlab import FAMILY, REGIME_IDS
from probe.truth import EvaluatorKey

KEY = EvaluatorKey("sealing tests")
AGENT_API = {
    "candidates",
    "clock",
    "execute",
    "exhausted",
    "metadata",
    "reset_oracle",
    "spent",
    "truth_commitment",
}


def _agent_view(env):
    md = env.metadata()
    return {
        "metadata": dataclasses.asdict(md),
        "candidates": [dataclasses.asdict(c) for c in env.candidates()],
        "repr": repr(env),
        "str": str(env),
    }


def _fixed_path(regime, n=20):
    env = FAMILY.make_world(regime, 0).env
    rng = np.random.default_rng(1)
    out = []
    for _ in range(n):
        cands = [c for c in env.candidates() if c.feasible]
        a = cands[int(rng.integers(len(cands)))].action
        env.execute(a)
        out.append(a)
    return out


@pytest.mark.parametrize("regime", REGIME_IDS)
def test_agent_view_does_not_depend_on_the_truth(regime):
    """Metadata and candidates are byte-identical for every seed (and so every truth),
    and observations differ across seeds only in ``value``."""
    views = {canonical_json(_agent_view(FAMILY.make_world(regime, s).env)) for s in range(12)}
    assert len(views) == 1
    path = _fixed_path(regime)
    streams = set()
    for s in range(12):
        env = FAMILY.make_world(regime, s).env
        obs = [dataclasses.asdict(env.execute(a)) for a in path]
        for o in obs:
            o.pop("value")
        streams.add(canonical_json({"obs": obs, "view": _agent_view(env)}))
    assert len(streams) == 1


def _truth_needles(world):
    opened = world.truth.open(KEY)
    p = opened["payload"]
    needles = [
        opened["salt"],
        world.truth.world_id,
        "true_hypothesis",
        "acceptable_verdicts",
        "true_grid_index",
        "deploy_at",
        "pre_deploy",
        "until_clock",
        "penalty",
        "out_of_set",
        "equivalence_class",
        canonical_json(p["service"]).decode(),
        canonical_json(p["service"]["params"]).decode(),
    ]
    if p.get("true_params"):
        needles.append(canonical_json(p["true_params"]).decode())
    return needles


@pytest.mark.parametrize("regime", REGIME_IDS)
@pytest.mark.parametrize("seed", [0, 5])
def test_no_truth_strings_in_agent_outputs(regime, seed):
    world = FAMILY.make_world(regime, seed)
    env = world.env
    blobs = [canonical_json(_agent_view(env)).decode()]
    for a in _fixed_path(regime, 15):
        blobs.append(canonical_json(dataclasses.asdict(env.execute(a))).decode())
        blobs.append(canonical_json(_agent_view(env)).decode())
    blobs.append(env.truth_commitment())
    text = "\n".join(blobs)
    for needle in _truth_needles(world):
        assert needle not in text, needle
    assert env.truth_commitment() == world.truth.commitment


def test_public_attributes_are_only_the_agent_api():
    env = FAMILY.make_world("R3", 2).env
    public = {n for n in dir(env) if not n.startswith("_")}
    assert public == AGENT_API
    for name, val in vars(env).items():
        assert name.startswith("_")
        assert "truth" not in type(val).__name__.lower()


@pytest.mark.parametrize("regime", REGIME_IDS)
def test_sealed_truth_contract(regime):
    world = FAMILY.make_world(regime, 7)
    truth = world.truth
    spec = FAMILY.regimes()[regime]
    assert "salt" not in repr(truth) and "payload" not in str(truth)
    with pytest.raises(PermissionError):
        truth.open("let me in")  # type: ignore[arg-type]
    opened = truth.open(KEY)
    assert truth.verify(opened)
    assert opened["world_id"] == f"systemlab/{regime}/7"
    assert len(opened["salt"]) == 32
    tid = truth.true_hypothesis_id(KEY)
    assert (tid is None) == (not spec.in_set)
    assert opened["true_hypothesis_id"] == tid
    p = opened["payload"]
    assert p["regime"] == regime and p["seed"] == 7
    assert p["acceptable_verdicts"]
    with pytest.raises(TypeError):
        pickle.dumps(truth)


def test_reset_oracle_is_one_metered_channel_per_world():
    """Regression: every reset_oracle() call used to hand out a fresh oracle with a zero
    cost meter, and oracle use never reached env.spent, so the P11 budget could be
    bypassed by asking again. Now the world has one oracle and its cost is spent."""
    env = FAMILY.make_world("R11", 1).env
    assert env.spent == 0.0
    oracle = env.reset_oracle()
    oracle.query("abcab")  # one reset + five accesses
    assert oracle.cost == 6.0
    again = env.reset_oracle()
    assert again is oracle and again.cost == 6.0
    assert env.spent == 6.0
    cand = env.candidates()[0]
    env.execute(cand.action)
    assert env.spent == 6.0 + cand.cost
    again.step("a")
    assert env.spent == 7.0 + cand.cost


def test_reset_oracle_only_where_declared():
    for regime in REGIME_IDS:
        env = FAMILY.make_world(regime, 0).env
        if FAMILY.regimes()[regime].id == "R11":
            assert env.reset_oracle().alphabet == ("a", "b", "c", "d")
            assert env.metadata().extra["resettable"] is True
        else:
            with pytest.raises(PermissionError):
                env.reset_oracle()
