"""Hidden-truth isolation (handover §12 H.1, H.11): nothing reachable through the
agent-facing environment may contain truth values or the true hypothesis id."""

import pickle
import re

import numpy as np
import pytest

from probe.determinism import canonical_json
from probe.envs.microclimate import FAMILY
from probe.truth import EvaluatorKey

KEY = EvaluatorKey("sealing tests")
ALL_REGIMES = [f"R{i}" for i in range(1, 11)]
FORBIDDEN_TOKENS = ("true_hypothesis", "beta", "payload", "expected_verdict", "mimic", "salt")


def _floats(obj, out):
    if isinstance(obj, bool):
        return
    if isinstance(obj, float):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _floats(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _floats(v, out)


def _agent_blob(world, steps, policy_seed):
    """Everything an agent could see: repr/str, metadata, every candidate list and
    every observation along a random trajectory."""
    env = world.env
    parts = [repr(env), str(env), repr(world.truth), canonical_json(env.metadata()).decode()]
    rng = np.random.default_rng(policy_seed)
    for _ in range(steps):
        cands = env.candidates()
        parts.append(canonical_json(cands).decode())
        feas = [c for c in cands if c.feasible]
        if not feas:
            break
        obs = env.execute(feas[int(rng.integers(len(feas)))].action)
        parts.append(canonical_json(obs).decode())
        parts.append(repr(obs))
    parts.append(repr(env))
    return "\n".join(parts)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@pytest.mark.parametrize("regime", ALL_REGIMES)
def test_no_truth_values_in_agent_facing_outputs(regime):
    for seed in (0, 1):
        world = FAMILY.make_world(regime, seed)
        opened = world.truth.open(KEY)
        payload = opened["payload"]
        blob = _agent_blob(world, steps=15, policy_seed=seed)
        numbers = set(_NUMBER.findall(blob))  # every numeric token an agent could see

        # 1. every full-precision truth number (draws, β, weather series, field, offsets)
        truth_floats: list[float] = []
        _floats(payload, truth_floats)
        long = {v for v in truth_floats if len(repr(v)) >= 10}
        assert len(long) > 50
        leaked = [v for v in long if repr(v) in numbers]
        assert not leaked, f"truth values leaked: {leaked[:5]}"

        # 2. structural parameters, also rounded to 5 or 6 decimals
        structural: list[float] = []
        _floats(payload["structure"], structural)
        _floats(payload["effects"], structural)
        for v in structural:
            if len(repr(v)) < 8:
                continue  # exact constants such as 0.0 or 0.08 are not secrets
            assert f"{v:.6f}" not in numbers and f"{v:.5f}" not in numbers, v

        # 3. identity of the true hypothesis and truth vocabulary
        assert not re.search(r"\bH[0-4]\b", blob)
        for tok in FORBIDDEN_TOKENS:
            assert tok not in blob, tok
        assert opened["salt"] not in blob


def test_sealed_truth_is_redacted_and_verifiable():
    world = FAMILY.make_world("R4", 2)
    text = repr(world.truth) + str(world.truth)
    assert "payload" not in text and "H1" not in text and "H4" not in text
    opened = world.truth.open(KEY)
    assert world.truth.verify(opened)
    assert opened["true_hypothesis_id"] == world.truth.true_hypothesis_id(KEY)
    assert world.env.truth_commitment() == world.truth.commitment
    assert world.env.metadata().extra["truth_commitment"] == world.truth.commitment
    with pytest.raises(PermissionError):
        world.truth.open("let me in")  # type: ignore[arg-type]


def test_oracle_methods_require_an_evaluator_key():
    env = FAMILY.make_world("R1", 0).env
    action = next(c for c in env.candidates() if c.feasible).action
    for bad in (None, "key", object()):
        with pytest.raises(PermissionError):
            env.oracle_expected(bad, action)  # type: ignore[arg-type]
        with pytest.raises(PermissionError):
            env.oracle_expected_at(bad, "S01", "shielded", 780.0)  # type: ignore[arg-type]
        with pytest.raises(PermissionError):
            env.oracle_structural_at(bad, "S01", "shielded", 780.0)  # type: ignore[arg-type]
    assert isinstance(env.oracle_expected(KEY, action), float)


def test_environment_and_truth_are_not_picklable():
    world = FAMILY.make_world("R1", 0)
    with pytest.raises(TypeError):
        pickle.dumps(world.env)
    with pytest.raises(TypeError):
        pickle.dumps(world.truth)


def test_metadata_extra_holds_only_public_design():
    md = FAMILY.make_world("R3", 0).env.metadata()
    assert set(md.extra) == {
        "world_id",
        "regime",
        "seed",
        "campaign_window_min",
        "start_position",
        "dwell_options_min",
        "wait_options_min",
        "settle_min",
        "shield_availability",
        "mast_sites",
        "nominal_direct_transmission",
        "solar",
        "campus",
        "scenario_manifest_hash",
        "truth_commitment",
    }
    site_keys = {k for s in md.extra["campus"]["sites"] for k in s}
    assert site_keys == {"id", "x_m", "y_m", "shade", "surface", "svf", "wall_m"}
