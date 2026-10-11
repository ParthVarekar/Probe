import json

import pytest

from probe.determinism import canonical_json, to_jsonable
from probe.run import run_world
from probe.select import POLICY_IDS
from probe.truth import EvaluatorKey

CASES = [("microclimate", "R1"), ("systemlab", "R2")]


def _json(record):
    return canonical_json(to_jsonable(record))


@pytest.mark.parametrize("family,regime", CASES)
def test_deterministic(family, regime):
    a, ma = run_world(family, regime, 3, "P8")
    b, mb = run_world(family, regime, 3, "P8")
    assert _json(a) == _json(b)
    assert ma == mb


@pytest.mark.parametrize("family,regime", CASES)
def test_all_policies_share_warmup_and_finish(family, regime):
    warm = None
    for pid in POLICY_IDS:
        rec, m = run_world(family, regime, 1, pid)
        w = [(s.action_id, s.value) for s in rec.steps if s.mode == "warmup"]
        assert warm is None or w == warm, pid
        warm = w
        assert rec.stop["reason"] is not None
        assert m["cost"] <= rec.protocol["budget"] + 1e-9
        assert all(0 < s.propensity <= 1 for s in rec.steps)
        assert abs(sum(rec.final_weights.values()) - 1) < 1e-9


@pytest.mark.parametrize("family,regime", CASES)
def test_commit_before_execute_and_no_truth(family, regime):
    from probe.run.world import get_family

    rec, _ = run_world(family, regime, 2, "P8")
    for i, s in enumerate(rec.steps):
        assert s.step == i and s.decision_hash
    world = get_family(family).make_world(regime, 2)
    opened = world.truth.open(EvaluatorKey("test"))
    text = _json(rec).decode()
    assert opened["salt"] not in text
    assert json.loads(text)["truth_commitment"] == world.truth.commitment


def test_easy_regime_decides_correctly():
    hits = sum(run_world("systemlab", "R2", s, "P8")[1]["correct"] for s in range(4))
    assert hits >= 3
