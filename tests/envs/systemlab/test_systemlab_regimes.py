"""Regime specifications, hypothesis sets, truth draws, and behaviour checks with the
exact reference filter (``systemlab_ref``): the regimes do what they pre-register."""

import numpy as np
import pytest
from systemlab_ref import RefPosterior, run_policy

from probe.engine.ir import GridHypothesis, HypothesisSet
from probe.engine.mechanisms import registered_mechanisms
from probe.envs.systemlab import FAMILY, REGIME_IDS
from probe.envs.systemlab.regimes import regime_config
from probe.truth import EvaluatorKey

KEY = EvaluatorKey("regime tests")


def _payload(world):
    return world.truth.open(KEY)["payload"]


@pytest.mark.parametrize("regime", REGIME_IDS)
def test_regime_spec_and_hypothesis_set(regime):
    spec = FAMILY.regimes()[regime]
    assert spec.expected and spec.title and spec.description
    b = spec.budgets
    assert set(b) == {"small", "medium", "large"}
    assert 0 < b["small"] < b["medium"] < b["large"]
    assert spec.warmup >= 0
    assert spec.in_set == (regime not in ("R3", "R11"))
    hs = FAMILY.hypothesis_set(regime)
    assert isinstance(hs, HypothesisSet)
    assert hs.residual.id == "residual"
    for h in hs.hypotheses:
        assert isinstance(h, GridHypothesis)
        assert h.mechanism in registered_mechanisms()
        assert h.grid_size() <= 5000  # keeps per-step EIG cheap
    cfg = regime_config(regime)
    assert hs.outcome.max_count == cfg.actions.max_count
    assert {c.name for c in hs.covariates} == set(cfg.actions.candidates(0)[0].record)


@pytest.mark.parametrize("regime", [r for r in REGIME_IDS if FAMILY.regimes()[r].in_set])
def test_in_set_truth_is_a_grid_point_with_declared_noise(regime):
    hs = FAMILY.hypothesis_set(regime)
    seen = set()
    for seed in range(40):
        world = FAMILY.make_world(regime, seed)
        tid = world.truth.true_hypothesis_id(KEY)
        p = _payload(world)
        hyp = hs.by_id(tid)
        assert isinstance(hyp, GridHypothesis) and not hyp.is_residual
        seen.add(tid)
        if regime == "R7":  # nonstationary: both segments are grid points, no single one
            assert "true_params" not in p
            assert p["acceptable_verdicts"] == [f"hypothesis:{tid}", "none_fits"]
            assert 40 <= p["extra"]["deploy_at"] <= 100
            assert p["extra"]["params_before"] != p["extra"]["params_after"]
            continue
        table = hyp.grid_table()
        gi = p["true_grid_index"]
        for k, v in p["true_params"].items():
            assert table[k][gi].item() == v
        assert table["eps"][gi].item() == p["noise"]["eps"]
        assert table["glitch"][gi].item() == p["noise"]["glitch"]
        assert p["service"]["params"] == {
            k: v for k, v in p["true_params"].items() if k not in ("eps", "glitch")
        } | ({"policy": hyp.fixed["policy"]} if "policy" in hyp.fixed else {})
        if regime == "R5":
            assert p["acceptable_verdicts"] == ["inconclusive"]
        else:
            assert p["acceptable_verdicts"] == [f"hypothesis:{tid}"]
    want = {h.id for h in hs.hypotheses if not h.is_residual}
    assert seen == want


def test_out_of_set_truths_cover_their_families():
    fams = {_payload(FAMILY.make_world("R3", s))["extra"]["out_of_set_family"] for s in range(30)}
    assert fams == {"dual", "penalty_box", "deploy"}
    pols = {_payload(FAMILY.make_world("R11", s))["service"]["params"]["policy"] for s in range(30)}
    assert pols == {"clock", "mru"}
    for r in ("R3", "R11"):
        for s in range(10):
            w = FAMILY.make_world(r, s)
            assert w.truth.true_hypothesis_id(KEY) is None
            assert _payload(w)["acceptable_verdicts"] == ["none_fits"]


def _random_run(regime, seed, steps):
    world = FAMILY.make_world(regime, seed)
    env = world.env
    post = RefPosterior(FAMILY.hypothesis_set(regime))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        cands = [c for c in env.candidates() if c.feasible]
        c = cands[int(rng.integers(len(cands)))]
        o = env.execute(c.action)
        post.update(o.record, int(o.value))
    return world, post


@pytest.mark.parametrize("seed", [0, 1])
def test_r5_variants_stay_exactly_tied(seed):
    world, post = _random_run("R5", seed, 25)
    ids = post.ids
    tb, lm = post.models[ids.index("token_bucket")], post.models[ids.index("leaky_meter")]
    assert tb.log_evidence == lm.log_evidence
    p = post.post()
    assert p[ids.index("token_bucket")] == p[ids.index("leaky_meter")]
    assert p[ids.index("residual")] < 1e-3  # the residual is separable
    u = post.eig([c.record for c in world.env.candidates()])
    assert np.all(u < 1e-3)  # nothing informative left between the twins


def test_r11_zero_noise_eliminates_exactly():
    _, post = _random_run("R11", 3, 12)
    ev = {m.id: m.log_evidence for m in post.models}
    assert np.isfinite(ev["residual"])
    assert any(np.isneginf(v) for k, v in ev.items() if k != "residual")


def test_r2_is_easy_for_every_design():
    spec = FAMILY.regimes()["R2"]
    hs = FAMILY.hypothesis_set("R2")
    for pol in ("random", "stratified"):
        correct = 0
        for seed in range(12):
            w = FAMILY.make_world("R2", seed)
            out = run_policy(
                w.env,
                hs,
                pol,
                spec.budgets["large"],
                np.random.default_rng(seed),
                warmup=spec.warmup,
                warmup_rng=np.random.default_rng(100 + seed),
            )
            correct += out.decided == w.truth.true_hypothesis_id(KEY)
        assert correct >= 11


def _median_cost(regime, policy, seeds, budget):
    spec = FAMILY.regimes()[regime]
    hs = FAMILY.hypothesis_set(regime)
    costs, wrong = [], 0
    for seed in seeds:
        w = FAMILY.make_world(regime, seed)
        out = run_policy(
            w.env,
            hs,
            policy,
            spec.budgets[budget],
            np.random.default_rng(1000 + seed),
            warmup=spec.warmup,
            warmup_rng=np.random.default_rng(seed),
        )
        costs.append(out.spent if out.decided is not None else np.inf)
        wrong += out.decided is not None and out.decided != w.truth.true_hypothesis_id(KEY)
    return float(np.median(costs)), wrong


def test_r10_positive_control_separates_policies_quick():
    seeds = range(8)
    eig, wrong_eig = _median_cost("R10", "eig_cost", seeds, "large")
    rnd, _ = _median_cost("R10", "random", seeds, "large")
    strat, _ = _median_cost("R10", "stratified", seeds, "large")
    assert eig <= 0.7 * min(rnd, strat)
    assert wrong_eig == 0


@pytest.mark.slow
def test_r10_positive_control_separates_policies():
    seeds = range(40)
    eig, wrong_eig = _median_cost("R10", "eig_cost", seeds, "large")
    rnd, _ = _median_cost("R10", "random", seeds, "large")
    strat, _ = _median_cost("R10", "stratified", seeds, "large")
    assert eig <= 0.6 * min(rnd, strat)
    assert wrong_eig <= 2


@pytest.mark.slow
def test_r1_well_specified_worlds_are_identified():
    spec = FAMILY.regimes()["R1"]
    hs = FAMILY.hypothesis_set("R1")
    correct = decided = 0
    for seed in range(60):
        w = FAMILY.make_world("R1", seed)
        out = run_policy(
            w.env,
            hs,
            "random",
            spec.budgets["large"],
            np.random.default_rng(seed),
            warmup=spec.warmup,
            warmup_rng=np.random.default_rng(100 + seed),
        )
        decided += out.decided is not None
        correct += out.decided == w.truth.true_hypothesis_id(KEY)
    assert decided >= 55
    assert correct >= decided - 3  # confidently-wrong rate ~ <= 5% at pi* = 0.95


@pytest.mark.slow
def test_r3_residual_can_detect_none_fits_with_a_spread_design():
    """With a full budget and no early stop, a random design puts the residual on top
    in a substantial share of out-of-set worlds (the rest is the job of H.7 alarms)."""
    spec = FAMILY.regimes()["R3"]
    hs = FAMILY.hypothesis_set("R3")
    leads = 0
    n = 24
    for seed in range(n):
        w = FAMILY.make_world("R3", seed)
        out = run_policy(
            w.env,
            hs,
            "random",
            spec.budgets["large"],
            np.random.default_rng(1000 + seed),
            pi_star=1.01,
        )
        leads += out.leader == "residual"
    assert leads >= n // 3
