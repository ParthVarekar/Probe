"""Regime specifications, hypothesis sets, truth draws, and behaviour checks with the
exact reference filter (``systemlab_ref``): the regimes do what they pre-register."""

import numpy as np
import pytest
from systemlab_ref import RefPosterior, run_policy

from probe.engine.ir import GridHypothesis, HypothesisSet
from probe.engine.mechanisms import get_mechanism, registered_mechanisms
from probe.envs.systemlab import (
    DEPLOY_TICKS,
    FAMILY,
    R11_VARIANTS,
    REGIME_IDS,
    ServiceSpec,
    acceptable_verdicts,
    build_service,
)
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
            at = p["extra"]["deploy_at"]
            assert DEPLOY_TICKS[0] <= at <= DEPLOY_TICKS[1]
            assert p["pre_deploy"] == {
                "until_clock": at,
                "acceptable_verdicts": [f"hypothesis:{tid}", "none_fits"],
            }
            assert p["extra"]["params_before"] != p["extra"]["params_after"]
            continue
        assert "pre_deploy" not in p
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


def _truth_class(regime, world):
    p = _payload(world)
    if regime == "R3":
        return p["extra"]["out_of_set_family"]
    if regime == "R11":
        return (p["service"]["params"]["policy"], p["service"]["params"]["size"])
    return world.truth.true_hypothesis_id(KEY)


@pytest.mark.parametrize("regime", REGIME_IDS)
def test_truth_classes_are_block_randomised(regime):
    """Every class appears exactly once per block of len(classes) consecutive seeds, in
    an order that differs between blocks."""
    hs = FAMILY.hypothesis_set(regime)
    if regime == "R3":
        classes = {"dual", "penalty_box", "deploy"}
    elif regime == "R11":
        classes = set(R11_VARIANTS)
    elif regime == "R5":
        classes = {"token_bucket", "leaky_meter"}
    else:
        classes = {h.id for h in hs.hypotheses if not h.is_residual}
    k = len(classes)
    orders = set()
    for block in range(8):
        got = [_truth_class(regime, FAMILY.make_world(regime, block * k + i)) for i in range(k)]
        assert set(got) == classes and len(got) == k
        orders.add(tuple(got))
    assert len(orders) > 1


def test_r6_budget_alone_allows_at_most_five_observations():
    spec = FAMILY.regimes()["R6"]
    env = FAMILY.make_world("R6", 0).env
    cheapest = min(c.cost for c in env.candidates())
    assert spec.budgets["large"] / cheapest <= 5
    assert spec.budgets["small"] / cheapest >= 3
    assert spec.warmup <= spec.budgets["small"] / cheapest
    assert FAMILY.hypothesis_set("R6") == FAMILY.hypothesis_set("R1")


@pytest.mark.parametrize("seed", [0, 1, 2])  # one block: every R11 variant once
def test_r11_every_registered_policy_is_refutable(seed):
    """'none fits' is reachable in R11: a fixed design cycling through the patterns
    refutes every registered policy exactly (zero likelihood), while the residual
    keeps finite evidence and ends with all the posterior mass."""
    world = FAMILY.make_world("R11", seed)
    env = world.env
    post = RefPosterior(FAMILY.hypothesis_set("R11"))
    actions = [c.action for c in env.candidates()]
    for _ in range(3):
        for a in actions:
            o = env.execute(a)
            post.update(o.record, int(o.value))
    ev = {m.id: m.log_evidence for m in post.models}
    assert np.isfinite(ev.pop("residual"))
    assert all(np.isneginf(v) for v in ev.values()), ev
    assert post.post()[post.ids.index("residual")] == 1.0


def _deploy_worlds(regime, n_seeds):
    for seed in range(n_seeds):
        world = FAMILY.make_world(regime, seed)
        p = _payload(world)
        if p["service"]["kind"] == "deploy":
            yield seed, world, p


@pytest.mark.parametrize("regime", ["R3", "R7"])
def test_every_deploy_falls_inside_every_budget(regime):
    """The hidden deploy comes after the first action's burst (so every run observes the
    pre-deploy service) and well inside the smallest budget (wall cost: clock == spent), so
    a run that spends any budget tier observes the post-deploy service too. Regression:
    deploy ticks used to be drawn up to 90 (R3) and 100 (R7), past the small budget of 60,
    where the observed process was exactly an in-set grid point."""
    cfg = regime_config(regime)
    spec = cfg.spec
    first_burst_max = max(cfg.actions.waits)  # type: ignore[union-attr]
    assert cfg.actions.cost_model == "wall"  # type: ignore[union-attr]
    n = 0
    for _, _, p in _deploy_worlds(regime, 60):
        at = p["service"]["params"]["at"]
        assert at == p["extra"]["deploy_at"] == p["pre_deploy"]["until_clock"]
        assert first_burst_max < at
        assert at + 12 <= min(spec.budgets.values())
        n += 1
    assert n >= 20


def _hyp_index(hs, kind, params):
    """The grid point of hypothesis ``kind`` equal to a service's parameters (a spec
    without ``phase`` has phase 0, as in ``build_service``)."""
    h = hs.by_id(kind)
    table = h.grid_table()
    want = {k: params.get(k, 0 if k == "phase" else None) for k in h.params}
    idx = [i for i in range(h.grid_size()) if all(table[k][i].item() == want[k] for k in want)]
    assert len(idx) == 1, (kind, params)
    return h, idx[0]


@pytest.mark.parametrize("regime", ["R3", "R7"])
def test_pre_deploy_verdicts_are_exactly_justified(regime):
    """A run that ends at clock <= deploy tick saw only the pre-deploy service, and that
    service is *exactly* a registered grid point (same noise-free counts along the path,
    hence the same law of every report): so its algorithm is an acceptable verdict then,
    and only then."""
    hs = FAMILY.hypothesis_set(regime)
    s = hs.outcome.max_count + 1
    checked = 0
    for seed, world, p in _deploy_worlds(regime, 24):
        before = ServiceSpec.from_dict(p["service"]["children"]["before"])
        h, gi = _hyp_index(hs, before.kind, before.params)
        mech = get_mechanism(h.mechanism, s)
        params = {k: v[gi : gi + 1] for k, v in h.grid_table().items()}
        state = mech.initial_state(params)
        planted = build_service(ServiceSpec.from_dict(p["service"]))
        env = world.env
        at = p["pre_deploy"]["until_clock"]
        rng = np.random.default_rng(seed)
        while True:
            cands = [c for c in env.candidates() if env.clock + c.record["wait"] + 1 <= at]
            if not cands:
                break
            c = cands[int(rng.integers(len(cands)))]
            obs = env.execute(c.action)
            assert obs.meta["clock_burst"] < at
            k = planted.burst(obs.meta["clock_burst"], int(c.record["burst"]))
            assert mech.noiseless_counts(state, params, obs.record)[0] == k
            state = mech.advance(state, params, obs.record)
        assert env.clock <= at
        verdicts = acceptable_verdicts(p, env.clock)
        assert f"hypothesis:{before.kind}" in verdicts
        assert acceptable_verdicts(p, at + 1) == tuple(p["acceptable_verdicts"])
        if regime == "R3":
            assert acceptable_verdicts(p, at + 1) == ("none_fits",)
        checked += 1
    assert checked >= 6


def test_acceptable_verdicts_without_a_deploy_ignore_the_clock():
    for regime in ("R1", "R3", "R5", "R11"):
        for seed in range(6):
            p = _payload(FAMILY.make_world(regime, seed))
            if "pre_deploy" in p:
                continue
            for clock in (0, 10, 10_000):
                assert acceptable_verdicts(p, clock) == tuple(p["acceptable_verdicts"])


def test_r2_description_matches_its_grids():
    """R2's pre-registered description, checked against the grids (it used to claim that
    almost any burst of four or more separates all three algorithms): at a rested service
    one burst of 12 gives disjoint admitted counts for the three, while bursts of 4-8 only
    single out the token bucket."""
    hs = FAMILY.hypothesis_set("R2")
    s = hs.outcome.max_count + 1

    def first_counts(burst):
        out = {}
        for h in hs.hypotheses:
            if h.is_residual:
                continue
            mech, p = get_mechanism(h.mechanism, s), h.grid_table()
            rec = {"wait": 0, "burst": burst, "clock": 0}
            out[h.id] = set(mech.noiseless_counts(mech.initial_state(p), p, rec).tolist())
        return out

    full = first_counts(12)
    assert full == {"token_bucket": {2, 3}, "fixed_window": {6, 8}, "sliding_window": {10, 12}}
    for burst in (4, 6, 8):
        c = first_counts(burst)
        assert c["token_bucket"].isdisjoint(c["fixed_window"] | c["sliding_window"])
        assert not c["fixed_window"].isdisjoint(c["sliding_window"])
    assert "burst of 12" in FAMILY.regimes()["R2"].description
