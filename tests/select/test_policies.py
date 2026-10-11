"""Policies on hand-made score tables: choices, propensities (exact and by simulation),
determinism, tie-breaking by stable key, the guarded policy's limits and alarm,
records and the registry."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from select_testkit import prepare, synthetic_context, synthetic_scored

from probe.determinism import canonical_json, make_rng, rng_state_hash
from probe.run.protocol import RunProtocol
from probe.select import POLICY_IDS, POLICY_NAMES, make_policy
from probe.select.base import Decision, stable_key
from probe.select.designs import audit_draw, audit_probabilities
from probe.select.policies import (
    GuardedEIGPolicy,
    NoCandidatesError,
    OraclePolicy,
    PolicyBase,
)
from probe.truth import EvaluatorKey

_KEY = EvaluatorKey("selection tests: oracle construction")
STRATA = ["s0", "s0", "s0", "s1", "s2", "s2"]
U = [0.10, 0.40, 0.25, 0.05, 0.30, 0.20]
COST = [1.0, 4.0, 1.0, 1.0, 2.0, 1.0]


def _proto(**kw: object) -> RunProtocol:
    return RunProtocol(budget=100.0, warmup=0, **kw)  # type: ignore[arg-type]


def _ctx(protocol: RunProtocol | None = None, alarm: bool = False, **kw: object):
    sc = synthetic_scored(U, cost=COST, strata=STRATA, **kw)  # type: ignore[arg-type]
    return synthetic_context(sc, protocol=protocol or _proto(), alarm=alarm)


def _decide(policy: PolicyBase, ctx, seed: int = 0) -> Decision:
    return policy.decide(ctx, make_rng(seed, "policy-test", policy.id))


def _decision_fields(d: Decision) -> dict:
    return d.as_record()


# --------------------------------------------------------------------------- P1, P3


def test_random_policy_propensity_and_frequencies() -> None:
    pol = make_policy("P1")
    ctx = _ctx()
    n = ctx.scored.n
    counts = Counter()
    for s in range(6000):
        d = _decide(pol, ctx, s)
        assert d.propensity == 1.0 / n and d.mode == "random"
        np.testing.assert_array_equal(d.scores, np.full(n, 1.0 / n))
        counts[d.chosen_index] += 1
    freq = np.array([counts[i] / 6000 for i in range(n)])
    assert np.all(np.abs(freq - 1 / n) < 4.5 * math.sqrt((1 / n) * (1 - 1 / n) / 6000))


def test_stratified_policy_least_sampled_first() -> None:
    pol = make_policy("P3")
    sc = synthetic_scored(U, cost=COST, strata=STRATA)
    ctx = synthetic_context(sc, protocol=_proto(), counts={"s0": 3, "s1": 1, "s2": 1})
    seen = set()
    for s in range(200):
        d = _decide(pol, ctx, s)
        stratum = STRATA[d.chosen_index]
        assert stratum in {"s1", "s2"}
        expect = 0.5 if stratum == "s1" else 0.25
        assert d.propensity == pytest.approx(expect, abs=1e-15)
        assert d.scores[d.chosen_index] == d.propensity
        seen.add(d.chosen_index)
    assert seen == {3, 4, 5}


# --------------------------------------------------------------------------- argmax family


def test_argmax_policies_choose_their_maximum() -> None:
    pv = [1.0, 3.0, 2.0, np.inf, 0.5, 4.0]
    up = [0.3, 0.1, 0.9, 0.2, 0.4, 0.0]
    ctx = _ctx(pred_var=pv, u_param=up)
    ids = ctx.scored.action_ids()
    expect = {"P4": 3, "P5": 2, "P6": 1, "P7": 2}  # P7: 0.25/1 beats 0.40/4 and 0.30/2
    for pid, idx in expect.items():
        d = _decide(make_policy(pid), ctx)
        assert d.chosen_index == idx and d.action_id == ids[idx], pid
        assert d.propensity == 1.0 and d.mode == "model" and d.tie_set == (ids[idx],)
        assert d.rng_state_before == d.rng_state_after  # deterministic: no draws
    d7 = _decide(make_policy("P7"), ctx)
    np.testing.assert_allclose(d7.scores, np.asarray(U) / np.asarray(COST), rtol=0)
    assert "highest U_M/cost: 0.25 nats / 1" in d7.reason and "next best" in d7.reason


@pytest.mark.parametrize("pid", ["P4", "P5", "P6", "P7", "P8"])
def test_argmax_ties_broken_by_stable_key(pid: str) -> None:
    ids = ["a00", "a01", "a02", "a03", "a04"]
    tied = [1, 2, 4]
    u = [0.1, 0.5, 0.5, 0.2, 0.5 * (1 + 1e-12)]  # a04 within δ of the others
    sc = synthetic_scored(u, ids=ids, u_param=u, pred_var=u)
    ctx = synthetic_context(sc, protocol=_proto(audit_eps=0.0))
    d = _decide(make_policy(pid), ctx)
    winner = min((ids[i] for i in tied), key=stable_key)
    assert d.action_id == winner
    assert d.tie_set == tuple(sorted(ids[i] for i in tied))
    assert "tied within δ" in d.reason
    # outside δ there is no tie
    u2 = [0.1, 0.5, 0.5 * (1 - 1e-6), 0.2, 0.3]
    sc2 = synthetic_scored(u2, ids=ids, u_param=u2, pred_var=u2)
    d2 = _decide(make_policy(pid), synthetic_context(sc2, protocol=_proto(audit_eps=0.0)))
    assert d2.action_id == "a01" and d2.tie_set == ("a01",)


def test_parameter_bald_requires_u_param() -> None:
    sc = synthetic_scored(U)
    from dataclasses import replace

    lazy = replace(sc, u_param=np.full(sc.n, np.nan), u_param_available=False)
    with pytest.raises(ValueError, match="u_param"):
        _decide(make_policy("P5"), synthetic_context(lazy, protocol=_proto()))


# --------------------------------------------------------------------------- P8


def test_guarded_with_eps_zero_equals_eig_per_cost() -> None:
    ctx = _ctx(protocol=_proto(audit_eps=0.0))
    p7 = _decide(make_policy("P7"), ctx, 5)
    for s in range(20):
        rng = make_rng(s, "p8")
        before = rng_state_hash(rng)
        d = make_policy("P8").decide(ctx, rng)
        assert (d.chosen_index, d.propensity, d.mode) == (p7.chosen_index, 1.0, "model")
        assert d.tie_set == p7.tie_set
        np.testing.assert_array_equal(d.scores, p7.scores)
        assert d.rng_state_before == d.rng_state_after == before  # ε = 0 draws nothing


def test_guarded_with_eps_one_is_the_audit_design() -> None:
    ctx = _ctx(protocol=_proto(audit_eps=1.0))
    q = audit_probabilities(ctx.scored)
    for s in range(200):
        d = make_policy("P8").decide(ctx, make_rng(s, "p8-audit"))
        ref = audit_draw(ctx.scored, make_rng(s, "p8-audit"))
        assert d.chosen_index == ref.index and d.mode == "audit"
        assert d.propensity == q[d.chosen_index]
        assert d.details["branch_draw"] is None and d.tie_set == ()


def test_guarded_propensity_is_the_mixture_probability_by_simulation() -> None:
    """P(d) = ε·q(d) + (1 − ε)·1[d = d*], checked against the empirical choice
    frequencies over many generator seeds."""
    eps = 0.3
    ctx = _ctx(protocol=_proto(audit_eps=eps))
    q = audit_probabilities(ctx.scored)
    star = 2  # argmax U/c
    expect = eps * q + (1 - eps) * (np.arange(ctx.scored.n) == star)
    n = 20_000
    pol = make_policy("P8")
    counts = Counter()
    logged: dict[int, float] = {}
    modes = Counter()
    for s in range(n):
        d = pol.decide(ctx, make_rng(s, "p8-sim"))
        counts[d.chosen_index] += 1
        modes[d.mode] += 1
        logged.setdefault(d.chosen_index, d.propensity)
        assert d.propensity == logged[d.chosen_index]  # a function of the action only
        assert d.details["model_choice"] == ctx.scored.action_ids()[star]
    np.testing.assert_allclose([logged[i] for i in range(ctx.scored.n)], expect, rtol=1e-15)
    assert math.fsum(expect) == pytest.approx(1.0, abs=1e-15)
    freq = np.array([counts[i] / n for i in range(ctx.scored.n)])
    sd = np.sqrt(expect * (1 - expect) / n)
    assert np.all(np.abs(freq - expect) < 4.5 * sd), (freq, expect)
    assert abs(modes["audit"] / n - eps) < 4.5 * math.sqrt(eps * (1 - eps) / n)


def test_guarded_alarm_raises_the_audit_fraction() -> None:
    proto = _proto(audit_eps=0.15)
    pol = GuardedEIGPolicy(alarm_audit_eps=0.6)
    assert pol.effective_eps(proto, alarm=False) == 0.15
    assert pol.effective_eps(proto, alarm=True) == 0.6
    assert GuardedEIGPolicy(alarm_audit_eps=0.05).effective_eps(proto, alarm=True) == 0.15
    n = 4000
    for alarm, eps in ((False, 0.15), (True, 0.6)):
        ctx = _ctx(protocol=proto, alarm=alarm)
        audits = 0
        for s in range(n):
            d = pol.decide(ctx, make_rng(s, "p8-alarm"))
            assert d.details["eps"] == eps and d.details["alarm"] is alarm
            audits += d.mode == "audit"
        assert abs(audits / n - eps) < 4.5 * math.sqrt(eps * (1 - eps) / n)
    # the default raises it all the way: the audit design while the alarm is up
    ctx = _ctx(protocol=proto, alarm=True)
    d = make_policy("P8").decide(ctx, make_rng(0, "p8-alarm-default"))
    assert d.mode == "audit" and d.details["eps"] == 1.0
    assert d.propensity == audit_probabilities(ctx.scored)[d.chosen_index]
    with pytest.raises(ValueError):
        GuardedEIGPolicy(alarm_audit_eps=1.5)


def test_guarded_lambda_uses_parameter_eig() -> None:
    up = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]  # λ·u_param makes candidate 3 best per cost
    ctx0 = _ctx(protocol=_proto(audit_eps=0.0), u_param=up)
    assert _decide(make_policy("P8"), ctx0).chosen_index == 2
    ctx1 = _ctx(protocol=_proto(audit_eps=0.0, lambda_param=0.5), u_param=up)
    d = _decide(make_policy("P8"), ctx1)
    assert d.chosen_index == 3 and d.score == pytest.approx(0.05 + 0.5 * 2.0)
    assert "U_M + 0.5·U_θ" in d.reason


# --------------------------------------------------------------------------- generic


@pytest.mark.parametrize("pid", ["P1", "P3", "P4", "P5", "P6", "P7", "P8"])
def test_deterministic_given_generator_seed(pid: str) -> None:
    ctx = _ctx(protocol=_proto(audit_eps=0.5))
    for s in range(10):
        a = _decide(make_policy(pid), ctx, s)
        b = _decide(make_policy(pid), ctx, s)
        assert _decision_fields(a) == _decision_fields(b)
        assert 0.0 < a.propensity <= 1.0
        assert 0 <= a.chosen_index < ctx.scored.n
        assert a.action_id == ctx.scored.action_ids()[a.chosen_index]
        assert a.reason and a.policy_id == pid


def test_decision_record_is_canonical_json() -> None:
    ctx = _ctx(pred_var=[1.0, np.inf, 2.0, 0.5, 0.1, 0.2])
    d4 = _decide(make_policy("P4"), ctx)
    rec = d4.as_record()
    assert rec["scores"][1] is None and rec["score"] is None  # +inf variance
    canonical_json(rec)
    canonical_json(_decide(make_policy("P8"), _ctx(protocol=_proto(audit_eps=0.5))).as_record())
    with pytest.raises(ValueError):
        d4.scores[0] = 3.0  # read-only


def test_empty_candidate_set_is_refused() -> None:
    sc = synthetic_scored([])
    for pid in ("P1", "P3", "P7", "P8"):
        with pytest.raises(NoCandidatesError):
            _decide(make_policy(pid), synthetic_context(sc, protocol=_proto()))


def test_planning_policies_need_reset() -> None:
    ctx = _ctx()
    st = prepare("systemlab", "R1", 0)
    opened = st.world.truth.open(_KEY)
    for pol in (make_policy("P2"), make_policy("P9"), make_policy("P10", opened_truth=opened)):
        with pytest.raises(RuntimeError, match="reset"):
            _decide(pol, ctx)


# --------------------------------------------------------------------------- oracle, registry


def test_oracle_construction_and_sealing() -> None:
    st = prepare("microclimate", "R3", 0)  # truth outside the set: targets the residual
    opened = st.world.truth.open(_KEY)
    assert opened["true_hypothesis_id"] is None
    pol = OraclePolicy.from_sealed(st.world.truth, _KEY)
    pol.reset(st.init)
    assert pol._target == [h.id for h in st.hs.hypotheses].index(st.hs.residual.id)
    with pytest.raises(PermissionError):
        OraclePolicy.from_sealed(st.world.truth, "not a key")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OraclePolicy({"payload": {}})
    with pytest.raises(TypeError, match="opened_truth"):
        make_policy("P10")
    in_set = prepare("microclimate", "R10", 0)
    tid = in_set.world.truth.true_hypothesis_id(_KEY)
    pol2 = make_policy("P10", opened_truth=in_set.world.truth.open(_KEY))
    pol2.reset(in_set.init)
    d = pol2.decide(in_set.context(in_set.score(need_u_param=False)), make_rng(0, "oracle"))
    assert d.mode == "oracle" and d.propensity == 1.0
    assert tid is not None and tid not in d.reason and tid not in repr(pol2)
    canonical_json(d.as_record())


def test_registry() -> None:
    assert POLICY_IDS == (*(f"P{i}" for i in range(1, 11)), "P8a")
    assert set(POLICY_NAMES) == set(POLICY_IDS)
    assert POLICY_NAMES["P8"] == "guarded EIG/cost" and POLICY_NAMES["P10"] == "oracle"
    for pid in POLICY_IDS[:9]:
        pol = make_policy(pid)
        assert pol.id == pid and pol.name == POLICY_NAMES[pid] and pol.version
        canonical_json(pol.config())
    assert make_policy("P9", max_plan_length=4).config() == {"max_plan_length": 4}
    assert make_policy("P8", alarm_audit_eps=0.7).config() == {"alarm_audit_eps": 0.7}
    with pytest.raises(KeyError):
        make_policy("P11")
    assert make_policy("P1") is not make_policy("P1")


def test_p8a_adaptive_audit_eps() -> None:
    from probe.run.protocol import RunProtocol

    pol = make_policy("P8a")
    proto = RunProtocol(budget=100, warmup=0, audit_eps=0.5)
    assert pol.effective_eps(proto, alarm=False) == 0.03  # protocol ε ignored
    assert pol.effective_eps(proto, alarm=True) == 1.0
