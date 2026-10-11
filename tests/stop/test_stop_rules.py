"""Stopping rules (handover §12 H.6): every branch, the precedence between
branches, determinism of the leader, and the JSON-able details."""

import math
from dataclasses import replace

import numpy as np
import pytest

from probe.determinism import canonical_json
from probe.run.protocol import RunProtocol
from probe.stop.rules import StoppingRule, StopState, information_rate

IDS = ("H1", "H2", "H3", "H0")
RES = "H0"
BIG = 10.0  # log E well above log(1/0.05) ≈ 3.0
SMALL = 1.0


def proto(**kw) -> RunProtocol:
    base = {
        "budget": 100.0,
        "warmup": 4,
        "max_steps": 50,
        "min_obs_before_inconclusive": 5,
        "min_obs_before_none_fits": 8,
    }
    base.update(kw)
    return RunProtocol(**base)


def log_e(leader: int = 0, value: float = BIG, others: float = -5.0) -> np.ndarray:
    m = np.full((4, 4), others)
    m[leader, :] = value
    np.fill_diagonal(m, 0.0)
    return m


def state(**kw) -> StopState:
    base = {
        "step": 20,
        "hypothesis_ids": IDS,
        "weights": np.array([0.4, 0.3, 0.2, 0.1]),
        "residual_id": RES,
        "n_obs": 20,
        "budget_remaining": 50.0,
        "feasible_count": 10,
        "max_score": 0.5,
        "log_e": log_e(),
    }
    base.update(kw)
    return StopState(**base)


W_H1 = np.array([0.97, 0.01, 0.01, 0.01])
W_RES = np.array([0.01, 0.01, 0.01, 0.97])
W_RES_LEADS = np.array([0.25, 0.2, 0.15, 0.4])


# --------------------------------------------------------------------------- rule 1


def test_decision_reached_with_posterior_and_eprocess():
    d = StoppingRule(proto()).evaluate(state(weights=W_H1))
    assert d.stop and d.reason == "decision_reached" and d.verdict == "hypothesis"
    assert d.leader == "H1"
    det = d.details
    assert det["rule"] == "1_decision"
    assert det["leader_weight"] == pytest.approx(0.97)
    assert det["log_e_threshold"] == pytest.approx(math.log(20.0))
    assert det["eprocess_passed"] is True
    assert set(det["log_e_vs_rivals"]) == {"H2", "H3", "H0"}
    canonical_json(det)


def test_posterior_alone_is_not_enough_when_eprocess_required():
    le = log_e()
    le[0, 2] = SMALL  # H1 vs H3 below 1/α
    d = StoppingRule(proto()).evaluate(state(weights=W_H1, log_e=le))
    assert not d.stop
    assert d.details["eprocess_passed"] is False
    assert d.details["checks"]["decision"] is False
    assert d.leader == "H1"


def test_eprocess_threshold_is_inclusive_and_alpha_dependent():
    le = log_e(value=math.log(20.0))
    assert StoppingRule(proto(alpha=0.05)).evaluate(state(weights=W_H1, log_e=le)).stop
    assert not StoppingRule(proto(alpha=0.01)).evaluate(state(weights=W_H1, log_e=le)).stop


def test_untracked_eprocess_blocks_decision_unless_not_required():
    s = state(weights=W_H1, log_e=None)
    d = StoppingRule(proto()).evaluate(s)
    assert not d.stop and d.details["eprocess_passed"] is None
    d2 = StoppingRule(proto(require_eprocess=False)).evaluate(s)
    assert d2.stop and d2.reason == "decision_reached" and d2.leader == "H1"


def test_nesting_rivals_are_exempt_from_the_eprocess():
    le = log_e()
    le[0, 1] = -3.0  # H1 ⊂ H2: this e-process cannot grow
    le[0, 2] = -2.0  # H1 ⊂ H3
    nests = np.eye(4, dtype=bool)
    nests[0, 1] = nests[0, 2] = True
    d = StoppingRule(proto()).evaluate(state(weights=W_H1, log_e=le, nests=nests))
    assert d.stop and d.reason == "decision_reached"
    assert d.details["eprocess_exempt_nesting"] == ["H2", "H3"]
    assert set(d.details["log_e_vs_rivals"]) == {"H0"}
    # without the nesting information the decision is withheld
    assert not StoppingRule(proto()).evaluate(state(weights=W_H1, log_e=le)).stop
    # nesting in the other direction (H2 ⊂ H1) does not exempt anything
    nests2 = np.eye(4, dtype=bool)
    nests2[1, 0] = True
    assert not StoppingRule(proto()).evaluate(state(weights=W_H1, log_e=le, nests=nests2)).stop


def test_infinite_e_values_are_reported_as_strings():
    le = log_e(value=math.inf)
    d = StoppingRule(proto()).evaluate(state(weights=W_H1, log_e=le))
    assert d.stop and d.details["log_e_vs_rivals"]["H2"] == "inf"
    canonical_json(d.details)


def test_residual_at_pi_star_is_none_fits_never_a_hypothesis():
    d = StoppingRule(proto()).evaluate(state(weights=W_RES))
    assert d.stop and d.reason == "none_fits" and d.verdict == "none_fits"
    assert d.leader == RES and d.details["rule"] == "1_residual_at_pi_star"


def test_residual_at_pi_star_waits_for_min_obs():
    s = state(weights=W_RES, n_obs=5, step=5)
    d = StoppingRule(proto()).evaluate(s)
    assert not d.stop
    assert d.details["residual_leads"] is True


# --------------------------------------------------------------------------- rule 2


@pytest.mark.parametrize(
    "kw",
    [
        {"budget_remaining": 0.0},
        {"budget_remaining": -1e-9},
        {"feasible_count": 0},
        {"exhausted": True},
    ],
)
def test_budget_exhausted_is_undecided(kw):
    if kw.get("feasible_count") == 0:
        kw = {**kw, "max_score": None}
    d = StoppingRule(proto()).evaluate(state(**kw))
    assert d.stop and d.reason == "budget_exhausted" and d.verdict == "undecided"
    assert d.leader == "H1"


def test_max_steps_is_undecided():
    d = StoppingRule(proto(max_steps=20)).evaluate(state(step=20))
    assert d.stop and d.reason == "max_steps" and d.verdict == "undecided"
    assert not StoppingRule(proto(max_steps=21)).evaluate(state(step=20)).stop


def test_decision_takes_precedence_over_budget():
    d = StoppingRule(proto()).evaluate(state(weights=W_H1, budget_remaining=0.0))
    assert d.reason == "decision_reached"
    d = StoppingRule(proto()).evaluate(state(weights=W_RES, budget_remaining=0.0))
    assert d.reason == "none_fits" and d.details["rule"] == "1_residual_at_pi_star"


def test_budget_takes_precedence_over_max_steps():
    d = StoppingRule(proto(max_steps=20)).evaluate(state(step=20, budget_remaining=0.0))
    assert d.reason == "budget_exhausted"


# --------------------------------------------------------------------------- rule 3


def test_nothing_informative_left_is_inconclusive():
    p = proto(eps_stop=1e-3)
    d = StoppingRule(p).evaluate(state(max_score=5e-4))
    assert d.stop and d.reason == "inconclusive" and d.verdict == "inconclusive"
    assert d.details["max_score"] == 5e-4 and d.details["eps_stop"] == 1e-3
    assert not StoppingRule(p).evaluate(state(max_score=1e-3)).stop  # strict <


def test_inconclusive_needs_min_obs_and_no_leader_at_pi_star():
    p = proto(eps_stop=1e-3)
    assert not StoppingRule(p).evaluate(state(max_score=0.0, n_obs=4, step=4)).stop
    le = log_e(value=SMALL)  # leader at π* but e-process short: not inconclusive
    d = StoppingRule(p).evaluate(state(weights=W_H1, log_e=le, max_score=0.0))
    assert not d.stop and d.details["checks"]["inconclusive"] is False


def test_budget_takes_precedence_over_inconclusive():
    d = StoppingRule(proto(eps_stop=1e-3)).evaluate(state(max_score=0.0, exhausted=True))
    assert d.reason == "budget_exhausted"


# --------------------------------------------------------------------------- rule 4


def test_residual_leading_after_min_obs_is_none_fits():
    d = StoppingRule(proto()).evaluate(state(weights=W_RES_LEADS))
    assert d.stop and d.reason == "none_fits" and d.verdict == "none_fits"
    assert d.leader == RES and d.details["rule"] == "4_residual_leads"
    early = state(weights=W_RES_LEADS, n_obs=7, step=7)
    assert not StoppingRule(proto()).evaluate(early).stop


def test_alarm_is_none_fits_after_min_obs():
    d = StoppingRule(proto()).evaluate(state(alarm=True))
    assert d.stop and d.reason == "none_fits" and d.details["rule"] == "4_alarm"
    assert d.leader == "H1"
    assert not StoppingRule(proto()).evaluate(state(alarm=True, n_obs=3, step=3)).stop


def test_inconclusive_takes_precedence_over_inadequacy():
    d = StoppingRule(proto(eps_stop=1e-3)).evaluate(
        state(weights=W_RES_LEADS, alarm=True, max_score=0.0)
    )
    assert d.reason == "inconclusive"
    assert d.details["checks"]["inadequate"] is True


def test_residual_tied_with_a_hypothesis_does_not_lead():
    w = np.array([0.35, 0.15, 0.15, 0.35])
    d = StoppingRule(proto()).evaluate(state(weights=w))
    assert not d.stop
    assert d.leader == "H1" and d.details["residual_leads"] is False
    # within the relative tie tolerance as well
    w2 = np.array([0.35, 0.15, 0.15, 0.35 * (1 + 1e-12)])
    w2 = w2 / w2.sum()
    d2 = StoppingRule(proto()).evaluate(state(weights=w2))
    assert d2.leader == "H1" and not d2.stop


def test_ties_between_hypotheses_go_to_set_order():
    w = np.array([0.1, 0.4, 0.4, 0.1])
    assert StoppingRule(proto()).evaluate(state(weights=w)).leader == "H2"


# --------------------------------------------------------------------------- rule 5


def test_user_stop_is_undecided_and_last():
    d = StoppingRule(proto()).evaluate(state(user_stop=True))
    assert d.stop and d.reason == "user_stopped" and d.verdict == "undecided"
    assert StoppingRule(proto()).evaluate(state(user_stop=True, weights=W_H1)).reason == (
        "decision_reached"
    )
    assert StoppingRule(proto()).evaluate(state(user_stop=True, alarm=True)).reason == "none_fits"


def test_go_on_carries_the_numbers():
    d = StoppingRule(proto()).evaluate(state())
    assert not d.stop and d.reason is None and d.verdict is None
    assert d.leader == "H1"
    det = d.details
    assert det["posterior"] == {"H1": 0.4, "H2": 0.3, "H3": 0.2, "H0": 0.1}
    assert det["rule"] is None
    assert not any(det["checks"].values())
    canonical_json(det)


def test_evaluation_is_deterministic():
    s = state(weights=W_RES_LEADS, alarm=True)
    a = StoppingRule(proto()).evaluate(s)
    b = StoppingRule(proto()).evaluate(replace(s))
    assert canonical_json(a.details) == canonical_json(b.details)


# --------------------------------------------------------------------------- state checks


@pytest.mark.parametrize(
    "kw",
    [
        {"weights": np.array([0.5, 0.5, 0.5, 0.5])},
        {"weights": np.array([0.5, 0.5, 0.0])},
        {"weights": np.array([1.1, -0.1, 0.0, 0.0])},
        {"residual_id": "nope"},
        {"hypothesis_ids": ("H1", "H1", "H3", "H0")},
        {"log_e": np.zeros((3, 3))},
        {"log_e": np.full((4, 4), math.nan)},
        {"nests": np.zeros((2, 2), dtype=bool)},
        {"max_score": -1.0},
        {"max_score": math.inf},
        {"budget_remaining": math.nan},
        {"n_obs": -1},
    ],
)
def test_stop_state_validation(kw):
    with pytest.raises(ValueError):
        state(**kw)


def test_information_rate():
    assert information_rate([], []) is None
    assert information_rate([0.2, 0.3], [2.0, 1.0]) == pytest.approx(0.3)
    # the error estimate is added (conservative for "nothing informative left")
    assert information_rate([0.2, 0.3], [2.0, 1.0], [0.4, 0.0]) == pytest.approx(0.3)
    assert information_rate([0.2, 0.3], [1.0, 1.0], [0.4, 0.0]) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        information_rate([0.1], [0.0])
    with pytest.raises(ValueError):
        information_rate([math.nan], [1.0])
    with pytest.raises(ValueError):
        information_rate([0.1, 0.2], [1.0])
