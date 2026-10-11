"""The headless run loop: the core of the Run Orchestrator (handover §12 Part F).

Per step, in a fixed order that makes the design auditable (H.8):

1. candidates from the environment; feasible subset within the remaining budget;
2. :func:`~probe.select.scoring.score_candidates`, computed once and shared by
   every policy;
3. the policy's :class:`~probe.select.base.Decision` is **committed** (hashed and
   emitted, with each hypothesis' predictive at the chosen action) *before*
   execution;
4. the environment executes exactly the committed action;
5. every model's ``log_predictive`` is taken **before** its ``update`` (the
   prequential evidence and the e-process numerators);
6. posterior weights, diagnostics and alarms, then the stopping rule.

Warm-up decisions come from a stratified design whose randomness is keyed
*without* the policy id, so every policy in a benchmark cell receives the same
warm-up (common random numbers). Policy randomness is keyed with the policy id.
Nothing here ever touches a :class:`~probe.truth.SealedTruth`.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from probe.determinism import content_hash, make_rng, rng_state_hash
from probe.diagnostics import AlarmMonitor, nesting_matrix, observation_diagnostics
from probe.engine.eig import log_prior_weights, posterior_weights
from probe.engine.factory import build_models
from probe.engine.ir import HypothesisSet, ir_hash
from probe.envs.base import Environment
from probe.run.protocol import RunProtocol
from probe.select.base import Decision, DecisionContext, PolicyInit, ScoredCandidates
from probe.select.designs import least_sampled_draw
from probe.select.scoring import requires_u_param, score_candidates
from probe.stop import EProcessTracker, StopDecision, StoppingRule, StopState, json_float
from probe.stop.rules import information_rate

__all__ = ["RunRecord", "StepRecord", "run_experiment"]

Event = Mapping[str, Any]


@dataclass(frozen=True)
class StepRecord:
    step: int
    mode: str
    policy_id: str
    action_id: str
    decision_hash: str
    propensity: float
    tie_set: tuple[str, ...]
    reason: str
    rng_state_before: str
    rng_state_after: str
    n_feasible: int
    chosen_u_model: float
    chosen_cost: float
    top: tuple[dict[str, Any], ...]
    predictions: dict[str, dict[str, float]]
    value: float | int | None
    missingness: str
    qc_flags: tuple[str, ...]
    cost: float
    clock: float
    weights: dict[str, float]
    log_e_leader: dict[str, float | str]
    diagnostics: dict[str, Any] | None
    alarm: bool


@dataclass(frozen=True)
class RunRecord:
    family: str
    env_version: str
    protocol: dict[str, Any]
    protocol_hash: str
    hypothesis_set_hash: str
    hypothesis_ids: tuple[str, ...]
    truth_commitment: str
    policy_id: str
    policy_version: str
    seed: int
    steps: tuple[StepRecord, ...]
    stop: dict[str, Any]
    spent: float
    n_obs: int
    final_weights: dict[str, float]
    decision_hash: str
    trajectory_hash: str
    nesting_known: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str | None:
        v = self.stop.get("verdict")
        return None if v is None else str(v)

    @property
    def leader(self) -> str | None:
        v = self.stop.get("leader")
        return None if v is None else str(v)


def _predictions(scored: ScoredCandidates, idx: int, ids: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, dict[str, float]] = {}
    for hid, p in zip(ids, scored.predictives, strict=True):
        sub = p.take([idx])
        out[hid] = {"mean": float(sub.mean()[0]), "var": json_float(float(sub.variance()[0]))}  # type: ignore[dict-item]
    return out


def _top(scored: ScoredCandidates, k: int = 5) -> tuple[dict[str, Any], ...]:
    rate = scored.u_model / scored.cost
    order = sorted(range(scored.n), key=lambda i: (-float(rate[i]), scored.action_ids()[i]))[:k]
    return tuple(
        {
            "action_id": scored.candidates[i].action.id,
            "u_model": float(scored.u_model[i]),
            "cost": float(scored.cost[i]),
            "rate": float(rate[i]),
        }
        for i in order
    )


def _warmup_decision(
    step: int, scored: ScoredCandidates, counts: Mapping[str, int], rng: np.random.Generator
) -> Decision:
    before = rng_state_hash(rng)
    draw, _ = least_sampled_draw(scored, counts, rng)
    i = int(draw.index)
    return Decision(
        step=step,
        policy_id="warmup",
        policy_version="1",
        chosen_index=i,
        action_id=scored.candidates[i].action.id,
        score=float("nan"),
        scores=np.full(scored.n, np.nan),
        propensity=float(draw.probs[i]),
        tie_set=(),
        mode="warmup",
        rng_state_before=before,
        rng_state_after=rng_state_hash(rng),
        reason=f"shared stratified warm-up: least-sampled stratum {draw.stratum!r}",
    )


def _decision_payload(d: Decision) -> dict[str, Any]:
    return {
        "step": d.step,
        "policy": d.policy_id,
        "action_id": d.action_id,
        "mode": d.mode,
        "propensity": float(d.propensity),
        "tie_set": list(d.tie_set),
        "rng_before": d.rng_state_before,
    }


def run_experiment(
    env: Environment,
    hypothesis_set: HypothesisSet,
    protocol: RunProtocol,
    policy: Any,
    seed: int,
    *,
    on_event: Callable[[Event], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    nesting: bool = True,
) -> RunRecord:
    """Run one experiment to a stop. ``policy`` implements
    :class:`~probe.select.base.Policy`. Deterministic given ``(env, seed, policy)``."""
    meta = env.metadata()
    emit = on_event or (lambda _e: None)
    models = build_models(hypothesis_set)
    ids = tuple(m.hypothesis_id for m in models)
    residual = hypothesis_set.residual.id
    log_prior = log_prior_weights(hypothesis_set)
    weights = posterior_weights(log_prior, np.zeros(len(models)), protocol.tempering_eta)
    tracker = EProcessTracker(ids)
    alarms = AlarmMonitor()
    rule = StoppingRule(protocol)
    need_u_param = requires_u_param(policy, protocol)
    warm_rng = make_rng(seed, meta.family, "warmup")
    pol_rng = make_rng(seed, meta.family, "policy", policy.id)
    first = env.candidates()
    policy.reset(
        PolicyInit(
            metadata=meta,
            hypothesis_set=hypothesis_set,
            protocol=protocol,
            initial_candidates=tuple(first),
        )
    )
    nests: np.ndarray | None = None
    if nesting:
        recs = [c.record for c in first if c.feasible]
        if recs:
            nests = nesting_matrix(models, recs)
    emit({"type": "run_started", "policy": policy.id, "hypotheses": list(ids), "seed": seed})

    steps: list[StepRecord] = []
    counts: Counter[str] = Counter()
    history: list[str] = []
    stop: StopDecision | None = None
    step = 0
    candidates = first
    while True:
        budget_left = protocol.budget - env.spent
        scored = score_candidates(
            models,
            weights,
            candidates,
            protocol,
            need_u_param=need_u_param,
            budget_remaining=budget_left,
        )
        user_stop = bool(stop_requested and stop_requested())
        state = StopState(
            step=step,
            hypothesis_ids=ids,
            weights=weights,
            residual_id=residual,
            n_obs=sum(m.n_obs for m in models[:1]),
            budget_remaining=budget_left,
            feasible_count=scored.n,
            max_score=information_rate(scored.u_model, scored.cost, scored.u_model_err)
            if scored.n
            else None,
            log_e=tracker.running_max_log_e() if tracker.n_steps else None,
            nests=nests,
            alarm=alarms.raised,
            exhausted=env.exhausted,
            user_stop=user_stop,
        )
        if step >= protocol.warmup or scored.n == 0 or user_stop:
            stop = rule.evaluate(state)
            if step >= protocol.max_steps and not stop.stop:
                stop = StopDecision(True, "max_steps", "undecided", ids[int(np.argmax(weights))])
            if stop.stop:
                break
        # --- decide and commit
        if step < protocol.warmup:
            decision = _warmup_decision(step, scored, counts, warm_rng)
        else:
            ctx = DecisionContext(
                step=step,
                hypothesis_set=hypothesis_set,
                weights=weights,
                scored=scored,
                budget_remaining=budget_left,
                spent=env.spent,
                protocol=protocol,
                stratum_counts=dict(counts),
                population=meta.population,
                alarm=alarms.raised,
                history=tuple(history),
            )
            decision = policy.decide(ctx, pol_rng)
        idx = decision.chosen_index
        chosen = scored.candidates[idx]
        if chosen.action.id != decision.action_id:
            raise RuntimeError("decision index and action id disagree")
        dhash = content_hash(_decision_payload(decision))
        preds = _predictions(scored, idx, ids)
        emit(
            {
                "type": "decision",
                "step": step,
                "action_id": decision.action_id,
                "mode": decision.mode,
                "decision_hash": dhash,
                "reason": decision.reason,
            }
        )
        # --- execute exactly the committed action
        obs = env.execute(chosen.action)
        if obs.action_id != decision.action_id:
            raise RuntimeError("environment executed a different action than committed")
        # --- prequential terms before updates
        diag = None
        if obs.value is not None:
            lp = np.array([m.log_predictive(obs.record, obs.value) for m in models])
            pre_preds = tuple(p.take([idx]) for p in scored.predictives)
            diag = observation_diagnostics(pre_preds, weights, obs.value, seed=seed, step=step)
        else:
            lp = None
        for m in models:
            m.update(obs.record, obs.value)
        tracker.observe_models(lp, models)
        weights = posterior_weights(
            log_prior, np.array([m.log_evidence() for m in models]), protocol.tempering_eta
        )
        if diag is not None:
            alarms.update(diag, float(weights[ids.index(residual)]))
        counts[chosen.stratum] += 1
        history.append(chosen.action.id)
        lead = ids[int(np.argmax(weights))]
        steps.append(
            StepRecord(
                step=step,
                mode=decision.mode,
                policy_id=decision.policy_id,
                action_id=decision.action_id,
                decision_hash=dhash,
                propensity=float(decision.propensity),
                tie_set=tuple(decision.tie_set),
                reason=decision.reason,
                rng_state_before=decision.rng_state_before,
                rng_state_after=decision.rng_state_after,
                n_feasible=scored.n,
                chosen_u_model=float(scored.u_model[idx]),
                chosen_cost=float(scored.cost[idx]),
                top=_top(scored),
                predictions=preds,
                value=obs.value,
                missingness=obs.missingness,
                qc_flags=tuple(obs.qc_flags),
                cost=float(obs.cost),
                clock=float(obs.clock),
                weights={h: float(w) for h, w in zip(ids, weights, strict=True)},
                log_e_leader={r: json_float(v) for r, v in tracker.versus_rivals(lead).items()},
                diagnostics=diag.to_dict() if diag is not None else None,
                alarm=alarms.raised,
            )
        )
        emit(
            {
                "type": "observation",
                "step": step,
                "value": obs.value,
                "missingness": obs.missingness,
                "weights": steps[-1].weights,
            }
        )
        step += 1
        candidates = env.candidates()

    assert stop is not None
    stop_d = {
        "reason": stop.reason,
        "verdict": stop.verdict,
        "leader": stop.leader,
        "details": _jsonable_details(stop.details),
    }
    emit({"type": "stopped", **stop_d})
    seq = [s.decision_hash for s in steps]
    traj = [[s.action_id, s.value, s.missingness] for s in steps]
    proto = protocol.model_dump(mode="json")
    return RunRecord(
        family=meta.family,
        env_version=meta.version,
        protocol=proto,
        protocol_hash=protocol.protocol_hash(),
        hypothesis_set_hash=ir_hash(hypothesis_set),
        hypothesis_ids=ids,
        truth_commitment=env.truth_commitment(),
        policy_id=policy.id,
        policy_version=policy.version,
        seed=seed,
        steps=tuple(steps),
        stop=stop_d,
        spent=float(env.spent),
        n_obs=int(models[0].n_obs),
        final_weights={h: float(w) for h, w in zip(ids, weights, strict=True)},
        decision_hash=content_hash(seq),
        trajectory_hash=content_hash(traj),
        nesting_known=nests is not None,
        extra={"residual_id": residual},
    )


def _jsonable_details(d: Mapping[str, Any]) -> dict[str, Any]:
    def conv(v: Any) -> Any:
        if isinstance(v, float | np.floating):
            f = float(v)
            return f if math.isfinite(f) else str(f)
        if isinstance(v, np.ndarray):
            return [conv(x) for x in v.tolist()]
        if isinstance(v, Mapping):
            return {str(k): conv(x) for k, x in v.items()}
        if isinstance(v, list | tuple):
            return [conv(x) for x in v]
        if isinstance(v, np.integer | np.bool_):
            return v.item()
        return v

    return {str(k): conv(v) for k, v in d.items()}
