"""Shared helpers for the selection tests: worlds after a stratified warm-up, decision
contexts, a minimal closed loop (decide -> execute -> update), and synthetic score
tables for unit tests.

The closed loop mirrors what the M3 run loop does around a policy (score the
feasible, affordable candidates once with :func:`score_candidates`, hand the table
to the policy, execute, update every engine), without stopping rules: it runs a
fixed number of steps or until nothing feasible and affordable is left.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from probe.determinism import make_rng
from probe.engine.eig import log_prior_weights, posterior_weights
from probe.engine.factory import EngineModel, build_models
from probe.engine.ir import HypothesisSet
from probe.engine.predictive import StudentTPredictive
from probe.envs.base import Action, Candidate, Environment, EnvironmentFamily, World
from probe.envs.microclimate import FAMILY as MICROCLIMATE
from probe.envs.systemlab import FAMILY as SYSTEMLAB
from probe.run.protocol import RunProtocol
from probe.select.base import Decision, DecisionContext, PolicyInit, ScoredCandidates
from probe.select.scoring import requires_u_param, score_candidates

FAMILIES: dict[str, EnvironmentFamily] = {"microclimate": MICROCLIMATE, "systemlab": SYSTEMLAB}


@dataclass
class State:
    family: str
    regime: str
    seed: int
    world: World
    hs: HypothesisSet
    models: list[EngineModel]
    protocol: RunProtocol
    init: PolicyInit
    counts: dict[str, int] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    step: int = 0

    @property
    def env(self) -> Environment:
        return self.world.env

    def weights(self) -> np.ndarray:
        ev = np.asarray([m.log_evidence() for m in self.models])
        return posterior_weights(log_prior_weights(self.hs), ev, self.protocol.tempering_eta)

    def budget_remaining(self) -> float:
        return self.protocol.budget - self.env.spent

    def score(self, need_u_param: bool | None = True) -> ScoredCandidates:
        return score_candidates(
            self.models,
            self.weights(),
            self.env.candidates(),
            self.protocol,
            need_u_param=need_u_param,
            budget_remaining=self.budget_remaining(),
        )

    def context(self, scored: ScoredCandidates, alarm: bool = False) -> DecisionContext:
        return DecisionContext(
            step=self.step,
            hypothesis_set=self.hs,
            weights=self.weights(),
            scored=scored,
            budget_remaining=self.budget_remaining(),
            spent=self.env.spent,
            protocol=self.protocol,
            stratum_counts=dict(self.counts),
            population=self.env.metadata().population,
            alarm=alarm,
            history=tuple(self.history),
        )

    def execute(self, cand: Candidate) -> None:
        obs = self.env.execute(Action(id=cand.action.id, params=dict(cand.action.params)))
        for m in self.models:
            m.update(obs.record, obs.value)
        self.counts[cand.stratum] = self.counts.get(cand.stratum, 0) + 1
        self.history.append(cand.action.id)
        self.step += 1


def make_protocol(family: str, regime: str, budget: str = "medium", **kw: object) -> RunProtocol:
    spec = FAMILIES[family].regimes()[regime]
    return RunProtocol(budget=spec.budgets[budget], warmup=spec.warmup, **kw)  # type: ignore[arg-type]


def prepare(
    family: str,
    regime: str,
    seed: int,
    protocol: RunProtocol | None = None,
    warmup: int | None = None,
) -> State:
    """World ``(regime, seed)`` with fresh engines, after the shared stratified warm-up
    (least-visited stratum first, uniform within, keyed generator). ``init`` holds
    the candidates at the moment the policy takes over."""
    fam = FAMILIES[family]
    proto = protocol or make_protocol(family, regime)
    world = fam.make_world(regime, seed)
    hs = fam.hypothesis_set(regime)
    st = State(family, regime, seed, world, hs, build_models(hs), proto, init=None)  # type: ignore[arg-type]
    n_warm = proto.warmup if warmup is None else warmup
    for k in range(n_warm):
        cands = [c for c in st.env.candidates() if c.feasible]
        least = min(st.counts.get(c.stratum, 0) for c in cands)
        pool = [c for c in cands if st.counts.get(c.stratum, 0) == least]
        rng = make_rng(seed, "select-tests", family, regime, "warmup", k)
        st.execute(pool[int(rng.integers(len(pool)))])
    st.init = PolicyInit(
        metadata=world.env.metadata(),
        hypothesis_set=hs,
        protocol=proto,
        initial_candidates=tuple(world.env.candidates()),
    )
    return st


def run_policy(
    policy: object,
    family: str,
    regime: str,
    seed: int,
    n_steps: int,
    protocol: RunProtocol | None = None,
    alarm: bool = False,
) -> tuple[State, list[Decision], list[ScoredCandidates]]:
    """Warm-up, ``policy.reset``, then up to ``n_steps`` decisions."""
    st = prepare(family, regime, seed, protocol)
    policy.reset(st.init)  # type: ignore[attr-defined]
    pid = policy.id  # type: ignore[attr-defined]
    rng = make_rng(seed, "select-tests", family, regime, "policy", pid)
    decisions: list[Decision] = []
    tables: list[ScoredCandidates] = []
    for _ in range(n_steps):
        scored = st.score(need_u_param=requires_u_param(policy, st.protocol))
        if scored.n == 0:
            break
        d = policy.decide(st.context(scored, alarm=alarm), rng)  # type: ignore[attr-defined]
        decisions.append(d)
        tables.append(scored)
        st.execute(scored.candidates[d.chosen_index])
    return st, decisions, tables


# --------------------------------------------------------------------------- synthetic


def synthetic_scored(
    u_model: Sequence[float],
    cost: Sequence[float] | None = None,
    strata: Sequence[str] | None = None,
    ids: Sequence[str] | None = None,
    u_param: Sequence[float] | None = None,
    pred_var: Sequence[float] | None = None,
) -> ScoredCandidates:
    """A hand-made score table (two dummy Student-t hypotheses) for unit tests.
    Candidates get ids ``a00, a01, ...`` (sorted) unless given."""
    n = len(u_model)
    ids = list(ids) if ids is not None else [f"a{i:02d}" for i in range(n)]
    assert ids == sorted(ids)
    strata = list(strata) if strata is not None else ["s0"] * n
    cost = list(cost) if cost is not None else [1.0] * n
    cands = tuple(
        Candidate(
            action=Action(id=ids[i], params={"i": i}),
            record={"x": float(i)},
            cost=float(cost[i]),
            stratum=strata[i],
        )
        for i in range(n)
    )
    pred = StudentTPredictive(np.full(n, 5.0), np.zeros(n), np.ones(n))
    up = np.asarray(u_param if u_param is not None else np.zeros(n), dtype=np.float64)
    pv = np.asarray(pred_var if pred_var is not None else np.ones(n), dtype=np.float64)
    return ScoredCandidates(
        candidates=cands,
        cost=np.asarray(cost, dtype=np.float64),
        u_model=np.asarray(u_model, dtype=np.float64),
        u_model_err=np.zeros(n),
        u_param=up,
        pred_var=pv,
        predictives=(pred, pred),
        u_param_available=True,
        u_model_upper=math.log(2.0),
    )


def synthetic_context(
    scored: ScoredCandidates,
    protocol: RunProtocol | None = None,
    counts: dict[str, int] | None = None,
    alarm: bool = False,
    hs: HypothesisSet | None = None,
) -> DecisionContext:
    hset = hs if hs is not None else MICROCLIMATE.hypothesis_set("R10")
    return DecisionContext(
        step=0,
        hypothesis_set=hset,
        weights=np.full(2, 0.5),
        scored=scored,
        budget_remaining=100.0,
        spent=0.0,
        protocol=protocol or RunProtocol(budget=100.0, warmup=0),
        stratum_counts=counts or {},
        population=tuple(sorted({c.stratum for c in scored.candidates})),
        alarm=alarm,
        history=(),
    )
