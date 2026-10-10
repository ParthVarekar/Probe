"""Reference exact grid inference for system-lab tests (independent of probe.engine).

A deliberately small, obviously-correct filter: posterior over grid points per
hypothesis, prequential log evidence, model posterior and the discrete
model-indicator EIG U_M(d) = H[Σ π_m p_m] − Σ π_m H[p_m]. Used to check that the
regimes behave as pre-registered (well-specified worlds are identified, the
residual can win out of set, R5 stays exactly tied, R10 separates policies).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import special

from probe.engine.ir import GridHypothesis, HypothesisSet, Record
from probe.engine.mechanisms import get_mechanism
from probe.envs.base import Candidate, Environment


class RefGridModel:
    def __init__(self, hyp: GridHypothesis, support_size: int) -> None:
        self.id = hyp.id
        self.mech = get_mechanism(hyp.mechanism, support_size)
        self.params = hyp.grid_table()
        self.state = self.mech.initial_state(self.params)
        g = hyp.grid_size()
        self.logw = np.full(g, -np.log(g))
        self.log_evidence = 0.0

    def weights(self) -> np.ndarray:
        return np.exp(self.logw - special.logsumexp(self.logw))

    def predictive(self, record: Record) -> np.ndarray:
        probs = self.mech.predict(self.state, self.params, record)
        return self.weights() @ probs

    def update(self, record: Record, y: int) -> None:
        probs = self.mech.predict(self.state, self.params, record)
        with np.errstate(divide="ignore"):
            lp = np.log(probs[:, y])
        joint = self.logw + lp
        le = special.logsumexp(joint)
        self.log_evidence += float(le)
        if np.isfinite(le):
            self.logw = joint - le
        self.state = self.mech.advance(self.state, self.params, record)


class RefPosterior:
    def __init__(self, hs: HypothesisSet) -> None:
        s = (hs.outcome.max_count or 0) + 1
        self.models = []
        for h in hs.hypotheses:
            assert isinstance(h, GridHypothesis)
            self.models.append(RefGridModel(h, s))
        self.log_prior = np.log([h.prior_weight for h in hs.hypotheses])
        self.ids = [h.id for h in hs.hypotheses]

    def post(self) -> np.ndarray:
        lp = self.log_prior + np.asarray([m.log_evidence for m in self.models])
        return np.exp(lp - special.logsumexp(lp))

    def update(self, record: Record, y: int) -> None:
        for m in self.models:
            m.update(record, y)

    def eig(self, records: Sequence[Record]) -> np.ndarray:
        pi = self.post()
        out = np.zeros(len(records))
        for j, r in enumerate(records):
            p = np.stack([m.predictive(r) for m in self.models])  # (K, S)
            mix = pi @ p
            h_mix = -np.sum(special.xlogy(mix, mix))
            h_each = -np.sum(special.xlogy(p, p), axis=1)
            out[j] = max(0.0, h_mix - float(pi @ h_each))
        return out


@dataclass
class RunResult:
    decided: str | None
    spent: float
    post: np.ndarray
    steps: int
    ids: list[str]

    @property
    def leader(self) -> str:
        return self.ids[int(np.argmax(self.post))]


def _stratified(
    cands: Sequence[Candidate], counts: dict[str, int], rng: np.random.Generator
) -> Candidate:
    least = min(counts.get(c.stratum, 0) for c in cands)
    pool = [c for c in cands if counts.get(c.stratum, 0) == least]
    return pool[int(rng.integers(len(pool)))]


def run_policy(
    env: Environment,
    hs: HypothesisSet,
    policy: str,
    budget: float,
    rng: np.random.Generator,
    *,
    pi_star: float = 0.95,
    warmup: int = 0,
    warmup_rng: np.random.Generator | None = None,
    min_obs_none_fits: int = 8,
) -> RunResult:
    """Run ``policy`` ∈ {random, stratified, eig_cost, eig} after ``warmup`` shared
    stratified steps (drawn from ``warmup_rng``, so identical across policies) until
    max posterior ≥ ``pi_star`` or the budget is spent. As in ``RunProtocol``
    (``min_obs_before_none_fits``), the residual cannot end a run before
    ``min_obs_none_fits`` observations."""
    post = RefPosterior(hs)
    wrng = warmup_rng if warmup_rng is not None else np.random.default_rng(0)
    steps = 0
    counts: dict[str, int] = {}
    while True:
        p = post.post()
        leader = post.ids[int(np.argmax(p))]
        early_none_fits = leader == "residual" and steps < min_obs_none_fits
        if p.max() >= pi_star and steps >= warmup and not early_none_fits:
            return RunResult(leader, env.spent, p, steps, post.ids)
        cands: list[Candidate] = [
            c for c in env.candidates() if c.feasible and env.spent + c.cost <= budget
        ]
        if not cands:
            return RunResult(None, env.spent, p, steps, post.ids)
        if steps < warmup:
            choice = _stratified(cands, counts, wrng)
        elif policy == "random":
            choice = cands[int(rng.integers(len(cands)))]
        elif policy == "stratified":
            choice = _stratified(cands, counts, rng)
        elif policy in ("eig", "eig_cost"):
            u = post.eig([c.record for c in cands])
            score = u / np.asarray([c.cost for c in cands]) if policy == "eig_cost" else u
            choice = cands[int(np.argmax(score))]
        else:
            raise ValueError(policy)
        obs = env.execute(choice.action)
        counts[choice.stratum] = counts.get(choice.stratum, 0) + 1
        assert obs.value is not None
        post.update(obs.record, int(obs.value))
        steps += 1
