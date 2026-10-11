"""Integration: both environment families driven through the real inference/EIG engine.

For each family (microclimate: ``linear_gaussian`` hypotheses, NIG engines and
Student-t predictives; system lab: ``finite_grid`` hypotheses, grid engines and
discrete predictives) and several regimes and seeds, a short closed loop runs exactly
as the M3 run loop will:

1. build the world and its hypothesis set, and one engine per hypothesis with
   :func:`probe.engine.factory.build_models`;
2. a deterministic stratified warm-up (``RegimeSpec.warmup`` observations: least
   visited stratum first, uniform within it from a counter-keyed generator);
3. then greedy model-discrimination steps: among feasible (and, when a budget is
   given, affordable) candidates, take the maximum of U_M(d) =
   :func:`probe.engine.eig.model_indicator_eig` with the deterministic tie-break
   :func:`probe.select.base.argmax_with_ties`, and update *every* engine with the
   observation, missing ones included (``y = None``).

Checked along every path: U_M finite and within [0, H(π)] (and H(π) ≤ log K) with
finite error estimates, a candidate's U_M independent of the batch it is scored in
(bitwise), the secondary scores (Σ π U_θ|m, mixture variance) finite, posteriors
finite, non-negative and summing to 1, and the prequential identity
Σ log p(y_t | D_<t) = log evidence per engine. The whole trajectory is
deterministic: two runs in one process, and a run in a fresh interpreter with a
different ``PYTHONHASHSEED``, give identical action ids and posteriors.

Recovery of the planted truth (opened with an :class:`~probe.truth.EvaluatorKey`,
evaluator side only) is measured in the easy in-set regimes (microclimate R10 and
system lab R2/R10; 8 seeds in the fast suite, 24 seeds and more regimes in the slow
survey). The asserted thresholds sit below the rates observed when the test was
written, which are recorded next to each case; the measured rate is printed.

Per-step wall time (an EIG step: predictives and U_M of every candidate, the choice,
execution and the update of every engine) is printed (``pytest -s``) and only loosely
bounded. Measured on the development laptop (Windows 11): microclimate 13-14 ms
(96 candidates × 5 NIG engines; 5 ms in R10 with 52 × 3), system lab 20-25 ms
(42 candidates × 4 grid engines; 35-45 ms on R8's 1680-point residual grid, 10 ms in
R11). The system-lab time is dominated by the grid mechanisms' per-record ``predict``.

Integration found one engine defect, fixed in :mod:`probe.engine.eig`: the discrete
U_M used ``einsum``, so a candidate's score could differ by an ulp between being
scored alone and in a batch (and so depend on which other candidates were feasible).
"""

from __future__ import annotations

import itertools
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from probe.determinism import content_hash, make_rng
from probe.engine.eig import (
    EIGResult,
    entropy,
    log_prior_weights,
    mixture_moments,
    model_indicator_eig,
    posterior_weights,
    weighted_parameter_eig,
)
from probe.engine.factory import EngineModel, build_models
from probe.engine.grid import GridModel
from probe.engine.ir import Record
from probe.engine.nig import NIGModel
from probe.engine.predictive import Predictive
from probe.envs.base import Candidate, EnvironmentFamily
from probe.envs.microclimate import FAMILY as MICROCLIMATE
from probe.envs.systemlab import FAMILY as SYSTEMLAB
from probe.select.base import argmax_with_ties
from probe.truth import EvaluatorKey

FAMILIES: dict[str, EnvironmentFamily] = {"microclimate": MICROCLIMATE, "systemlab": SYSTEMLAB}
_KEY = EvaluatorKey("integration tests: score recovery of the planted truth")
_TIE_REL_TOL = 1e-9  # RunProtocol.tie_rel_tol default
_WEIGHT_SUM_TOL = 1e-12
_PREQUENTIAL_TOL = 1e-9


# --------------------------------------------------------------------------- the loop


@dataclass(frozen=True)
class Step:
    mode: str
    """``"warmup"`` or ``"eig"``."""
    action_id: str
    value: float | int | None
    weights: np.ndarray
    """Posterior over hypotheses after this observation (hypothesis-set order)."""
    u_max: float
    """Max U_M over the scored candidates (NaN in warm-up)."""
    n_candidates: int
    seconds: float
    """Wall time of an EIG step: predictives and U_M of every candidate, the choice,
    executing it and updating every engine (the test-only invariant checks excluded).
    Warm-up steps time only execution and the update."""


@dataclass(frozen=True)
class Episode:
    family: str
    regime: str
    seed: int
    hypothesis_ids: tuple[str, ...]
    steps: tuple[Step, ...]
    log_evidence: np.ndarray
    spent: float
    exhausted: bool
    """The world had no feasible action left when the loop ended."""

    @property
    def action_ids(self) -> list[str]:
        return [s.action_id for s in self.steps]

    @property
    def final_weights(self) -> np.ndarray:
        return self.steps[-1].weights

    @property
    def leader(self) -> str:
        return self.hypothesis_ids[int(np.argmax(self.final_weights))]

    def eig_seconds(self) -> list[float]:
        return [s.seconds for s in self.steps if s.mode == "eig"]

    def digest(self) -> str:
        """Hash of the decision sequence and the exact posterior bits."""
        return content_hash(
            {
                "actions": self.action_ids,
                "values": [s.value for s in self.steps],
                "weights": [s.weights.tobytes().hex() for s in self.steps],
                "evidence": self.log_evidence.tobytes().hex(),
            }
        )


def _check_weights(w: np.ndarray, k: int) -> None:
    assert w.shape == (k,)
    assert np.all(np.isfinite(w)) and np.all(w >= 0.0)
    assert abs(math.fsum(w.tolist()) - 1.0) <= _WEIGHT_SUM_TOL


def _weights(models: Sequence[EngineModel], log_prior: np.ndarray) -> np.ndarray:
    ev = np.asarray([m.log_evidence() for m in models])
    assert not np.any(np.isnan(ev)) and not np.any(np.isposinf(ev))
    w = posterior_weights(log_prior, ev)
    _check_weights(w, len(models))
    return w


def _warmup_choice(
    cands: Sequence[Candidate], counts: dict[str, int], rng: np.random.Generator
) -> int:
    """Least-visited stratum first; uniform within it (candidates are in id order)."""
    least = min(counts.get(c.stratum, 0) for c in cands)
    pool = [i for i, c in enumerate(cands) if counts.get(c.stratum, 0) == least]
    return pool[int(rng.integers(len(pool)))]


def _check_scores(
    models: Sequence[EngineModel],
    w: np.ndarray,
    records: Sequence[Record],
    preds: Sequence[Predictive],
    res: EIGResult,
    chosen: int,
) -> None:
    """The invariants of one scored step (test-only work, not timed)."""
    n, k = len(records), len(models)
    for arr in (res.value, res.error, res.raw):
        assert arr.shape == (n,) and np.all(np.isfinite(arr))
    assert res.upper_bound == pytest.approx(entropy(w), abs=1e-12)
    assert res.upper_bound <= math.log(k) + 1e-12
    assert np.all(res.value >= 0.0) and np.all(res.value <= res.upper_bound)
    assert np.all(res.error >= 0.0)
    slack = res.error + 1e-12
    assert np.all(res.raw >= -slack) and np.all(res.raw <= res.upper_bound + slack)
    # The secondary scores the run loop logs for every candidate.
    u_param = weighted_parameter_eig(models, w, records)
    assert np.all(np.isfinite(u_param)) and np.all(u_param >= 0.0)
    _, var = mixture_moments(preds, w)
    assert not np.any(np.isnan(var)) and np.all(var > 0.0)
    # A candidate's U_M does not depend on the batch it is scored in: the chosen one
    # alone, and every one in the reversed batch, get bitwise the same value and error.
    alone = model_indicator_eig([m.predictive([records[chosen]]) for m in models], w)
    assert (alone.raw[0], alone.error[0]) == (res.raw[chosen], res.error[chosen])
    rev = model_indicator_eig([m.predictive(records[::-1]) for m in models], w)
    np.testing.assert_array_equal(rev.raw[::-1], res.raw)
    np.testing.assert_array_equal(rev.error[::-1], res.error)


def run_episode(
    family: str,
    regime: str,
    seed: int,
    n_steps: int = 15,
    budget: float | None = None,
    check: bool = True,
) -> Episode:
    """Warm-up, then up to ``n_steps`` max-U_M steps. With ``budget`` only candidates
    whose cost fits the remaining budget are eligible; the episode also ends early when
    nothing feasible (or affordable) is left. ``check=False`` skips the per-step score
    invariants (not the posterior and prequential checks), e.g. for a replay."""
    fam = FAMILIES[family]
    spec = fam.regimes()[regime]
    hs = fam.hypothesis_set(regime)
    env = fam.make_world(regime, seed).env
    models = build_models(hs)
    k = len(models)
    assert [m.hypothesis_id for m in models] == [h.id for h in hs.hypotheses]
    log_prior = log_prior_weights(hs)
    w = _weights(models, log_prior)
    prequential = np.zeros(k)
    counts: dict[str, int] = {}
    steps: list[Step] = []
    while len(steps) < spec.warmup + n_steps:
        t0 = time.perf_counter()
        cands = [c for c in env.candidates() if c.feasible]
        if budget is not None:
            cands = [c for c in cands if env.spent + c.cost <= budget + 1e-9]
        if not cands:
            break
        ids = [c.action.id for c in cands]
        if len(steps) < spec.warmup:
            mode, u_max = "warmup", math.nan
            rng = make_rng(seed, "integration", family, regime, "warmup", len(steps))
            chosen = _warmup_choice(cands, counts, rng)
            elapsed = 0.0
        else:
            mode = "eig"
            records = [c.record for c in cands]
            preds = [m.predictive(records) for m in models]
            res = model_indicator_eig(preds, w)
            chosen, ties = argmax_with_ties(res.value, ids, _TIE_REL_TOL)
            elapsed = time.perf_counter() - t0
            u_max = float(res.value[chosen])
            assert ids == sorted(ids) and ids[chosen] in ties
            best = float(np.max(res.value))  # the tie rule of argmax_with_ties
            assert u_max >= best - (_TIE_REL_TOL * max(best, 1e-300) + 1e-300)
            if check:
                _check_scores(models, w, records, preds, res, chosen)
        cand = cands[chosen]
        t1 = time.perf_counter()
        obs = env.execute(cand.action)
        y = obs.value
        n_before = [m.n_obs for m in models]
        lps = []
        for m in models:
            if y is not None:  # prequential term, before the update
                lps.append(m.log_predictive(obs.record, y))
            m.update(obs.record, y)  # missing outcomes too
        w = _weights(models, log_prior)
        elapsed += time.perf_counter() - t1

        assert obs.action_id == cand.action.id and obs.record == cand.record
        assert (y is None) == (obs.missingness != "observed")
        assert [m.n_obs for m in models] == [n + (y is not None) for n in n_before]
        if y is not None:
            assert not any(math.isnan(lp) or lp == math.inf for lp in lps)
            prequential += np.asarray(lps)
        counts[cand.stratum] = counts.get(cand.stratum, 0) + 1
        steps.append(Step(mode, cand.action.id, y, w.copy(), u_max, len(cands), elapsed))

    ev = np.asarray([m.log_evidence() for m in models])
    for i, m in enumerate(models):
        if math.isinf(ev[i]):  # a refuted grid hypothesis
            assert isinstance(m, GridModel) and m.refuted and prequential[i] == -math.inf
        else:
            tol = _PREQUENTIAL_TOL
            assert prequential[i] == pytest.approx(ev[i], rel=tol, abs=tol)
    hids = tuple(h.id for h in hs.hypotheses)
    return Episode(family, regime, seed, hids, tuple(steps), ev, env.spent, env.exhausted)


def true_id(family: str, regime: str, seed: int) -> str | None:
    """Evaluator side: the planted hypothesis of world ``(regime, seed)``."""
    return FAMILIES[family].make_world(regime, seed).truth.true_hypothesis_id(_KEY)


def _timing_line(label: str, seconds: Sequence[float]) -> str:
    ms = [1e3 * s for s in seconds]
    return (
        f"{label}: {len(ms)} EIG steps, per step mean {statistics.fmean(ms):.1f} ms, "
        f"median {statistics.median(ms):.1f} ms, max {max(ms):.1f} ms"
    )


# --------------------------------------------------------------------------- cases

# (family, regime, seed). Regimes cover in-set and out-of-set truths, heavy tails,
# drift, nuisance offsets, flat landscapes, tiny budgets, zero-noise refutation (R11)
# and indistinguishable twins (systemlab R5).
LOOP_CASES = [
    ("microclimate", "R1", 0),
    ("microclimate", "R1", 5),
    ("microclimate", "R3", 1),
    ("microclimate", "R4", 2),
    ("microclimate", "R6", 3),
    ("microclimate", "R8", 4),
    ("microclimate", "R9", 6),
    ("microclimate", "R10", 7),
    ("systemlab", "R1", 0),
    ("systemlab", "R1", 5),
    ("systemlab", "R2", 1),
    ("systemlab", "R3", 2),
    ("systemlab", "R5", 3),
    ("systemlab", "R7", 4),
    ("systemlab", "R8", 6),
    ("systemlab", "R10", 7),
    ("systemlab", "R11", 2),
]


_MAX_MEAN_STEP_SECONDS = 0.5
"""Loose bound on the mean EIG step; measured 5-50 ms (see the module docstring)."""


@pytest.mark.parametrize(("family", "regime", "seed"), LOOP_CASES)
def test_loop_invariants_and_determinism(family: str, regime: str, seed: int) -> None:
    first = run_episode(family, regime, seed)
    second = run_episode(family, regime, seed, check=False)
    spec = FAMILIES[family].regimes()[regime]
    n_eig = sum(s.mode == "eig" for s in first.steps)
    # Without a budget only the world's horizon (campaign window) can end the loop early.
    assert n_eig == 15 or (first.exhausted and n_eig >= 5)
    assert [s.mode for s in first.steps[: spec.warmup]] == ["warmup"] * spec.warmup
    # The whole trajectory replays: same decisions, same data, same posterior.
    assert first.action_ids == second.action_ids
    assert [s.value for s in first.steps] == [s.value for s in second.steps]
    for a, b in zip(first.steps, second.steps, strict=True):
        assert np.max(np.abs(a.weights - b.weights)) <= 1e-12
    np.testing.assert_array_equal(first.log_evidence, second.log_evidence)
    assert first.digest() == second.digest()  # in fact bitwise identical
    secs = first.eig_seconds()
    print(_timing_line(f"{family}/{regime}/seed={seed}", secs))
    assert statistics.fmean(secs) < _MAX_MEAN_STEP_SECONDS


def test_missing_observations_reach_every_engine() -> None:
    """Microclimate drops readings (MAR logger dropouts, MNAR overheat cut-outs). Every
    engine is updated with ``y = None``; n_obs counts only observed values (checked in
    the loop), and since NIG engines ignore a missing outcome the posterior over
    hypotheses is bitwise unchanged by it. These two paths contain 5 dropouts."""
    missing = 0
    for regime, seed in (("R4", 2), ("R3", 1)):
        ep = run_episode("microclimate", regime, seed, n_steps=25)
        for prev, step in itertools.pairwise(ep.steps):
            if step.value is None:
                missing += 1
                np.testing.assert_array_equal(step.weights, prev.weights)
    assert missing >= 1  # otherwise the missing branch went untested


_CHILD = """
import sys
sys.path.insert(0, {test_dir!r})
from test_envs_engine import run_episode
for family, regime, seed in {cases!r}:
    print(run_episode(family, regime, seed, n_steps=6, check=False).digest())
"""


def test_trajectories_are_identical_in_a_fresh_interpreter() -> None:
    """Same digests (actions, values and posterior bits) in a new process with another
    hash seed: nothing depends on Python's salted ``hash()`` or on process state."""
    cases = [("microclimate", "R1", 0), ("systemlab", "R8", 6), ("systemlab", "R11", 2)]
    here = [run_episode(f, r, s, n_steps=6, check=False).digest() for f, r, s in cases]
    code = _CHILD.format(test_dir=str(Path(__file__).parent), cases=cases)
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.split() == here


def test_engines_are_the_expected_kinds() -> None:
    """Microclimate sets are all linear-Gaussian (NIG); system-lab sets are all grids;
    each set has exactly one residual."""
    for family, kind in (("microclimate", NIGModel), ("systemlab", GridModel)):
        fam = FAMILIES[family]
        for regime in fam.regimes():
            models = build_models(fam.hypothesis_set(regime))
            assert models and all(isinstance(m, kind) for m in models)
            assert sum(m.is_residual for m in models) == 1


# --------------------------------------------------------------------------- recovery


def _recovery(family: str, regime: str, level: str, seeds: range) -> tuple[float, list[str]]:
    """Fraction of seeds whose planted hypothesis leads the final posterior of a
    budgeted max-U_M run, with a per-seed table and the step timing."""
    budget = FAMILIES[family].regimes()[regime].budgets[level]
    hits, lines, secs = 0, [], []
    for seed in seeds:
        ep = run_episode(family, regime, seed, n_steps=10_000, budget=budget, check=False)
        truth = true_id(family, regime, seed)
        assert truth is not None and truth in ep.hypothesis_ids
        p_true = float(ep.final_weights[ep.hypothesis_ids.index(truth)])
        hit = ep.leader == truth
        hits += hit
        secs += ep.eig_seconds()
        lines.append(
            f"  seed {seed:2d}: truth {truth:<14} leader {ep.leader:<14} "
            f"P(truth)={p_true:.3f} steps={len(ep.steps)} spent={ep.spent:g} "
            f"{'ok' if hit else 'MISS'}"
        )
    head = f"{family}/{regime} ({level} budget {budget:g}): truth leads in {hits}/{len(seeds)}"
    lines.insert(0, head)
    lines.append(_timing_line(f"  {family}/{regime}", secs))
    return hits / len(seeds), lines


# Easy in-set regimes: the planted hypothesis should lead the posterior at the end of a
# budgeted max-U_M run for most seeds. (family, regime, budget level, min rate); the rate
# observed over seeds 0-23 when this was written is in the comment.
EASY_CASES = [
    ("microclimate", "R10", "large", 0.75),  # 24/24 (medium 23/24, small 20/24)
    ("systemlab", "R2", "large", 0.75),  # 24/24 (small and medium 24/24 too)
    ("systemlab", "R10", "medium", 0.75),  # 24/24 (small 22/24, large 24/24)
]


@pytest.mark.parametrize(("family", "regime", "level", "min_rate"), EASY_CASES)
def test_truth_leads_in_easy_regimes(family: str, regime: str, level: str, min_rate: float) -> None:
    rate, lines = _recovery(family, regime, level, range(8))
    print("\n".join(lines))
    assert rate >= min_rate, "\n".join(lines)


# The same over 24 seeds and more in-set regimes. Thresholds sit about two binomial
# standard errors below the observed rate (comment), so a real regression trips them.
# Misses in microclimate R1/R8 are Occam effects, not engine faults: with a prior sd
# of about 1.8 degC per extra coefficient, a small true surface or shade-by-hour
# effect loses to shade-only H1 within the budget (R1 is the "small interaction
# effect" regime).
SURVEY_CASES = [
    ("microclimate", "R10", "large", 0.85),  # 24/24
    ("microclimate", "R2", "large", 0.85),  # 24/24
    ("microclimate", "R4", "large", 0.85),  # 24/24
    ("microclimate", "R1", "large", 0.45),  # 16/24 (8 misses: H1 leads a true H2/H3)
    ("microclimate", "R8", "large", 0.45),  # 16/24 (heavy tails; H1/H0/H4 lead)
    ("systemlab", "R2", "large", 0.85),  # 24/24
    ("systemlab", "R10", "medium", 0.85),  # 24/24
    ("systemlab", "R1", "medium", 0.85),  # 24/24
    ("systemlab", "R8", "small", 0.6),  # 20/24 (contamination noise, 100-tick budget)
]


@pytest.mark.slow
@pytest.mark.parametrize(("family", "regime", "level", "min_rate"), SURVEY_CASES)
def test_truth_recovery_survey(family: str, regime: str, level: str, min_rate: float) -> None:
    rate, lines = _recovery(family, regime, level, range(24))
    print("\n".join(lines))
    assert rate >= min_rate, "\n".join(lines)


def test_indistinguishable_twins_never_become_confident() -> None:
    """System lab R5: token bucket and GCRA leaky meter predict identically, so however
    long the max-U_M run, neither twin may reach the 0.95 decision level and their
    posteriors stay exactly equal ("inconclusive" is the correct verdict)."""
    for seed in range(4):
        ep = run_episode("systemlab", "R5", seed, n_steps=10_000, budget=150.0, check=False)
        w = dict(zip(ep.hypothesis_ids, ep.final_weights.tolist(), strict=True))
        assert w["token_bucket"] == w["leaky_meter"], (seed, w)
        assert max(w.values()) < 0.95, (seed, w)
