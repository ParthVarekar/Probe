"""System-lab regimes: pre-registered expectations, hypothesis sets and truth draws.

Regimes follow handover §12 Part I (R4 and R9 are microclimate-specific and have
no system-lab analogue). Two coherent hypothesis families are used, one per
regime:

* **rate limiters** (R1, R2, R3, R5, R6, R7, R8, R10): token bucket, fixed
  window, sliding window (R5: token bucket vs. its GCRA twin, the leaky meter),
  each a :class:`~probe.engine.ir.GridHypothesis` over its parameters, plus the
  structure-free ``allowance`` residual. Actions: wait, then burst; cost is wall
  time (heterogeneous), except in R2 (uniform).
* **caches** (R11): LRU, FIFO and LFU over a size grid, plus the stack-distance
  ``reuse`` residual. Actions: key-reuse patterns (uniform cost). The planted
  cache is a deterministic CLOCK or MRU cache, a finite-state machine outside the
  set: the home turf of automata learning, where the hypothesis-driven engine can
  at best say "none fits".

Truth draws use ``make_rng(seed, "systemlab", regime, "truth")``. In-set truths
are a registered hypothesis at one of its grid points, with the declared noise,
so the hypothesis set is exactly well specified. Each draw also states the
verdicts that count as correct (``acceptable_verdicts``) for the scorer: the
true hypothesis id; ``"none_fits"`` when the truth is outside the set;
``"inconclusive"`` in R5, where the two structured hypotheses are
observationally equivalent.

Budgets are in the regime's cost unit (ticks, actions or accesses). They were
sized with the exact reference filter in ``tests/envs/systemlab`` so that
"medium" lets even a random design decide in most well-specified worlds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal

import numpy as np

from probe.engine.ir import GridHypothesis, GridValue, HypothesisSet
from probe.envs.base import RegimeSpec
from probe.envs.systemlab.actions import CacheActionSpace, RateActionSpace, cache_patterns
from probe.envs.systemlab.noise import ReportNoise
from probe.envs.systemlab.services import ServiceSpec

__all__ = ["REGIME_IDS", "RegimeConfig", "TruthDraw", "regime_config"]

Domain = Literal["ratelimit", "cache"]

# --------------------------------------------------------------------------- truth draws


@dataclass(frozen=True)
class TruthDraw:
    service: ServiceSpec
    noise: ReportNoise
    true_hypothesis_id: str | None
    true_params: Mapping[str, GridValue] | None
    """Grid coordinates of the truth in its hypothesis (in-set only)."""
    acceptable_verdicts: tuple[str, ...]
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeConfig:
    spec: RegimeSpec
    domain: Domain
    actions: RateActionSpace | CacheActionSpace
    hypotheses: HypothesisSet
    sample_truth: Callable[[np.random.Generator], TruthDraw]
    resettable: bool = False
    """A reset-capable, noise-free input/output oracle is offered (R11 only)."""
    oracle_reset_cost: float = 1.0


# --------------------------------------------------------------------------- grids

_LIMITS = (2, 3, 4, 6, 8, 12)
_PERIODS = (1, 2, 4, 8)
_WINDOWS = (4, 8, 16, 32)
_PHASES = (0, 1, 2, 3)
_WAITS = (0, 1, 2, 4, 8, 16, 32)
_BURSTS = (1, 2, 4, 6, 8, 12)
_ALPHA = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
_BETA = (0.0, 0.5, 1.0, 2.0)
_Q = (0.05, 0.25, 0.5, 0.75, 0.95)
_AMAX = (2, 4, 6, 8, 12)
_GAMMA = (0.0, 0.5, 1.0, 2.0)

_RATE_TEXT: dict[str, tuple[str, str, str]] = {
    "token_bucket": (
        "Token bucket",
        "Requests spend tokens that refill continuously, one every `period` ticks, up to "
        "`capacity`.",
        "After draining, the admitted burst grows by one per `period` ticks of waiting, "
        "smoothly, up to the capacity.",
    ),
    "leaky_meter": (
        "Leaky bucket (GCRA meter)",
        "A generic-cell-rate meter: a request conforms when its theoretical arrival time is "
        "at most (capacity - 1) periods ahead.",
        "Same admitted counts as a token bucket with the same capacity and period.",
    ),
    "fixed_window": (
        "Fixed window",
        "At most `limit` requests per aligned window of `window` ticks; the count resets at "
        "each boundary (offset `phase` quarter-windows).",
        "Admits reset all at once at window boundaries, regardless of how long you waited "
        "inside a window.",
    ),
    "sliding_window": (
        "Sliding window log",
        "At most `limit` admitted requests in any trailing interval of `window` ticks.",
        "A slot frees exactly `window` ticks after the admission it holds, independent of "
        "any boundary.",
    ),
}


def _noise_fixed(noise: ReportNoise) -> dict[str, GridValue]:
    return {"eps": float(noise.eps), "glitch": float(noise.glitch)}


def _rate_hyp(
    kind: str,
    grid: Mapping[str, tuple[GridValue, ...]],
    noise_fixed: Mapping[str, GridValue],
    noise_grid: Mapping[str, tuple[GridValue, ...]],
) -> GridHypothesis:
    label, statement, prediction = _RATE_TEXT[kind]
    return GridHypothesis(
        id=kind,
        label=label,
        statement=statement,
        qualitative_prediction=prediction,
        applicability="Single-client request bursts; integer-tick clock.",
        mechanism=f"systemlab.{kind}",
        params={**dict(grid), **dict(noise_grid)},
        fixed=dict(noise_fixed),
    )


GridMode = Literal["standard", "narrow", "separated"]


def _rate_grids(mode: GridMode = "standard") -> dict[str, dict[str, tuple[GridValue, ...]]]:
    if mode == "separated":  # R2: disjoint capacity scales, so most bursts separate them
        return {
            "token_bucket": {"capacity": (2, 3), "period": (2, 4)},
            "fixed_window": {"limit": (6, 8), "window": (16, 32), "phase": (0, 2)},
            "sliding_window": {"limit": (10, 12), "window": (16, 32)},
            "leaky_meter": {"capacity": (2, 3), "period": (2, 4)},
        }
    if mode == "narrow":  # R10 positive control
        return {
            "token_bucket": {"capacity": (4, 6), "period": (2, 4)},
            "fixed_window": {"limit": (4, 6), "window": (8, 16), "phase": (0,)},
            "sliding_window": {"limit": (4, 6), "window": (8, 16)},
            "leaky_meter": {"capacity": (4, 6), "period": (2, 4)},
        }
    return {
        "token_bucket": {"capacity": _LIMITS, "period": _PERIODS},
        "fixed_window": {"limit": _LIMITS, "window": _WINDOWS, "phase": _PHASES},
        "sliding_window": {"limit": _LIMITS, "window": _WINDOWS},
        "leaky_meter": {"capacity": _LIMITS, "period": _PERIODS},
    }


def _residual_rate(
    noise_fixed: Mapping[str, GridValue],
    noise_grid: Mapping[str, tuple[GridValue, ...]],
    amax: int,
) -> GridHypothesis:
    return GridHypothesis(
        id="residual",
        label="None of the above (load-dependent allowance)",
        statement="No registered algorithm: each burst is admitted a random prefix whose "
        "expected size rises with the preceding wait and falls with the requests sent in the "
        "last `memory` ticks; no algorithmic state.",
        qualitative_prediction="Admitted counts vary between repeats of the same action and "
        "history.",
        mechanism="systemlab.allowance",
        params={
            "alpha": _ALPHA,
            "beta": _BETA,
            "gamma": _GAMMA,
            "amax": tuple(a for a in _AMAX if a <= amax),
            **dict(noise_grid),
        },
        fixed={"memory": 16, **dict(noise_fixed)},
        is_residual=True,
    )


def _rate_set(
    actions: RateActionSpace,
    kinds: tuple[str, ...],
    noise_fixed: Mapping[str, GridValue],
    noise_grid: Mapping[str, tuple[GridValue, ...]] | None = None,
    mode: GridMode = "standard",
) -> HypothesisSet:
    ng = dict(noise_grid or {})
    nf = {k: v for k, v in noise_fixed.items() if k not in ng}
    grids = _rate_grids(mode)
    hyps = [_rate_hyp(k, grids[k], nf, ng) for k in kinds]
    hyps.append(_residual_rate(nf, ng, actions.max_count))
    return HypothesisSet(
        outcome=actions.outcome(), covariates=actions.covariates(), hypotheses=tuple(hyps)
    )


_CACHE_TEXT = {
    "lru": ("LRU cache", "Evicts the least recently used key.", "Re-accessing a key protects it."),
    "fifo": (
        "FIFO cache",
        "Evicts the earliest inserted key; hits do not change the order.",
        "Re-accessing a key does not protect it from eviction.",
    ),
    "lfu": (
        "LFU cache",
        "Evicts the least frequently used resident key (ties: least recently used).",
        "A key accessed many times survives a scan of new keys.",
    ),
}


def _cache_set(
    actions: CacheActionSpace, sizes: tuple[int, ...], noise: ReportNoise
) -> HypothesisSet:
    nf = _noise_fixed(noise)
    hyps: list[GridHypothesis] = []
    for pol in ("lru", "fifo", "lfu"):
        label, statement, prediction = _CACHE_TEXT[pol]
        hyps.append(
            GridHypothesis(
                id=pol,
                label=label,
                statement=statement,
                qualitative_prediction=prediction,
                applicability="Single-client access sequences over a small key set.",
                mechanism="systemlab.cache",
                params={"size": sizes},
                fixed={"policy": pol, **nf},
            )
        )
    hyps.append(
        GridHypothesis(
            id="residual",
            label="None of the above (generic locality)",
            statement="No registered eviction policy: each access hits with a probability that "
            "depends only on its reuse (stack) distance.",
            qualitative_prediction="Hit counts vary between repeats of the same history.",
            mechanism="systemlab.reuse",
            params={"q_near": _Q, "q_mid": _Q, "q_far": _Q},
            fixed=nf,
            is_residual=True,
        )
    )
    return HypothesisSet(
        outcome=actions.outcome(), covariates=actions.covariates(), hypotheses=tuple(hyps)
    )


# --------------------------------------------------------------------------- samplers


def _pick(rng: np.random.Generator, options: tuple[Any, ...]) -> Any:
    return options[int(rng.integers(0, len(options)))]


def _grid_point(
    rng: np.random.Generator, hyp: GridHypothesis, exclude: Mapping[str, GridValue] | None = None
) -> dict[str, GridValue]:
    """A uniformly drawn grid point of ``hyp`` (structural parameters plus any gridded
    noise), optionally different from ``exclude`` on the structural parameters."""
    table = hyp.grid_table()
    g = hyp.grid_size()
    for _ in range(1000):
        i = int(rng.integers(0, g))
        point = {k: table[k][i].item() for k in sorted(hyp.params)}
        if exclude is None or any(point[k] != exclude[k] for k in exclude):
            return point
    raise RuntimeError("could not draw a distinct grid point")


def _structural(point: Mapping[str, GridValue]) -> dict[str, GridValue]:
    return {k: v for k, v in point.items() if k not in ("eps", "glitch")}


def _in_set_sampler(
    hs: HypothesisSet, kinds: tuple[str, ...], noise: ReportNoise
) -> Callable[[np.random.Generator], TruthDraw]:
    def sample(rng: np.random.Generator) -> TruthDraw:
        kind = _pick(rng, kinds)
        hyp = hs.by_id(kind)
        assert isinstance(hyp, GridHypothesis)
        point = _grid_point(rng, hyp)
        truth_noise = ReportNoise(
            eps=float(point.get("eps", noise.eps)), glitch=float(point.get("glitch", noise.glitch))
        )
        if hyp.mechanism == "systemlab.cache":
            params = {"policy": hyp.fixed["policy"], **_structural(point)}
            svc = ServiceSpec("cache", params)
        else:
            svc = ServiceSpec(kind, _structural(point))
        return TruthDraw(
            service=svc,
            noise=truth_noise,
            true_hypothesis_id=kind,
            true_params=point,
            acceptable_verdicts=(f"hypothesis:{kind}",),
        )

    return sample


# R3: three families of services outside the rate-limiter hypothesis set.
_R3_DUAL = (
    ({"capacity": 6, "period": 2}, {"limit": 10, "window": 32, "phase": 0}),
    ({"capacity": 4, "period": 1}, {"limit": 8, "window": 16, "phase": 2}),
    ({"capacity": 8, "period": 2}, {"limit": 12, "window": 32, "phase": 1}),
)
_R3_PENALTY = (
    ({"capacity": 4, "period": 2}, 16),
    ({"capacity": 6, "period": 4}, 24),
    ({"capacity": 3, "period": 1}, 12),
)
_R3_SWITCH = (
    (("token_bucket", {"capacity": 6, "period": 2}), ("fixed_window", {"limit": 4, "window": 16})),
    (("sliding_window", {"limit": 4, "window": 8}), ("token_bucket", {"capacity": 8, "period": 4})),
    (("fixed_window", {"limit": 6, "window": 16}), ("sliding_window", {"limit": 3, "window": 8})),
)


def _r3_sampler(noise: ReportNoise) -> Callable[[np.random.Generator], TruthDraw]:
    def sample(rng: np.random.Generator) -> TruthDraw:
        family = _pick(rng, ("dual", "penalty_box", "deploy"))
        extra: dict[str, Any] = {"out_of_set_family": family}
        if family == "dual":
            tb, fw = _pick(rng, _R3_DUAL)
            svc = ServiceSpec(
                "dual",
                {},
                {
                    "first": ServiceSpec("token_bucket", tb),
                    "second": ServiceSpec("fixed_window", fw),
                },
            )
        elif family == "penalty_box":
            tb, pen = _pick(rng, _R3_PENALTY)
            svc = ServiceSpec(
                "penalty_box", {"penalty": pen}, {"inner": ServiceSpec("token_bucket", tb)}
            )
        else:
            (k1, p1), (k2, p2) = _pick(rng, _R3_SWITCH)
            at = int(rng.integers(30, 91))
            extra["deploy_at"] = at
            svc = ServiceSpec(
                "deploy", {"at": at}, {"before": ServiceSpec(k1, p1), "after": ServiceSpec(k2, p2)}
            )
        return TruthDraw(
            service=svc,
            noise=noise,
            true_hypothesis_id=None,
            true_params=None,
            acceptable_verdicts=("none_fits",),
            extra=extra,
        )

    return sample


def _r5_sampler(
    hs: HypothesisSet, noise: ReportNoise
) -> Callable[[np.random.Generator], TruthDraw]:
    base = _in_set_sampler(hs, ("token_bucket", "leaky_meter"), noise)

    def sample(rng: np.random.Generator) -> TruthDraw:
        d = base(rng)
        return TruthDraw(
            service=d.service,
            noise=d.noise,
            true_hypothesis_id=d.true_hypothesis_id,
            true_params=d.true_params,
            acceptable_verdicts=("inconclusive",),
            extra={"equivalence_class": ["leaky_meter", "token_bucket"]},
        )

    return sample


def _r7_sampler(
    hs: HypothesisSet, noise: ReportNoise
) -> Callable[[np.random.Generator], TruthDraw]:
    def sample(rng: np.random.Generator) -> TruthDraw:
        kind = _pick(rng, ("token_bucket", "fixed_window", "sliding_window"))
        hyp = hs.by_id(kind)
        assert isinstance(hyp, GridHypothesis)
        before = _structural(_grid_point(rng, hyp))
        after = _structural(_grid_point(rng, hyp, exclude=before))
        at = int(rng.integers(40, 101))
        svc = ServiceSpec(
            "deploy",
            {"at": at},
            {"before": ServiceSpec(kind, before), "after": ServiceSpec(kind, after)},
        )
        return TruthDraw(
            service=svc,
            noise=noise,
            true_hypothesis_id=kind,
            true_params=None,
            acceptable_verdicts=(f"hypothesis:{kind}", "none_fits"),
            extra={"deploy_at": at, "params_before": before, "params_after": after},
        )

    return sample


# --------------------------------------------------------------------------- regimes

_RATE_KINDS = ("token_bucket", "fixed_window", "sliding_window")
_STD_NOISE = ReportNoise(eps=0.03)


def _rate_actions() -> RateActionSpace:
    return RateActionSpace(waits=_WAITS, bursts=_BURSTS)


def _build_r1() -> RegimeConfig:
    a = _rate_actions()
    hs = _rate_set(a, _RATE_KINDS, _noise_fixed(_STD_NOISE))
    return RegimeConfig(
        spec=RegimeSpec(
            id="R1",
            title="Well-specified rate limiter, heterogeneous cost",
            description="A planted token bucket, fixed window or sliding window at a grid point "
            "of its hypothesis, 3% per-request status flips. Waiting is what reveals refill "
            "behaviour, and waiting costs wall time.",
            expected="adaptive wins: guarded EIG/cost (P8) reaches the decision criterion at >=15% "
            "lower median cost than the best of P1-P5/P9, with confidently-wrong rate <= best "
            "baseline + 1pp",
            budgets={"small": 60.0, "medium": 150.0, "large": 400.0},
            warmup=3,
            in_set=True,
        ),
        domain="ratelimit",
        actions=a,
        hypotheses=hs,
        sample_truth=_in_set_sampler(hs, _RATE_KINDS, _STD_NOISE),
    )


def _build_r2() -> RegimeConfig:
    a = RateActionSpace(waits=_WAITS, bursts=_BURSTS, cost_model="uniform")
    noise = ReportNoise(eps=0.01)
    hs = _rate_set(a, _RATE_KINDS, _noise_fixed(noise), mode="separated")
    return RegimeConfig(
        spec=RegimeSpec(
            id="R2",
            title="Large, easy differences, uniform cost",
            description="The three limiter algorithms live on disjoint capacity scales "
            "(token bucket 2-3, fixed window 6-8, sliding window 10-12), so almost any burst "
            "of four or more separates them; 1% status flips; every action costs one unit.",
            expected="tie with stratified: |cost difference| < 5% between P8 and P3; all "
            "policies decide correctly",
            budgets={"small": 4.0, "medium": 8.0, "large": 16.0},
            warmup=2,
            in_set=True,
        ),
        domain="ratelimit",
        actions=a,
        hypotheses=hs,
        sample_truth=_in_set_sampler(hs, _RATE_KINDS, noise),
    )


def _build_r3() -> RegimeConfig:
    a = _rate_actions()
    hs = _rate_set(a, _RATE_KINDS, _noise_fixed(_STD_NOISE))
    return RegimeConfig(
        spec=RegimeSpec(
            id="R3",
            title="Truth outside the hypothesis set",
            description="The planted service is a dual limiter (token bucket AND a fixed-window "
            "quota), a token bucket with a penalty box, or a mid-run deploy that switches "
            "algorithm. None is a registered hypothesis.",
            expected="unguarded adaptive (P6) loses on safety (higher confidently-wrong rate); "
            "guarded P8 ~ baselines; the correct verdict is none_fits",
            budgets={"small": 60.0, "medium": 150.0, "large": 400.0},
            warmup=3,
            in_set=False,
            notes="Sealed payload names the out-of-set family; acceptable verdict: none_fits.",
        ),
        domain="ratelimit",
        actions=a,
        hypotheses=hs,
        sample_truth=_r3_sampler(_STD_NOISE),
    )


def _build_r5() -> RegimeConfig:
    a = _rate_actions()
    hs = _rate_set(a, ("token_bucket", "leaky_meter"), _noise_fixed(_STD_NOISE))
    return RegimeConfig(
        spec=RegimeSpec(
            id="R5",
            title="Indistinguishable variants",
            description="Token bucket vs. leaky bucket as a meter (GCRA) with the same "
            "parameter grid: the two give identical admitted counts for every action, so no "
            "design can tell them apart.",
            expected="all policies tie; 'inconclusive' is the correct verdict; any decision for "
            "one variant is scored a false decision",
            budgets={"small": 60.0, "medium": 150.0, "large": 400.0},
            warmup=3,
            in_set=True,
            notes="The residual is still separable from both; once it is ruled out the "
            "model-indicator EIG is exactly 0 for every candidate.",
        ),
        domain="ratelimit",
        actions=a,
        hypotheses=hs,
        sample_truth=_r5_sampler(hs, _STD_NOISE),
    )


def _build_r6() -> RegimeConfig:
    base = _build_r1()
    return RegimeConfig(
        spec=RegimeSpec(
            id="R6",
            title="Tiny budget",
            description="R1's worlds and hypotheses with a budget of at most a handful of "
            "observations.",
            expected="static maximin design (P9) >= adaptive",
            budgets={"small": 8.0, "medium": 16.0, "large": 30.0},
            warmup=0,
            in_set=True,
            notes="Run with RunProtocol.max_steps=5 (<= 5 observations, handover H.12).",
        ),
        domain=base.domain,
        actions=base.actions,
        hypotheses=base.hypotheses,
        sample_truth=base.sample_truth,
    )


def _build_r7() -> RegimeConfig:
    a = _rate_actions()
    hs = _rate_set(a, _RATE_KINDS, _noise_fixed(_STD_NOISE))
    return RegimeConfig(
        spec=RegimeSpec(
            id="R7",
            title="Nonstationary service (mid-run deploy)",
            description="The planted limiter keeps its algorithm but a deploy at a hidden tick "
            "(40-100) changes its parameters to another grid point and restarts it.",
            expected="stratified >= adaptive",
            budgets={"small": 60.0, "medium": 150.0, "large": 400.0},
            warmup=3,
            in_set=True,
            notes="Structurally in the set, parametrically nonstationary: no single grid point "
            "fits the whole run. Acceptable verdicts: the true algorithm or none_fits.",
        ),
        domain="ratelimit",
        actions=a,
        hypotheses=hs,
        sample_truth=_r7_sampler(hs, _STD_NOISE),
    )


def _build_r8() -> RegimeConfig:
    a = _rate_actions()
    noise = ReportNoise(eps=0.05, glitch=0.25)
    glitch_grid: dict[str, tuple[GridValue, ...]] = {"glitch": (0.02, 0.1, 0.25)}
    hs = _rate_set(a, _RATE_KINDS, _noise_fixed(noise), noise_grid=glitch_grid)
    in_set = _in_set_sampler(hs, _RATE_KINDS, noise)

    def sample(rng: np.random.Generator) -> TruthDraw:
        d = in_set(rng)
        point = {**dict(d.true_params or {}), "glitch": noise.glitch}
        return TruthDraw(
            service=d.service,
            noise=noise,
            true_hypothesis_id=d.true_hypothesis_id,
            true_params=point,
            acceptable_verdicts=d.acceptable_verdicts,
        )

    return RegimeConfig(
        spec=RegimeSpec(
            id="R8",
            title="Heavy-tailed reporting noise",
            description="R1's services with 5% status flips and 25% garbage reports (uniform "
            "count). The contamination level is unknown to the hypotheses (gridded "
            "0.02/0.1/0.25), i.e. a robust likelihood.",
            expected="robust (contamination) likelihood required; all policies slower; adaptive "
            "may lose; confidently-wrong rate must stay <= best baseline + 1pp",
            budgets={"small": 100.0, "medium": 300.0, "large": 800.0},
            warmup=3,
            in_set=True,
        ),
        domain="ratelimit",
        actions=a,
        hypotheses=hs,
        sample_truth=sample,
    )


def _build_r10() -> RegimeConfig:
    a = _rate_actions()
    noise = ReportNoise(eps=0.02)
    hs = _rate_set(a, _RATE_KINDS, _noise_fixed(noise), mode="narrow")
    return RegimeConfig(
        spec=RegimeSpec(
            id="R10",
            title="Positive control",
            description="Three limiters with narrow grids (capacity/limit 4 or 6). Long waits "
            "are expensive and uninformative (every hypothesis refills fully), while cheap "
            "short-wait bursts after draining separate them sharply.",
            expected="adaptive wins (EIG/cost needs clearly less cost than random/stratified to "
            "decide); otherwise the benchmark is invalid",
            budgets={"small": 30.0, "medium": 80.0, "large": 200.0},
            warmup=2,
            in_set=True,
        ),
        domain="ratelimit",
        actions=a,
        hypotheses=hs,
        sample_truth=_in_set_sampler(hs, _RATE_KINDS, noise),
    )


def _build_r11() -> RegimeConfig:
    noise = ReportNoise(eps=0.0, glitch=0.0)
    a = CacheActionSpace(patterns=cache_patterns("abcd", 6))
    hs = _cache_set(a, (1, 2, 3), noise)

    def sample(rng: np.random.Generator) -> TruthDraw:
        policy = _pick(rng, ("clock", "mru"))
        size = _pick(rng, (2, 3))
        return TruthDraw(
            service=ServiceSpec("cache", {"policy": policy, "size": size}),
            noise=noise,
            true_hypothesis_id=None,
            true_params=None,
            acceptable_verdicts=("none_fits",),
            extra={"finite_state": True},
        )

    return RegimeConfig(
        spec=RegimeSpec(
            id="R11",
            title="Deterministic finite-state cache (automata-learning home turf)",
            description="A noise-free cache of size 2-3 over keys a-d whose eviction policy "
            "(CLOCK or MRU) is not registered. A reset-capable input/output oracle is offered "
            "for automata learning (AALpy L*).",
            expected="AALpy (P11) wins: it learns an exact Mealy machine of the service, while "
            "the hypothesis-driven policies can at best report none_fits",
            budgets={"small": 48.0, "medium": 96.0, "large": 192.0},
            warmup=1,
            in_set=False,
            notes="Zero noise: a registered policy is eliminated exactly (zero likelihood) at "
            "its first wrong prediction. Oracle cost: 1 per access, 1 per reset. The "
            "comparison is on what each method delivers (an exact, verified automaton vs. a "
            "verdict), reported with costs side by side: L*/KV need thousands of oracle steps "
            "(MRU) to ~10^5 (CLOCK, size 3), far above the EIG budgets.",
        ),
        domain="cache",
        actions=a,
        hypotheses=hs,
        sample_truth=sample,
        resettable=True,
        oracle_reset_cost=1.0,
    )


_BUILDERS: dict[str, Callable[[], RegimeConfig]] = {
    "R1": _build_r1,
    "R2": _build_r2,
    "R3": _build_r3,
    "R5": _build_r5,
    "R6": _build_r6,
    "R7": _build_r7,
    "R8": _build_r8,
    "R10": _build_r10,
    "R11": _build_r11,
}
REGIME_IDS: tuple[str, ...] = tuple(_BUILDERS)


@cache
def regime_config(regime: str) -> RegimeConfig:
    try:
        builder = _BUILDERS[regime]
    except KeyError as e:
        raise KeyError(f"unknown system-lab regime {regime!r}; known: {list(REGIME_IDS)}") from e
    return builder()
