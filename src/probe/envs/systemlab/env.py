"""The system-lab environment family (handover §12 H.2): planted services as worlds.

A world is a pure function of ``(regime, seed)``:

* the truth (service spec, noise, true hypothesis) is drawn from
  ``make_rng(seed, "systemlab", regime, "truth")``, with its class
  block-randomised by seed (:mod:`probe.envs.systemlab.regimes`), and sealed in a
  :class:`~probe.truth.SealedTruth` whose salt comes from
  ``make_rng(seed, "systemlab", regime, "salt")``;
* the environment holds a private, deterministic instance of the service and
  draws reporting noise from a **counter-based** generator
  ``make_rng(seed, "systemlab", regime, "noise", action_id, clock)``: two policies
  that take the same action at the same clock in the same world consume the same
  uniforms (common random numbers), regardless of their other choices.

Agent-facing surface (:class:`SystemLabEnv`): ``metadata()``, ``candidates()``,
``execute()``, ``spent``, ``clock``, ``exhausted``, ``truth_commitment()`` and, in
reset-capable regimes only, ``reset_oracle()`` (one oracle per world, whose cost is
counted in ``spent``). None of these returns truth fields; metadata and candidates
are identical for every seed of a regime, and observations differ across seeds only
in ``value`` (both are tested). The regime configuration (regime id, ``in_set``, the
truth sampler) is held privately like the service. An environment cannot be copied or
pickled: a copy would execute actions against the same planted service with the same
counter-based noise, i.e. preview every outcome free of charge. Worlds are rebuilt
from ``(regime, seed)`` instead.

Evaluator side: the opened payload's ``acceptable_verdicts`` (and, in deploy worlds,
``pre_deploy``) are resolved for a finished run by
:func:`probe.envs.systemlab.regimes.acceptable_verdicts`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from probe.determinism import make_rng
from probe.engine.ir import GridHypothesis, HypothesisSet
from probe.envs.base import (
    Action,
    Candidate,
    Environment,
    EnvMetadata,
    InfeasibleActionError,
    Observation,
    RegimeSpec,
    World,
)
from probe.envs.systemlab.actions import CacheActionSpace, RateActionSpace
from probe.envs.systemlab.noise import ReportNoise, sample_report
from probe.envs.systemlab.oracle import MealyOracle
from probe.envs.systemlab.regimes import REGIME_IDS, RegimeConfig, TruthDraw, regime_config
from probe.envs.systemlab.services import Cache, RateLimiter, ServiceSpec, build_service
from probe.truth import SealedTruth

__all__ = ["FAMILY", "SystemLabEnv", "SystemLabFamily", "noise_rng"]

FAMILY_NAME = "systemlab"
FAMILY_VERSION = "0.1.0"


_DESCRIPTION = {
    "ratelimit": "System lab: a black-box rate-limited service. Each action idles for `wait` "
    "ticks, then sends `burst` simultaneous requests; the outcome is the reported number "
    "admitted.",
    "cache": "System lab: a black-box cache. Each action replays a key sequence; the outcome "
    "is the reported number of cache hits.",
}


def noise_rng(seed: int, regime: str, action_id: str, clock: int) -> np.random.Generator:
    """The counter-based generator for one measurement (world seed, action, clock)."""
    return make_rng(seed, FAMILY_NAME, regime, "noise", action_id, clock)


class SystemLabEnv(Environment):
    """What the investigating agent sees of one system-lab world."""

    def __init__(
        self, config: RegimeConfig, seed: int, service: ServiceSpec, noise: ReportNoise, commit: str
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative int")
        is_cache = config.domain == "cache"
        if service.is_cache() != is_cache:
            raise ValueError("service kind does not match the regime's domain")
        self.__config = config
        self.__seed = seed
        self.__spec = service
        self.__noise = noise
        self.__service: RateLimiter | Cache = build_service(service)
        self.__commitment = commit
        self.__oracle: MealyOracle | None = None
        self._clock = 0
        self._spent = 0.0
        self._step = 0

    # -- public surface -------------------------------------------------------
    def metadata(self) -> EnvMetadata:
        """Family, outcome, covariates, cost unit and population. Deliberately free of
        the regime id and title, which would tell a policy regime-level facts such as
        "the truth is outside the set"; the run loop knows the regime separately."""
        cfg = self.__config
        return EnvMetadata(
            family=FAMILY_NAME,
            version=FAMILY_VERSION,
            description=_DESCRIPTION[cfg.domain],
            outcome=cfg.actions.outcome(),
            covariates=cfg.actions.covariates(),
            cost_unit=cfg.actions.cost_unit,
            origin="simulated",
            population=cfg.actions.population(),
            extra={
                "domain": cfg.domain,
                "horizon": cfg.actions.horizon,
                "noise_model": "per-item status flips (eps) plus whole-report glitches "
                "(glitch); values as declared in the hypothesis parameters",
                "resettable": cfg.resettable,
            },
        )

    def candidates(self) -> list[Candidate]:
        return self.__config.actions.candidates(self._clock)

    def execute(self, action: Action) -> Observation:
        cand = self.__config.actions.candidate(action.id, self._clock)
        if cand is None:
            raise InfeasibleActionError(f"unknown action {action.id!r}")
        if dict(action.params) != dict(cand.action.params):
            raise InfeasibleActionError(f"action {action.id!r} has unexpected params")
        if not cand.feasible:
            raise InfeasibleActionError(f"{action.id}: {cand.infeasible_reason}")
        start = self._clock
        rng = noise_rng(self.__seed, self.__config.spec.id, action.id, start)
        svc = self.__service
        if isinstance(svc, Cache):
            keys = str(cand.action.params["keys"])
            n = len(keys)
            k = svc.run(keys)
            end = start + n
            meta: dict[str, Any] = {"clock_start": start, "clock_end": end}
        else:
            wait = int(cand.action.params["wait"])
            n = int(cand.action.params["burst"])
            t_burst = start + wait
            k = svc.burst(t_burst, n)
            end = start + RateActionSpace.duration(wait, n)
            meta = {"clock_start": start, "clock_burst": t_burst, "clock_end": end}
        y = sample_report(k, n, self.__noise, rng)
        obs = Observation(
            step=self._step,
            action_id=action.id,
            value=int(y),
            missingness="observed",
            record=dict(cand.record),
            cost=cand.cost,
            clock=float(end),
            origin="simulated",
            meta=meta,
        )
        self._clock = end
        self._spent += cand.cost
        self._step += 1
        return obs

    @property
    def spent(self) -> float:
        """Cost of executed actions plus, in reset-capable worlds, everything charged by
        the world's oracle (:meth:`reset_oracle`): one budget for both channels."""
        oracle = self.__oracle
        return self._spent + (oracle.cost if oracle is not None else 0.0)

    @property
    def clock(self) -> float:
        return float(self._clock)

    @property
    def exhausted(self) -> bool:
        return not any(c.feasible for c in self.candidates())

    def truth_commitment(self) -> str:
        return self.__commitment

    def reset_oracle(self) -> MealyOracle:
        """The world's reset-capable, noise-free input/output oracle on fresh copies of the
        planted service (automata-learning baseline P11). Only offered by regimes that
        declare ``resettable`` (R11): elsewhere the service is noisy or nonstationary and a
        reset would be an unfair extra capability.

        There is exactly one oracle per world (every call returns the same object), and
        its cost is part of :attr:`spent`, so asking again cannot reset the meter."""
        cfg = self.__config
        if not cfg.resettable:
            raise PermissionError("this world offers no reset-capable oracle")
        if self.__oracle is None:
            spec = self.__spec
            if isinstance(cfg.actions, CacheActionSpace):
                alphabet: tuple[str, ...] = tuple(cfg.actions.keys())
            else:
                alphabet = MealyOracle.RATE_ALPHABET
            self.__oracle = MealyOracle(
                lambda: build_service(spec),
                alphabet,
                step_cost=1.0,
                reset_cost=cfg.oracle_reset_cost,
            )
        return self.__oracle

    def __repr__(self) -> str:
        return f"SystemLabEnv(step={self._step}, clock={self._clock}, spent={self._spent})"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        # Also blocks copy.copy/copy.deepcopy: a copy previews outcomes without paying.
        raise TypeError("environments are not copyable or picklable; rebuild from (regime, seed)")


# --------------------------------------------------------------------------- family


def _grid_index(hyp: GridHypothesis, point: Mapping[str, Any]) -> int:
    table = hyp.grid_table()
    names = sorted(hyp.params)
    for i in range(hyp.grid_size()):
        if all(table[n][i].item() == point[n] for n in names):
            return i
    raise ValueError(f"truth {dict(point)} is not a grid point of {hyp.id}")


class SystemLabFamily:
    """``EnvironmentFamily`` implementation for the black-box system lab."""

    name = FAMILY_NAME
    version = FAMILY_VERSION

    def regimes(self) -> Mapping[str, RegimeSpec]:
        return {r: regime_config(r).spec for r in REGIME_IDS}

    def hypothesis_set(self, regime: str) -> HypothesisSet:
        return regime_config(regime).hypotheses

    def make_world(self, regime: str, seed: int) -> World:
        cfg = regime_config(regime)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative int")
        draw: TruthDraw = cfg.sample_truth(make_rng(seed, FAMILY_NAME, regime, "truth"), seed)
        salt = make_rng(seed, FAMILY_NAME, regime, "salt").bytes(16).hex()
        world_id = f"{FAMILY_NAME}/{regime}/{seed}"
        payload: dict[str, Any] = {
            "family": FAMILY_NAME,
            "family_version": FAMILY_VERSION,
            "regime": regime,
            "seed": seed,
            "service": draw.service.as_dict(),
            "noise": draw.noise.as_dict(),
            "acceptable_verdicts": list(draw.acceptable_verdicts),
            "extra": dict(draw.extra),
        }
        if draw.pre_deploy is not None:
            payload["pre_deploy"] = dict(draw.pre_deploy)
        if draw.true_hypothesis_id is not None and draw.true_params is not None:
            hyp = cfg.hypotheses.by_id(draw.true_hypothesis_id)
            assert isinstance(hyp, GridHypothesis)
            payload["true_params"] = dict(draw.true_params)
            payload["true_grid_index"] = _grid_index(hyp, draw.true_params)
        truth = SealedTruth(world_id, draw.true_hypothesis_id, payload, salt)
        env = SystemLabEnv(cfg, seed, draw.service, draw.noise, truth.commitment)
        return World(env=env, truth=truth)


FAMILY = SystemLabFamily()
