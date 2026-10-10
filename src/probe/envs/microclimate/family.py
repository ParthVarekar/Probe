"""``MicroclimateFamily``: the :class:`~probe.envs.base.EnvironmentFamily` for M2.

``make_world(regime, seed)`` is a pure, fast function of its arguments:

1. campus from ``make_rng(seed, "microclimate", regime, "campus")`` (public);
2. hidden SCM (:class:`~.scm.HiddenWorld`) from its own keyed streams;
3. :class:`~probe.truth.SealedTruth` over the hidden payload, with salt
   ``make_rng(seed, "microclimate", regime, "salt").bytes(16).hex()``;
4. the agent-facing :class:`~.env.MicroclimateEnv`, which carries only the
   commitment.

The first call in a process also computes the shared pvlib solar day (cached);
later calls take a few milliseconds.
"""

from __future__ import annotations

from collections.abc import Mapping

from probe.determinism import make_rng
from probe.engine.ir import HypothesisSet
from probe.envs.base import RegimeSpec, World
from probe.truth import SealedTruth

from .campus import LAYOUTS, generate_campus
from .env import VERSION, MicroclimateEnv
from .hypotheses import build_hypothesis_set
from .scenarios import REGIME_CONFIGS, RegimeConfig, regime_specs
from .scm import FAMILY_KEY, HiddenWorld
from .solar import solar_day

__all__ = ["FAMILY", "MicroclimateFamily", "make_world"]


class MicroclimateFamily:
    name: str = FAMILY_KEY
    version: str = VERSION

    def __repr__(self) -> str:
        return f"MicroclimateFamily(version={self.version!r})"

    def regimes(self) -> Mapping[str, RegimeSpec]:
        return regime_specs()

    def regime_config(self, regime: str) -> RegimeConfig:
        try:
            return REGIME_CONFIGS[regime]
        except KeyError as e:
            known = sorted(REGIME_CONFIGS)
            raise KeyError(f"unknown microclimate regime {regime!r}; known: {known}") from e

    def hypothesis_set(self, regime: str) -> HypothesisSet:
        return build_hypothesis_set(self.regime_config(regime).hypothesis_ids)

    def make_world(self, regime: str, seed: int) -> World:
        cfg = self.regime_config(regime)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative int, got {seed!r}")
        hset = build_hypothesis_set(cfg.hypothesis_ids)
        solar = solar_day()
        campus = generate_campus(make_rng(seed, FAMILY_KEY, regime, "campus"), LAYOUTS[cfg.layout])
        hidden = HiddenWorld(cfg, campus, seed, hset)
        world_id = f"{FAMILY_KEY}/{regime}/{seed}"
        salt = make_rng(seed, FAMILY_KEY, regime, "salt").bytes(16).hex()
        truth = SealedTruth(world_id, hidden.true_hypothesis_id, hidden.payload(), salt)
        env = MicroclimateEnv(
            cfg=cfg,
            seed=seed,
            world_id=world_id,
            campus=campus,
            solar=solar,
            covariates=hset.covariates,
            outcome=hset.outcome,
            hidden=hidden,
            commitment=truth.commitment,
        )
        return World(env=env, truth=truth)


FAMILY = MicroclimateFamily()


def make_world(regime: str, seed: int) -> World:
    """Shorthand for ``FAMILY.make_world(regime, seed)``."""
    return FAMILY.make_world(regime, seed)
