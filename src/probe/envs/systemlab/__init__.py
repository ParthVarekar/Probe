"""Black-box system lab: planted services probed through controlled requests (handover §12 H.2).

Second, evidence-bearing environment of PROBE. Mystery services (rate limiters:
token bucket, fixed window, sliding window and out-of-set composites; caches:
LRU, FIFO, LFU) are deterministic simulators whose *reports* are noisy. The
hypotheses are ``finite_grid`` models over registered mechanisms
(:mod:`probe.envs.systemlab.mechanisms`), with a structure-light residual in
every set.

Entry point: :data:`FAMILY` (``EnvironmentFamily`` named ``"systemlab"``).
Importing this package registers the grid mechanisms.
"""

from probe.envs.systemlab import mechanisms as mechanisms  # registers the mechanisms
from probe.envs.systemlab.env import FAMILY, SystemLabEnv, SystemLabFamily, noise_rng
from probe.envs.systemlab.mechanisms import MECHANISM_NAMES
from probe.envs.systemlab.noise import ReportNoise
from probe.envs.systemlab.oracle import MealyOracle, aalpy_sul
from probe.envs.systemlab.regimes import REGIME_IDS, RegimeConfig, regime_config
from probe.envs.systemlab.services import ServiceSpec, build_service

__all__ = [
    "FAMILY",
    "MECHANISM_NAMES",
    "REGIME_IDS",
    "MealyOracle",
    "RegimeConfig",
    "ReportNoise",
    "ServiceSpec",
    "SystemLabEnv",
    "SystemLabFamily",
    "aalpy_sul",
    "build_service",
    "mechanisms",
    "noise_rng",
    "regime_config",
]
