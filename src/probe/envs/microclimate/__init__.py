"""Microclimate structural causal simulator (handover §12 H.1), milestone M2.

A seeded schematic campus (24 sites with shade, surface, sky-view factor and wall
distance; a walking-time matrix; a reference station at base), a hidden SCM
driven by pvlib solar geometry, and an explicit sensor model (radiation error by
shield type, unit offsets, quantisation, MAR dropouts, MNAR overheating
cut-outs). The observable is the difference from the reference station.

Public entry points::

    from probe.envs.microclimate import FAMILY
    world = FAMILY.make_world("R1", seed=17)       # World(env, truth)
    hset = FAMILY.hypothesis_set("R1")             # H1-H4 + residual H0
    FAMILY.regimes()["R10"].expected               # pre-registered expectation

Module map: :mod:`.solar` (pvlib, cached), :mod:`.campus` (public map),
:mod:`.records` (agent-knowable covariates and strata), :mod:`.hypotheses` (IR),
:mod:`.scenarios` ([U] assumptions and regimes R1-R10), :mod:`.scm` (hidden
truth and sensor model), :mod:`.env` (agent-facing environment),
:mod:`.family` (``make_world``).
"""

from .env import DWELLS, VERSION, WAITS, MicroclimateEnv
from .family import FAMILY, MicroclimateFamily, make_world
from .hypotheses import COVARIATES, HYPOTHESIS_IDS, OUTCOME, build_hypothesis_set
from .scenarios import (
    REGIME_CONFIGS,
    REGIME_IDS,
    SCENARIOS,
    RegimeConfig,
    regime_specs,
    scenario_manifest,
    scenario_manifest_hash,
)

__all__ = [
    "COVARIATES",
    "DWELLS",
    "FAMILY",
    "HYPOTHESIS_IDS",
    "OUTCOME",
    "REGIME_CONFIGS",
    "REGIME_IDS",
    "SCENARIOS",
    "VERSION",
    "WAITS",
    "MicroclimateEnv",
    "MicroclimateFamily",
    "RegimeConfig",
    "build_hypothesis_set",
    "make_world",
    "regime_specs",
    "scenario_manifest",
    "scenario_manifest_hash",
]
