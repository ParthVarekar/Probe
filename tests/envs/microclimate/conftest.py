"""Evaluator-side helpers shared by the microclimate tests."""

import json

import pytest

from probe.determinism import canonical_json, make_rng
from probe.envs.microclimate import FAMILY
from probe.envs.microclimate.campus import LAYOUTS, generate_campus
from probe.envs.microclimate.scm import HiddenWorld
from probe.truth import EvaluatorKey

_KEY = EvaluatorKey("microclimate test fixtures")


def build_hidden(regime: str, seed: int) -> HiddenWorld:
    """The hidden SCM of world ``(regime, seed)``, rebuilt exactly as ``make_world``
    builds it; its payload is checked against the sealed commitment of that world."""
    cfg = FAMILY.regime_config(regime)
    campus = generate_campus(make_rng(seed, "microclimate", regime, "campus"), LAYOUTS[cfg.layout])
    hidden = HiddenWorld(cfg, campus, seed, FAMILY.hypothesis_set(regime))
    opened = FAMILY.make_world(regime, seed).truth.open(_KEY)
    assert opened["payload"] == json.loads(canonical_json(hidden.payload()))
    return hidden


@pytest.fixture
def hidden_world():
    """Factory fixture: ``hidden_world(regime, seed) -> HiddenWorld``."""
    return build_hidden
