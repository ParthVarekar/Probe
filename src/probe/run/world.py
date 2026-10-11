"""Convenience: build a world, run one policy on it, score it (benchmark and CLI).

This is evaluator-side code: the P10 oracle is constructed here with the opened
truth; :func:`~probe.run.loop.run_experiment` itself never sees the truth.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from probe.envs.base import EnvironmentFamily
from probe.run.evaluate import evaluate_run
from probe.run.loop import RunRecord, run_experiment
from probe.run.protocol import RunProtocol
from probe.select import make_policy
from probe.truth import EvaluatorKey

__all__ = ["FAMILIES", "get_family", "make_protocol", "run_world"]

FAMILIES = ("microclimate", "systemlab")


def get_family(name: str) -> EnvironmentFamily:
    if name not in FAMILIES:
        raise KeyError(f"unknown family {name!r}; known: {FAMILIES}")
    fam: EnvironmentFamily = importlib.import_module(f"probe.envs.{name}").FAMILY
    return fam


def make_protocol(
    family: str, regime: str, budget_key: str = "medium", overrides: Mapping[str, Any] | None = None
) -> RunProtocol:
    spec = get_family(family).regimes()[regime]
    base: dict[str, Any] = {"budget": float(spec.budgets[budget_key]), "warmup": spec.warmup}
    base.update(calibrated_alarms(family))
    base.update(overrides or {})
    return RunProtocol(**base)


def calibrated_alarms(family: str) -> dict[str, Any]:
    """Calibrated alarm thresholds for ``family`` (``src/probe/diagnostics/calibration.json``,
    written by ``probe calibrate``), or ``{}`` to keep the protocol's uncalibrated defaults."""
    import json
    from pathlib import Path

    f = Path(__file__).parents[1] / "diagnostics" / "calibration.json"
    if not f.is_file():
        return {}
    entry = json.loads(f.read_text(encoding="utf-8")).get(family)
    if not entry:
        return {}
    keys = (
        "alarm_residual_weight",
        "alarm_surprise_mean",
        "alarm_pit_ks",
        "alarm_persistence",
        "alarm_min_obs",
    )
    out = {k: entry[k] for k in keys}
    out["alarm_calibration"] = entry["calibration_id"]
    return out


def run_world(
    family: str,
    regime: str,
    seed: int,
    policy_id: str,
    budget_key: str = "medium",
    protocol_overrides: Mapping[str, Any] | None = None,
    **run_kwargs: Any,
) -> tuple[RunRecord, dict[str, Any]]:
    fam = get_family(family)
    world = fam.make_world(regime, seed)
    hs = fam.hypothesis_set(regime)
    protocol = make_protocol(family, regime, budget_key, protocol_overrides)
    if policy_id == "P10":
        policy = make_policy("P10", opened_truth=world.truth.open(EvaluatorKey("oracle policy")))
    else:
        policy = make_policy(policy_id)
    record = run_experiment(world.env, hs, protocol, policy, seed, **run_kwargs)
    metrics = evaluate_run(record, world.truth)
    metrics.update(
        {
            "family": family,
            "regime": regime,
            "seed": seed,
            "policy": policy_id,
            "budget": budget_key,
        }
    )
    return record, metrics
