"""Run protocol, the headless run loop and evaluator-side scoring.

Exports are resolved lazily: :mod:`probe.select` imports
:mod:`probe.run.protocol`, and the run loop imports :mod:`probe.select`, so an
eager import here would be circular.
"""

from __future__ import annotations

import importlib
from typing import Any

from probe.run.protocol import RunProtocol

_LAZY = {
    "acceptable_verdicts": "probe.run.evaluate",
    "evaluate_run": "probe.run.evaluate",
    "verdict_string": "probe.run.evaluate",
    "RunRecord": "probe.run.loop",
    "StepRecord": "probe.run.loop",
    "run_experiment": "probe.run.loop",
    "FAMILIES": "probe.run.world",
    "get_family": "probe.run.world",
    "make_protocol": "probe.run.world",
    "run_world": "probe.run.world",
}

__all__ = ["RunProtocol", *sorted(_LAZY)]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(name)
