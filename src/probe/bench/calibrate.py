"""Alarm-threshold calibration on the simulator (handover §12 H.7).

The diagnostics are not uniform or zero-mean for a *fixed* truth, so alarm
thresholds cannot come from theory. They are calibrated here:

* run the claimed policy (P8) with **alarms disabled** on calibration seeds
  ``10_000+`` (disjoint from benchmark seeds ``0..n``) in well-specified regimes
  (the truth is exactly one registered hypothesis, nominal noise) and in
  misspecified regimes (truth outside the set, drift, heavy tails, unmodelled
  sensor offsets, deploys);
* pick thresholds with false-alarm rate ≤ 10% on the well-specified runs and the
  highest detection on the misspecified runs
  (:func:`probe.diagnostics.calibrate_thresholds`);
* write them, with the achieved rates, to
  ``src/probe/diagnostics/calibration.json``. :func:`probe.run.make_protocol` reads
  that file, so new protocols carry the calibrated values and their provenance id.

The result is reported as *calibrated alarms*, never as anytime-valid tests.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from probe.determinism import content_hash
from probe.diagnostics import DiagnosticTrace, calibrate_thresholds

__all__ = ["CALIBRATION_FILE", "REGIMES", "calibrate"]

CALIBRATION_FILE = Path(__file__).parents[1] / "diagnostics" / "calibration.json"
SEED_BASE = 10_000
REGIMES = {
    "microclimate": {"well": ["R1", "R2", "R4", "R10"], "mis": ["R3", "R7", "R8", "R9"]},
    "systemlab": {"well": ["R1", "R2", "R10"], "mis": ["R3", "R7", "R11"]},
}
OFF = {"alarm_residual_weight": None, "alarm_surprise_mean": None, "alarm_pit_ks": None}


def _trace(job: tuple[str, str, int, str]) -> dict[str, list[float]]:
    from probe.run import run_world

    family, regime, seed, budget = job
    rec, _ = run_world(family, regime, seed, "P8", budget, protocol_overrides=OFF)
    t: dict[str, list[float]] = rec.extra["alarm_trace"]
    return t


def _to_trace(d: dict[str, list[float]]) -> DiagnosticTrace:
    return DiagnosticTrace(
        n_obs=np.asarray(d["n_obs"], dtype=np.int64),
        residual_weight=np.asarray(d["residual_weight"]),
        surprise_mean=np.asarray(d["surprise_mean"]),
        pit_ks=np.asarray(d["pit_ks"]),
    )


def calibrate(
    seeds: int = 30, workers: int = 1, budget: str = "large", write: bool = True
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family, groups in REGIMES.items():
        traces: dict[str, list[DiagnosticTrace]] = {}
        for kind in ("well", "mis"):
            jobs = [(family, r, SEED_BASE + s, budget) for r in groups[kind] for s in range(seeds)]
            if workers > 1:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    raw = list(ex.map(_trace, jobs, chunksize=2))
            else:
                raw = [_trace(j) for j in jobs]
            traces[kind] = [_to_trace(d) for d in raw if d["n_obs"]]
        cal = calibrate_thresholds(traces["well"], traces["mis"])
        th = cal.thresholds
        entry = {
            "alarm_residual_weight": None
            if not np.isfinite(th.residual_weight)
            else float(th.residual_weight),
            "alarm_surprise_mean": None
            if not np.isfinite(th.surprise_mean)
            else float(th.surprise_mean),
            "alarm_pit_ks": None if not np.isfinite(th.pit_ks) else float(th.pit_ks),
            "alarm_persistence": th.persistence,
            "alarm_min_obs": th.min_obs,
            "report": cal.to_dict(),
            "regimes": groups,
            "seeds": [SEED_BASE, SEED_BASE + seeds - 1],
            "budget": budget,
            "n_traces": {k: len(v) for k, v in traces.items()},
        }
        entry["calibration_id"] = f"{family}-" + content_hash(entry)[:12]
        out[family] = entry
    if write:
        CALIBRATION_FILE.write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    return out
