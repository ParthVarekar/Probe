"""Calibrated misspecification alarms (handover §12 H.7).

Three signals are monitored after every observed outcome:

* ``residual_weight``: the posterior weight of the residual hypothesis;
* ``surprise_mean``: the running mean of the standardized surprise
  z_t = (−log f_t(y_t) − H_t)/√V_t (:mod:`probe.diagnostics.pit`): realized minus
  expected information, in units of its own predictive sd;
* ``pit_ks``: the Kolmogorov–Smirnov distance of the PIT values so far from
  uniform.

A signal *exceeds* when its value is ≥ its threshold (an infinite threshold
disables the signal) and at least ``min_obs`` outcomes have been observed. A
signal is *persistent* once it has exceeded at each of the last ``persistence``
updates. The alarm is **raised** when any signal is persistent and, by default,
stays raised (latched): the guarded policy's fallback (raise the audit fraction,
H.7 (a)) must not flicker, and the stopping rule turns a raised alarm into
"none fits" (H.6 rule 4).

**These are calibrated alarms, NOT anytime-valid tests.** z_t and the PIT are
exactly centred/uniform only under the prior predictive of the hypothesis set
(:mod:`probe.diagnostics.pit`); for a fixed true parameter, and under the repeated
looks of sequential monitoring, no closed-form error rate holds. Thresholds are
therefore chosen by simulation (:func:`calibrate_thresholds`): from diagnostic
traces of well-specified runs (truth inside the set) and misspecified runs
(truth outside), pick the thresholds that maximise the detection rate subject
to a false-alarm rate ≤ the target (H.7: ≤ 10 % false alarms, ≥ 70 % detection
targeted). The resulting rates are *empirical, simulator-specific* properties
and must be reported as such, together with the simulator, regimes and seeds
used. Nothing here enters a likelihood.

The monitor and the calibration use the same rule: a run's alarm fires at
threshold c for a signal iff its *critical value* max_t min_{s ∈ window(t)} x_s
(over the windows of ``persistence`` consecutive eligible updates) is ≥ c, which
:func:`critical_values` computes and the tests check against the incremental
:class:`AlarmMonitor`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from probe.diagnostics.pit import ObservationDiagnostics, RunningDiagnostics

__all__ = [
    "SIGNALS",
    "AlarmCalibration",
    "AlarmMonitor",
    "AlarmState",
    "AlarmThresholds",
    "DiagnosticTrace",
    "calibrate_thresholds",
    "critical_values",
]

SIGNALS: tuple[str, ...] = ("residual_weight", "surprise_mean", "pit_ks")


def _jf(x: float) -> float | str:
    f = float(x)
    if math.isfinite(f):
        return f
    return "nan" if math.isnan(f) else ("inf" if f > 0 else "-inf")


@dataclass(frozen=True)
class AlarmThresholds:
    """Alarm thresholds (defaults are placeholders until calibrated on the
    simulator; ``math.inf`` disables a signal)."""

    residual_weight: float = 0.5
    surprise_mean: float = 1.0
    pit_ks: float = 0.4
    persistence: int = 3
    """Consecutive exceeding updates before a signal counts."""
    min_obs: int = 8
    """Observed outcomes before any signal may exceed."""

    def __post_init__(self) -> None:
        for name in SIGNALS:
            v = float(getattr(self, name))
            if math.isnan(v) or v <= 0.0:
                raise ValueError(f"threshold {name} must be positive (inf disables it)")
        if self.persistence < 1 or self.min_obs < 0:
            raise ValueError("persistence must be >= 1 and min_obs >= 0")

    def value(self, signal: str) -> float:
        if signal not in SIGNALS:
            raise KeyError(signal)
        return float(getattr(self, signal))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {s: _jf(self.value(s)) for s in SIGNALS}
        out["persistence"] = self.persistence
        out["min_obs"] = self.min_obs
        return out


@dataclass(frozen=True)
class DiagnosticTrace:
    """Per-update values of the three signals over one run (one entry per observed
    outcome, in order). ``n_obs`` is the observation count after each update."""

    n_obs: np.ndarray
    residual_weight: np.ndarray
    surprise_mean: np.ndarray
    pit_ks: np.ndarray

    def __post_init__(self) -> None:
        n = np.asarray(self.n_obs).shape
        if len(n) != 1:
            raise ValueError("trace arrays must be 1-D")
        for s in SIGNALS:
            a = np.asarray(getattr(self, s), dtype=np.float64)
            if a.shape != n:
                raise ValueError("trace arrays must have equal length")
            if np.any(np.isnan(a)):
                raise ValueError(f"NaN in trace signal {s}")

    @property
    def length(self) -> int:
        return int(np.asarray(self.n_obs).shape[0])

    def signal(self, name: str) -> np.ndarray:
        if name not in SIGNALS:
            raise KeyError(name)
        return np.asarray(getattr(self, name), dtype=np.float64)


@dataclass(frozen=True)
class AlarmState:
    """The monitor after one update."""

    n_obs: int
    raised: bool
    values: Mapping[str, float]
    exceeding: Mapping[str, bool]
    persistent: Mapping[str, bool]
    first_raised_n_obs: int | None
    """Observation count at which the alarm was first raised."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_obs": self.n_obs,
            "raised": self.raised,
            "values": {k: _jf(v) for k, v in self.values.items()},
            "exceeding": dict(self.exceeding),
            "persistent": dict(self.persistent),
            "first_raised_n_obs": self.first_raised_n_obs,
        }


class AlarmMonitor:
    """Incremental alarm (module docstring). Call :meth:`update` once per *observed*
    outcome with that outcome's :class:`~probe.diagnostics.pit.ObservationDiagnostics`
    and the residual's posterior weight after the update; a missing outcome changes
    nothing and is not passed."""

    def __init__(self, thresholds: AlarmThresholds | None = None, *, latch: bool = True) -> None:
        self.thresholds = thresholds or AlarmThresholds()
        self.latch = latch
        self._running = RunningDiagnostics()
        self._consecutive = dict.fromkeys(SIGNALS, 0)
        self._raised = False
        self._first: int | None = None
        self._trace: dict[str, list[float]] = {s: [] for s in SIGNALS}
        self._n_trace: list[int] = []
        self._state = AlarmState(
            n_obs=0,
            raised=False,
            values=dict.fromkeys(SIGNALS, 0.0),
            exceeding=dict.fromkeys(SIGNALS, False),
            persistent=dict.fromkeys(SIGNALS, False),
            first_raised_n_obs=None,
        )

    @property
    def state(self) -> AlarmState:
        return self._state

    @property
    def raised(self) -> bool:
        return self._state.raised

    @property
    def running(self) -> RunningDiagnostics:
        return self._running

    def update(self, diagnostics: ObservationDiagnostics, residual_weight: float) -> AlarmState:
        rw = float(residual_weight)
        if not 0.0 <= rw <= 1.0:
            raise ValueError("residual weight must lie in [0, 1]")
        self._running.update(diagnostics)
        n = self._running.n
        values = {
            "residual_weight": rw,
            "surprise_mean": self._running.surprise_mean(),
            "pit_ks": self._running.pit_ks(),
        }
        th = self.thresholds
        eligible = n >= th.min_obs
        exceeding: dict[str, bool] = {}
        persistent: dict[str, bool] = {}
        for s in SIGNALS:
            c = th.value(s)
            ex = eligible and math.isfinite(c) and values[s] >= c
            self._consecutive[s] = self._consecutive[s] + 1 if ex else 0
            exceeding[s] = ex
            persistent[s] = self._consecutive[s] >= th.persistence
        now = any(persistent.values())
        raised = now or (self.latch and self._raised)
        if raised and self._first is None:
            self._first = n
        self._raised = raised
        for s in SIGNALS:
            self._trace[s].append(values[s])
        self._n_trace.append(n)
        self._state = AlarmState(
            n_obs=n,
            raised=raised,
            values=values,
            exceeding=exceeding,
            persistent=persistent,
            first_raised_n_obs=self._first,
        )
        return self._state

    def trace(self) -> DiagnosticTrace:
        """The signal values so far, for calibration and reports."""
        return DiagnosticTrace(
            n_obs=np.asarray(self._n_trace, dtype=np.int64),
            residual_weight=np.asarray(self._trace["residual_weight"]),
            surprise_mean=np.asarray(self._trace["surprise_mean"]),
            pit_ks=np.asarray(self._trace["pit_ks"]),
        )


# --------------------------------------------------------------------------- calibration


def critical_values(trace: DiagnosticTrace, *, persistence: int, min_obs: int) -> dict[str, float]:
    """Per signal, the largest threshold at which the alarm would fire on this run:
    max over windows of ``persistence`` consecutive eligible updates (n_obs ≥
    ``min_obs``) of the window minimum; −inf when the run has no full window."""
    if persistence < 1 or min_obs < 0:
        raise ValueError("persistence must be >= 1 and min_obs >= 0")
    eligible = np.asarray(trace.n_obs) >= min_obs
    out: dict[str, float] = {}
    for s in SIGNALS:
        x = trace.signal(s)
        best = -math.inf
        run: list[float] = []
        for t in range(x.shape[0]):
            if not eligible[t]:
                run = []
                continue
            run.append(float(x[t]))
            if len(run) >= persistence:
                best = max(best, min(run[-persistence:]))
        out[s] = best
    return out


@dataclass(frozen=True)
class AlarmCalibration:
    """Result of :func:`calibrate_thresholds` (empirical rates on the given traces)."""

    thresholds: AlarmThresholds
    false_alarm_rate: float
    detection_rate: float
    n_well_specified: int
    n_misspecified: int
    target_false_alarm: float
    target_detection: float
    meets_detection_target: bool
    per_signal: Mapping[str, Mapping[str, float | str]] = field(default_factory=dict)
    """For each signal alone at its chosen threshold: false-alarm and detection rates."""
    exact: bool = True
    """False when the candidate thresholds were subsampled (``max_candidates``)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": self.thresholds.to_dict(),
            "false_alarm_rate": self.false_alarm_rate,
            "detection_rate": self.detection_rate,
            "n_well_specified": self.n_well_specified,
            "n_misspecified": self.n_misspecified,
            "target_false_alarm": self.target_false_alarm,
            "target_detection": self.target_detection,
            "meets_detection_target": self.meets_detection_target,
            "per_signal": {k: dict(v) for k, v in self.per_signal.items()},
            "exact": self.exact,
            "note": "calibrated on simulated traces; not an anytime-valid test",
        }


def _candidates(mis: np.ndarray, max_candidates: int) -> tuple[np.ndarray, bool]:
    """Candidate thresholds for one signal: the misspecified runs' finite, positive
    critical values (optimal thresholds can always be raised to one of them without
    losing detection), subsampled by rank if there are too many, plus +inf."""
    vals = np.unique(mis[np.isfinite(mis) & (mis > 0.0)])
    exact = vals.size <= max_candidates
    if not exact:
        idx = np.unique(np.round(np.linspace(0, vals.size - 1, max_candidates)).astype(np.int64))
        vals = vals[idx]
    return np.concatenate([vals, [math.inf]]), exact


def calibrate_thresholds(
    well_specified: Sequence[DiagnosticTrace],
    misspecified: Sequence[DiagnosticTrace],
    *,
    target_false_alarm: float = 0.10,
    target_detection: float = 0.70,
    persistence: int = 3,
    min_obs: int = 8,
    signals: Sequence[str] = SIGNALS,
    max_candidates: int = 64,
) -> AlarmCalibration:
    """Thresholds maximising the detection rate (fraction of misspecified runs whose
    alarm is ever raised) subject to the false-alarm rate (fraction of well-specified
    runs whose alarm is ever raised) ≤ ``target_false_alarm``; ties go to the lower
    false-alarm rate, then to the larger (more conservative) thresholds. Signals not
    in ``signals`` are disabled (+inf). Disabling everything is always feasible, so a
    result always exists; ``meets_detection_target`` says whether the H.7 target was
    reached. Deterministic (exhaustive search over the candidate grid)."""
    if not well_specified or not misspecified:
        raise ValueError("need at least one well-specified and one misspecified trace")
    if not 0.0 <= target_false_alarm < 1.0:
        raise ValueError("target_false_alarm must lie in [0, 1)")
    unknown = set(signals) - set(SIGNALS)
    if unknown:
        raise ValueError(f"unknown signals {sorted(unknown)}")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")

    def crit(traces: Sequence[DiagnosticTrace]) -> np.ndarray:
        rows = [critical_values(t, persistence=persistence, min_obs=min_obs) for t in traces]
        return np.asarray([[r[s] for s in SIGNALS] for r in rows])  # (N, 3)

    cw, cm = crit(well_specified), crit(misspecified)
    grids: list[np.ndarray] = []
    exact = True
    for k, s in enumerate(SIGNALS):
        if s in signals:
            g, ex = _candidates(cm[:, k], max_candidates)
            exact = exact and ex
        else:
            g = np.asarray([math.inf])
        grids.append(g)

    def fires(c: np.ndarray, g: np.ndarray) -> np.ndarray:
        # (G, N): run fires at threshold g[i] (an infinite threshold never fires)
        return np.isfinite(g)[:, None] & (c[None, :] >= g[:, None])

    fw = [fires(cw[:, k], grids[k]) for k in range(3)]
    fm = [fires(cm[:, k], grids[k]) for k in range(3)]
    # exhaustive search, one slab of the (G0, G1, G2) grid at a time (memory G1·G2·N)
    best: tuple[float, float, int, tuple[int, int, int]] | None = None
    best_rates = (0.0, 0.0)
    for i0 in range(grids[0].size):
        slab_w = fw[0][i0][None, None, :] | fw[1][:, None, :] | fw[2][None, :, :]
        slab_m = fm[0][i0][None, None, :] | fm[1][:, None, :] | fm[2][None, :, :]
        fa = slab_w.mean(axis=-1)
        det = slab_m.mean(axis=-1)
        for i1, i2 in zip(*np.nonzero(fa <= target_false_alarm + 1e-12), strict=True):
            i = (i0, int(i1), int(i2))
            key = (-float(det[i1, i2]), float(fa[i1, i2]), -(i[0] + i[1] + i[2]), i)
            if best is None or key < best:
                best = key
                best_rates = (float(fa[i1, i2]), float(det[i1, i2]))
    assert best is not None  # all-disabled is always feasible
    ii = best[3]
    chosen = {s: float(grids[k][ii[k]]) for k, s in enumerate(SIGNALS)}
    per_signal: dict[str, dict[str, float | str]] = {
        s: {
            "threshold": _jf(chosen[s]),
            "false_alarm_rate": float(fw[k][ii[k]].mean()),
            "detection_rate": float(fm[k][ii[k]].mean()),
        }
        for k, s in enumerate(SIGNALS)
    }
    thresholds = AlarmThresholds(
        residual_weight=chosen["residual_weight"],
        surprise_mean=chosen["surprise_mean"],
        pit_ks=chosen["pit_ks"],
        persistence=persistence,
        min_obs=min_obs,
    )
    fa_best, d = best_rates
    return AlarmCalibration(
        thresholds=thresholds,
        false_alarm_rate=fa_best,
        detection_rate=d,
        n_well_specified=len(well_specified),
        n_misspecified=len(misspecified),
        target_false_alarm=target_false_alarm,
        target_detection=target_detection,
        meets_detection_target=d >= target_detection,
        per_signal=per_signal,
        exact=exact,
    )
