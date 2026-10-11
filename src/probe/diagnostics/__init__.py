"""Misspecification diagnostics, alarms and identifiability (handover §12 H.7, H.11).

* :mod:`probe.diagnostics.pit`: PIT, surprise and predictive entropy of each
  observation under the model-averaged predictive; running KS / surprise means.
* :mod:`probe.diagnostics.alarms`: :class:`AlarmMonitor` (residual weight,
  standardized surprise, PIT KS) and :func:`calibrate_thresholds`. Calibrated
  alarms, not anytime-valid tests.
* :mod:`probe.diagnostics.identifiability`: pairwise predictive divergences and
  structural nesting over the feasible designs, with warnings (H.11).
"""

from probe.diagnostics.alarms import (
    AlarmCalibration,
    AlarmMonitor,
    AlarmState,
    AlarmThresholds,
    DiagnosticTrace,
    calibrate_thresholds,
)
from probe.diagnostics.identifiability import (
    IdentifiabilityReport,
    identifiability_report,
    nesting_matrix,
)
from probe.diagnostics.pit import (
    ObservationDiagnostics,
    RunningDiagnostics,
    observation_diagnostics,
)

__all__ = [
    "AlarmCalibration",
    "AlarmMonitor",
    "AlarmState",
    "AlarmThresholds",
    "DiagnosticTrace",
    "IdentifiabilityReport",
    "ObservationDiagnostics",
    "RunningDiagnostics",
    "calibrate_thresholds",
    "identifiability_report",
    "nesting_matrix",
    "observation_diagnostics",
]
