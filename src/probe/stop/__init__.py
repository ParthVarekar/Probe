"""Stopping rules, e-processes and confidence sequences (handover §12 H.6, H.8).

* :mod:`probe.stop.base`: the verdict contract (Verdict, StopReason, StopDecision).
* :mod:`probe.stop.rules`: :class:`StoppingRule` (the H.6 rules, in order) over a
  :class:`StopState`; :func:`information_rate` for the "nothing informative" rule.
* :mod:`probe.stop.eprocess`: universal-inference e-processes between hypotheses
  (:class:`EProcessTracker`), e-BH and e-to-p conversion.
* :mod:`probe.stop.confseq`: anytime-valid confidence sequences for means
  (sub-Gaussian normal mixture, bounded empirical Bernstein / Hoeffding).
"""

from probe.stop.base import StopDecision, StopReason, Verdict
from probe.stop.confseq import (
    BoundedMeanCS,
    ConfidenceSequence,
    SubGaussianMeanCS,
    bounded_cs,
    subgaussian_cs,
)
from probe.stop.eprocess import EProcessTracker, e_to_p, ebh, json_float, log_e_value
from probe.stop.rules import StoppingRule, StopState, information_rate

__all__ = [
    "BoundedMeanCS",
    "ConfidenceSequence",
    "EProcessTracker",
    "StopDecision",
    "StopReason",
    "StopState",
    "StoppingRule",
    "SubGaussianMeanCS",
    "Verdict",
    "bounded_cs",
    "e_to_p",
    "ebh",
    "information_rate",
    "json_float",
    "log_e_value",
    "subgaussian_cs",
]
