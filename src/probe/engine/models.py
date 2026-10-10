"""The interface every hypothesis engine implements.

Concrete engines: :class:`probe.engine.nig.NIGModel` (``linear_gaussian``) and
:class:`probe.engine.grid.GridModel` (``finite_grid``). Build them with
:func:`probe.engine.factory.build_model`.

Contract (all engines):

* A model is created at the hypothesis's *registration* and only ever sees data
  passed to :meth:`update` afterwards, which gives registration-time evidence
  (handover §12 H.4) by construction.
* :meth:`predictive` is the one-step-ahead posterior predictive at each
  candidate record, from the *current* state, without mutating it.
* :meth:`log_predictive` is evaluated **before** :meth:`update` with the same
  observation; the sum of these prequential terms equals :meth:`log_evidence`
  (checked in tests).
* :meth:`update` with ``y=None`` records that the action happened but produced no
  usable outcome (missing). Linear models ignore it; stateful grid mechanisms
  still advance their internal state.
* :meth:`max_log_likelihood` is sup_θ log p(y_obs | θ, m) over all observed data,
  for universal-inference e-process denominators. It may be ``+inf`` when the
  model can interpolate the data (e.g. n ≤ p with free σ²); callers treat
  that as "cannot reject".
* Everything is deterministic: no hidden randomness, no wall-clock, no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from probe.engine.ir import Record
from probe.engine.predictive import Predictive

__all__ = ["HypothesisModel"]


@runtime_checkable
class HypothesisModel(Protocol):
    hypothesis_id: str
    is_residual: bool

    @property
    def n_obs(self) -> int:
        """Number of non-missing observations conditioned on."""
        ...

    def predictive(self, records: Sequence[Record]) -> Predictive: ...

    def log_predictive(self, record: Record, y: float | int) -> float: ...

    def update(self, record: Record, y: float | int | None) -> None: ...

    def log_evidence(self) -> float: ...

    def max_log_likelihood(self) -> float: ...

    def parameter_eig(self, records: Sequence[Record]) -> np.ndarray:
        """I(θ; Y_d | m, D) per record (nats), the secondary objective U_θ|m."""
        ...

    def clone(self) -> HypothesisModel:
        """An independent deep copy (for look-ahead and replay checks)."""
        ...

    def summary(self) -> dict[str, Any]:
        """JSON-able posterior summary for reports (means, sds, n, evidence)."""
        ...
