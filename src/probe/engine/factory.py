"""Build hypothesis engines from IR.

``build_model`` maps one registered hypothesis to its exact engine:
``linear_gaussian`` → :class:`~probe.engine.nig.NIGModel` (needs the set's
covariate schema) and ``finite_grid`` → :class:`~probe.engine.grid.GridModel`
(needs the set's count outcome for the support 0..max_count). A fresh model has
seen no data; registering it later and feeding it only subsequent observations
gives registration-time evidence (handover §12 H.4).
"""

from __future__ import annotations

from probe.engine.grid import GridModel
from probe.engine.ir import GridHypothesis, Hypothesis, HypothesisSet, LinearGaussianHypothesis
from probe.engine.nig import NIGModel

__all__ = ["EngineModel", "build_model", "build_models"]

EngineModel = NIGModel | GridModel


def build_model(h: Hypothesis, hypothesis_set: HypothesisSet) -> EngineModel:
    """A fresh engine for ``h``, which must be a member of ``hypothesis_set``
    (registered hypotheses only, so the covariate schema and outcome match)."""
    try:
        registered = hypothesis_set.by_id(h.id)
    except KeyError as e:
        raise ValueError(f"hypothesis {h.id!r} is not registered in this set") from e
    if registered != h:
        raise ValueError(f"hypothesis {h.id!r} differs from the registered version")
    if isinstance(h, LinearGaussianHypothesis):
        return NIGModel(h, hypothesis_set.covariates)
    if isinstance(h, GridHypothesis):
        return GridModel(h, hypothesis_set.outcome)
    raise TypeError(f"no engine for hypothesis kind {type(h).__name__}")


def build_models(hypothesis_set: HypothesisSet) -> list[EngineModel]:
    """One fresh engine per hypothesis, in set order."""
    return [build_model(h, hypothesis_set) for h in hypothesis_set.hypotheses]
