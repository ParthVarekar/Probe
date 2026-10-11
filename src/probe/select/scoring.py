"""Per-step candidate scoring, shared by every policy (handover §12 H.5).

:func:`score_candidates` turns the engines' current posteriors into one
:class:`~probe.select.base.ScoredCandidates` table over the *feasible* candidates
(sorted by ``action.id``). The run loop computes it once per step and hands the
same table to whichever policy runs, so policies differ only in how they choose,
never in what they know, and every decision record can carry the full table.

Work per step, each piece computed exactly once:

1. **Predictives.** ``model.predictive(records)`` once per hypothesis over all
   feasible candidates (hypothesis-set order). Everything below reuses them.
2. **U_M** (``u_model``, ``u_model_err``) by
   :func:`probe.engine.eig.model_indicator_eig` on those predictives: exact for
   discrete outcomes, adaptive Gauss–Kronrod with an error estimate for Student-t.
   Its bound 0 ≤ U_M ≤ H(π) ≤ log K is asserted by the kernel; H(π) is kept as
   ``u_model_upper``.
3. **Mixture variance** (``pred_var``) by :func:`probe.engine.eig.mixture_moments`
   on the same predictives.
4. **Parameter EIG** (``u_param`` = Σ_m π_m U_θ|m(d)), which is the secondary objective:

   * ``linear_gaussian`` (NIG) hypotheses: computed *from the predictive already in
     hand*, at no extra cost. I((β, σ²); Y_d) = H[t_{2a}(xᵀμ, s_d)] − ½(log 2πe +
     E[log σ²]) with E[log σ²] = log b − ψ(a), the closed form of
     :meth:`probe.engine.nig.NIGModel.parameter_eig`; (a, b) come from the public
     ``posterior``. The arithmetic is the engine's, operation for operation, so
     the result is bitwise equal to :func:`probe.engine.eig.weighted_parameter_eig`
     (tested).
   * ``finite_grid`` hypotheses: U_θ|m needs the per-grid-point outcome PMFs, which
     :meth:`~probe.engine.grid.GridModel.predictive` mixes away, so it can only come
     from :meth:`~probe.engine.grid.GridModel.parameter_eig`, which re-runs the
     mechanism's ``predict`` for every record and doubles the step cost (measured on
     system lab R1: predictives 21 ms, parameter EIG another 25 ms). It is
     therefore computed **only on request**: ``need_u_param=True``, or by default
     when the protocol's combined objective uses it (``lambda_param > 0``). The run
     loop passes ``need_u_param=requires_u_param(policy, protocol)``, which is True
     only for parameter-BALD (P5) and for the guarded policy with λ > 0. When it
     is skipped, ``u_param`` is all NaN and ``u_param_available`` is False; a
     policy that needs it raises instead of reading NaN.

Candidates are filtered to ``feasible`` and, when ``budget_remaining`` is given, to
those whose cost fits it (a relative slack of 1e-12 absorbs rounding in the spent
total). An empty feasible set gives an empty table (n = 0); the run loop stops
before asking a policy to choose from it.

The table's arrays are read-only: one table is shared by the policy, the stopping
rules and the decision record, so none of them can alter what the others see.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import special

from probe.engine.eig import QuadratureSettings, mixture_moments, model_indicator_eig
from probe.engine.ir import Record
from probe.engine.models import HypothesisModel
from probe.engine.nig import NIGModel
from probe.engine.predictive import Predictive, StudentTPredictive
from probe.envs.base import Candidate
from probe.run.protocol import RunProtocol
from probe.select.base import ScoredCandidates

__all__ = [
    "nig_parameter_eig",
    "requires_u_param",
    "score_candidates",
]

_LOG_2PI_E = math.log(2.0 * math.pi * math.e)
"""The engine's constant, spelled identically (see :mod:`probe.engine.nig`)."""
_BUDGET_SLACK = 1e-12


def requires_u_param(policy: object, protocol: RunProtocol) -> bool:
    """Whether ``policy`` reads ``u_param`` under ``protocol``. Policies of this
    package answer through ``needs_u_param``; any other object is assumed to need
    it (the safe default: compute rather than hand over NaN)."""
    fn = getattr(policy, "needs_u_param", None)
    if fn is None:
        return True
    return bool(fn(protocol))


def nig_parameter_eig(model: NIGModel, predictive: StudentTPredictive) -> np.ndarray:
    """U_θ|m(d) = I((β, σ²); Y_d) for a NIG model from its own current predictive at
    the candidates. Identical arithmetic to :meth:`NIGModel.parameter_eig` (which
    would recompute the predictive)."""
    post = model.posterior
    e_log_sigma2 = math.log(post.b) - float(special.digamma(post.a))
    return np.asarray(predictive.entropy() - 0.5 * (_LOG_2PI_E + e_log_sigma2))


def _feasible(candidates: Sequence[Candidate], budget_remaining: float | None) -> list[Candidate]:
    feas = [c for c in candidates if c.feasible]
    if budget_remaining is not None:
        if math.isnan(budget_remaining):
            raise ValueError("budget_remaining must not be NaN")
        limit = budget_remaining + _BUDGET_SLACK * max(1.0, abs(budget_remaining))
        feas = [c for c in feas if c.cost <= limit]
    feas.sort(key=lambda c: c.action.id)
    ids = [c.action.id for c in feas]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate action ids among the candidates")
    for c in feas:
        if not (math.isfinite(c.cost) and c.cost > 0.0):
            raise ValueError(f"candidate {c.action.id!r} has invalid cost {c.cost!r}")
    return feas


def _frozen(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=np.float64)  # a private copy
    out.setflags(write=False)
    return out


def score_candidates(
    models: Sequence[HypothesisModel],
    weights: np.ndarray | Sequence[float],
    candidates: Sequence[Candidate],
    protocol: RunProtocol,
    *,
    need_u_param: bool | None = None,
    budget_remaining: float | None = None,
    quadrature: QuadratureSettings | None = None,
) -> ScoredCandidates:
    """Score the feasible ``candidates`` under the current posterior ``weights``
    (hypothesis-set order, aligned with ``models``). See the module docstring for
    what is computed and when ``u_param`` is skipped (``need_u_param=None`` means
    "only if ``protocol.lambda_param > 0``"; NIG hypotheses always get it, free)."""
    if len(models) == 0:
        raise ValueError("need at least one hypothesis model")
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (len(models),):
        raise ValueError(f"weights must have shape ({len(models)},), got {w.shape}")
    feas = _feasible(candidates, budget_remaining)
    records = [c.record for c in feas]
    preds: tuple[Predictive, ...] = tuple(m.predictive(records) for m in models)
    res = model_indicator_eig(preds, w, quadrature)
    _, var = mixture_moments(preds, w)
    want = protocol.lambda_param > 0.0 if need_u_param is None else bool(need_u_param)
    u_param, available = _weighted_u_param(models, w, records, preds, want)
    return ScoredCandidates(
        candidates=tuple(feas),
        cost=_frozen(np.asarray([c.cost for c in feas], dtype=np.float64)),
        u_model=_frozen(res.value),
        u_model_err=_frozen(res.error),
        u_param=_frozen(u_param),
        pred_var=_frozen(var),
        predictives=preds,
        u_param_available=available,
        u_model_upper=float(res.upper_bound),
    )


def _weighted_u_param(
    models: Sequence[HypothesisModel],
    w: np.ndarray,
    records: Sequence[Record],
    preds: Sequence[Predictive],
    want: bool,
) -> tuple[np.ndarray, bool]:
    """Σ_m π_m U_θ|m with the accumulation order of
    :func:`probe.engine.eig.weighted_parameter_eig` (so the sums agree bitwise)."""
    n = len(records)
    s = float(w.sum())
    wn = w / s
    active = [i for i in range(len(models)) if wn[i] > 0.0]
    free = all(isinstance(models[i], NIGModel) for i in active)
    if not (want or free):
        return np.full(n, np.nan), False
    out = np.zeros(n)
    for i in active:
        m, p = models[i], preds[i]
        if isinstance(m, NIGModel) and isinstance(p, StudentTPredictive):
            u = nig_parameter_eig(m, p)
        else:
            u = np.asarray(m.parameter_eig(records), dtype=np.float64)
        out = out + wn[i] * u
    return out, True
