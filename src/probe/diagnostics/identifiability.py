"""Identifiability check at protocol validation (handover §12 H.11, H.12).

Before any data are collected, can the registered hypotheses be told apart with
the *available* actions? Two complementary answers are computed over a set of
design records (the feasible design space, e.g. the design records of the
declared population; choosing a representative set is the caller's job):

1. **Predictive divergence.** For every design d and ordered pair (i, j), the
   one-step KL divergence KL(p_i(· | d) ‖ p_j(· | d)) between the hypotheses'
   current predictives (the *prior* predictives when the models are fresh):

   * discrete outcomes: exact, Σ_y p_i(y) log(p_i(y)/p_j(y)) (+inf when p_j
     misses an outcome p_i allows);
   * Student-t outcomes: an approximation, the closed-form KL between the
     moment-matched Gaussians N(μ, v),
     ½[log(v_j/v_i) + (v_i + (μ_i − μ_j)²)/v_j − 1], with v the predictive
     variance (scale² when ν ≤ 2, where the variance is infinite; a cruder
     approximation, flagged in the report).

   Reported per pair: the mean over designs, and the maximum over designs of the
   symmetric Jeffreys divergence J = KL_ij + KL_ji (what the single best action
   separates per observation). A pair whose best J is below ``min_divergence`` is
   warned as **weak** (H.12 "flat information landscape").

2. **Observational equivalence / nesting** (:func:`nesting_matrix`). Prior
   predictives can differ only because the priors differ, which data cannot
   resolve. The structural question is whether one hypothesis's *model class*
   contains the other's on the design space:

   * ``linear_gaussian``: class i ⊆ class j iff the columns of Φ_i (features at
     the designs) lie in the column span of Φ_j (both have a free σ²). Checked
     by orthogonal projection with a relative tolerance.
   * ``finite_grid``: i ⊆ j iff both use the same mechanism and every grid point
     of i (parameters including fixed ones) is a grid point of j. Exact, and
     valid for stateful mechanisms, since equal parameters give equal state
     evolution.

   Mutual nesting means the hypotheses are **observationally equivalent** with
   these actions: no data can tell them apart (e.g. a sensor-artefact hypothesis
   and a shade hypothesis when no shielded readings are possible, H.11). One-way
   nesting is reported as information: the e-process of the nested hypothesis
   against its container cannot grow (:mod:`probe.stop.eprocess`), so a decision
   for it rests on the posterior alone; the stopping rule exempts that pair via
   ``StopState.nests``.

All quantities are deterministic. The divergences are per *single* observation
at the models' current state; for stateful grid mechanisms they describe the
next action only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import special

from probe.engine.grid import GridModel
from probe.engine.ir import Record
from probe.engine.models import HypothesisModel
from probe.engine.nig import NIGModel
from probe.engine.predictive import DiscretePredictive, Predictive, StudentTPredictive

__all__ = [
    "IdentifiabilityReport",
    "IdentifiabilityWarning",
    "identifiability_report",
    "nesting_matrix",
    "predictive_kl",
]

_EPS = float(np.finfo(np.float64).eps)


def _gauss_moments(p: StudentTPredictive) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, matched variance, infinite-variance flag) per design."""
    var = p.variance()
    inf = ~np.isfinite(var)
    return p.loc, np.where(inf, p.scale**2, var), inf


def predictive_kl(p: Predictive, q: Predictive) -> np.ndarray:
    """KL(p ‖ q) per design (aligned batches): exact for discrete predictives,
    moment-matched Gaussians for Student-t (module docstring)."""
    if p.n != q.n:
        raise ValueError("predictive batches must cover the same designs")
    if isinstance(p, DiscretePredictive) and isinstance(q, DiscretePredictive):
        if p.probs.shape != q.probs.shape:
            raise ValueError("discrete predictives must share the support")
        a = np.clip(p.probs, 0.0, None)
        b = np.clip(q.probs, 0.0, None)
        pos = a > 0.0
        with np.errstate(divide="ignore"):
            terms = np.where(pos, special.xlogy(a, a) - a * np.log(np.where(pos, b, 1.0)), 0.0)
        kl = np.sum(terms, axis=1)
        missing = np.any(pos & (b <= 0.0), axis=1)
        return np.where(missing, np.inf, np.maximum(kl, 0.0))
    if isinstance(p, StudentTPredictive) and isinstance(q, StudentTPredictive):
        mp, vp, _ = _gauss_moments(p)
        mq, vq, _ = _gauss_moments(q)
        d = mp - mq
        kl = 0.5 * (np.log(vq / vp) + (vp + d * d) / vq - 1.0)
        return np.maximum(kl, 0.0)
    raise TypeError("predictives must both be discrete or both Student-t")


def _orth_basis(x: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column span of x (numpy's rank rule)."""
    if x.size == 0:
        return np.zeros((x.shape[0], 0))
    u, sv, _ = np.linalg.svd(x, full_matrices=False)
    tol = (float(sv[0]) if sv.size else 0.0) * max(x.shape) * _EPS
    return u[:, sv > tol]


def _grid_points(m: GridModel) -> tuple[str, tuple[str, ...], set[tuple[Any, ...]]]:
    """(mechanism, parameter names, set of grid points incl. fixed parameters)."""
    table = m.grid_table()
    names = tuple(sorted(table))
    rows = {tuple(_plain(table[n][g]) for n in names) for g in range(m.grid_size)}
    return str(m.summary()["mechanism"]), names, rows


def _plain(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def nesting_matrix(
    models: Sequence[HypothesisModel], records: Sequence[Record], *, rtol: float = 1e-8
) -> np.ndarray:
    """(K, K) bool, ``[i, j]`` True when hypothesis j's model class contains
    hypothesis i's on the given designs (module docstring). The diagonal is True.
    Pairs of different engine kinds are never nested."""
    k = len(models)
    out = np.eye(k, dtype=bool)
    if not records:
        raise ValueError("need at least one design record")
    feats: list[np.ndarray | None] = []
    grids: list[tuple[str, tuple[str, ...], set[tuple[Any, ...]]] | None] = []
    for m in models:
        if isinstance(m, NIGModel):
            feats.append(np.asarray(m.features(records), dtype=np.float64))
            grids.append(None)
        elif isinstance(m, GridModel):
            feats.append(None)
            grids.append(_grid_points(m))
        else:
            raise TypeError(f"no nesting check for {type(m).__name__}")
    bases = [None if f is None else _orth_basis(f) for f in feats]
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            fi, bj = feats[i], bases[j]
            if fi is not None and bj is not None:
                resid = fi - bj @ (bj.T @ fi)
                scale = float(np.linalg.norm(fi))
                out[i, j] = float(np.linalg.norm(resid)) <= rtol * max(scale, 1e-300)
            gi, gj = grids[i], grids[j]
            if gi is not None and gj is not None:
                out[i, j] = gi[0] == gj[0] and gi[1] == gj[1] and gi[2] <= gj[2]
    return out


@dataclass(frozen=True)
class IdentifiabilityWarning:
    pair: tuple[str, str]
    kind: Literal["equivalent", "weak", "nested"]
    severity: Literal["warning", "info"]
    max_jeffreys: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": list(self.pair),
            "kind": self.kind,
            "severity": self.severity,
            "max_jeffreys": _jf(self.max_jeffreys),
            "message": self.message,
        }


def _jf(x: float) -> float | str:
    f = float(x)
    if math.isfinite(f):
        return f
    return "nan" if math.isnan(f) else ("inf" if f > 0 else "-inf")


@dataclass(frozen=True)
class IdentifiabilityReport:
    hypothesis_ids: tuple[str, ...]
    n_designs: int
    kl_mean: np.ndarray
    """(K, K) mean over designs of KL(p_i ‖ p_j)."""
    kl_max: np.ndarray
    """(K, K) max over designs of KL(p_i ‖ p_j)."""
    jeffreys_max: np.ndarray
    """(K, K) symmetric, max over designs of KL_ij + KL_ji (0 on the diagonal)."""
    best_design: np.ndarray
    """(K, K) index of the design attaining ``jeffreys_max`` (first on ties; −1 diag)."""
    nests: np.ndarray
    """(K, K) bool, see :func:`nesting_matrix`."""
    equivalent: np.ndarray
    """(K, K) bool, mutual nesting off the diagonal."""
    approximate: bool
    """True when Student-t KLs were moment-matched (always, for continuous outcomes)."""
    infinite_variance: bool
    """True when some predictive had ν ≤ 2 (scale² used as the matched variance)."""
    min_divergence: float
    warnings: tuple[IdentifiabilityWarning, ...]

    @property
    def indistinguishable_pairs(self) -> list[tuple[str, str]]:
        """Pairs warned as equivalent or weak (severity ``warning``)."""
        return [w.pair for w in self.warnings if w.severity == "warning"]

    def to_dict(self) -> dict[str, Any]:
        ids = self.hypothesis_ids

        def mat(a: np.ndarray) -> dict[str, dict[str, Any]]:
            return {
                x: {y: _jf(float(a[i, j])) for j, y in enumerate(ids) if j != i}
                for i, x in enumerate(ids)
            }

        return {
            "hypothesis_ids": list(ids),
            "n_designs": self.n_designs,
            "kl_mean": mat(self.kl_mean),
            "kl_max": mat(self.kl_max),
            "jeffreys_max": mat(self.jeffreys_max),
            "nests": {
                x: [y for j, y in enumerate(ids) if j != i and bool(self.nests[i, j])]
                for i, x in enumerate(ids)
            },
            "approximate": self.approximate,
            "infinite_variance": self.infinite_variance,
            "min_divergence": self.min_divergence,
            "warnings": [w.to_dict() for w in self.warnings],
        }


def identifiability_report(
    models: Sequence[HypothesisModel],
    records: Sequence[Record],
    *,
    min_divergence: float = 1e-2,
    nests: np.ndarray | None = None,
) -> IdentifiabilityReport:
    """Pairwise predictive divergences over ``records`` and structural nesting, with
    warnings for pairs that cannot (equivalent) or can hardly (best Jeffreys
    divergence per observation < ``min_divergence`` nats) be told apart with these
    actions. ``nests`` may be passed if already computed."""
    k = len(models)
    if k < 2:
        raise ValueError("need at least two hypotheses")
    if not records:
        raise ValueError("need at least one design record")
    if not min_divergence > 0.0:
        raise ValueError("min_divergence must be positive")
    ids = tuple(m.hypothesis_id for m in models)
    preds = [m.predictive(records) for m in models]
    n = len(records)
    kl = np.zeros((k, k, n))
    for i in range(k):
        for j in range(k):
            if i != j:
                kl[i, j] = predictive_kl(preds[i], preds[j])
    jeff = kl + np.transpose(kl, (1, 0, 2))
    jmax = np.max(jeff, axis=2)
    best = np.argmax(jeff, axis=2)
    np.fill_diagonal(best, -1)
    nm = nesting_matrix(models, records) if nests is None else np.asarray(nests, dtype=bool)
    if nm.shape != (k, k):
        raise ValueError(f"nests must be ({k}, {k})")
    equiv = nm & nm.T & ~np.eye(k, dtype=bool)
    continuous = isinstance(preds[0], StudentTPredictive)
    inf_var = continuous and any(
        bool(np.any(_gauss_moments(p)[2])) for p in preds if isinstance(p, StudentTPredictive)
    )
    warns: list[IdentifiabilityWarning] = []
    for i in range(k):
        for j in range(i + 1, k):
            pair = (ids[i], ids[j])
            jm = float(jmax[i, j])
            if equiv[i, j]:
                warns.append(
                    IdentifiabilityWarning(
                        pair,
                        "equivalent",
                        "warning",
                        jm,
                        f"{ids[i]} and {ids[j]} span the same model class on the {n} designs: "
                        "no data from these actions can tell them apart (only their priors "
                        "differ). Add actions that separate them or merge the hypotheses.",
                    )
                )
            elif jm < min_divergence:
                warns.append(
                    IdentifiabilityWarning(
                        pair,
                        "weak",
                        "warning",
                        jm,
                        f"the best available action separates {ids[i]} and {ids[j]} by only "
                        f"{jm:.3g} nats per observation (Jeffreys divergence of the current "
                        f"predictives, threshold {min_divergence:.3g}); expect an "
                        "inconclusive result.",
                    )
                )
            if not equiv[i, j] and (nm[i, j] or nm[j, i]):
                small, big = (i, j) if nm[i, j] else (j, i)
                warns.append(
                    IdentifiabilityWarning(
                        (ids[small], ids[big]),
                        "nested",
                        "info",
                        jm,
                        f"{ids[small]} is nested in {ids[big]}: the e-process of {ids[small]} "
                        f"against {ids[big]} cannot grow, so a decision for {ids[small]} rests "
                        "on the posterior (Occam factor) alone.",
                    )
                )
    return IdentifiabilityReport(
        hypothesis_ids=ids,
        n_designs=n,
        kl_mean=np.mean(kl, axis=2),
        kl_max=np.max(kl, axis=2),
        jeffreys_max=jmax,
        best_design=best,
        nests=nm,
        equivalent=equiv,
        approximate=continuous,
        infinite_variance=inf_var,
        min_divergence=min_divergence,
        warnings=tuple(warns),
    )
