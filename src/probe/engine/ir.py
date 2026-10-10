"""Hypothesis intermediate representation (IR), schema v0.

A hypothesis is *data*, never code. Two families exist in v0 (handover §12 H.3):

* ``linear_gaussian``: y = φ_m(d)ᵀβ + ε with a Normal-Inverse-Gamma prior. The
  feature map φ_m is built only from the closed term vocabulary below, evaluated
  against *design records* (the agent-knowable covariates of a candidate action).
* ``finite_grid``: a named mechanism from the in-tree registry
  (:mod:`probe.engine.mechanisms`) with an explicit parameter grid and a discrete
  outcome likelihood. The IR names the mechanism; it never carries code.

A :class:`HypothesisSet` bundles the outcome, the covariate schema and the
competing hypotheses, and validates the cross-references (levels exist, exactly
one residual, one family per set). Everything is hashable with
:func:`probe.determinism.content_hash`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from probe.determinism import content_hash, make_rng

__all__ = [
    "IR_SCHEMA_VERSION",
    "CovariateSpec",
    "FourierTerm",
    "GridHypothesis",
    "Hypothesis",
    "HypothesisSet",
    "IndicatorTerm",
    "InteractionTerm",
    "InterceptTerm",
    "LinearGaussianHypothesis",
    "NIGPrior",
    "NumericTerm",
    "OutcomeSpec",
    "RFFInput",
    "RFFTerm",
    "Record",
    "Term",
    "column_names",
    "design_matrix",
    "ir_hash",
    "prior_arrays",
    "term_signature",
]

IR_SCHEMA_VERSION: Literal["probe.ir/v0"] = "probe.ir/v0"

Record = Mapping[str, float | int | str | bool]
"""A design record: agent-knowable covariate values for one candidate action."""

_ID = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.\-]+$", min_length=1, max_length=64)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- schema


class CovariateSpec(_Frozen):
    name: _ID
    kind: Literal["categorical", "numeric", "binary"]
    levels: tuple[str, ...] | None = None
    unit: str | None = None
    description: str = ""

    @model_validator(mode="after")
    def _levels(self) -> CovariateSpec:
        if self.kind == "categorical" and not self.levels:
            raise ValueError(f"categorical covariate {self.name!r} needs levels")
        if self.kind != "categorical" and self.levels is not None:
            raise ValueError(f"only categorical covariates take levels ({self.name!r})")
        return self


class OutcomeSpec(_Frozen):
    name: _ID
    kind: Literal["continuous", "count"]
    unit: str | None = None
    max_count: int | None = Field(default=None, ge=1, le=10_000)
    description: str = ""

    @model_validator(mode="after")
    def _support(self) -> OutcomeSpec:
        if self.kind == "count" and self.max_count is None:
            raise ValueError("count outcomes need max_count (support is 0..max_count)")
        if self.kind == "continuous" and self.max_count is not None:
            raise ValueError("continuous outcomes take no max_count")
        return self


class _TermBase(_Frozen):
    prior_mean: float = 0.0
    prior_sd: float | None = Field(default=None, gt=0)
    """Prior sd multiplier for this term's coefficients (β | σ² ~ N(m, σ² sd²));
    ``None`` uses the hypothesis default."""


class InterceptTerm(_TermBase):
    kind: Literal["intercept"] = "intercept"


class IndicatorTerm(_TermBase):
    """1[record[covariate] == level] (categorical) or record[covariate] (binary)."""

    kind: Literal["indicator"] = "indicator"
    covariate: _ID
    level: str | None = None


class NumericTerm(_TermBase):
    """(record[covariate] - center) / scale."""

    kind: Literal["numeric"] = "numeric"
    covariate: _ID
    center: float = 0.0
    scale: float = Field(default=1.0, gt=0)


class FourierTerm(_TermBase):
    """sin/cos(2πk·x/period) for k = 1..harmonics (2·harmonics columns)."""

    kind: Literal["fourier"] = "fourier"
    covariate: _ID = "hour"
    period: float = Field(default=24.0, gt=0)
    harmonics: int = Field(default=1, ge=1, le=6)


class RFFInput(_Frozen):
    covariate: _ID
    scale: float = Field(default=1.0, gt=0)
    """Numeric inputs are divided by ``scale``; categorical inputs are one-hot."""


class RFFTerm(_TermBase):
    """Fixed random Fourier features approximating an RBF-kernel GP:
    sqrt(2/D)·cos(W z + b), W ~ N(0, 1/lengthscale²), b ~ U(0, 2π), drawn from
    ``make_rng(seed, "rff")``. Used for the flexible residual hypothesis."""

    kind: Literal["rff"] = "rff"
    inputs: tuple[RFFInput, ...] = Field(min_length=1)
    n_features: int = Field(default=20, ge=1, le=200)
    lengthscale: float = Field(default=1.0, gt=0)
    seed: int = Field(default=0, ge=0)


class InteractionTerm(_TermBase):
    """Elementwise products over the Cartesian product of the factors' columns."""

    kind: Literal["interaction"] = "interaction"
    factors: tuple[IndicatorTerm | NumericTerm | FourierTerm, ...] = Field(
        min_length=2, max_length=3
    )


Term = Annotated[
    InterceptTerm | IndicatorTerm | NumericTerm | FourierTerm | RFFTerm | InteractionTerm,
    Field(discriminator="kind"),
]


class NIGPrior(_Frozen):
    """β | σ² ~ N(μ₀, σ² V₀) with V₀ diagonal built from per-term ``prior_sd``
    (default ``default_sd``); σ² ~ InvGamma(a0, b0)."""

    default_sd: float = Field(default=1.0, gt=0)
    a0: float = Field(default=2.0, gt=0)
    b0: float = Field(default=1.0, gt=0)


class _HypothesisBase(_Frozen):
    id: _ID
    label: str = Field(min_length=1, max_length=120)
    statement: str = Field(min_length=1, max_length=2000)
    applicability: str = ""
    qualitative_prediction: str = ""
    prior_weight: float = Field(default=1.0, gt=0)
    is_residual: bool = False


class LinearGaussianHypothesis(_HypothesisBase):
    kind: Literal["linear_gaussian"] = "linear_gaussian"
    terms: tuple[Term, ...] = Field(min_length=1)
    prior: NIGPrior = NIGPrior()


GridValue = float | int | str


class GridHypothesis(_HypothesisBase):
    kind: Literal["finite_grid"] = "finite_grid"
    mechanism: _ID
    params: dict[str, tuple[GridValue, ...]] = Field(default_factory=dict)
    """Cartesian parameter grid, uniform prior over grid points."""
    fixed: dict[str, GridValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _grid(self) -> GridHypothesis:
        overlap = set(self.params) & set(self.fixed)
        if overlap:
            raise ValueError(f"parameters both gridded and fixed: {sorted(overlap)}")
        for k, vals in self.params.items():
            if len(vals) == 0:
                raise ValueError(f"empty grid for parameter {k!r}")
            if len(set(vals)) != len(vals):
                raise ValueError(f"duplicate grid values for parameter {k!r}")
        n = int(np.prod([len(v) for v in self.params.values()])) if self.params else 1
        if n > 200_000:
            raise ValueError(f"grid too large ({n} points > 200000)")
        return self

    def grid_size(self) -> int:
        return int(np.prod([len(v) for v in self.params.values()])) if self.params else 1

    def grid_table(self) -> dict[str, np.ndarray]:
        """Column-oriented table of all grid points (C order over sorted names),
        including fixed parameters broadcast to the grid length."""
        names = sorted(self.params)
        points = list(product(*(self.params[n] for n in names))) if names else [()]
        table: dict[str, np.ndarray] = {
            n: np.asarray([p[i] for p in points]) for i, n in enumerate(names)
        }
        for k, v in self.fixed.items():
            table[k] = np.asarray([v] * len(points))
        return table


Hypothesis = Annotated[LinearGaussianHypothesis | GridHypothesis, Field(discriminator="kind")]


class HypothesisSet(_Frozen):
    schema_version: Literal["probe.ir/v0"] = IR_SCHEMA_VERSION
    outcome: OutcomeSpec
    covariates: tuple[CovariateSpec, ...]
    hypotheses: tuple[Hypothesis, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _cross_refs(self) -> HypothesisSet:
        ids = [h.id for h in self.hypotheses]
        if len(set(ids)) != len(ids):
            raise ValueError("hypothesis ids must be unique")
        residuals = [h.id for h in self.hypotheses if h.is_residual]
        if len(residuals) != 1:
            raise ValueError(f"exactly one residual hypothesis required, got {residuals}")
        kinds = {h.kind for h in self.hypotheses}
        if len(kinds) != 1:
            raise ValueError(f"one hypothesis family per set in v0, got {sorted(kinds)}")
        if "linear_gaussian" in kinds and self.outcome.kind != "continuous":
            raise ValueError("linear_gaussian hypotheses need a continuous outcome")
        if "finite_grid" in kinds and self.outcome.kind != "count":
            raise ValueError("finite_grid hypotheses need a count outcome in v0")
        cov = {c.name: c for c in self.covariates}
        if len(cov) != len(self.covariates):
            raise ValueError("covariate names must be unique")
        for h in self.hypotheses:
            if isinstance(h, LinearGaussianHypothesis):
                for t in h.terms:
                    _check_term(t, cov, h.id)
        return self

    def by_id(self, hid: str) -> Hypothesis:
        for h in self.hypotheses:
            if h.id == hid:
                return h
        raise KeyError(hid)

    @property
    def residual(self) -> Hypothesis:
        return next(h for h in self.hypotheses if h.is_residual)

    def covariate_map(self) -> dict[str, CovariateSpec]:
        return {c.name: c for c in self.covariates}


def _check_term(t: object, cov: Mapping[str, CovariateSpec], hid: str) -> None:
    if isinstance(t, InteractionTerm):
        for f in t.factors:
            _check_term(f, cov, hid)
        return
    if isinstance(t, RFFTerm):
        for inp in t.inputs:
            if inp.covariate not in cov:
                raise ValueError(f"{hid}: unknown covariate {inp.covariate!r} in rff term")
        return
    name = getattr(t, "covariate", None)
    if name is None:
        return
    if name not in cov:
        raise ValueError(f"{hid}: unknown covariate {name!r}")
    spec = cov[name]
    if isinstance(t, IndicatorTerm):
        if spec.kind == "categorical":
            if t.level is None or spec.levels is None or t.level not in spec.levels:
                raise ValueError(f"{hid}: indicator {name}={t.level!r} is not a declared level")
        elif spec.kind == "binary":
            if t.level is not None:
                raise ValueError(f"{hid}: binary indicator {name!r} takes no level")
        else:
            raise ValueError(f"{hid}: indicator on numeric covariate {name!r}")
    elif isinstance(t, NumericTerm | FourierTerm) and spec.kind == "categorical":
        raise ValueError(f"{hid}: numeric/fourier term on categorical covariate {name!r}")


# --------------------------------------------------------------------------- evaluation


def _num(record: Record, name: str) -> float:
    try:
        v = record[name]
    except KeyError as e:
        raise KeyError(f"design record lacks covariate {name!r}") from e
    if isinstance(v, str):
        raise TypeError(f"covariate {name!r} is categorical in the record, expected a number")
    return float(v)


def _term_columns(t: object) -> list[str]:
    if isinstance(t, InterceptTerm):
        return ["1"]
    if isinstance(t, IndicatorTerm):
        return [f"{t.covariate}={t.level}" if t.level is not None else f"{t.covariate}"]
    if isinstance(t, NumericTerm):
        return [f"num({t.covariate};c={t.center!r};s={t.scale!r})"]
    if isinstance(t, FourierTerm):
        out = []
        for k in range(1, t.harmonics + 1):
            out += [f"sin{k}({t.covariate}/{t.period!r})", f"cos{k}({t.covariate}/{t.period!r})"]
        return out
    if isinstance(t, RFFTerm):
        tag = content_hash(t.model_dump(mode="json", exclude={"prior_mean", "prior_sd"}))[:12]
        return [f"rff[{tag}]{i}" for i in range(t.n_features)]
    if isinstance(t, InteractionTerm):
        parts = [_term_columns(f) for f in t.factors]
        return ["*".join(sorted(combo)) for combo in product(*parts)]
    raise TypeError(f"unknown term {t!r}")


def _rff_params(t: RFFTerm, cov: Mapping[str, CovariateSpec]) -> tuple[np.ndarray, np.ndarray]:
    dim = 0
    for inp in t.inputs:
        spec = cov[inp.covariate]
        dim += len(spec.levels) if spec.kind == "categorical" and spec.levels else 1
    rng = make_rng(t.seed, "rff")
    w = rng.standard_normal((dim, t.n_features)) / t.lengthscale
    b = rng.uniform(0.0, 2.0 * np.pi, size=t.n_features)
    return w, b


def _rff_inputs(
    t: RFFTerm, records: Sequence[Record], cov: Mapping[str, CovariateSpec]
) -> np.ndarray:
    cols: list[np.ndarray] = []
    for inp in t.inputs:
        spec = cov[inp.covariate]
        if spec.kind == "categorical" and spec.levels:
            vals = [str(r[inp.covariate]) for r in records]
            for lev in spec.levels:
                cols.append(np.asarray([1.0 if v == lev else 0.0 for v in vals]) / inp.scale)
        else:
            cols.append(np.asarray([_num(r, inp.covariate) for r in records]) / inp.scale)
    return np.column_stack(cols)


def _term_values(
    t: object, records: Sequence[Record], cov: Mapping[str, CovariateSpec]
) -> np.ndarray:
    n = len(records)
    if isinstance(t, InterceptTerm):
        return np.ones((n, 1))
    if isinstance(t, IndicatorTerm):
        if t.level is None:  # binary covariate
            return np.asarray([[1.0 if bool(r[t.covariate]) else 0.0] for r in records])
        return np.asarray([[1.0 if str(r[t.covariate]) == t.level else 0.0] for r in records])
    if isinstance(t, NumericTerm):
        return np.asarray([[(_num(r, t.covariate) - t.center) / t.scale] for r in records])
    if isinstance(t, FourierTerm):
        x = np.asarray([_num(r, t.covariate) for r in records])
        out = np.empty((n, 2 * t.harmonics))
        for k in range(1, t.harmonics + 1):
            ang = 2.0 * np.pi * k * x / t.period
            out[:, 2 * k - 2] = np.sin(ang)
            out[:, 2 * k - 1] = np.cos(ang)
        return out
    if isinstance(t, RFFTerm):
        w, b = _rff_params(t, cov)
        z = _rff_inputs(t, records, cov)
        return np.sqrt(2.0 / t.n_features) * np.cos(z @ w + b)
    if isinstance(t, InteractionTerm):
        blocks = [_term_values(f, records, cov) for f in t.factors]
        names = [_term_columns(f) for f in t.factors]
        cols, labels = [], []
        for combo in product(*(range(b.shape[1]) for b in blocks)):
            col = np.ones(n)
            for blk, j in zip(blocks, combo, strict=True):
                col = col * blk[:, j]
            cols.append(col)
            labels.append("*".join(sorted(names[i][j] for i, j in enumerate(combo))))
        return np.column_stack(cols) if cols else np.ones((n, 0))
    raise TypeError(f"unknown term {t!r}")


def column_names(h: LinearGaussianHypothesis) -> list[str]:
    out: list[str] = []
    for t in h.terms:
        out += _term_columns(t)
    if len(set(out)) != len(out):
        raise ValueError(f"{h.id}: duplicate design columns {out}")
    return out


def design_matrix(
    h: LinearGaussianHypothesis, records: Sequence[Record], covariates: Sequence[CovariateSpec]
) -> np.ndarray:
    """φ_m evaluated at each record: an (n, p_m) float64 matrix, columns in
    :func:`column_names` order. Deterministic and platform-independent."""
    cov = {c.name: c for c in covariates}
    if not records:
        return np.zeros((0, len(column_names(h))))
    blocks = [_term_values(t, records, cov) for t in h.terms]
    x = np.column_stack(blocks).astype(np.float64, copy=False)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{h.id}: non-finite design values")
    return x


def prior_arrays(h: LinearGaussianHypothesis) -> tuple[np.ndarray, np.ndarray]:
    """(μ₀, diag V₀) per design column."""
    mu, var = [], []
    for t in h.terms:
        k = len(_term_columns(t))
        sd = t.prior_sd if t.prior_sd is not None else h.prior.default_sd
        mu += [t.prior_mean] * k
        var += [sd * sd] * k
    return np.asarray(mu, dtype=np.float64), np.asarray(var, dtype=np.float64)


def term_signature(h: Hypothesis) -> frozenset[str]:
    """Column-name set of a linear hypothesis (name-level nesting proxy; numeric
    column-space checks live in the stopping module). Grid hypotheses return
    ``{mechanism:<name>}``."""
    if isinstance(h, GridHypothesis):
        return frozenset({f"mechanism:{h.mechanism}"})
    return frozenset(column_names(h))


def ir_hash(obj: BaseModel) -> str:
    return content_hash(obj.model_dump(mode="json"))
