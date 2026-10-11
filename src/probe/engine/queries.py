"""Binary user queries for goal-oriented information gain (handover §12 H.5).

A query q is a yes/no question about the world, e.g. "is the shaded − sunlit
difference below −1 °C at 14 h?". Goal-oriented EIG scores a candidate design d
by U_q(d) = I(1[q]; Y_d | D): how much observing Y_d is expected to tell us about
the answer, averaged over the hypotheses with their posterior weights.

A query is *data*, never code, and each hypothesis interprets it in its own
parameterization:

* :class:`ContrastQuery` (``linear_gaussian`` family): q = 1[Σ_j c_j·E[Y | r_j] < τ]
  (or ``>``), a linear contrast of the mean response at fixed design records r_j.
  Under hypothesis m, E[Y | r] = φ_m(r)ᵀβ, so q is the event cₘᵀβ < τ with
  cₘ = Σ_j c_j φ_m(r_j). A hypothesis whose feature map makes the contrast
  vanish (cₘ = 0) answers the query deterministically (0 < τ).
* :class:`GridQuery` (``finite_grid`` family): q = 1[θ[param] <op> value] on the
  grid parameters. A hypothesis without that parameter answers ``missing``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from probe.engine.ir import Record

__all__ = ["ContrastQuery", "GridQuery", "Query"]


@dataclass(frozen=True)
class ContrastQuery:
    records: tuple[Record, ...]
    coefs: tuple[float, ...]
    threshold: float
    direction: Literal["<", ">"] = "<"

    def __post_init__(self) -> None:
        if len(self.records) == 0 or len(self.records) != len(self.coefs):
            raise ValueError("ContrastQuery needs one coefficient per record (at least one)")
        if not all(math.isfinite(float(c)) for c in self.coefs):
            raise ValueError("ContrastQuery coefficients must be finite")
        if not math.isfinite(float(self.threshold)):
            raise ValueError("ContrastQuery threshold must be finite")
        if self.direction not in ("<", ">"):
            raise ValueError("direction must be '<' or '>'")


_GridOp = Literal["<", "<=", ">", ">=", "==", "!="]


@dataclass(frozen=True)
class GridQuery:
    param: str
    op: _GridOp
    value: float | int | str
    missing: bool = False
    """Answer for hypotheses whose grid has no parameter ``param``."""

    def __post_init__(self) -> None:
        if self.op not in ("<", "<=", ">", ">=", "==", "!="):
            raise ValueError(f"unknown comparison {self.op!r}")
        if isinstance(self.value, str) and self.op not in ("==", "!="):
            raise ValueError("string-valued grid queries support only == and !=")

    def mask(self, table: Mapping[str, np.ndarray], size: int) -> np.ndarray:
        """Boolean answer per grid point for a grid table of ``size`` points."""
        if self.param not in table:
            return np.full(size, bool(self.missing))
        col = np.asarray(table[self.param])
        numeric_col = col.dtype.kind in "biuf"
        if numeric_col == isinstance(self.value, str):
            # numpy compares "4" with 4 as plain inequality, which would silently answer
            # the query "no" at every grid point
            raise TypeError(
                f"grid query value {self.value!r} does not match the type of parameter "
                f"{self.param!r} ({'numeric' if numeric_col else 'categorical'})"
            )
        if self.op == "==":
            return np.asarray(col == self.value, dtype=bool)
        if self.op == "!=":
            return np.asarray(col != self.value, dtype=bool)
        if col.dtype.kind not in "biuf":
            raise TypeError(f"ordered comparison on non-numeric grid parameter {self.param!r}")
        v = float(self.value)
        c = col.astype(np.float64)
        out = {"<": c < v, "<=": c <= v, ">": c > v, ">=": c >= v}[self.op]
        return np.asarray(out, dtype=bool)


Query = ContrastQuery | GridQuery
