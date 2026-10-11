"""CSV import (data mode B, milestone M8; handover §12 Parts G, M).

Untrusted input, so:

* size and row limits; UTF-8 only; the raw bytes are stored first as an
  immutable CAS blob and every derived row points back to it by hash;
* an explicit mapping (outcome column + covariate columns) validated against a
  :class:`~probe.engine.ir.HypothesisSet`'s schema: categorical levels must be
  declared, numerics must parse, booleans are ``true/false/1/0``;
* empty cells are **missing** (never zero) and are reported, not imputed;
* rejected rows are kept with a reason (a logged transformation, not a silent drop);
* spreadsheet formula injection (cells starting ``= + - @``) is flagged and
  neutralized on any re-export (:func:`neutralize_cell`).

Analysis of imported data is **non-adaptive** and observational: the hypotheses'
evidence is computed exactly, but the claim ceiling is L1 (descriptive and
correlational only) because the sampling design is unknown.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from probe.determinism import sha256_hex
from probe.engine.eig import log_prior_weights, posterior_weights
from probe.engine.factory import build_models
from probe.engine.ir import HypothesisSet

__all__ = ["ImportResult", "analyze_import", "neutralize_cell", "parse_csv"]

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 200_000
_FORMULA = ("=", "+", "-", "@", "\t", "\r")
_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}


def neutralize_cell(value: str) -> str:
    """Prefix a quote so spreadsheets never evaluate the cell as a formula."""
    if value and value[0] in _FORMULA:
        try:
            float(value)
            return value  # a plain negative number is fine
        except ValueError:
            return "'" + value
    return value


@dataclass
class ImportResult:
    raw_sha256: str
    columns: list[str]
    records: list[dict[str, Any]] = field(default_factory=list)
    outcomes: list[float | None] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    missing_outcome: int = 0
    formula_cells: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "raw_sha256": self.raw_sha256,
            "rows_accepted": len(self.records),
            "rows_rejected": len(self.rejected),
            "missing_outcome": self.missing_outcome,
            "formula_cells_flagged": self.formula_cells,
            "rejections": self.rejected[:20],
        }


def parse_csv(data: bytes, hs: HypothesisSet, mapping: Mapping[str, str]) -> ImportResult:
    """``mapping`` maps schema names (the outcome and every covariate) to CSV columns."""
    if len(data) > MAX_BYTES:
        raise ValueError(f"file too large ({len(data)} bytes > {MAX_BYTES})")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("CSV must be UTF-8") from e
    reader = csv.DictReader(io.StringIO(text))
    cols = list(reader.fieldnames or [])
    needed = [hs.outcome.name, *(c.name for c in hs.covariates)]
    for name in needed:
        if name not in mapping:
            raise ValueError(f"mapping lacks {name!r}")
        if mapping[name] not in cols:
            raise ValueError(f"column {mapping[name]!r} (for {name}) not in CSV header")
    res = ImportResult(raw_sha256=sha256_hex(data), columns=cols)
    for i, row in enumerate(reader, start=2):
        if i - 1 > MAX_ROWS:
            raise ValueError(f"too many rows (> {MAX_ROWS})")
        res.formula_cells += sum(
            1
            for v in row.values()
            if isinstance(v, str) and v[:1] in _FORMULA and not _is_number(v)
        )
        try:
            rec: dict[str, Any] = {}
            for c in hs.covariates:
                raw = (row.get(mapping[c.name]) or "").strip()
                if raw == "":
                    raise ValueError(f"missing covariate {c.name}")
                if c.kind == "categorical":
                    if c.levels and raw not in c.levels:
                        raise ValueError(f"{c.name}={raw!r} not a declared level")
                    rec[c.name] = raw
                elif c.kind == "binary":
                    low = raw.lower()
                    if low not in _TRUE | _FALSE:
                        raise ValueError(f"{c.name}={raw!r} is not boolean")
                    rec[c.name] = low in _TRUE
                else:
                    v = float(raw)
                    if not np.isfinite(v):
                        raise ValueError(f"{c.name} is not finite")
                    rec[c.name] = v
            yraw = (row.get(mapping[hs.outcome.name]) or "").strip()
            y: float | None
            if yraw == "":
                y = None
                res.missing_outcome += 1
            else:
                y = float(yraw)
                if not np.isfinite(y):
                    raise ValueError("outcome is not finite")
                if hs.outcome.kind == "count":
                    if y != int(y) or not 0 <= y <= (hs.outcome.max_count or 0):
                        raise ValueError("count outcome out of support")
                    y = int(y)
        except ValueError as e:
            res.rejected.append({"line": i, "reason": str(e)})
            continue
        res.records.append(rec)
        res.outcomes.append(y)
    return res


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def analyze_import(res: ImportResult, hs: HypothesisSet) -> dict[str, Any]:
    """Exact posterior over the hypothesis set for imported rows (in file order)."""
    models = build_models(hs)
    for rec, y in zip(res.records, res.outcomes, strict=True):
        for m in models:
            m.update(rec, y)
    w = posterior_weights(log_prior_weights(hs), np.array([m.log_evidence() for m in models]))
    ids = [m.hypothesis_id for m in models]
    return {
        "origin": "imported",
        "claim_ceiling": "L1",
        "note": "Observational data with unknown sampling: descriptive and correlational "
        "statements only; the posterior is conditional on the registered hypotheses.",
        "n_obs": int(models[0].n_obs),
        "posterior": {h: float(x) for h, x in zip(ids, w, strict=True)},
        "import": res.summary(),
    }
