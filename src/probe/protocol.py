"""Protocol validation before lock (handover §12 G.2 "validate", H.11 identifiability).

:func:`validate_protocol` returns a report that the UI shows *before* any data is
collected:

* schema: protocol fields, the hypothesis set, exactly one residual;
* identifiability: expected pairwise divergence between the hypotheses' prior
  predictives over the actions actually available, warning about pairs that the
  available actions cannot tell apart (e.g. a shade hypothesis vs a sensor-artefact
  hypothesis when no shielded sensor can be deployed);
* feasibility: candidates exist and at least one fits the budget.

The candidate list comes from a fresh world's public ``candidates()`` (the same
for every seed's action *space*); no truth is read.
"""

from __future__ import annotations

from typing import Any

from probe.determinism import content_hash
from probe.diagnostics import identifiability_report
from probe.engine.factory import build_models
from probe.engine.ir import ir_hash

__all__ = ["validate_protocol"]


def validate_protocol(family: str, regime: str, budget_key: str = "medium") -> dict[str, Any]:
    from probe.run import get_family, make_protocol

    fam = get_family(family)
    errors: list[str] = []
    if regime not in fam.regimes():
        return {"ok": False, "errors": [f"unknown regime {regime!r}"], "warnings": []}
    protocol = make_protocol(family, regime, budget_key)
    hs = fam.hypothesis_set(regime)
    env = fam.make_world(regime, 0).env
    cands = [c for c in env.candidates() if c.feasible]
    if not cands:
        errors.append("no feasible action at the start of the campaign")
    if cands and min(c.cost for c in cands) > protocol.budget:
        errors.append("no action fits in the budget")
    warnings: list[dict[str, Any]] = []
    ident: dict[str, Any] | None = None
    if cands:
        rep = identifiability_report(build_models(hs), [c.record for c in cands])
        ident = rep.to_dict()
        warnings = [w.to_dict() for w in rep.warnings]
    body = {"family": family, "regime": regime, "protocol": protocol.model_dump(mode="json")}
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "protocol_hash": protocol.protocol_hash(),
        "hypothesis_set_hash": ir_hash(hs),
        "validation_hash": content_hash(body),
        "n_candidates": len(cands),
        "residual": hs.residual.id,
        "identifiability": ident,
    }
