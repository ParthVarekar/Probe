"""Deterministic claim-guard (handover §12 H.9).

The claim level is *computed* from the evidence design, never written by hand
or by an LLM. Report prose comes only from the templates below, filled with
engine numbers, and :func:`check_wording` rejects text that exceeds the level.

Levels: L0 simulated · L1 convenience observational · L2 probability sampling ·
L2a adaptive (model-conditional) · L3 conditional causal · L4 qualified causal ·
L5 randomized. v0 data is always simulated, so every claim is L0 and scoped to
"simulator world W (seed S)". Downgrades (one level each, never below L0) are
recorded with their reasons even at L0, so the report shows what would apply to
real data.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["LEVELS", "Claim", "check_wording", "claims_for_run"]

LEVELS = ("L0", "L1", "L2", "L2a", "L3", "L4", "L5")
CAUSAL_WORDS = ("causes", "caused", "reduces", "leads to", "effect of", "because", "impact")


@dataclass(frozen=True)
class Claim:
    claim_type: str
    level: str
    text: str
    template_id: str
    values: Mapping[str, Any]
    downgrades: tuple[str, ...] = ()
    scope: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)


def _downgrades(record: Mapping[str, Any]) -> list[str]:
    out = []
    if any(s.get("alarm") for s in record["steps"]):
        out.append("misspecification alarm raised during the run (PIT/surprise/residual)")
    missing = sum(1 for s in record["steps"] if s["missingness"] != "observed")
    if record["steps"] and missing / len(record["steps"]) > 0.2:
        out.append(f"high missingness ({missing}/{len(record['steps'])} readings)")
    if record["n_obs"] < 8:
        out.append(f"small sample (n={record['n_obs']} < 8)")
    return out


def claims_for_run(record: Mapping[str, Any], *, world_label: str) -> list[Claim]:
    """Claims supported by a (JSON) run record. Origin is simulated ⇒ level L0."""
    stop = record["stop"]
    w = record["final_weights"]
    leader = max(w, key=lambda h: (w[h], h))
    scope = f"in simulator world {world_label} (seed {record['seed']})"
    downs = tuple(_downgrades(record))
    level = "L0"
    out: list[Claim] = []
    verdict = stop.get("verdict")
    vals = {
        "leader": leader,
        "posterior": round(w[leader], 3),
        "n": record["n_obs"],
        "cost": round(record["spent"], 1),
        "reason": stop.get("reason"),
    }
    if verdict == "hypothesis":
        e = stop.get("details", {})
        text = (
            f"{scope.capitalize()}, the data favour hypothesis {leader} "
            f"(posterior {vals['posterior']}, n={vals['n']}), and the e-process test "
            f"rejected every registered rival at the pre-declared level."
        )
        out.append(Claim("predictive", level, text, "decision@v0", vals, downs, scope, e))
    elif verdict == "none_fits":
        text = (
            f"{scope.capitalize()}, none of the registered hypotheses fits the data "
            f"adequately (the residual model or a calibrated alarm led; n={vals['n']})."
        )
        out.append(Claim("inadequacy", level, text, "none_fits@v0", vals, downs, scope))
    elif verdict == "inconclusive":
        text = (
            f"{scope.capitalize()}, the registered hypotheses could not be told apart "
            f"with the available actions (stopped as inconclusive after n={vals['n']})."
        )
        out.append(Claim("descriptive", level, text, "inconclusive@v0", vals, downs, scope))
    else:
        text = (
            f"{scope.capitalize()}, no decision was reached within the budget "
            f"(spent {vals['cost']}); the leading hypothesis was {leader} with posterior "
            f"{vals['posterior']}, which is not a supported conclusion."
        )
        out.append(Claim("descriptive", level, text, "undecided@v0", vals, downs, scope))
    return out


def check_wording(text: str, level: str) -> list[str]:
    """Violations of the H.9 lexicon for ``level`` (empty = allowed)."""
    if level not in LEVELS:
        raise ValueError(level)
    bad = []
    low = text.lower()
    if level in ("L0", "L1", "L2", "L2a"):
        bad += [
            f"causal wording {w!r} not permitted at {level}"
            for w in CAUSAL_WORDS
            if re.search(rf"\b{re.escape(w)}\b", low)
        ]
    if level == "L0" and "simulat" not in low:
        bad.append("L0 claims must be scoped to the simulator world")
    return bad


def claim_dicts(claims: Sequence[Claim]) -> list[dict[str, Any]]:
    return [
        {
            "type": c.claim_type,
            "level": c.level,
            "text": c.text,
            "template": c.template_id,
            "values": dict(c.values),
            "downgrades": list(c.downgrades),
            "violations": check_wording(c.text, c.level),
        }
        for c in claims
    ]
