"""Evaluator-side scoring of a :class:`~probe.run.loop.RunRecord` (handover §12 Part I).

Only evaluator code opens the truth. The two environment families state the
correct answer differently, and :func:`acceptable_verdicts` unifies them into
strings ``"hypothesis:<id>"``, ``"none_fits"`` or ``"inconclusive"``:

* microclimate: ``payload.expected_verdict`` plus ``true_hypothesis_id``;
* systemlab: :func:`probe.envs.systemlab.regimes.acceptable_verdicts`
  (which also handles runs that ended before a mid-run deploy).

An ``undecided`` stop (budget ran out) is never "correct", but it is not
confidently wrong either. *Confidently wrong* means the final posterior puts at
least ``0.95`` on a hypothesis that is not an acceptable answer, or the run
declared a verdict that is not acceptable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from probe.run.loop import RunRecord
from probe.truth import EvaluatorKey, SealedTruth

__all__ = ["acceptable_verdicts", "evaluate_run", "verdict_string"]

CONFIDENT = 0.95


def verdict_string(record: RunRecord) -> str:
    v = record.verdict
    if v == "hypothesis":
        return f"hypothesis:{record.leader}"
    return str(v)


def acceptable_verdicts(opened: Mapping[str, Any], final_clock: float) -> tuple[str, ...]:
    payload = opened["payload"]
    if "acceptable_verdicts" in payload:
        from probe.envs.systemlab.regimes import acceptable_verdicts as sl

        return tuple(sl(payload, final_clock))
    tid = opened["true_hypothesis_id"]
    expected = payload.get("expected_verdict", "hypothesis" if tid else "none_fits")
    if expected == "hypothesis" and tid is not None:
        return (f"hypothesis:{tid}",)
    return (str(expected),)


def evaluate_run(record: RunRecord, truth: SealedTruth) -> dict[str, Any]:
    opened = truth.open(EvaluatorKey("benchmark scoring"))
    if record.truth_commitment != truth.commitment:
        raise ValueError("run record does not belong to this world")
    final_clock = record.steps[-1].clock if record.steps else 0.0
    ok = acceptable_verdicts(opened, final_clock)
    ok_ids = {v.split(":", 1)[1] for v in ok if v.startswith("hypothesis:")}
    verdict = verdict_string(record)
    w = record.final_weights
    leader = max(w, key=lambda h: (w[h], h))
    mass_ok = sum(w.get(h, 0.0) for h in ok_ids)
    declared_wrong = verdict not in ok and verdict != "undecided"
    residual = record.extra.get("residual_id")
    fine = set(ok_ids)
    if "none_fits" in ok and residual is not None:
        fine.add(residual)  # confidence in the residual is the right answer here
    confident_wrong_post = any(p >= CONFIDENT and h not in fine for h, p in w.items())
    target = [1.0 if h in ok_ids else 0.0 for h in w]
    brier = sum((p - t) ** 2 for p, t in zip(w.values(), target, strict=True)) if ok_ids else None
    logscore = math.log(max(mass_ok, 1e-300)) if ok_ids else None
    decided = record.stop.get("reason") == "decision_reached"
    return {
        "verdict": verdict,
        "acceptable": list(ok),
        "correct": verdict in ok,
        "declared_wrong": declared_wrong,
        "confidently_wrong": bool(declared_wrong or confident_wrong_post),
        "top1_correct": leader in ok_ids,
        "mass_on_truth": mass_ok if ok_ids else None,
        "brier": brier,
        "log_score": logscore,
        "cost": record.spent,
        "n_steps": len(record.steps),
        "n_obs": record.n_obs,
        "decided": decided,
        "cost_to_decision": record.spent if decided else None,
        "stop_reason": record.stop.get("reason"),
    }
