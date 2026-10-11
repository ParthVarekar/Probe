"""Candidate scoring and selection policies P1-P10 (handover §12 H.5, Part I).

* :mod:`probe.select.base`: the contract (ScoredCandidates, PolicyInit,
  DecisionContext, Decision, Policy, argmax_with_ties, stable_key).
* :mod:`probe.select.scoring`: :func:`score_candidates`, computed once per step.
* :mod:`probe.select.designs`: audit, stratified, fixed-grid and static maximin
  designs; nearest-feasible fallback.
* :mod:`probe.select.oracle`: the oracle's truth-targeted information.
* :mod:`probe.select.policies` and :mod:`probe.select.registry`: P1-P10,
  :func:`make_policy`, :data:`POLICY_IDS`, :data:`POLICY_NAMES`.
"""

from probe.select.registry import POLICY_IDS, POLICY_NAMES, make_policy
from probe.select.scoring import requires_u_param, score_candidates

__all__ = ["POLICY_IDS", "POLICY_NAMES", "make_policy", "requires_u_param", "score_candidates"]
