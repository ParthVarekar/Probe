"""Pre-declared run protocol: every setting that can change a conclusion.

A :class:`RunProtocol` is fixed and hashed *before* any observation (handover
§12 H.6, H.11). The same protocol applies to every policy in a benchmark cell;
only the selection policy differs. In particular the stopping rules, including
"nothing informative left", are evaluated identically for all policies, using
engine-computed information gain, so a baseline is never stopped by a different
rule than the adaptive policy.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from probe.determinism import content_hash

__all__ = ["RunProtocol"]


class RunProtocol(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    budget: float = Field(gt=0, description="Total cost budget in the family's cost unit.")
    warmup: int = Field(
        ge=0,
        description="Initial observations taken by the shared stratified warm-up design "
        "(identical across policies via common random numbers).",
    )
    max_steps: int = Field(default=500, ge=1)

    # --- decision rule (H.6 rule 1)
    pi_star: float = Field(default=0.95, gt=0.5, lt=1.0)
    alpha: float = Field(default=0.05, gt=0.0, lt=0.5)
    require_eprocess: bool = True
    """Also require the universal-inference e-process against every non-nesting
    rival to reach 1/alpha before declaring a decision."""

    # --- inconclusive rule (H.6 rule 3)
    eps_stop: float = Field(
        default=1e-4,
        ge=0.0,
        description="Stop as inconclusive when max feasible U_M(d)/c(d) is below this "
        "(nats per cost unit) while no hypothesis has reached pi_star.",
    )
    min_obs_before_inconclusive: int = Field(default=5, ge=0)

    # --- inadequacy rule (H.6 rule 4)
    min_obs_before_none_fits: int = Field(default=8, ge=0)

    # --- inference settings
    tempering_eta: float = Field(default=1.0, gt=0.0, le=1.0)

    # --- selection settings shared by the EIG family of policies
    tie_rel_tol: float = Field(default=1e-9, ge=0.0, description="Relative tie tolerance δ.")
    audit_eps: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Guarded policy audit fraction ε."
    )
    lambda_param: float = Field(default=0.0, ge=0.0, description="Weight λ on parameter EIG.")

    def protocol_hash(self) -> str:
        return content_hash(self.model_dump(mode="json"))
