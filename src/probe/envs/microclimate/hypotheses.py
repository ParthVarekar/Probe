"""The registered microclimate hypothesis set (handover §12 H.3), as IR data.

Outcome: ``dT_ref``, the sensor reading minus the reference-station reading
(°C). Differencing absorbs the synoptic drift shared by every site.

Design-record covariates (all agent-knowable, see :mod:`.env`):

========== =========== ===================================================
name       kind        meaning
========== =========== ===================================================
shade      categorical open / tree / built (site survey)
surface    categorical asphalt / concrete / grass / cool_pavement
hour       numeric     local standard time at the middle of the reading, h
unshielded binary      True when the reading uses an unshielded sensor
irr_proxy  numeric     clear-sky shortwave at the site, kW/m² (public sun
                       position × nominal shade transmission + diffuse·svf)
svf        numeric     sky-view factor of the site
========== =========== ===================================================

Hypotheses (one family, ``linear_gaussian``; open shade and asphalt are the
reference levels; every hypothesis has an intercept as column 0):

* **H1 shade-only**: ΔT = c + β_tree·1[tree] + β_built·1[built].
* **H2 shade + surface**: H1 + additive surface offsets.
* **H3 shade × hour**: H1 + shade-level × (sin, cos)(2π·hour/24): the shade
  effect varies smoothly over the day.
* **H4 sensor radiation artefact**: ΔT = c + β·unshielded·irr_proxy. Air
  temperature does not depend on shade; unshielded sensors read warm in
  proportion to irradiance (which is why sunlit sites *look* warmer).
* **H0 residual**: a wide-prior random-Fourier-feature GP over every covariate
  (``is_residual=True``), competing on evidence so "none fits" can win.

Every regime registers all five except the positive control R10, which registers
the focused subset {H1, H4, H0} (same objects; see its RegimeSpec notes).

Priors: β | σ² ~ N(0, σ²·sd²) with sd = 6 per column (intercept and the artefact
slope 10, residual features 8), and σ² ~ InvGamma(2, 0.1), whose mean 0.1 °C²
matches the simulator's noise scale (σ ≈ 0.3 °C). With σ ≈ 0.3 the implied
prior sd of a shade or surface effect is ≈ 1.8 °C, wide enough for every
regime's effect sizes (see :mod:`.scenarios`).
"""

from __future__ import annotations

from functools import lru_cache

from probe.engine.ir import (
    CovariateSpec,
    FourierTerm,
    HypothesisSet,
    IndicatorTerm,
    InteractionTerm,
    InterceptTerm,
    LinearGaussianHypothesis,
    NIGPrior,
    NumericTerm,
    OutcomeSpec,
    RFFInput,
    RFFTerm,
)

from .campus import SHADE_LEVELS, SURFACE_LEVELS

__all__ = [
    "COVARIATES",
    "HYPOTHESIS_IDS",
    "OUTCOME",
    "RESIDUAL_ID",
    "build_hypothesis_set",
]

OUTCOME = OutcomeSpec(
    name="dT_ref",
    kind="continuous",
    unit="degC",
    description="Sensor reading minus the reference-station reading at the same time.",
)

COVARIATES: tuple[CovariateSpec, ...] = (
    CovariateSpec(
        name="shade", kind="categorical", levels=SHADE_LEVELS, description="Site shade class."
    ),
    CovariateSpec(
        name="surface",
        kind="categorical",
        levels=SURFACE_LEVELS,
        description="Ground surface material at the site.",
    ),
    CovariateSpec(
        name="hour",
        kind="numeric",
        unit="h",
        description="Local standard time at the middle of the reading.",
    ),
    CovariateSpec(
        name="unshielded", kind="binary", description="Reading taken with an unshielded sensor."
    ),
    CovariateSpec(
        name="irr_proxy",
        kind="numeric",
        unit="kW/m2",
        description="Clear-sky shortwave at the site from public sun position, nominal shade "
        "transmission and sky-view factor (no cloud information).",
    ),
    CovariateSpec(name="svf", kind="numeric", unit="1", description="Sky-view factor."),
)

HYPOTHESIS_IDS: tuple[str, ...] = ("H1", "H2", "H3", "H4", "H0")
RESIDUAL_ID = "H0"
RFF_SEED = 20251011

_PRIOR = NIGPrior(default_sd=6.0, a0=2.0, b0=0.1)
_TREE = IndicatorTerm(covariate="shade", level="tree")
_BUILT = IndicatorTerm(covariate="shade", level="built")
_DAY = FourierTerm(covariate="hour", period=24.0, harmonics=1)


def _intercept() -> InterceptTerm:
    return InterceptTerm(prior_sd=10.0)


@lru_cache(maxsize=8)
def build_hypothesis_set(ids: tuple[str, ...] = HYPOTHESIS_IDS) -> HypothesisSet:
    """The registered set, optionally restricted to ``ids`` (in the order given). The
    residual H0 must be included. Hypotheses are identical objects across subsets."""
    full = _full_set()
    unknown = set(ids) - set(HYPOTHESIS_IDS)
    if unknown or len(set(ids)) != len(ids):
        raise ValueError(f"bad hypothesis ids {ids!r}")
    if ids == HYPOTHESIS_IDS:
        return full
    return HypothesisSet(
        outcome=OUTCOME,
        covariates=COVARIATES,
        hypotheses=tuple(full.by_id(i) for i in ids),
    )


@lru_cache(maxsize=1)
def _full_set() -> HypothesisSet:
    h1 = LinearGaussianHypothesis(
        id="H1",
        label="Shade only",
        statement="Shade alone shifts air temperature: tree and built shade lower dT_ref by "
        "constant amounts relative to open sites. Surface material and the sensor shield "
        "do not matter.",
        applicability="Daytime campaign window, any sensor.",
        qualitative_prediction="dT_ref(tree) - dT_ref(open) < 0 at every hour, constant over "
        "the day; shielded and unshielded readings agree.",
        terms=(_intercept(), _TREE, _BUILT),
        prior=_PRIOR,
    )
    h2 = LinearGaussianHypothesis(
        id="H2",
        label="Shade + surface",
        statement="Shade and surface material shift air temperature additively: shade cools, "
        "and concrete, grass and cool pavement are cooler than asphalt.",
        applicability="Daytime campaign window, any sensor.",
        qualitative_prediction="At equal shade, dT_ref(grass) < dT_ref(asphalt); the shade "
        "contrast is the same on every surface.",
        terms=(
            _intercept(),
            _TREE,
            _BUILT,
            IndicatorTerm(covariate="surface", level="concrete"),
            IndicatorTerm(covariate="surface", level="grass"),
            IndicatorTerm(covariate="surface", level="cool_pavement"),
        ),
        prior=_PRIOR,
    )
    h3 = LinearGaussianHypothesis(
        id="H3",
        label="Shade x hour",
        statement="The cooling effect of shade changes smoothly with time of day (a 24-hour "
        "harmonic per shade class), e.g. small in the morning and largest in the afternoon.",
        applicability="Daytime campaign window, any sensor.",
        qualitative_prediction="dT_ref(tree) - dT_ref(open) is more negative at 14-16 h than "
        "at 9-10 h.",
        terms=(
            _intercept(),
            _TREE,
            _BUILT,
            InteractionTerm(factors=(_TREE, _DAY)),
            InteractionTerm(factors=(_BUILT, _DAY)),
        ),
        prior=_PRIOR,
    )
    h4 = LinearGaussianHypothesis(
        id="H4",
        label="Sensor radiation artefact",
        statement="Shade does not change air temperature. Unshielded sensors read warm in "
        "proportion to the shortwave reaching them, so sunlit sites only appear warmer.",
        applicability="Daytime campaign window; distinguishes itself only with shielded "
        "readings or across irradiance levels.",
        qualitative_prediction="Shielded readings show no shade contrast; unshielded minus "
        "shielded grows with irradiance.",
        terms=(
            _intercept(),
            InteractionTerm(
                factors=(
                    IndicatorTerm(covariate="unshielded"),
                    NumericTerm(covariate="irr_proxy", center=0.0, scale=1.0),
                ),
                prior_sd=10.0,
            ),
        ),
        prior=_PRIOR,
    )
    h0 = LinearGaussianHypothesis(
        id=RESIDUAL_ID,
        label="Residual (none of the above)",
        statement="None of the registered mechanisms: dT_ref is some smooth function of shade, "
        "surface, hour, shield, irradiance and sky-view factor (random-Fourier-feature GP "
        "with a wide prior).",
        applicability="Always; competes on evidence.",
        qualitative_prediction="No specific sign or shape.",
        is_residual=True,
        terms=(
            _intercept(),
            RFFTerm(
                inputs=(
                    RFFInput(covariate="shade"),
                    RFFInput(covariate="surface"),
                    RFFInput(covariate="hour", scale=3.0),
                    RFFInput(covariate="unshielded"),
                    RFFInput(covariate="irr_proxy", scale=0.3),
                    RFFInput(covariate="svf", scale=0.3),
                ),
                n_features=40,
                lengthscale=1.0,
                seed=RFF_SEED,
                prior_sd=8.0,
            ),
        ),
        prior=_PRIOR,
    )
    return HypothesisSet(outcome=OUTCOME, covariates=COVARIATES, hypotheses=(h1, h2, h3, h4, h0))
