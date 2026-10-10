"""Scenario manifest and benchmark regimes R1–R10 (handover §12 H.1, Part I).

``SCENARIOS`` lists every physical magnitude the simulator uses. **All are
marked [U] (unverified).** They are order-of-magnitude assumptions chosen to be
plausible for a hot-arid campus in summer; none is a literature citation and
none should be quoted as one. The manifest is hashed (``scenario_manifest_hash``)
and the hash is published in every world's metadata, so a change of assumptions
is visible in every result.

``REGIME_CONFIGS`` are the knobs behind each regime. They are public design
(what kind of world a regime is), not hidden truth; the realised draws of a world
(which hypothesis is true, effect sizes, noise, nuisance series) are sealed in its
:class:`~probe.truth.SealedTruth`. Each regime's pre-registered expectation is in
its :class:`~probe.envs.base.RegimeSpec` ``expected`` field, written before any
run. Budgets are in field-effort minutes (travel + settling + dwell; waiting is
free but the campaign window bounds the clock).

How truths are chosen: for in-set regimes the true hypothesis is
block-randomised by seed over the regime's pool (each pool member appears exactly
once in every block of ``len(pool)`` consecutive seeds, in a seeded random
order), and its parameters are drawn from the seed. R3's truth is the full
nonlinear SCM, which lies outside the registered set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from probe.determinism import content_hash
from probe.envs.base import RegimeSpec

__all__ = [
    "REGIME_CONFIGS",
    "REGIME_IDS",
    "SCENARIOS",
    "Assumption",
    "ExpectedVerdict",
    "RegimeConfig",
    "regime_specs",
    "scenario_manifest",
    "scenario_manifest_hash",
]

ExpectedVerdict = Literal["hypothesis", "inconclusive", "none_fits"]


@dataclass(frozen=True)
class Assumption:
    name: str
    value: Any
    unit: str
    basis: str
    """Why this value: a stated assumption, never a citation."""
    status: str = "[U]"


SCENARIOS: tuple[Assumption, ...] = (
    Assumption(
        "location",
        {"latitude": 33.42, "longitude": -111.93, "altitude_m": 360.0, "tz": "UTC-7"},
        "deg / m",
        "A hot-arid desert-city campus setting; fixed so solar geometry is shared and cached.",
    ),
    Assumption("date", "2025-07-15", "", "Mid-summer clear-sky season."),
    Assumption(
        "clear_sky_model",
        {"model": "ineichen", "linke_turbidity": 3.5},
        "",
        "pvlib Ineichen-Perez clear sky with a fixed, moderately clear turbidity.",
    ),
    Assumption(
        "nominal_direct_transmission",
        {"open": 1.0, "tree": 0.15, "built": 0.0},
        "1",
        "What an investigator would assume per shade class (public proxy only).",
    ),
    Assumption(
        "tree_canopy_direct_transmissivity",
        [0.06, 0.30],
        "1",
        "Per-site truth; dense to sparse canopies let a small fraction of beam through.",
    ),
    Assumption(
        "built_shadow_sector",
        {"center_deg": [160.0, 220.0], "half_width_deg": [60.0, 100.0]},
        "deg azimuth",
        "Built-shade sites are shaded while the sun is in a site-specific azimuth sector "
        "around midday and sunlit outside it (R3 physics only).",
    ),
    Assumption(
        "surface_albedo",
        {"asphalt": 0.08, "concrete": 0.35, "grass": 0.23, "cool_pavement": 0.45},
        "1",
        "Typical broadband albedos; cool pavement coatings are markedly more reflective.",
    ),
    Assumption(
        "surface_evaporative_fraction",
        {"asphalt": 0.0, "concrete": 0.0, "grass": 0.6, "cool_pavement": 0.0},
        "1",
        "Irrigated grass converts much of its absorbed energy to latent heat.",
    ),
    Assumption(
        "surface_heating_gain",
        40.0,
        "degC per kW/m2 absorbed",
        "Steady-state surface excess over air; gives asphalt surfaces roughly +30-35 degC "
        "at noon (R3 physics).",
    ),
    Assumption(
        "surface_lag_minutes",
        {"asphalt": 75.0, "concrete": 60.0, "grass": 25.0, "cool_pavement": 60.0},
        "min",
        "First-order thermal response time (thermal inertia) of each surface.",
    ),
    Assumption(
        "shade_profile_lag_minutes",
        60.0,
        "min",
        "Lag of the time-of-day profile of shade effects behind clear-sky GHI.",
    ),
    Assumption(
        "air_coupling_in_set",
        [1.0, 2.2],
        "degC per unit absorbed-fraction difference at peak forcing",
        "Converts differences in absorbed shortwave into 2 m air-temperature differences; "
        "yields shade cooling of roughly 0.5-2.5 degC and surface contrasts of 0.2-1.2 degC "
        "at medium scale. Air differences are far smaller than surface differences.",
    ),
    Assumption(
        "tree_evapotranspiration_cooling",
        [0.2, 0.6],
        "degC at peak",
        "Additional cooling under trees beyond shading.",
    ),
    Assumption(
        "convective_coupling_r3",
        {"kappa": [0.05, 0.09], "threshold_degC": [6.0, 12.0]},
        "degC air per degC surface excess",
        "R3: surface heat reaches 2 m air only above a convective-onset threshold "
        "(a deliberate nonlinearity outside every registered hypothesis).",
    ),
    Assumption(
        "unshielded_radiation_error",
        [1.5, 3.0],
        "degC per kW/m2 at 1 m/s wind",
        "Unshielded low-cost sensors in sun read warm; ventilation reduces the error by a "
        "factor 2/(1 + wind speed).",
    ),
    Assumption(
        "shielded_radiation_error_r3",
        [0.1, 0.3],
        "degC per kW/m2 at 1 m/s wind",
        "Passive shields reduce but do not remove radiation error (R3 physics only).",
    ),
    Assumption(
        "wind",
        {"mean_m_s": [1.0, 3.0], "ar1_phi_per_5min": 0.95, "sd_m_s": 0.6, "min_m_s": 0.3},
        "m/s",
        "Light afternoon winds; unobserved by the investigator.",
    ),
    Assumption(
        "synoptic_temperature",
        {"daily_mean_degC": [32.0, 35.0], "amplitude_degC": 6.0, "peak_hour": 16.0},
        "degC",
        "Hot summer day; the shared signal cancels in the difference from the reference.",
    ),
    Assumption(
        "campus_offset",
        [-0.2, 0.6],
        "degC",
        "Open asphalt campus sites relative to the park-sited reference station.",
    ),
    Assumption(
        "noise",
        {"turbulent_sd": 0.25, "instrument_sd_at_5min": 0.12, "reference_sd": 0.08},
        "degC",
        "Micro-scale turbulent fluctuations do not average out within a dwell; instrument "
        "noise scales with 1/sqrt(dwell).",
    ),
    Assumption("quantization", 0.05, "degC", "Logger resolution of each raw reading."),
    Assumption(
        "settling_minutes",
        {"unshielded": 2.0, "shielded": 5.0},
        "min",
        "Shields add thermal mass and a longer time constant (regimes may override).",
    ),
    Assumption(
        "walking",
        {"speed_m_per_min": 75.0, "detour_factor": 1.3, "overhead_min": 0.5},
        "",
        "Walking pace with path detours and door/crossing overhead.",
    ),
    Assumption(
        "dropouts",
        {"mar_rate": 0.03, "cutout_degC": 46.0, "cutout_width_degC": 0.5, "cutout_max_p": 0.9},
        "",
        "Random logger dropouts (MAR) and overheating cut-outs of the sensor body (MNAR: "
        "more likely the hotter the reading).",
    ),
)


@dataclass(frozen=True)
class RegimeConfig:
    id: str
    title: str
    description: str
    expected: str
    expected_verdict: ExpectedVerdict
    budgets: tuple[float, float, float]
    """(small, medium, large) field-effort minutes."""
    warmup: int
    truth_pool: tuple[str, ...]
    """Hypotheses the truth is block-randomised over; empty = physics-native truth (out of set)."""
    hypothesis_ids: tuple[str, ...] = ("H1", "H2", "H3", "H4", "H0")
    """The registered hypothesis set for this regime (all regimes but R10 use all five)."""
    effect_scale: float = 1.0
    rho_range: tuple[float, float] = (0.6, 1.0)
    """Time-varying share of the shade effect in H3 truths."""
    mimicry: bool = False
    """H1 truths mimic the unshielded pattern of the same seed's H4 world."""
    window: tuple[float, float] = (540.0, 1080.0)
    layout: str = "spread"
    shield_sites: Literal["all", "masts"] = "all"
    settle_unshielded: float = 2.0
    settle_shielded: float = 5.0
    sd_turb: float = 0.25
    sd_inst: float = 0.12
    sd_ref: float = 0.08
    heavy_df: float | None = None
    spike_rate: float = 0.0
    n_units: int = 1
    offset_sd: float = 0.0
    unit_drift_sd: float = 0.0
    mar_rate: float = 0.03
    cutout_c: float = 46.0
    field_sd: float = 0.0
    gradient_sd: float = 0.0
    front: bool = False
    cloud_level: float = 0.02
    cloud_ramp: bool = False
    cloud_modulates: bool = False
    synoptic_drift_sd: float = 0.4
    notes: str = ""

    @property
    def in_set(self) -> bool:
        return bool(self.truth_pool)

    def spec(self) -> RegimeSpec:
        small, medium, large = self.budgets
        return RegimeSpec(
            id=self.id,
            title=self.title,
            description=self.description,
            expected=self.expected,
            budgets={"small": small, "medium": medium, "large": large},
            warmup=self.warmup,
            in_set=self.in_set,
            notes=self.notes,
        )


_ALL4 = ("H1", "H2", "H3", "H4")
_STD_BUDGETS = (120.0, 240.0, 420.0)

REGIME_CONFIGS: dict[str, RegimeConfig] = {
    c.id: c
    for c in (
        RegimeConfig(
            id="R1",
            title="Well specified, small interaction effect, heterogeneous cost",
            description="In-set worlds; the truth is block-randomised over H1-H4 by seed. "
            "Effects are small (0.7x the medium scale; in H3 truths 50-80% of the shade effect "
            "varies with time of day, an interaction of a few tenths of a degree, comparable to "
            "one reading's noise). Spread campus with walking times of 1-20 min and "
            "longer settling for shielded readings, so costs are heterogeneous. Clean sensors: "
            "Gaussian noise, no unit offsets, MAR dropouts only.",
            expected="Adaptive wins: guarded cost-aware model-indicator EIG (P8) reaches the "
            "decision criterion at lower median cost than the best of P1-P5 and P9, with a "
            "confidently-wrong rate no worse than the best baseline + 1 point.",
            expected_verdict="hypothesis",
            budgets=_STD_BUDGETS,
            warmup=4,
            truth_pool=_ALL4,
            effect_scale=0.7,
            rho_range=(0.5, 0.8),
        ),
        RegimeConfig(
            id="R2",
            title="Large known main effects, uniform cost",
            description="In-set worlds; the truth is block-randomised over H1 and H2. Large "
            "main effects (1.4x medium scale). Compact campus (walks of 1-4 min) and equal "
            "settling for both shields, so every reading costs about the same.",
            expected="Tie with stratified sampling: |difference in cost-to-decision| < 5% "
            "between P8 and P3.",
            expected_verdict="hypothesis",
            budgets=(100.0, 200.0, 360.0),
            warmup=4,
            truth_pool=("H1", "H2"),
            effect_scale=1.4,
            layout="compact",
            settle_unshielded=3.0,
            settle_shielded=3.0,
        ),
        RegimeConfig(
            id="R3",
            title="Truth outside the hypothesis set",
            description="Physics-native SCM: surface heating through a first-order surface "
            "energy balance with lag and a convective-onset threshold, an unobserved wind "
            "channel (ventilation by wind speed x site exposure), true canopy transmissivity "
            "and building-shadow sectors that differ from the nominal shade classes, partly "
            "cloudy sky, a smooth spatial field, physical radiation error for both shield "
            "types, two sensor units per shield with small offsets, and MNAR overheating "
            "cut-outs on hot afternoons.",
            expected="Unguarded adaptive (P6) loses on safety (higher confidently-wrong rate "
            "than baselines); guarded adaptive (P8) is about equal to baselines. The correct "
            "verdict is 'none fits'.",
            expected_verdict="none_fits",
            budgets=_STD_BUDGETS,
            warmup=4,
            truth_pool=(),
            field_sd=0.25,
            gradient_sd=0.3,
            cloud_level=0.08,
            n_units=2,
            offset_sd=0.1,
            unit_drift_sd=0.05,
            mar_rate=0.05,
            cutout_c=43.0,
        ),
        RegimeConfig(
            id="R4",
            title="Sensor radiation-artefact confound",
            description="In-set worlds; the truth is block-randomised over H1 and H4. In H1 "
            "worlds the shade effects mimic what unshielded sensors would show in the H4 "
            "world of the same seed, so unshielded readings separate the two only through "
            "the time-of-day shape of irradiance. Shielded readings are possible at every "
            "site but settle longer (8 min vs 2 min).",
            expected="Adaptive wins if shield actions exist (they do in this regime); without "
            "shield actions 'inconclusive' would be the correct answer.",
            expected_verdict="hypothesis",
            budgets=_STD_BUDGETS,
            warmup=4,
            truth_pool=("H1", "H4"),
            mimicry=True,
            settle_shielded=8.0,
        ),
        RegimeConfig(
            id="R5",
            title="Flat information landscape",
            description="In-set worlds with the truth block-randomised over H1-H4 but every "
            "effect tiny (4% of medium scale, well below the noise), so all hypotheses "
            "predict nearly the same readings everywhere.",
            expected="All policies tie; 'inconclusive' is the correct verdict and any decision "
            "for a single hypothesis counts as a false decision.",
            expected_verdict="inconclusive",
            budgets=_STD_BUDGETS,
            warmup=4,
            truth_pool=_ALL4,
            effect_scale=0.04,
        ),
        RegimeConfig(
            id="R6",
            title="Tiny budget",
            description="In-set worlds (truth block-randomised over H1-H4, large effects) on a "
            "sparse campus where every move takes at least 3 minutes, so a reading costs at "
            "least 10 minutes; budgets allow at most 3, 4 and 5 readings, warm-up included.",
            expected="The static maximin design (P9) does at least as well as adaptive selection.",
            expected_verdict="hypothesis",
            budgets=(30.0, 45.0, 55.0),
            warmup=1,
            truth_pool=_ALL4,
            effect_scale=1.4,
            layout="sparse",
        ),
        RegimeConfig(
            id="R7",
            title="Strong drift / nonstationarity",
            description="In-set structure (truth block-randomised over H1-H4, medium effects) "
            "plus unobserved nonstationary nuisance: a time-varying cross-campus temperature "
            "gradient (AR(1), sd 1.2 degC/km, with a front that shifts it by 0.5-1.5 degC/km "
            "at a random hour between 12 and 15 h), afternoon convective cloud that scales "
            "every non-intercept effect by the clear-sky index, a 0.1 degC spatial field and "
            "strong synoptic drift (removed by differencing, but it drives cut-outs).",
            expected="Stratified sampling does at least as well as adaptive selection.",
            expected_verdict="hypothesis",
            budgets=_STD_BUDGETS,
            warmup=4,
            truth_pool=_ALL4,
            field_sd=0.1,
            gradient_sd=1.2,
            front=True,
            cloud_level=0.18,
            cloud_ramp=True,
            cloud_modulates=True,
            synoptic_drift_sd=1.2,
            cutout_c=44.0,
        ),
        RegimeConfig(
            id="R8",
            title="Heavy-tailed noise",
            description="In-set structure (truth block-randomised over H1-H4, medium effects) "
            "with heavy-tailed turbulent noise (0.15 x Student-t with 2.5 degrees of freedom) "
            "and 4% positive spikes of +1.5 to +4 degC (e.g. exhaust plumes or vent outflow).",
            expected="A robust fallback is required; adaptive selection with the Gaussian "
            "likelihood may lose.",
            expected_verdict="hypothesis",
            budgets=_STD_BUDGETS,
            warmup=4,
            truth_pool=_ALL4,
            heavy_df=2.5,
            spike_rate=0.04,
            cutout_c=44.0,
        ),
        RegimeConfig(
            id="R9",
            title="Nuisance-heavy (many sensor offsets)",
            description="In-set structure (truth block-randomised over H1-H4, medium effects) "
            "measured with eight sensor units per shield type, each with an unknown offset "
            "(sd 0.4 degC) and linear drift (sd 0.3 degC over the campaign). The unit is fixed "
            "by the site (site k uses unit k mod 8), so offsets act as a site-level nuisance "
            "confounded with site attributes.",
            expected="Negative interference is possible (Sloman et al., UAI 2024): adaptive "
            "selection may do worse than baselines.",
            expected_verdict="hypothesis",
            budgets=_STD_BUDGETS,
            warmup=4,
            truth_pool=_ALL4,
            n_units=8,
            offset_sd=0.4,
            unit_drift_sd=0.3,
            cutout_c=44.0,
        ),
        RegimeConfig(
            id="R10",
            title="Positive control",
            description="In-set worlds; the truth is block-randomised over H1 and H4, with H1 "
            "truths mimicking the same seed's H4 unshielded pattern. The campaign runs "
            "11:00-15:00, when clear-sky irradiance is nearly flat, so unshielded readings "
            "barely separate the two. Shielded readings are possible only at two remote open "
            "'instrumented mast' sites (the open sites farthest from base): a few expensive "
            "designs carry almost all the discriminating information. The registered set is "
            "focused on the question being controlled for: H1 (shade), H4 (artefact) and the "
            "residual H0.",
            expected="Adaptive selection wins clearly; if it does not, the benchmark is invalid.",
            expected_verdict="hypothesis",
            budgets=(60.0, 120.0, 200.0),
            warmup=3,
            truth_pool=("H1", "H4"),
            hypothesis_ids=("H1", "H4", "H0"),
            effect_scale=1.4,
            mimicry=True,
            window=(660.0, 900.0),
            shield_sites="masts",
            notes="H2 and H3 are not registered here: they nest H1, and over a 4-hour window "
            "their extra columns are nearly collinear with the shade indicators, so no policy "
            "could separate them from H1 within budget and the control would measure Occam "
            "ambiguity instead of design value.",
        ),
    )
}

REGIME_IDS: tuple[str, ...] = tuple(REGIME_CONFIGS)


def regime_specs() -> dict[str, RegimeSpec]:
    return {k: c.spec() for k, c in REGIME_CONFIGS.items()}


def scenario_manifest() -> dict[str, Any]:
    """JSON-able manifest of every assumption (all [U]) and every regime's knobs."""
    return {
        "assumptions": [
            {"name": a.name, "value": a.value, "unit": a.unit, "basis": a.basis, "status": a.status}
            for a in SCENARIOS
        ],
        "regimes": {
            k: {
                "title": c.title,
                "expected": c.expected,
                "expected_verdict": c.expected_verdict,
                "in_set": c.in_set,
                "truth_pool": list(c.truth_pool),
                "hypothesis_ids": list(c.hypothesis_ids),
                "budgets": list(c.budgets),
                "warmup": c.warmup,
                "window": list(c.window),
                "layout": c.layout,
                "shield_sites": c.shield_sites,
            }
            for k, c in REGIME_CONFIGS.items()
        },
    }


def scenario_manifest_hash() -> str:
    return content_hash(scenario_manifest())
