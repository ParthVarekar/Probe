"""Hidden structural causal model and sensor model of one microclimate world.

Nothing in this module is agent-facing. :class:`HiddenWorld` is held privately
by :class:`~.env.MicroclimateEnv`; its parameters are committed in the world's
:class:`~probe.truth.SealedTruth` via :meth:`HiddenWorld.payload`.

Observable (per reading of site i with shield k, dwell D, centred at time t)::

    y = q(T_syn(t) + μ_k,i(t) + o_u + δ_u·f(t) + ε_site) − q(T_syn(t) + ε_ref)

* ``T_syn``: synoptic air temperature, a diurnal sinusoid (peak 16 h) plus an
  AR(1) drift. It is common to the site and the reference, so differencing
  removes it; it still matters for overheating cut-outs (MNAR).
* ``μ``: the structural site anomaly relative to the reference station
  (including sensor radiation error, which belongs to the measurement).
* ``o_u, δ_u``: offset and linear drift of the sensor unit used (unit fixed by
  site and shield; ``f`` is the elapsed fraction of the campaign window).
* ``ε_site = s_turb·z₁ + s_inst·√(5/D)·z₂ (+ spike)``, ``ε_ref = s_ref·z₃``;
  z ~ N(0, 1) (Student-t for the turbulent term in R8). ``q`` quantises each
  raw reading to 0.05 °C.
* Missingness: MNAR overheating cut-out with probability
  0.9·logistic((T_site_raw − T_cut)/0.5), then MAR logger dropout at a fixed
  rate. Missing readings have ``value=None``.

**Common random numbers.** All noise and missingness draws of a reading come
from ``make_rng(seed, "microclimate", regime, "obs", site, shield, dwell,
start_tick)``: a pure function of the world and the *measurement identity*,
never of how many draws happened before. Weather series are drawn once per world
from their own keyed streams.

**Structural part μ.**

*In-set worlds* (every regime except R3) are built so the truth lies exactly in
the span of the true hypothesis m*: μ = φ_m*(record)ᵀβ*, using the very feature
map of :mod:`probe.engine.ir`. β* is the least-squares projection, over the
declared population (sites × shields × 15-minute times in the window), of a
physics-shaped target whose magnitudes come from a few [U] parameters (see
:mod:`.scenarios`):

* shade cooling at peak forcing: tree κ·0.8·(1 − T_tree) + e_tree, built
  κ·0.8·(1 − T_built), with κ the air-coupling coefficient and T the effective
  shortwave transmission;
* its time profile g(t): a first-order lag (τ = 60 min) of clear-sky GHI from
  pvlib (a surface-energy-balance response), so H3 truths are
  −peak·((1 − ρ)·ḡ + ρ·g(t)); H1/H2 truths use the window mean ḡ (ρ = 0);
* surface contrasts κ·(A_s − A_asphalt)·ḡ, with absorbed fraction
  A_s = (1 − albedo_s)(1 − evaporative fraction_s);
* radiation-artefact slope r_u·2/(1 + mean wind) °C per kW/m² (H4 truths only;
  in H1-H3 truths both shield types read air temperature exactly, a deliberate
  idealisation that keeps those worlds well specified).

"Mimicry" (R4, R10): an H1 truth's β is the projection of the same seed's H4
pattern *restricted to unshielded rows*, so unshielded readings look alike under
both and only shielded readings separate them.

Optional nuisance added on top (regime knobs; zero in the clean regimes R1, R2,
R4, R5, R6, R10): a smooth spatial Gaussian random field over site coordinates,
a time-varying cross-campus gradient with an optional front (R7), cloud
modulation of every non-intercept effect by the clear-sky index (R7), sensor-unit
offsets and drifts (R3, R9).

*Out-of-set worlds* (R3) run the nonlinear physics directly: true site shortwave
= clear-sky × cloud index × (beam·true transmission + diffuse·svf), with
per-site canopy transmissivity and building-shadow azimuth sectors; surface
excess temperature from a first-order surface energy balance with
surface-specific lag; air heating only above a convective threshold, divided by
a wind-ventilation factor (an unobserved wind channel); tree evapotranspiration;
radiation error for both shield types depending on true shortwave and wind.

Not modelled (v0): mean radiant temperature / UTCI (pythermalcomfort is not a
dependency); portable-shade interventions; a residual first-order sensor lag
(the charged settling time is assumed to absorb it; a lag towards the previous
site's temperature would make a reading depend on history and break
measurement-identity CRN).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from probe.determinism import make_rng
from probe.engine.ir import HypothesisSet, LinearGaussianHypothesis, column_names, design_matrix
from probe.envs.base import Missingness

from .campus import SURFACE_LEVELS, Campus
from .records import SHIELDS, design_records
from .scenarios import RegimeConfig
from .solar import GRID_STEP_MIN, SolarDay, lagged_forcing, solar_day

__all__ = ["FAMILY_KEY", "HiddenWorld", "Measurement"]

FAMILY_KEY = "microclimate"
QUANTUM_C = 0.05
DWELL_REF_MIN = 5.0
SHADE_LAG_MIN = 60.0
POPULATION_STEP_MIN = 15.0
CUTOUT_WIDTH_C = 0.5
CUTOUT_MAX_P = 0.9
SPIKE_RANGE_C = (1.5, 4.0)
HEAVY_SCALE = 0.15
SYNOPTIC_AMPLITUDE_C = 6.0
SYNOPTIC_PEAK_H = 16.0

ALBEDO = {"asphalt": 0.08, "concrete": 0.35, "grass": 0.23, "cool_pavement": 0.45}
EVAP_FRACTION = {"asphalt": 0.0, "concrete": 0.0, "grass": 0.6, "cool_pavement": 0.0}
SURFACE_LAG_MIN = {"asphalt": 75.0, "concrete": 60.0, "grass": 25.0, "cool_pavement": 60.0}
SURFACE_GAIN_C = 40.0
REF_SVF = 0.9
REF_EXPOSURE = 0.9
ABSORB_REF = 0.8

Measurement = tuple[float | None, Missingness, tuple[str, ...]]
"""(value or None, missingness, qc_flags)."""


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    """Stationary AR(1) with marginal sd ``sd``."""
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = sd * e[0]
    s = sd * math.sqrt(1.0 - phi * phi)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + s * e[k]
    return x


def _lag(eq: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """First-order lag along the last axis (exact discretisation); ``tau`` has
    shape ``eq.shape[:-1]`` (one time constant per row)."""
    a = 1.0 - np.exp(-GRID_STEP_MIN / np.asarray(tau, dtype=np.float64))
    out = np.empty_like(eq)
    out[..., 0] = eq[..., 0]
    for k in range(1, eq.shape[-1]):
        out[..., k] = out[..., k - 1] + a * (eq[..., k - 1] - out[..., k - 1])
    return out


def _r(a: np.ndarray, decimals: int = 12) -> np.ndarray:
    """Round derived per-world series so last-ulp libm/LAPACK differences between
    platforms (exp, Cholesky, least squares) cannot change a world."""
    return np.round(np.asarray(a, dtype=np.float64), decimals)


def _quantize(x: float) -> float:
    return round(x / QUANTUM_C) * QUANTUM_C


class HiddenWorld:
    """The full hidden state of one world. Evaluator/simulator use only."""

    def __init__(self, cfg: RegimeConfig, campus: Campus, seed: int, hset: HypothesisSet) -> None:
        self._cfg = cfg
        self._campus = campus
        self._seed = seed
        self._solar: SolarDay = solar_day()
        self._covariates = hset.covariates
        grid = self._solar.minutes
        n_t = grid.shape[0]
        key = (FAMILY_KEY, cfg.id)
        w0, w1 = cfg.window
        self._g = lagged_forcing(SHADE_LAG_MIN)
        win = (grid >= w0) & (grid <= w1)
        self._gbar = float(np.mean(self._g[win]))

        # ---- weather (each process from its own keyed stream) -------------------
        def wrng(name: str) -> np.random.Generator:
            return make_rng(seed, *key, "weather", name)

        self._t_mean = float(wrng("tmean").uniform(32.0, 35.0))
        self._drift = _r(_ar1(wrng("drift"), n_t, 0.98, cfg.synoptic_drift_sd))
        z_cloud = _ar1(wrng("cloud"), n_t, 0.95, 1.0)
        ramp = np.ones(n_t)
        if cfg.cloud_ramp:  # convective cloud building through the afternoon
            ramp = 0.4 + 1.2 * np.clip((grid - 600.0) / 420.0, 0.0, 1.0)
        self._kc = _r(
            np.clip(1.0 - cfg.cloud_level * ramp * np.exp(0.8 * z_cloud - 0.32), 0.3, 1.0)
        )
        wind_rng = wrng("wind")
        self._wind_mean = float(wind_rng.uniform(1.0, 3.0))
        self._wind = _r(np.maximum(0.3, self._wind_mean + _ar1(wind_rng, n_t, 0.95, 0.6)))
        grad_rng = wrng("gradient")
        gradient = _ar1(grad_rng, n_t, 0.97, cfg.gradient_sd)
        front_hour = float(grad_rng.uniform(12.0, 15.0))
        front_mag = float(grad_rng.uniform(0.5, 1.5)) * (1.0 if grad_rng.random() < 0.5 else -1.0)
        if cfg.front:
            gradient = gradient + front_mag / (1.0 + np.exp(-(grid - 60.0 * front_hour) / 10.0))
        self._gradient = _r(gradient)
        self._front = {"enabled": cfg.front, "hour": front_hour, "magnitude": front_mag}

        # ---- spatial field and sensor units ---------------------------------------
        xy = np.asarray([[s.x_m, s.y_m] for s in campus.sites])
        self._x_rel_km = (xy[:, 0] - campus.base_xy[0]) / 1000.0
        frng = make_rng(seed, *key, "field")
        zf = frng.standard_normal(campus.n_sites)
        if cfg.field_sd > 0:
            d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)
            cov = cfg.field_sd**2 * np.exp(-d2 / (2.0 * 250.0**2))
            chol = np.linalg.cholesky(cov + 1e-10 * np.eye(campus.n_sites))
            self._field = _r(chol @ zf)
        else:
            self._field = np.zeros(campus.n_sites)
        urng = make_rng(seed, *key, "units")
        self._unit_offset = {
            sh: cfg.offset_sd * urng.standard_normal(cfg.n_units) for sh in SHIELDS
        }
        self._unit_drift = {
            sh: cfg.unit_drift_sd * urng.standard_normal(cfg.n_units) for sh in SHIELDS
        }

        # ---- effect magnitudes (always drawn in the same order) ------------------
        erng = make_rng(seed, *key, "effects")
        s = cfg.effect_scale
        c0 = float(erng.uniform(-0.2, 0.6))
        kappa = float(erng.uniform(1.0, 2.2))
        t_tree = float(erng.uniform(0.12, 0.28))
        t_built = float(erng.uniform(0.15, 0.35))
        e_tree = float(erng.uniform(0.2, 0.6))
        rho = float(erng.uniform(*cfg.rho_range))
        albedo = {k: ALBEDO[k] + float(erng.uniform(-0.03, 0.03)) for k in SURFACE_LEVELS}
        r_unsh = float(erng.uniform(1.5, 3.0))
        absorb = {k: (1.0 - albedo[k]) * (1.0 - EVAP_FRACTION[k]) for k in SURFACE_LEVELS}
        self._effects: dict[str, Any] = {
            "c0": c0,
            "effect_scale": s,
            "air_coupling": kappa,
            "tree_transmission": t_tree,
            "built_transmission": t_built,
            "tree_evap": e_tree,
            "rho": rho,
            "albedo": albedo,
            "peak_tree": s * (kappa * ABSORB_REF * (1.0 - t_tree) + e_tree),
            "peak_built": s * kappa * ABSORB_REF * (1.0 - t_built),
            "surface_peak": {k: s * kappa * (absorb[k] - absorb["asphalt"]) for k in absorb},
            "artefact_slope": s * r_unsh * 2.0 / (1.0 + self._wind_mean),
            "gbar_window": self._gbar,
        }

        # ---- structure ----------------------------------------------------------
        self._true_id: str | None = None
        self._hyp: LinearGaussianHypothesis | None = None
        self._beta = np.zeros(0)
        self._columns: list[str] = []
        self._mimic_fit_rmse: float | None = None
        self._physics: dict[str, Any] = {}
        self._phys_mean: np.ndarray | None = None
        if cfg.in_set:
            self._build_linear(hset)
        else:
            self._build_physics(c0)

    # ------------------------------------------------------------------ structure
    def _draw_true_id(self) -> str:
        pool = self._cfg.truth_pool
        block, pos = divmod(self._seed, len(pool))
        perm = make_rng(0, FAMILY_KEY, self._cfg.id, "truth-block", block).permutation(len(pool))
        return pool[int(perm[pos])]

    def _population(self) -> tuple[list[dict[str, float | int | str | bool]], np.ndarray]:
        """Declared population rows for the projection: sites × shields × times every
        15 minutes in the window (reading midpoints)."""
        w0, w1 = self._cfg.window
        times = np.arange(w0 + 2.5, w1, POPULATION_STEP_MIN)
        items = [
            (site, shield, float(t))
            for site in self._campus.sites
            for shield in SHIELDS
            for t in times
        ]
        rows = design_records(items, self._solar)
        return rows, np.asarray([m for _, _, m in items])

    def _targets(
        self, rows: list[dict[str, float | int | str | bool]], minutes: np.ndarray
    ) -> dict[str, np.ndarray]:
        e = self._effects
        gbar = self._gbar
        g_t = self._solar.interp(self._g, minutes)
        rho = e["rho"]
        shade = np.asarray([str(r["shade"]) for r in rows])
        surface = np.asarray([str(r["surface"]) for r in rows])
        unsh = np.asarray([bool(r["unshielded"]) for r in rows], dtype=np.float64)
        irr = np.asarray([float(r["irr_proxy"]) for r in rows])
        peak = np.where(shade == "tree", e["peak_tree"], 0.0) + np.where(
            shade == "built", e["peak_built"], 0.0
        )
        surf = np.zeros(len(rows))
        for k, v in e["surface_peak"].items():
            surf = surf + np.where(surface == k, v * gbar, 0.0)
        c0 = e["c0"]
        h1 = c0 - peak * gbar
        return {
            "H1": h1,
            "H2": h1 + surf,
            "H3": c0 - peak * ((1.0 - rho) * gbar + rho * g_t),
            "H4": c0 + e["artefact_slope"] * unsh * irr,
            "_unshielded": unsh,
        }

    def _build_linear(self, hset: HypothesisSet) -> None:
        true_id = self._draw_true_id()
        h = hset.by_id(true_id)
        if not isinstance(h, LinearGaussianHypothesis):  # pragma: no cover - set is linear
            raise TypeError("microclimate hypotheses are linear_gaussian")
        rows, minutes = self._population()
        targets = self._targets(rows, minutes)
        x = design_matrix(h, rows, self._covariates)
        target = targets[true_id]
        mask = np.ones(len(rows), dtype=bool)
        if self._cfg.mimicry and true_id != "H4":
            target = targets["H4"]
            mask = targets["_unshielded"] > 0.5
        beta = _r(np.linalg.lstsq(x[mask], target[mask], rcond=None)[0])
        if self._cfg.mimicry and true_id != "H4":
            self._mimic_fit_rmse = float(np.sqrt(np.mean((x[mask] @ beta - target[mask]) ** 2)))
        self._true_id = true_id
        self._hyp = h
        self._beta = beta
        self._columns = column_names(h)

    def _build_physics(self, c0: float) -> None:
        cfg, campus, sol = self._cfg, self._campus, self._solar
        prng = make_rng(self._seed, FAMILY_KEY, cfg.id, "physics")
        n = campus.n_sites
        tree_tau = prng.uniform(0.06, 0.30, n)
        sector_center = prng.uniform(160.0, 220.0, n)
        sector_half = prng.uniform(60.0, 100.0, n)
        kappa = float(prng.uniform(0.05, 0.09))
        theta = float(prng.uniform(6.0, 12.0))
        e_tree = float(prng.uniform(0.2, 0.6))
        r_unsh = float(prng.uniform(1.5, 3.0))
        r_sh = float(prng.uniform(0.1, 0.3))
        shade = [s.shade for s in campus.sites]
        surf = [s.surface for s in campus.sites]
        svf = np.asarray([s.svf for s in campus.sites])
        wall = np.asarray([s.wall_m for s in campus.sites])

        bh = sol.beam_horizontal[None, :]
        az = sol.azimuth[None, :]
        ang = np.abs((az - sector_center[:, None] + 180.0) % 360.0 - 180.0)
        t_dir = np.ones((n, bh.shape[1]))
        for i, sh in enumerate(shade):
            if sh == "tree":
                t_dir[i] = tree_tau[i]
            elif sh == "built":
                t_dir[i] = np.where(ang[i] <= sector_half[i], 0.0, 1.0)
        kc = self._kc[None, :]
        sw = kc * (bh * t_dir + sol.dhi[None, :] * svf[:, None]) / 1000.0
        sw_ref = self._kc * (sol.beam_horizontal + sol.dhi * REF_SVF) / 1000.0

        absorb = np.asarray([(1 - ALBEDO[k]) * (1 - EVAP_FRACTION[k]) for k in surf])
        tau = np.asarray([SURFACE_LAG_MIN[k] for k in surf])
        ts = _lag(SURFACE_GAIN_C * absorb[:, None] * sw, tau)
        absorb_ref = (1 - ALBEDO["grass"]) * (1 - EVAP_FRACTION["grass"])
        ts_ref = _lag(
            (SURFACE_GAIN_C * absorb_ref * sw_ref)[None, :],
            np.asarray([SURFACE_LAG_MIN["grass"]]),
        )[0]
        exposure = svf * np.minimum(1.0, wall / 8.0)
        vent = 1.0 + 0.25 * self._wind[None, :] * exposure[:, None]
        vent_ref = 1.0 + 0.25 * self._wind * REF_EXPOSURE
        heat = kappa * np.maximum(0.0, ts - theta) / vent
        heat_ref = kappa * np.maximum(0.0, ts_ref - theta) / vent_ref
        is_tree = np.asarray([1.0 if sh == "tree" else 0.0 for sh in shade])[:, None]
        air = c0 + heat - heat_ref[None, :] - e_tree * is_tree * self._g[None, :]
        vent_rad = 2.0 / (1.0 + self._wind[None, :])
        self._phys_mean = _r(
            np.stack([air + r_sh * sw * vent_rad, air + r_unsh * sw * vent_rad]), 10
        )  # (shield, site, time) in SHIELDS order
        self._physics = {
            "tree_canopy_transmissivity": tree_tau,
            "built_sector_center_deg": sector_center,
            "built_sector_half_width_deg": sector_half,
            "convective_kappa": kappa,
            "convective_threshold_c": theta,
            "tree_evap": e_tree,
            "radiation_error": {"unshielded": r_unsh, "shielded": r_sh},
            "surface_gain_c": SURFACE_GAIN_C,
            "albedo": ALBEDO,
            "evaporative_fraction": EVAP_FRACTION,
            "surface_lag_min": SURFACE_LAG_MIN,
        }

    # ------------------------------------------------------------------ evaluation
    @property
    def true_hypothesis_id(self) -> str | None:
        return self._true_id

    def unit_index(self, site_index: int) -> int:
        return site_index % self._cfg.n_units

    def unit_id(self, site_index: int, shield: str) -> str:
        return f"{shield}-{self.unit_index(site_index):02d}"

    def structural(self, site_index: int, shield: str, minute: float, record: Any) -> float:
        """Noise-free site anomaly μ (before sensor-unit offsets and nuisance)."""
        if self._phys_mean is not None:
            k = SHIELDS.index(shield)
            return float(self._solar.interp(self._phys_mean[k, site_index], minute))
        assert self._hyp is not None
        mu = float(design_matrix(self._hyp, [record], self._covariates)[0] @ self._beta)
        if self._cfg.cloud_modulates:
            kc = float(self._solar.interp(self._kc, minute))
            mu = float(self._beta[0]) + kc * (mu - float(self._beta[0]))
        return mu

    def nuisance(self, site_index: int, shield: str, minute: float) -> float:
        w0, w1 = self._cfg.window
        frac = (minute - w0) / (w1 - w0)
        u = self.unit_index(site_index)
        grad = float(self._solar.interp(self._gradient, minute))
        return float(
            self._field[site_index]
            + grad * self._x_rel_km[site_index]
            + self._unit_offset[shield][u]
            + self._unit_drift[shield][u] * frac
        )

    def expected(self, site_index: int, shield: str, minute: float, record: Any) -> float:
        """E[y] for a reading centred at ``minute`` (excluding the mean of R8 spikes)."""
        return self.structural(site_index, shield, minute, record) + self.nuisance(
            site_index, shield, minute
        )

    def synoptic(self, minute: float) -> float:
        diurnal = SYNOPTIC_AMPLITUDE_C * math.sin(
            2.0 * math.pi * (minute / 60.0 - (SYNOPTIC_PEAK_H - 6.0)) / 24.0
        )
        return self._t_mean + diurnal + float(self._solar.interp(self._drift, minute))

    def measure(
        self, site_index: int, shield: str, dwell: float, t_start: float, record: Any
    ) -> Measurement:
        """One reading. Randomness is keyed by the measurement identity (CRN)."""
        cfg = self._cfg
        tick = round(float(t_start) * 4.0)
        rng = make_rng(
            self._seed,
            FAMILY_KEY,
            cfg.id,
            "obs",
            site_index,
            SHIELDS.index(shield),
            round(float(dwell)),
            tick,
        )
        z = rng.standard_normal(3)
        u = rng.random(4)
        turb = cfg.sd_turb * float(z[0])
        if cfg.heavy_df is not None:
            turb = HEAVY_SCALE * float(rng.standard_t(cfg.heavy_df))
        minute = t_start + 0.5 * dwell
        noise = turb + cfg.sd_inst * math.sqrt(DWELL_REF_MIN / dwell) * float(z[1])
        if u[2] < cfg.spike_rate:
            lo, hi = SPIKE_RANGE_C
            noise += lo + (hi - lo) * float(u[3])
        base = self.synoptic(minute)
        site_raw = base + self.expected(site_index, shield, minute, record) + noise
        ref_raw = base + cfg.sd_ref * float(z[2])
        p_cut = CUTOUT_MAX_P / (1.0 + math.exp(-(site_raw - cfg.cutout_c) / CUTOUT_WIDTH_C))
        if u[1] < p_cut:
            return None, "missing", ("overheat_cutout",)
        if u[0] < cfg.mar_rate:
            return None, "missing", ("logger_dropout",)
        return round(_quantize(site_raw) - _quantize(ref_raw), 2), "observed", ()

    # ------------------------------------------------------------------ sealing
    def payload(self) -> dict[str, Any]:
        """Everything that defines this world's truth (JSON-able), for SealedTruth."""
        cfg = self._cfg
        structure: dict[str, Any]
        if self._hyp is not None:
            structure = {
                "kind": "linear",
                "hypothesis_id": self._true_id,
                "columns": self._columns,
                "beta": self._beta,
                "mimicry": cfg.mimicry and self._true_id != "H4",
                "mimic_fit_rmse": self._mimic_fit_rmse,
                "cloud_modulated": cfg.cloud_modulates,
            }
        else:
            structure = {"kind": "physics", "hypothesis_id": None, "physics": self._physics}
        return {
            "family": FAMILY_KEY,
            "regime": cfg.id,
            "seed": self._seed,
            "expected_verdict": cfg.expected_verdict,
            "structure": structure,
            "effects": self._effects,
            "weather": {
                "t_mean": self._t_mean,
                "synoptic_drift": self._drift,
                "clear_sky_index": self._kc,
                "wind_mean": self._wind_mean,
                "wind": self._wind,
                "gradient_c_per_km": self._gradient,
                "front": self._front,
            },
            "nuisance": {
                "field": self._field,
                "unit_offsets": self._unit_offset,
                "unit_drifts": self._unit_drift,
            },
            "sensor": {
                "sd_turb": cfg.sd_turb,
                "sd_inst_5min": cfg.sd_inst,
                "sd_ref": cfg.sd_ref,
                "heavy_df": cfg.heavy_df,
                "heavy_scale": HEAVY_SCALE if cfg.heavy_df is not None else None,
                "spike_rate": cfg.spike_rate,
                "mar_rate": cfg.mar_rate,
                "cutout_c": cfg.cutout_c,
                "quantum_c": QUANTUM_C,
            },
        }
