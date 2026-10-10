"""Solar geometry and clear-sky irradiance for the schematic campus (pvlib).

Every microclimate world shares one fixed location and summer date, so the pvlib
calls (NREL SPA solar position, Ineichen–Perez clear sky with a fixed Linke
turbidity, which avoids pvlib's turbidity-file lookup) run **once per process**
(``functools.lru_cache``) on a 5-minute grid from 05:00 to 20:00. Everything
else interpolates linearly on that grid, so ``make_world`` stays fast.

Values are rounded before caching (zenith/azimuth to 1e-6 degree, irradiance to
1e-4 W/m²) so that last-ulp libm differences between platforms cannot leak into
world generation, and the cached arrays are read-only.

Clock convention: minutes after local *standard* midnight (MST, UTC-7, no DST).

Two derived quantities are used by the simulator:

* :func:`irradiance_proxy` is **agent-knowable**: clear-sky shortwave reaching a
  site computed only from public information (sun position, the site's declared
  shade class through a *nominal* direct-beam transmission, and its sky-view
  factor). It never sees the hidden cloud process or the true canopy
  transmissivity. It is the ``irr_proxy`` covariate in design records.
* :func:`lagged_forcing` is a first-order lag (time constant τ) of clear-sky GHI,
  normalised to a daily maximum of 1: the shape of a surface-energy-balance
  response to the sun, used to give shade effects their time-of-day profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "ALTITUDE_M",
    "CAMPAIGN_DATE",
    "GRID_END_MIN",
    "GRID_START_MIN",
    "GRID_STEP_MIN",
    "LATITUDE",
    "LINKE_TURBIDITY",
    "LONGITUDE",
    "NOMINAL_DIRECT_TRANSMISSION",
    "TIMEZONE",
    "SolarDay",
    "irradiance_proxy",
    "lagged_forcing",
    "solar_day",
]

LATITUDE = 33.42
LONGITUDE = -111.93
ALTITUDE_M = 360.0
TIMEZONE = "Etc/GMT+7"  # MST, UTC-7 (POSIX sign convention), no daylight saving
CAMPAIGN_DATE = "2025-07-15"
LINKE_TURBIDITY = 3.5

GRID_START_MIN = 300  # 05:00
GRID_END_MIN = 1200  # 20:00
GRID_STEP_MIN = 5

NOMINAL_DIRECT_TRANSMISSION: dict[str, float] = {"open": 1.0, "tree": 0.15, "built": 0.0}
"""Direct-beam transmission an investigator would assume from the shade class
alone (public). True per-site canopy transmissivity is hidden and differs."""


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class SolarDay:
    """Clear-sky solar quantities on the shared grid (arrays of equal length)."""

    minutes: np.ndarray
    zenith: np.ndarray
    """Apparent solar zenith, degrees."""
    azimuth: np.ndarray
    """Solar azimuth, degrees clockwise from north."""
    ghi: np.ndarray
    dni: np.ndarray
    dhi: np.ndarray
    """Clear-sky global horizontal, direct normal and diffuse horizontal, W/m²."""
    beam_horizontal: np.ndarray
    """Direct beam on a horizontal surface, DNI·max(cos Z, 0), W/m² (precomputed)."""

    def interp(self, values: np.ndarray, minute: float | np.ndarray) -> np.ndarray:
        """Linear interpolation of a grid series at ``minute`` (clamped at the ends)."""
        return np.interp(np.asarray(minute, dtype=np.float64), self.minutes, values)


@lru_cache(maxsize=4)
def solar_day(
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
    altitude_m: float = ALTITUDE_M,
    date: str = CAMPAIGN_DATE,
    timezone: str = TIMEZONE,
    linke_turbidity: float = LINKE_TURBIDITY,
) -> SolarDay:
    """pvlib solar position + Ineichen clear sky on the 5-minute grid (cached)."""
    import pandas as pd  # pvlib dependency; imported lazily (slow import)
    import pvlib

    minutes = np.arange(GRID_START_MIN, GRID_END_MIN + 1, GRID_STEP_MIN, dtype=np.float64)
    times = pd.DatetimeIndex(pd.Timestamp(date, tz=timezone) + pd.to_timedelta(minutes, unit="min"))
    loc = pvlib.location.Location(latitude, longitude, tz=timezone, altitude=altitude_m)
    sp = loc.get_solarposition(times)
    cs = loc.get_clearsky(
        times, model="ineichen", solar_position=sp, linke_turbidity=linke_turbidity
    )
    zen = np.round(np.asarray(sp["apparent_zenith"], dtype=np.float64), 6)
    azi = np.round(np.asarray(sp["azimuth"], dtype=np.float64), 6)

    def irr(col: str) -> np.ndarray:
        v = np.nan_to_num(np.asarray(cs[col], dtype=np.float64), nan=0.0)
        return np.round(np.clip(v, 0.0, None), 4)

    dni = irr("dni")
    beam = np.round(dni * np.clip(np.cos(np.radians(zen)), 0.0, None), 4)
    return SolarDay(
        minutes=_readonly(minutes),
        zenith=_readonly(zen),
        azimuth=_readonly(azi),
        ghi=_readonly(irr("ghi")),
        dni=_readonly(dni),
        dhi=_readonly(irr("dhi")),
        beam_horizontal=_readonly(beam),
    )


def irradiance_proxy(
    solar: SolarDay,
    minute: float | np.ndarray,
    direct_transmission: float | np.ndarray,
    svf: float | np.ndarray,
) -> np.ndarray:
    """Agent-knowable clear-sky shortwave at a site, kW/m² (rounded to 1e-4):
    (beam_horizontal·T_nominal(shade) + DHI·svf) / 1000."""
    bh = solar.interp(solar.beam_horizontal, minute)
    dh = solar.interp(solar.dhi, minute)
    val = (bh * np.asarray(direct_transmission) + dh * np.asarray(svf)) / 1000.0
    return np.round(val, 4)


@lru_cache(maxsize=8)
def lagged_forcing(tau_min: float = 60.0) -> np.ndarray:
    """First-order lag of clear-sky GHI with time constant ``tau_min`` (exact
    discretisation on the grid), normalised so the daily maximum is 1."""
    sd = solar_day()
    u = sd.ghi / 1000.0
    a = 1.0 - math.exp(-GRID_STEP_MIN / tau_min)
    x = np.empty_like(u)
    x[0] = u[0]
    for k in range(1, u.shape[0]):
        x[k] = x[k - 1] + a * (u[k - 1] - x[k - 1])
    return _readonly(np.round(x / float(np.max(x)), 9))
