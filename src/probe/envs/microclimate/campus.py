"""Schematic campus: sites, their public attributes and the walking-time matrix.

A campus is drawn from a keyed RNG (``make_rng(seed, "microclimate", regime,
"campus")``) and is **public**: everything here is something an investigator
would know from a site survey (shade class, surface material, sky-view factor,
distance to the nearest wall, map coordinates, walking times). Nothing here is
hidden truth; hidden per-site physics (true canopy transmissivity, building
shadow sectors, spatial temperature field) lives in :mod:`.scm`.

Confounding is designed in (handover §12 H.1): surface material is drawn
conditionally on shade, so sunlit sites tend to be asphalt and tree-shaded
sites tend to be grass.

Walking time between points a and b (minutes, rounded to 0.5):
``max(min_walk, round_half(0.5 + detour·dist/speed))`` with speed 75 m/min and
detour factor 1.3 ([U] assumptions). Index ``n_sites`` of the matrix is the
base, where the reference station stands and every campaign starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "LAYOUTS",
    "SHADE_LEVELS",
    "SURFACE_LEVELS",
    "Campus",
    "CampusLayout",
    "Site",
    "generate_campus",
]

SHADE_LEVELS: tuple[str, ...] = ("open", "tree", "built")
SURFACE_LEVELS: tuple[str, ...] = ("asphalt", "concrete", "grass", "cool_pavement")

N_SITES = 24
_SHADE_COUNTS = {"open": 10, "tree": 8, "built": 6}
_SURFACE_GIVEN_SHADE: dict[str, tuple[float, float, float, float]] = {
    # P(asphalt, concrete, grass, cool_pavement | shade): sunlit sites tend to be asphalt.
    "open": (0.50, 0.22, 0.16, 0.12),
    "tree": (0.15, 0.20, 0.55, 0.10),
    "built": (0.25, 0.50, 0.10, 0.15),
}
_SVF_RANGE = {"open": (0.75, 0.95), "tree": (0.15, 0.45), "built": (0.25, 0.55)}
_WALL_RANGE = {"open": (6.0, 40.0), "tree": (4.0, 30.0), "built": (1.0, 6.0)}
WALK_SPEED_M_PER_MIN = 75.0
WALK_DETOUR = 1.3


@dataclass(frozen=True)
class CampusLayout:
    name: str
    width_m: float
    height_m: float
    n_clusters: int
    cluster_sd_m: float
    min_walk_min: float


LAYOUTS: dict[str, CampusLayout] = {
    # Heterogeneous walking costs (1 to ~10-20 min): clustered buildings on a large campus.
    "spread": CampusLayout("spread", 1200.0, 800.0, 5, 70.0, 1.0),
    # Near-uniform costs (1 to 5 min): one compact quad.
    "compact": CampusLayout("compact", 240.0, 160.0, 3, 30.0, 1.0),
    # Expensive moves (>= 3 min each): tiny-budget regime.
    "sparse": CampusLayout("sparse", 1600.0, 1100.0, 6, 60.0, 3.0),
}


@dataclass(frozen=True)
class Site:
    id: str
    index: int
    x_m: float
    y_m: float
    shade: str
    surface: str
    svf: float
    """Sky-view factor (0..1)."""
    wall_m: float
    """Distance to the nearest building wall, metres."""

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "shade": self.shade,
            "surface": self.surface,
            "svf": self.svf,
            "wall_m": self.wall_m,
        }


@dataclass(frozen=True)
class Campus:
    layout: CampusLayout
    sites: tuple[Site, ...]
    base_xy: tuple[float, float]
    walk: np.ndarray
    """(n+1, n+1) walking minutes; index n is the base / reference station."""

    @property
    def n_sites(self) -> int:
        return len(self.sites)

    @property
    def base_index(self) -> int:
        return len(self.sites)

    def walk_minutes(self, origin: int, dest: int) -> float:
        return float(self.walk[origin, dest])

    def public(self) -> dict[str, Any]:
        """JSON-able public map: layout, sites, base and walking matrix."""
        return {
            "layout": self.layout.name,
            "sites": [s.public() for s in self.sites],
            "base": {"id": "BASE", "x_m": self.base_xy[0], "y_m": self.base_xy[1]},
            "walk_minutes": self.walk.tolist(),
            "walk_index": [s.id for s in self.sites] + ["BASE"],
        }


def _assign_surfaces(rng: np.random.Generator, shades: list[str]) -> list[str]:
    surfaces = [
        SURFACE_LEVELS[int(rng.choice(len(SURFACE_LEVELS), p=_SURFACE_GIVEN_SHADE[s]))]
        for s in shades
    ]
    # Deterministic repair: every surface level appears at least twice, so surface
    # effects are estimable. Take sites from the most common level (lowest index first).
    for _ in range(4 * len(SURFACE_LEVELS)):
        counts = {lev: surfaces.count(lev) for lev in SURFACE_LEVELS}
        short = [lev for lev in SURFACE_LEVELS if counts[lev] < 2]
        if not short:
            break
        donor = max(SURFACE_LEVELS, key=lambda lev: (counts[lev], -SURFACE_LEVELS.index(lev)))
        idx = next(i for i, s in enumerate(surfaces) if s == donor)
        surfaces[idx] = short[0]
    return surfaces


def generate_campus(rng: np.random.Generator, layout: CampusLayout) -> Campus:
    shades: list[str] = []
    for lev in SHADE_LEVELS:
        shades += [lev] * _SHADE_COUNTS[lev]
    shades = [shades[i] for i in rng.permutation(len(shades))]
    surfaces = _assign_surfaces(rng, shades)

    w, h = layout.width_m, layout.height_m
    centers = rng.uniform([0.1 * w, 0.1 * h], [0.9 * w, 0.9 * h], size=(layout.n_clusters, 2))
    cluster_of = rng.permutation(N_SITES) % layout.n_clusters
    xy = centers[cluster_of] + rng.normal(0.0, layout.cluster_sd_m, size=(N_SITES, 2))
    xy[:, 0] = np.clip(xy[:, 0], 0.0, w)
    xy[:, 1] = np.clip(xy[:, 1], 0.0, h)
    xy = np.round(xy, 1)
    svf_u = rng.random(N_SITES)
    wall_u = rng.random(N_SITES)

    sites = []
    for i in range(N_SITES):
        lo, hi = _SVF_RANGE[shades[i]]
        wlo, whi = _WALL_RANGE[shades[i]]
        sites.append(
            Site(
                id=f"S{i + 1:02d}",
                index=i,
                x_m=float(xy[i, 0]),
                y_m=float(xy[i, 1]),
                shade=shades[i],
                surface=surfaces[i],
                svf=round(lo + (hi - lo) * float(svf_u[i]), 2),
                wall_m=round(wlo + (whi - wlo) * float(wall_u[i]), 1),
            )
        )
    base = (round(0.5 * w, 1), round(0.5 * h, 1))
    pts = np.vstack([xy, np.asarray([base])])
    dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1))
    walk = np.round((0.5 + WALK_DETOUR * dist / WALK_SPEED_M_PER_MIN) * 2.0) / 2.0
    walk = np.maximum(walk, layout.min_walk_min)
    np.fill_diagonal(walk, 0.0)
    walk.setflags(write=False)
    return Campus(layout=layout, sites=tuple(sites), base_xy=base, walk=walk)
