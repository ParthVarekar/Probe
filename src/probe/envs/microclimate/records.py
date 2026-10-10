"""Agent-knowable design records and sampling strata.

A design record is computed **only** from public information: the site survey
(shade, surface, sky-view factor), the shield chosen, the clock (deterministic
travel/settling/dwell arithmetic) and public solar geometry. It never reads the
hidden SCM, so it cannot leak truth. The same function builds candidate records,
observation records and the population rows the simulator projects onto.

Strata (the declared sampling population used by stratified and audit designs)
are ``shade × surface × hour-block`` with hour blocks starting at 9, 11, 13
and 15 h (the last block runs to the end of the day), e.g.
``"shade=tree|surface=grass|hour=13"``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .campus import SHADE_LEVELS, SURFACE_LEVELS, Site
from .solar import NOMINAL_DIRECT_TRANSMISSION, SolarDay, irradiance_proxy

__all__ = [
    "HOUR_BLOCKS",
    "SHIELDS",
    "design_record",
    "design_records",
    "hour_block",
    "population_strata",
    "stratum_key",
]

SHIELDS: tuple[str, ...] = ("shielded", "unshielded")
HOUR_BLOCKS: tuple[int, ...] = (9, 11, 13, 15)
_DAY_END_H = 24


Rec = dict[str, float | int | str | bool]


def design_records(items: Sequence[tuple[Site, str, float]], solar: SolarDay) -> list[Rec]:
    """Covariates of readings ``(site, shield, minute)``, ``minute`` being the middle
    of the reading. Vectorised over items; :func:`design_record` is the one-item case,
    so both give bit-identical records."""
    for _, shield, _ in items:
        if shield not in SHIELDS:
            raise ValueError(f"unknown shield {shield!r}")
    if not items:
        return []
    minutes = np.fromiter((m for _, _, m in items), dtype=np.float64, count=len(items))
    trans = np.fromiter(
        (NOMINAL_DIRECT_TRANSMISSION[s.shade] for s, _, _ in items),
        dtype=np.float64,
        count=len(items),
    )
    svf = np.fromiter((s.svf for s, _, _ in items), dtype=np.float64, count=len(items))
    irr = irradiance_proxy(solar, minutes, trans, svf).tolist()
    return [
        {
            "shade": site.shade,
            "surface": site.surface,
            "hour": round(minute / 60.0, 6),
            "unshielded": shield == "unshielded",
            "irr_proxy": float(v),
            "svf": site.svf,
        }
        for (site, shield, minute), v in zip(items, irr, strict=True)
    ]


def design_record(site: Site, shield: str, minute: float, solar: SolarDay) -> Rec:
    """Covariates of a reading of ``site`` with ``shield`` centred at ``minute``."""
    return design_records([(site, shield, float(minute))], solar)[0]


def hour_block(minute: float) -> int:
    hour = minute / 60.0
    block = HOUR_BLOCKS[0]
    for b in HOUR_BLOCKS:
        if hour >= b:
            block = b
    return block


def stratum_key(site: Site, minute: float) -> str:
    return f"shade={site.shade}|surface={site.surface}|hour={hour_block(minute):02d}"


def population_strata(sites: Sequence[Site], window: tuple[float, float]) -> tuple[str, ...]:
    """All stratum keys of the declared population: (shade, surface) combinations
    present on campus × hour blocks overlapping the campaign window."""
    w0, w1 = window[0] / 60.0, window[1] / 60.0
    ends = (*HOUR_BLOCKS[1:], _DAY_END_H)
    blocks = [b for b, e in zip(HOUR_BLOCKS, ends, strict=True) if b < w1 and e > w0]
    combos = {(s.shade, s.surface) for s in sites}
    keys = [
        f"shade={sh}|surface={su}|hour={b:02d}"
        for sh in SHADE_LEVELS
        for su in SURFACE_LEVELS
        if (sh, su) in combos
        for b in blocks
    ]
    return tuple(sorted(keys))
