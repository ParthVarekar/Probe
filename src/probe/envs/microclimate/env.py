"""The agent-facing microclimate environment (one world).

State: the team's position (a site, or ``BASE`` where the reference station
stands and the campaign starts), the clock (minutes after local standard
midnight) and the effort spent.

Action ``site=S07|shield=unshielded|dwell=05`` (handover §12 H.1: site, shield,
dwell): walk to the site, let the sensor settle, then record for the dwell time.
With travel τ from the current position, settling s(shield) and dwell D::

    t_start = clock + τ + s,   t_end = t_start + D,   cost = τ + s + D

so the clock advances by exactly the cost of each action and
``clock == window_start + spent`` at all times. There is no free waiting: every
regime's budgets fit inside its campaign window (``window_start + budget <=
window_end``), so the effort budget, not the window, is what binds for every
policy. (An earlier draft offered a free 60-minute wait; uniformly random
baselines then spent half their actions waiting and ran out of window long before
their budget, which made the large budget level meaningless and the baselines
strawmen.)

The reading is centred at t_mid = t_start + D/2; its design record (shade,
surface, hour, unshielded, irr_proxy, svf) is evaluated there. An action is
feasible iff ``t_end`` is within the campaign window and the shield is available
at the site (R10 has shielded masts at two sites only; other shielded actions do
not exist there). Time only moves forward; when no action fits before the window
closes the environment is exhausted.

Candidates per step: 24 sites × shields × dwell {5, 15}, i.e. 96 (52 in R10),
listed feasible or not and sorted by action id. A candidate's ``stratum`` is
``shade=…|surface=…|hour=HH`` (hour blocks 9/11/13/15 h); readings that would end
after the window (infeasible) are assigned the window's last block, so every
stratum is a key of the declared population.

Observations carry ``value`` (°C, on a 0.05 grid, or ``None`` when missing with
``qc_flags`` ``("logger_dropout",)`` or ``("overheat_cutout",)``), the realised
record (identical to the candidate's), ``clock`` = ``t_end``, and ``meta`` with the
site, shield, dwell, travel, settle, start/end times and sensor-unit id.

Sealing: the environment holds its :class:`~.scm.HiddenWorld` privately and
exposes only metadata (public campus map, schema, regime knobs that are public
design), candidates and observations. Neither the metadata nor ``repr`` contains
the world seed: the true hypothesis is a deterministic function of
``(regime, seed)``, and official benchmark seeds are private (handover §12 H.11).
Every call to :meth:`MicroclimateEnv.candidates` and
:meth:`MicroclimateEnv.metadata` returns fresh copies, so a caller mutating a
record or the metadata cannot change what the next caller sees. The
evaluator-only ``oracle_*`` methods and :attr:`MicroclimateEnv.world_id` are for
the run loop and scorer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

from probe.engine.ir import CovariateSpec, OutcomeSpec
from probe.envs.base import (
    Action,
    Candidate,
    Environment,
    EnvMetadata,
    InfeasibleActionError,
    Observation,
)
from probe.truth import EvaluatorKey

from .campus import Campus, Site
from .records import (
    SHIELDS,
    design_record,
    design_records,
    population_strata,
    stratum_key,
)
from .scenarios import RegimeConfig, scenario_manifest_hash
from .scm import HiddenWorld
from .solar import (
    ALTITUDE_M,
    CAMPAIGN_DATE,
    LATITUDE,
    LONGITUDE,
    NOMINAL_DIRECT_TRANSMISSION,
    SolarDay,
)

__all__ = ["DWELLS", "VERSION", "MicroclimateEnv", "mast_sites"]

VERSION = "0.2.0"
DWELLS: tuple[int, ...] = (5, 15)
N_MASTS = 2


@dataclass(frozen=True)
class _Plan:
    site_index: int
    shield: str
    dwell: float
    travel: float
    settle: float
    t_start: float
    t_end: float

    @property
    def t_mid(self) -> float:
        return self.t_start + 0.5 * self.dwell


def mast_sites(campus: Campus) -> tuple[int, ...]:
    """The open sites farthest (by walking time) from base: R10 shielded masts."""
    base = campus.base_index
    open_sites = [s for s in campus.sites if s.shade == "open"]
    ranked = sorted(open_sites, key=lambda s: (-campus.walk_minutes(base, s.index), s.index))
    return tuple(sorted(s.index for s in ranked[:N_MASTS]))


class MicroclimateEnv(Environment):
    def __init__(
        self,
        *,
        cfg: RegimeConfig,
        seed: int,
        world_id: str,
        campus: Campus,
        solar: SolarDay,
        covariates: tuple[CovariateSpec, ...],
        outcome: OutcomeSpec,
        hidden: HiddenWorld,
        commitment: str,
    ) -> None:
        self.__cfg = cfg
        self.__hidden = hidden
        self._world_id = world_id
        self._seed = seed
        self._campus = campus
        self._solar = solar
        self._covariates = covariates
        self._outcome = outcome
        self._commitment = commitment
        self._window = (float(cfg.window[0]), float(cfg.window[1]))
        self._settle = {"unshielded": cfg.settle_unshielded, "shielded": cfg.settle_shielded}
        masts = mast_sites(campus) if cfg.shield_sites == "masts" else ()
        self._shields: dict[int, tuple[str, ...]] = {
            s.index: (SHIELDS if cfg.shield_sites == "all" or s.index in masts else ("unshielded",))
            for s in campus.sites
        }
        self._masts = masts
        self._population = population_strata(campus.sites, self._window)
        self._last_block_minute = _last_block_minute(self._window)
        self._clock = self._window[0]
        self._pos = campus.base_index
        self._spent = 0.0
        self._step = 0
        self._cache: tuple[list[Candidate], dict[str, tuple[Candidate, _Plan]]] | None = None
        self._metadata = self._build_metadata()

    # ------------------------------------------------------------------ public API
    def __repr__(self) -> str:
        # No seed / world id: the seed determines the truth (see module docstring).
        return (
            f"MicroclimateEnv(regime={self.__cfg.id!r}, commitment={self._commitment[:12]}…, "
            f"clock={self._clock!r}, position={self.position!r}, spent={self._spent!r}, "
            f"step={self._step})"
        )

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("environments are not picklable; rebuild them from (regime, seed)")

    def metadata(self) -> EnvMetadata:
        return copy.deepcopy(self._metadata)

    @property
    def world_id(self) -> str:
        """``microclimate/<regime>/<seed>``: for the run loop and evaluator only (it
        contains the seed), never placed in metadata, candidates or observations."""
        return self._world_id

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def clock(self) -> float:
        return self._clock

    @property
    def position(self) -> str:
        if self._pos == self._campus.base_index:
            return "BASE"
        return self._campus.sites[self._pos].id

    @property
    def step(self) -> int:
        return self._step

    @property
    def exhausted(self) -> bool:
        return not any(c.feasible for c in self.candidates())

    def truth_commitment(self) -> str:
        return self._commitment

    def candidates(self) -> list[Candidate]:
        # Fresh records/params per call: the cached ones are never handed out.
        return [
            replace(
                c,
                action=Action(id=c.action.id, params=dict(c.action.params)),
                record=dict(c.record),
            )
            for c in self._state_candidates()[0]
        ]

    def execute(self, action: Action) -> Observation:
        if not isinstance(action, Action):
            raise TypeError("execute() takes an Action")
        cand, plan = self._lookup(action)
        site = self._campus.sites[plan.site_index]
        # Rebuild the record from the plan: never trust a (mutable) cached candidate
        # record that the caller could have modified.
        record = design_record(site, plan.shield, plan.t_mid, self._solar)
        value, missingness, flags = self.__hidden.measure(
            plan.site_index, plan.shield, plan.dwell, plan.t_start, record
        )
        obs = Observation(
            step=self._step,
            action_id=cand.action.id,
            value=value,
            missingness=missingness,
            record=record,
            cost=cand.cost,
            clock=plan.t_end,
            origin="simulated",
            qc_flags=flags,
            meta={
                "site": site.id,
                "shield": plan.shield,
                "dwell": plan.dwell,
                "travel": plan.travel,
                "settle": plan.settle,
                "t_start": plan.t_start,
                "t_end": plan.t_end,
                "unit": self.__hidden.unit_id(plan.site_index, plan.shield),
            },
        )
        self._clock = plan.t_end
        self._pos = plan.site_index
        self._spent += cand.cost
        self._step += 1
        self._cache = None
        return obs

    # ------------------------------------------------------------------ evaluator only
    def oracle_expected(self, key: EvaluatorKey, action: Action) -> float:
        """Noise-free expected observation of ``action`` from the current state."""
        _require(key)
        _, plan = self._lookup(action, allow_infeasible=True)
        site = self._campus.sites[plan.site_index]
        record = design_record(site, plan.shield, plan.t_mid, self._solar)
        return self.__hidden.expected(plan.site_index, plan.shield, plan.t_mid, record)

    def oracle_expected_at(
        self, key: EvaluatorKey, site_id: str, shield: str, minute: float
    ) -> float:
        """Noise-free expected observation of a reading centred at ``minute``."""
        _require(key)
        site = self._site(site_id)
        record = design_record(site, shield, minute, self._solar)
        return self.__hidden.expected(site.index, shield, minute, record)

    def oracle_structural_at(
        self, key: EvaluatorKey, site_id: str, shield: str, minute: float
    ) -> float:
        """The structural anomaly μ only (no sensor-unit offsets or nuisance)."""
        _require(key)
        site = self._site(site_id)
        record = design_record(site, shield, minute, self._solar)
        return self.__hidden.structural(site.index, shield, minute, record)

    # ------------------------------------------------------------------ internals
    def _site(self, site_id: str) -> Site:
        for s in self._campus.sites:
            if s.id == site_id:
                return s
        raise KeyError(site_id)

    def _lookup(self, action: Action, allow_infeasible: bool = False) -> tuple[Candidate, _Plan]:
        table = self._state_candidates()[1]
        hit = table.get(action.id)
        if hit is None:
            raise InfeasibleActionError(f"unknown action {action.id!r} in this state")
        cand, plan = hit
        if dict(action.params) != dict(cand.action.params):
            raise InfeasibleActionError(f"action params do not match id {action.id!r}")
        if not cand.feasible and not allow_infeasible:
            raise InfeasibleActionError(f"{action.id}: {cand.infeasible_reason}")
        return cand, plan

    def _state_candidates(self) -> tuple[list[Candidate], dict[str, tuple[Candidate, _Plan]]]:
        if self._cache is not None:
            return self._cache
        plans: list[_Plan] = []
        for site in self._campus.sites:
            travel = self._campus.walk_minutes(self._pos, site.index)
            for shield in self._shields[site.index]:
                settle = self._settle[shield]
                for dwell in DWELLS:
                    t_start = self._clock + travel + settle
                    plans.append(
                        _Plan(
                            site_index=site.index,
                            shield=shield,
                            dwell=float(dwell),
                            travel=travel,
                            settle=settle,
                            t_start=t_start,
                            t_end=t_start + dwell,
                        )
                    )
        sites = self._campus.sites
        records = design_records(
            [(sites[p.site_index], p.shield, p.t_mid) for p in plans], self._solar
        )
        end = self._window[1]
        out: list[tuple[Candidate, _Plan]] = []
        for plan, record in zip(plans, records, strict=True):
            site = sites[plan.site_index]
            dwell = int(plan.dwell)
            feasible = plan.t_end <= end
            cand = Candidate(
                action=Action(
                    id=f"site={site.id}|shield={plan.shield}|dwell={dwell:02d}",
                    params={"site": site.id, "shield": plan.shield, "dwell": dwell},
                ),
                record=record,
                cost=plan.travel + plan.settle + plan.dwell,
                stratum=stratum_key(site, min(plan.t_mid, self._last_block_minute)),
                feasible=feasible,
                infeasible_reason=None
                if feasible
                else "reading would end after the campaign window closes",
            )
            out.append((cand, plan))
        out.sort(key=lambda cp: cp[0].action.id)
        cands = [c for c, _ in out]
        table = {c.action.id: (c, p) for c, p in out}
        self._cache = (cands, table)
        return self._cache

    def _build_metadata(self) -> EnvMetadata:
        cfg = self.__cfg
        return EnvMetadata(
            family="microclimate",
            version=VERSION,
            description="Synthetic campus microclimate (SIMULATED). Observable: sensor reading "
            "minus the reference-station reading, degC. Actions: (site, shield, dwell); cost is "
            "field-effort minutes (travel + settling + dwell) and the clock advances by exactly "
            "the cost, forward only, within the campaign window.",
            outcome=self._outcome,
            covariates=self._covariates,
            cost_unit="min",
            origin="simulated",
            population=self._population,
            extra={
                "regime": cfg.id,
                "campaign_window_min": list(self._window),
                "start_position": "BASE",
                "dwell_options_min": list(DWELLS),
                "settle_min": dict(self._settle),
                "shield_availability": {
                    self._campus.sites[i].id: list(v) for i, v in self._shields.items()
                },
                "mast_sites": [self._campus.sites[i].id for i in self._masts],
                "nominal_direct_transmission": dict(NOMINAL_DIRECT_TRANSMISSION),
                "solar": {
                    "latitude": LATITUDE,
                    "longitude": LONGITUDE,
                    "altitude_m": ALTITUDE_M,
                    "date": CAMPAIGN_DATE,
                    "utc_offset_h": -7,
                },
                "campus": self._campus.public(),
                "scenario_manifest_hash": scenario_manifest_hash(),
                "truth_commitment": self._commitment,
            },
        )


def _last_block_minute(window: tuple[float, float]) -> float:
    """A minute inside the last hour block that overlaps the window: strata of
    readings centred after the window closes are clamped to it."""
    w0, w1 = window
    return max(w0, w1 - 1e-6)


def _require(key: EvaluatorKey) -> None:
    if not isinstance(key, EvaluatorKey):
        raise PermissionError("oracle access requires an EvaluatorKey")
