"""The uniform environment interface (handover §12 Part F, "Environment Adapters").

An *environment family* (microclimate, system lab, ...) builds *worlds*. A world
is a pair: an :class:`Environment` the investigating agent may use, and a
:class:`~probe.truth.SealedTruth` that only the evaluator may open. The
agent never receives the truth object, and nothing reachable through the public
``Environment`` methods may contain truth fields (tested per family).

Determinism and common random numbers (CRN):

* A world is a pure function of ``(family, regime, seed)``.
* Observation noise is **counter-based**: a deterministic function of the world
  seed and the *identity of the measurement* (action id and the clock at
  execution), never of how many draws other policies made. Two policies that
  take the same action at the same simulated time observe the same value.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from probe.engine.ir import CovariateSpec, HypothesisSet, OutcomeSpec, Record
from probe.truth import SealedTruth

__all__ = [
    "Action",
    "Candidate",
    "EnvMetadata",
    "Environment",
    "EnvironmentFamily",
    "InfeasibleActionError",
    "Missingness",
    "Observation",
    "RegimeSpec",
    "World",
]

Missingness = Literal["observed", "missing", "censored", "rejected"]
Origin = Literal["simulated", "executed"]


class InfeasibleActionError(RuntimeError):
    """Raised by :meth:`Environment.execute` for an action not currently feasible."""


@dataclass(frozen=True)
class Action:
    """A concrete, JSON-able action. ``id`` is stable across runs and platforms
    (it is the tie-break key), e.g. ``"site=S07|shield=unshielded|dwell=10"``."""

    id: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class Candidate:
    action: Action
    record: Record
    """Agent-knowable covariates used to build φ_m(d). Must satisfy the family's
    covariate schema. Never contains truth."""
    cost: float
    """Total cost to take this action from the current state (travel + settling +
    dwell, in the family's cost unit). Always > 0."""
    stratum: str
    """Key of the declared sampling population cell (e.g. ``"shade=tree|surface=grass|hour=13"``),
    used by stratified and audit designs."""
    feasible: bool = True
    infeasible_reason: str | None = None


@dataclass(frozen=True)
class Observation:
    step: int
    action_id: str
    value: float | int | None
    missingness: Missingness
    record: Record
    """The realized design record (the record models are updated with)."""
    cost: float
    clock: float
    origin: Origin
    qc_flags: tuple[str, ...] = ()
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.value is None) != (self.missingness != "observed"):
            raise ValueError("value is None exactly when missingness != 'observed'")


@dataclass(frozen=True)
class EnvMetadata:
    family: str
    version: str
    description: str
    outcome: OutcomeSpec
    covariates: tuple[CovariateSpec, ...]
    cost_unit: str
    origin: Origin
    population: tuple[str, ...]
    """All stratum keys of the declared sampling population."""
    extra: Mapping[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """What the investigating agent sees. Mutable: executing an action advances
    the clock, the position and the spent budget."""

    @abstractmethod
    def metadata(self) -> EnvMetadata: ...

    @abstractmethod
    def candidates(self) -> list[Candidate]:
        """All candidate actions from the current state, feasible or not, in a
        deterministic order (sorted by ``action.id``)."""

    @abstractmethod
    def execute(self, action: Action) -> Observation:
        """Execute a feasible action. Raises :class:`InfeasibleActionError`."""

    @property
    @abstractmethod
    def spent(self) -> float:
        """Cost spent so far."""

    @property
    @abstractmethod
    def clock(self) -> float:
        """Current simulated time (family-defined unit; only moves forward)."""

    @property
    @abstractmethod
    def exhausted(self) -> bool:
        """True when no further action can ever be feasible (e.g. campaign over)."""

    @abstractmethod
    def truth_commitment(self) -> str:
        """``SealedTruth.commitment`` for this world, to be logged at start."""


@dataclass(frozen=True)
class World:
    env: Environment
    truth: SealedTruth


@dataclass(frozen=True)
class RegimeSpec:
    id: str
    title: str
    description: str
    expected: str
    """Pre-registered expected outcome (handover §12 Part I), written before any run."""
    budgets: Mapping[str, float]
    """Cost budgets keyed ``small``/``medium``/``large`` in the family's cost unit."""
    warmup: int
    """Number of initial stratified warm-up observations given to every policy."""
    in_set: bool = True
    """False when the truth lies outside the registered hypothesis set."""
    notes: str = ""


class EnvironmentFamily(Protocol):
    name: str
    version: str

    def regimes(self) -> Mapping[str, RegimeSpec]: ...

    def hypothesis_set(self, regime: str) -> HypothesisSet: ...

    def make_world(self, regime: str, seed: int) -> World: ...


def check_records(records: Sequence[Record], covariates: Sequence[CovariateSpec]) -> None:
    """Validate design records against a covariate schema (used by family tests)."""
    cov = {c.name: c for c in covariates}
    for r in records:
        missing = set(cov) - set(r)
        if missing:
            raise ValueError(f"record lacks covariates {sorted(missing)}")
        for k, spec in cov.items():
            v = r[k]
            if spec.kind == "categorical":
                if not isinstance(v, str) or (spec.levels and v not in spec.levels):
                    raise ValueError(f"{k}={v!r} not a declared level")
            elif spec.kind == "binary":
                if not isinstance(v, bool):
                    raise ValueError(f"{k} must be bool")
            elif isinstance(v, str | bool):
                raise ValueError(f"{k} must be numeric")
