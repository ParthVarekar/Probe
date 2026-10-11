"""Universal-inference e-processes between registered hypotheses (handover §12 H.6, H.8).

For an ordered pair of hypotheses (m, m′) and the observations y_1..y_t taken
*after registration* at designs d_1..d_t, the split-likelihood-ratio e-process of
Wasserman, Ramdas & Balakrishnan (2020, "Universal inference", PNAS 117(29)) is

    E_t(m : m′) = Π_{i ≤ t} q_m(y_i | d_i, D_{<i})  /  sup_{θ ∈ Θ_{m′}} L_{m′}(θ; D_t),

* **numerator**: the *prequential* (one-step-ahead posterior predictive) density of
  m, each factor q_m(y_i | d_i, D_{<i}) evaluated **before** m is updated with
  y_i. Its product is m's marginal likelihood of the window, but what matters for
  validity is only that each factor is a density fixed before y_i is seen;
* **denominator**: the maximised likelihood of the rival m′ over the *same*
  observations (``HypothesisModel.max_log_likelihood``). A value of +inf (m′
  interpolates the data, e.g. n ≤ p with free σ²) gives log E = −inf ("cannot
  reject"); −inf (every parameter of a finite grid gives the data probability 0)
  gives log E = +inf.

**Validity (why sup_t E_t ≥ 1/α has probability ≤ α under the null).** Suppose
the data come from m′ with some fixed θ₀ ∈ Θ_{m′} (the null "m′ is true"). Then
L_{m′}(θ₀; D_t) ≤ sup_θ L_{m′}(θ; D_t), so E_t(m : m′) ≤ M_t := Π q_m(y_i | ·) /
Π p_{m′}(y_i | d_i, θ₀, ·). Each factor of M_t has conditional expectation 1
given the past and d_i, so M_t is a non-negative martingale with M_0 = 1 and, by
Ville's inequality, P(∃t: E_t ≥ 1/α) ≤ P(∃t: M_t ≥ 1/α) ≤ α. Stopping at the
first crossing, or at any data-dependent time, keeps the guarantee. The
assumptions are exactly these:

1. **Predictable design.** d_i may depend on D_{<i} and on independent
   randomisation (audit draws, ties), but not on y_i or on the hidden truth. The
   design's own probabilities then cancel between numerator and denominator.
   Missing outcomes must be missing for reasons that do not depend on the
   unobserved value (they contribute no factor to either side).
2. **The null class contains the truth.** When m′ is the null, its model class
   (likelihood family *and* parameter set Θ_{m′}, e.g. the finite grid) contains
   the data-generating distribution. For a stateful grid mechanism the state
   evolution is part of the likelihood; it may depend on the actions and θ but not
   on the noisy outcomes. Under misspecification of m′ the guarantee does not
   transfer, which is why the misspecification alarms of H.7 exist.
3. **Same window.** Numerator and denominator cover the same observations, all
   taken after both hypotheses were registered. The tracker therefore requires
   fresh models (no data) when it starts; a hypothesis registered later needs a
   tracker of its own started at its registration, with models created then.

The numerator needs no correctness: any predictable density sequence keeps the
guarantee (a poor numerator only loses power). Tempering (η < 1) changes the
posterior weights, never these quantities; the numerator uses the untempered
predictive of m.

**Decisions over several rivals.** Declaring "m̂" requires E(m̂ : m′) ≥ 1/α for
every rival m′ whose class does not contain m̂'s (a rival that nests m̂ also
contains the truth when m̂ does, so that e-process cannot grow). The decision is
wrong only if the true hypothesis m′ was rejected, and that one event has
probability ≤ α, so no multiplicity correction is needed for the decision
itself (an intersection-union argument). For *reporting* several separate
claims, :func:`ebh` gives e-BH false-discovery-rate control (Wang & Ramdas 2022).

Everything here is deterministic arithmetic on the numbers the engine reports.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from probe.engine.models import HypothesisModel

__all__ = [
    "EProcessTracker",
    "e_to_p",
    "ebh",
    "json_float",
    "log_e_value",
]


def log_e_value(log_numerator: float, max_log_likelihood: float) -> float:
    """log E = log numerator − log sup L_{m′}, with the conventions of the module
    docstring: a refuted numerator (−inf) or an unbounded denominator (+inf) gives
    −inf ("cannot reject"); otherwise a refuted rival (denominator −inf) gives +inf."""
    num = float(log_numerator)
    den = float(max_log_likelihood)
    if math.isnan(num) or math.isnan(den):
        raise ValueError("NaN in an e-process term")
    if num == math.inf:
        raise ValueError("log numerator is +inf: not a valid predictive density")
    if num == -math.inf or den == math.inf:
        return -math.inf
    if den == -math.inf:
        return math.inf
    return num - den


def json_float(x: float) -> float | str:
    """A float for canonical JSON: finite values unchanged, ±inf and NaN as the
    strings ``"inf"``, ``"-inf"`` and ``"nan"`` (canonical JSON rejects them)."""
    f = float(x)
    if math.isfinite(f):
        return f
    if math.isnan(f):
        return "nan"
    return "inf" if f > 0 else "-inf"


class EProcessTracker:
    """E_t(i : j) for every ordered pair of hypotheses (tracker order), fed once per
    executed action.

    Per step the run loop passes, for every hypothesis k, the log predictive
    log p_k(y_t | d_t, D_{<t}) computed **before** updating the models, and then
    each model's ``max_log_likelihood()`` **after** the update::

        lp = [m.log_predictive(record, y) for m in models]   # before update
        for m in models:
            m.update(record, y)
        tracker.observe(lp, [m.max_log_likelihood() for m in models])

    (:meth:`observe_models` reads the second list from the models and checks their
    ids.) A missing outcome is passed as ``log_predictives=None``: it adds nothing
    to any numerator.

    ``log_e()[i, j]`` is log E_t(i : j); the diagonal is 0 by convention (unused).
    :meth:`running_max_log_e` is sup_{s ≤ t} log E_s (with E_0 = 1), the quantity
    Ville's inequality bounds.
    """

    def __init__(self, hypothesis_ids: Sequence[str]) -> None:
        ids = tuple(str(h) for h in hypothesis_ids)
        if not ids:
            raise ValueError("an e-process tracker needs at least one hypothesis")
        if len(set(ids)) != len(ids):
            raise ValueError("hypothesis ids must be unique")
        k = len(ids)
        self._ids = ids
        self._index = {h: i for i, h in enumerate(ids)}
        self._log_num = np.zeros(k)
        self._mll = np.zeros(k)
        self._log_e = np.zeros((k, k))
        self._max_log_e = np.zeros((k, k))
        self._n_steps = 0
        self._n_obs = 0

    @classmethod
    def for_models(cls, models: Sequence[HypothesisModel]) -> EProcessTracker:
        """A tracker over ``models`` (their order), which must not have seen data yet:
        the denominators come from the models' own data, so the window has to start
        at the models' registration."""
        used = [m.hypothesis_id for m in models if m.n_obs > 0]
        if used:
            raise ValueError(
                f"models {used} already conditioned on data; an e-process window must "
                "start at registration (create the tracker with fresh models)"
            )
        return cls([m.hypothesis_id for m in models])

    # -- feeding -------------------------------------------------------------
    def observe(
        self,
        log_predictives: Sequence[float] | np.ndarray | None,
        max_log_likelihoods: Sequence[float] | np.ndarray,
    ) -> None:
        """Record one executed action (see the class docstring). Validated first, so
        a bad input leaves the tracker unchanged."""
        k = len(self._ids)
        mll = np.asarray(max_log_likelihoods, dtype=np.float64)
        if mll.shape != (k,):
            raise ValueError(f"max_log_likelihoods must have shape ({k},)")
        if np.any(np.isnan(mll)):
            raise ValueError("NaN maximum log-likelihood")
        lp: np.ndarray | None = None
        if log_predictives is not None:
            lp = np.asarray(log_predictives, dtype=np.float64)
            if lp.shape != (k,):
                raise ValueError(f"log_predictives must have shape ({k},)")
            if np.any(np.isnan(lp)) or np.any(np.isposinf(lp)):
                raise ValueError("log predictive densities must be < +inf and not NaN")
        if lp is not None:
            self._log_num = self._log_num + lp  # −inf stays −inf (a refuted numerator)
            self._n_obs += 1
        self._mll = mll.copy()
        self._n_steps += 1
        self._log_e = self._matrix()
        self._max_log_e = np.maximum(self._max_log_e, self._log_e)

    def observe_models(
        self,
        log_predictives: Sequence[float] | np.ndarray | None,
        models: Sequence[HypothesisModel],
    ) -> None:
        """:meth:`observe` with the denominators read from ``models`` (already updated
        with this step's outcome), which must be in tracker order."""
        ids = tuple(m.hypothesis_id for m in models)
        if ids != self._ids:
            raise ValueError(f"models {ids} do not match the tracker's hypotheses {self._ids}")
        self.observe(log_predictives, [m.max_log_likelihood() for m in models])

    def _matrix(self) -> np.ndarray:
        k = len(self._ids)
        out = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                if i != j:
                    out[i, j] = log_e_value(float(self._log_num[i]), float(self._mll[j]))
        return out

    # -- reading -------------------------------------------------------------
    @property
    def hypothesis_ids(self) -> tuple[str, ...]:
        return self._ids

    @property
    def n_steps(self) -> int:
        """Executed actions fed so far (including missing outcomes)."""
        return self._n_steps

    @property
    def n_obs(self) -> int:
        """Observed (non-missing) outcomes in the window."""
        return self._n_obs

    def index(self, hypothesis_id: str) -> int:
        return self._index[hypothesis_id]

    def log_numerators(self) -> np.ndarray:
        """Σ log q_k(y_i | ·) per hypothesis (the prequential log evidence of the window)."""
        return self._log_num.copy()

    def max_log_likelihoods(self) -> np.ndarray:
        return self._mll.copy()

    def log_e(self) -> np.ndarray:
        """(K, K) matrix of log E_t(i : j); diagonal 0."""
        return self._log_e.copy()

    def running_max_log_e(self) -> np.ndarray:
        """(K, K) matrix of sup_{s ≤ t} log E_s(i : j) (≥ 0, since E_0 = 1)."""
        return self._max_log_e.copy()

    def log_e_value(self, numerator_id: str, rival_id: str) -> float:
        return float(self._log_e[self._index[numerator_id], self._index[rival_id]])

    def versus_rivals(self, numerator_id: str) -> dict[str, float]:
        """log E_t(numerator : j) for every other hypothesis j."""
        i = self._index[numerator_id]
        return {h: float(self._log_e[i, j]) for j, h in enumerate(self._ids) if j != i}

    def summary(self) -> dict[str, Any]:
        """JSON-able state (non-finite values as strings, see :func:`json_float`)."""
        ids = self._ids
        return {
            "hypothesis_ids": list(ids),
            "n_steps": self._n_steps,
            "n_obs": self._n_obs,
            "log_numerator": {h: json_float(v) for h, v in zip(ids, self._log_num, strict=True)},
            "max_log_likelihood": {h: json_float(v) for h, v in zip(ids, self._mll, strict=True)},
            "log_e": {
                a: {b: json_float(self._log_e[i, j]) for j, b in enumerate(ids) if j != i}
                for i, a in enumerate(ids)
            },
            "running_max_log_e": {
                a: {b: json_float(self._max_log_e[i, j]) for j, b in enumerate(ids) if j != i}
                for i, a in enumerate(ids)
            },
        }


def e_to_p(log_e: float | np.ndarray) -> np.ndarray:
    """The valid p-value min(1, 1/E) of an e-value (Markov's inequality), from log E.
    Use it to combine e-values with p-value procedures such as Holm."""
    le = np.asarray(log_e, dtype=np.float64)
    if np.any(np.isnan(le)):
        raise ValueError("NaN log e-value")
    with np.errstate(over="ignore"):
        return np.minimum(1.0, np.exp(-le))


def ebh(e_values: Sequence[float] | np.ndarray, alpha: float) -> np.ndarray:
    """e-BH (Wang & Ramdas 2022, JRSS-B 84(3)): with K e-values and level α, let
    k* = max{k : e_(k) ≥ K/(α·k)} (e_(1) ≥ e_(2) ≥ ... the sorted e-values; k* = 0
    if none) and reject every hypothesis with e ≥ K/(α·k*). This controls the false
    discovery rate at α·K₀/K ≤ α under *arbitrary* dependence between the e-values.

    Returns a boolean rejection mask aligned with ``e_values``. Ties are handled by
    the threshold form, so the result does not depend on the order of equal values.
    e-values may be +inf (certainly rejected) but not negative or NaN."""
    e = np.asarray(e_values, dtype=np.float64)
    if e.ndim != 1:
        raise ValueError("e_values must be 1-D")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if np.any(np.isnan(e)) or np.any(e < 0.0):
        raise ValueError("e-values must be non-negative and not NaN")
    k = e.shape[0]
    if k == 0:
        return np.zeros(0, dtype=bool)
    srt = np.sort(e)[::-1]
    ranks = np.arange(1, k + 1, dtype=np.float64)
    ok = srt >= k / (alpha * ranks)
    if not ok.any():
        return np.zeros(k, dtype=bool)
    k_star = int(np.max(np.flatnonzero(ok))) + 1
    return np.asarray(e >= k / (alpha * k_star))
