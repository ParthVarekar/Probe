"""Exact finite-grid engine for ``finite_grid`` hypotheses.

A :class:`~probe.engine.ir.GridHypothesis` names a registered mechanism
(:mod:`probe.engine.mechanisms`) and a Cartesian parameter grid θ₁..θ_G with a
uniform prior. Each mechanism gives, for the current per-grid-point state and a
design record, the outcome PMF p_g(y | d) over the count support 0..S−1, and a
deterministic state transition that depends on the action and θ but never on
the noisy outcome. Inference over the grid index is therefore exact:

* log w_g ← log w_g + log p_g(y_t | d_t) − log p(y_t | D)  (normalised in log space)
* log p(y_t | D) = logsumexp_g(log w_g + log p_g(y_t | d_t))  (prequential evidence term)
* the predictive is the mixture Σ_g w_g p_g(· | d)
* I(θ; Y_d | D) = Σ_g w_g KL(p_g ‖ Σ_h w_h p_h), exact (no sampling)

The state advances through ``advance`` after *every* executed action, including
those whose outcome is missing.

Zero-probability outcomes are first-class: a point with p_g(y) = 0 gets log
weight −inf (never NaN). If an observation has probability 0 under *every*
grid point still in the posterior, the hypothesis is **refuted**: its evidence
becomes −inf (its posterior model weight is exactly 0), the observation still
enters the per-point log-likelihoods used by :meth:`max_log_likelihood`, and it
is excluded from the grid posterior, which stays the last well-defined one so the
predictive remains a valid distribution for diagnostics.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import special

from probe.engine.ir import GridHypothesis, OutcomeSpec, Record
from probe.engine.mechanisms import Mechanism, get_mechanism
from probe.engine.predictive import DiscretePredictive
from probe.engine.queries import GridQuery

__all__ = ["GridModel", "logsumexp"]

_PROB_TOL = 1e-9


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable log Σ exp(a) that maps an all −inf slice to −inf without warnings."""
    a = np.asarray(a, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        s = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)) + m_safe
    s = np.where(np.isneginf(m), -np.inf, s)
    return np.squeeze(s, axis=axis) if axis is not None else s.reshape(())


class GridModel:
    def __init__(self, h: GridHypothesis, outcome: OutcomeSpec) -> None:
        if outcome.kind != "count" or outcome.max_count is None:
            raise ValueError("finite_grid hypotheses need a count outcome with max_count")
        self.hypothesis_id = h.id
        self.is_residual = h.is_residual
        self._h = h
        self._support = int(outcome.max_count) + 1
        self._mech: Mechanism = get_mechanism(h.mechanism, self._support)
        if int(self._mech.support_size) != self._support:
            raise ValueError(
                f"mechanism {h.mechanism!r} has support {self._mech.support_size}, "
                f"outcome needs {self._support}"
            )
        table = h.grid_table()
        _freeze(table)
        self._params: Mapping[str, np.ndarray] = table
        self._g = h.grid_size()
        self._state: Any = self._mech.initial_state(self._params)
        self._log_w = np.full(self._g, -math.log(self._g))
        self._loglik = np.zeros(self._g)
        self._log_evidence = 0.0
        self._n_obs = 0
        self._n_actions = 0
        self._refuted = False

    # -- basic properties ----------------------------------------------------
    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def n_actions(self) -> int:
        """Executed actions so far (including those with a missing outcome)."""
        return self._n_actions

    @property
    def grid_size(self) -> int:
        return self._g

    @property
    def support_size(self) -> int:
        return self._support

    @property
    def refuted(self) -> bool:
        return self._refuted

    def grid_table(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._params.items()}

    def grid_weights(self) -> np.ndarray:
        """Posterior over grid points (sums to 1)."""
        w = np.exp(self._log_w - logsumexp(self._log_w))
        return w / w.sum()

    # -- mechanism access ----------------------------------------------------
    def _probs(self, record: Record) -> np.ndarray:
        p = np.asarray(self._mech.predict(self._state, self._params, record), dtype=np.float64)
        if p.shape != (self._g, self._support):
            raise ValueError(
                f"mechanism {self._h.mechanism!r} returned shape {p.shape}, "
                f"expected {(self._g, self._support)}"
            )
        if not np.all(np.isfinite(p)) or np.any(p < -_PROB_TOL):
            raise ValueError(f"mechanism {self._h.mechanism!r} returned invalid probabilities")
        p = np.clip(p, 0.0, None)
        sums = p.sum(axis=1)
        if np.any(np.abs(sums - 1.0) > _PROB_TOL):
            raise ValueError(f"mechanism {self._h.mechanism!r} rows do not sum to 1")
        return p / sums[:, None]

    def _outcome_index(self, y: float | int) -> int:
        yf = float(y)
        if not math.isfinite(yf) or yf != math.floor(yf):
            raise ValueError(f"count outcome must be an integer, got {y!r}")
        k = int(yf)
        if not 0 <= k < self._support:
            raise ValueError(f"count outcome {k} outside support 0..{self._support - 1}")
        return k

    # -- predictive ----------------------------------------------------------
    def _mixture(self, probs: np.ndarray, w: np.ndarray) -> np.ndarray:
        row = w @ probs
        return row / row.sum()

    def predictive(self, records: Sequence[Record]) -> DiscretePredictive:
        w = self.grid_weights()
        rows = [self._mixture(self._probs(r), w) for r in records]
        if not rows:
            return DiscretePredictive(np.zeros((0, self._support)))
        return DiscretePredictive(np.vstack(rows))

    def log_predictive(self, record: Record, y: float | int) -> float:
        k = self._outcome_index(y)
        with np.errstate(divide="ignore"):
            lp = np.log(self._probs(record)[:, k])
        return float(logsumexp(self._log_w + lp))

    # -- update --------------------------------------------------------------
    def update(self, record: Record, y: float | int | None) -> None:
        """Condition on y (if not missing), then advance the mechanism state for the
        executed action. Atomic: everything is computed first, so an invalid outcome
        or a failing mechanism leaves the model exactly as it was."""
        lp: np.ndarray | None = None
        lpred = 0.0
        if y is not None:
            k = self._outcome_index(y)
            with np.errstate(divide="ignore"):
                lp = np.log(self._probs(record)[:, k])
            lpred = float(logsumexp(self._log_w + lp))
        new_state = self._mech.advance(self._state, self._params, record)
        if lp is not None:
            self._log_evidence += lpred
            self._loglik = self._loglik + lp
            self._n_obs += 1
            if lpred == -math.inf:
                self._refuted = True  # posterior kept; evidence is −inf from now on
            else:
                log_w = self._log_w + lp
                self._log_w = log_w - logsumexp(log_w)
        self._state = new_state
        self._n_actions += 1

    # -- evidence ------------------------------------------------------------
    def log_evidence(self) -> float:
        return self._log_evidence

    def max_log_likelihood(self) -> float:
        """max_g Σ_t log p_g(y_t | d_t): the grid MLE (−inf if every point is refuted)."""
        if self._n_obs == 0:
            return 0.0
        return float(np.max(self._loglik))

    def log_likelihoods(self) -> np.ndarray:
        """Per-grid-point log-likelihood of all observed data, shape (G,)."""
        return self._loglik.copy()

    # -- information gain ----------------------------------------------------
    def parameter_eig(self, records: Sequence[Record]) -> np.ndarray:
        """I(θ; Y_d | m, D) per record: exact mutual information between the grid
        index and the outcome, Σ_g w_g KL(p_g ‖ mixture)."""
        w = self.grid_weights()
        active = w > 0.0
        wa = w[active]
        h_w = float(0.0 - np.sum(special.xlogy(wa, wa)))  # +0.0 for a point mass
        out = np.empty(len(records))
        for i, r in enumerate(records):
            p = self._probs(r)[active]
            mix = wa @ p
            out[i] = _mutual_information(wa, p, mix)
        if np.any(out < -1e-10) or np.any(out > h_w + 1e-10):
            raise ArithmeticError("grid parameter EIG outside [0, H(w)] beyond rounding")
        return np.clip(out, 0.0, h_w)

    # -- goal-oriented queries -----------------------------------------------
    def query_mask(self, query: GridQuery) -> np.ndarray:
        return query.mask(self._params, self._g)

    def query_prob(self, query: GridQuery) -> float:
        """Posterior probability of the query under this hypothesis."""
        w = self.grid_weights()
        return float(np.clip(np.sum(w[self.query_mask(query)]), 0.0, 1.0))

    def query_joint(self, records: Sequence[Record], query: GridQuery) -> np.ndarray:
        """P_m(1[q] = a, Y_d = y) for a ∈ {1, 0}: shape (n, 2, S)."""
        w = self.grid_weights()
        q = self.query_mask(query).astype(np.float64)
        out = np.empty((len(records), 2, self._support))
        for i, r in enumerate(records):
            p = self._probs(r)
            out[i, 0] = (w * q) @ p
            out[i, 1] = (w * (1.0 - q)) @ p
        return out

    # -- oracle helpers (brute-force nested Monte Carlo; tests and diagnostics) ---
    def sample_parameters(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Grid indices drawn from the posterior (inverse-CDF on fixed order)."""
        cdf = np.cumsum(self.grid_weights())
        idx = np.searchsorted(cdf, rng.random(size) * cdf[-1], side="right")
        return np.minimum(idx, self._g - 1)

    def simulate(self, params: np.ndarray, record: Record, rng: np.random.Generator) -> np.ndarray:
        """One outcome per sampled grid index."""
        p = self._probs(record)[np.asarray(params)]
        cdf = np.cumsum(p, axis=1)
        u = rng.random(p.shape[0])[:, None] * cdf[:, -1:]
        return np.minimum(np.sum(cdf <= u, axis=1), self._support - 1)

    def conditional_logpdf(self, params: np.ndarray, record: Record, y: np.ndarray) -> np.ndarray:
        """log p_g(y_i | d) for sampled grid indices g, shape (len(y), S_samples)."""
        p = self._probs(record)
        yi = np.asarray(y, dtype=np.int64)
        with np.errstate(divide="ignore"):
            return np.log(p[np.asarray(params)][:, yi].T)

    # -- misc ----------------------------------------------------------------
    def clone(self) -> GridModel:
        c = copy.deepcopy(self)
        _freeze(c._params)  # deepcopy returns writeable arrays
        return c

    def summary(self) -> dict[str, Any]:
        """JSON-able summary. Non-finite numbers are reported as ``None`` with the
        ``refuted`` flag, because canonical JSON rejects ±inf."""
        w = self.grid_weights()
        g_map = int(np.argmax(w))
        # The grid is the C-order Cartesian product over sorted names, so a marginal is
        # a sum over the other axes of the reshaped weights. (Matching values instead,
        # ``col == v``, fails for mixed-type parameters such as (1, "lru"): numpy stores
        # the column as strings and the numeric values never matched.)
        names = sorted(self._h.params)
        shaped = w.reshape([len(self._h.params[n]) for n in names]) if names else w
        marginals: dict[str, dict[str, float]] = {}
        for axis, name in enumerate(names):
            others = tuple(i for i in range(len(names)) if i != axis)
            sums = np.sum(shaped, axis=others) if others else shaped
            marginals[name] = {
                str(v): float(s) for v, s in zip(self._h.params[name], sums, strict=True)
            }
        mll = self.max_log_likelihood()
        return {
            "hypothesis_id": self.hypothesis_id,
            "kind": "finite_grid",
            "mechanism": self._h.mechanism,
            "n": self._n_obs,
            "n_actions": self._n_actions,
            "grid_size": self._g,
            "refuted": self._refuted,
            "log_evidence": self._log_evidence if math.isfinite(self._log_evidence) else None,
            "max_log_likelihood": mll if math.isfinite(mll) else None,
            "map": {k: _plain(v[g_map]) for k, v in sorted(self._params.items())},
            "map_weight": float(w[g_map]),
            "posterior_entropy": float(0.0 - np.sum(special.xlogy(w, w))),
            "marginals": marginals,
        }


def _freeze(table: Mapping[str, np.ndarray]) -> None:
    """Make the grid parameter columns read-only, so a mechanism that writes to
    ``params`` fails loudly instead of corrupting the grid."""
    for col in table.values():
        col.setflags(write=False)


def _plain(v: Any) -> float | int | str:
    if isinstance(v, np.generic):
        out = v.item()
        return out if isinstance(out, float | int | str) else str(out)
    return v if isinstance(v, float | int | str) else str(v)


def _mutual_information(w: np.ndarray, p: np.ndarray, mix: np.ndarray) -> float:
    """Σ_g w_g Σ_y p_gy (log p_gy − log mix_y), with 0·log 0 = 0 (mix_y > 0 wherever
    some p_gy > 0 with w_g > 0)."""
    pos = p > 0.0
    with np.errstate(divide="ignore"):
        logp = np.where(pos, np.log(np.where(pos, p, 1.0)), 0.0)
        logm = np.log(np.where(mix > 0.0, mix, 1.0))
    terms = np.where(pos, p * (logp - logm[None, :]), 0.0)
    return float(w @ terms.sum(axis=1))
