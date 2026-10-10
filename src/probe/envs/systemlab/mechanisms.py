"""Grid mechanisms of the system lab, registered with :mod:`probe.engine.mechanisms`.

Each mechanism is the hypothesis-side model of one service type, vectorized over
the G points of a :class:`~probe.engine.ir.GridHypothesis` parameter grid. It
follows the registry contract exactly:

* ``initial_state(params)``: dict of arrays with leading dimension G (the service
  as first seen: limiters full, caches empty, at tick 0).
* ``predict(state, params, record) -> probs`` of shape (G, S), S = max_count + 1:
  the exact PMF of the *reported* count for the action in ``record``, i.e. the
  noise-free service count pushed through the reporting channel of
  :mod:`probe.envs.systemlab.noise` (parameters ``eps`` and ``glitch``, fixed or
  gridded; 0 when absent). Never mutates ``state``.
* ``advance(state, params, record) -> new_state``: the service state after the
  action. Services are deterministic and the noise is on the report only, so the
  new state depends on the action and the parameters, never on the outcome.

Design records (agent-knowable, no truth):

* rate limiters: ``{"wait": w, "burst": n, "clock": c}`` (integer ticks). The
  burst of ``n`` simultaneous requests is sent at tick ``c + w``; the outcome is
  the reported number admitted, in ``0..n``.
* caches: ``{"keys": "abcab...", "clock": c}``; each lowercase letter is one
  access, the outcome is the reported number of hits, in ``0..len(keys)``.

Mechanisms (names as registered):

=============================  ==========================================  ==========
name                           parameters                                   kind
=============================  ==========================================  ==========
``systemlab.token_bucket``     capacity, period (ticks per token)           service
``systemlab.leaky_meter``      capacity, period (GCRA; ≡ token bucket)      service
``systemlab.fixed_window``     limit, window, phase (quarters, 0..3)        service
``systemlab.sliding_window``   limit, window                                service
``systemlab.allowance``        alpha, beta, gamma, amax, memory             residual
``systemlab.cache``            policy (lru/fifo/lfu), size                  service
``systemlab.reuse``            q_near, q_mid, q_far                         residual
=============================  ==========================================  ==========

All mechanisms also read the reporting-noise parameters ``eps`` and ``glitch``.

The two residual mechanisms are deliberately structure-light (no algorithm, no
memory of *admissions*, only of the agent's own actions) so that they win on
evidence only when every structured hypothesis misfits (handover §12 H.3
residual hypothesis):

* **allowance**: each burst is admitted a random prefix ``K = min(n, A)``,
  ``A ~ Bin(amax, q)``, independently across bursts given the action history, with
  ``q = expit(alpha + beta·log1p(w) − gamma·log1p(L))``: rising with the wait
  ``w`` and falling with the load ``L``, the number of requests the agent sent in
  the last ``memory`` ticks (a deterministic function of past actions).
* **reuse**: each access is a hit independently with a probability set by its
  LRU stack distance d (distinct other keys since its last access): ``q_near``
  for d ≤ 1, ``q_mid`` for 2 ≤ d ≤ 3, ``q_far`` for d ≥ 4; first-ever accesses
  always miss. The stack distances are a deterministic function of the access
  sequence, so the state still evolves from actions alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import special

from probe.engine.ir import Record
from probe.engine.mechanisms import Mechanism, register_mechanism
from probe.envs.systemlab.noise import report_probs, report_probs_point

__all__ = [
    "MECHANISM_NAMES",
    "AllowanceMechanism",
    "CacheMechanism",
    "FixedWindowMechanism",
    "LeakyMeterMechanism",
    "ReuseMechanism",
    "SlidingWindowMechanism",
    "TokenBucketMechanism",
]

Params = Mapping[str, np.ndarray]
State = dict[str, np.ndarray]

_NEG = -(2**40)  # "never" timestamp (an empty sliding-window slot)
_BIG = 2**40  # padding for slots beyond a grid point's limit
_POLICY_CODE = {"lru": 0, "fifo": 1, "lfu": 2}
_N_KEYS = 26  # cache keys are the lowercase letters a..z
_HIST = 64  # bursts remembered by the allowance residual (>= its memory in ticks)


# --------------------------------------------------------------------------- helpers


def _grid_len(params: Params) -> int:
    lens = {np.asarray(v).shape[0] for v in params.values()}
    if len(lens) != 1:
        raise ValueError(f"parameter columns must share one length, got {sorted(lens)}")
    return lens.pop()


def _int_col(params: Params, name: str, *, minimum: int = 1) -> np.ndarray:
    try:
        raw = np.asarray(params[name])
    except KeyError as e:
        raise KeyError(f"mechanism needs parameter {name!r}") from e
    vals = raw.astype(np.float64)
    if np.any(vals != np.round(vals)) or np.any(vals < minimum):
        raise ValueError(f"parameter {name!r} must be integers >= {minimum}")
    return vals.astype(np.int64)


def _float_col(params: Params, name: str, g: int, default: float | None = None) -> np.ndarray:
    if name not in params:
        if default is None:
            raise KeyError(f"mechanism needs parameter {name!r}")
        return np.full(g, float(default))
    return np.asarray(params[name], dtype=np.float64)


def _noise(params: Params, g: int) -> tuple[np.ndarray, np.ndarray]:
    eps = _float_col(params, "eps", g, 0.0)
    glitch = _float_col(params, "glitch", g, 0.0)
    if np.any(eps < 0) or np.any(eps >= 0.5) or np.any(glitch < 0) or np.any(glitch >= 1):
        raise ValueError("noise parameters need 0 <= eps < 0.5 and 0 <= glitch < 1")
    return eps, glitch


def _rec_int(record: Record, name: str) -> int:
    try:
        v = record[name]
    except KeyError as e:
        raise KeyError(f"system-lab record lacks {name!r}") from e
    if isinstance(v, str | bool) or float(v) != int(v) or int(v) < 0:
        raise ValueError(f"record field {name!r} must be a non-negative integer, got {v!r}")
    return int(v)


def _burst_of(record: Record, support_size: int) -> tuple[int, int, int]:
    """(burst tick, burst size, wait) from a rate-limiter record."""
    wait = _rec_int(record, "wait")
    n = _rec_int(record, "burst")
    t = _rec_int(record, "clock") + wait
    if n > support_size - 1:
        raise ValueError(f"burst of {n} exceeds the outcome support 0..{support_size - 1}")
    return t, n, wait


def _keys_of(record: Record, support_size: int) -> np.ndarray:
    try:
        seq = record["keys"]
    except KeyError as e:
        raise KeyError("cache record lacks 'keys'") from e
    if not isinstance(seq, str) or not seq or not all("a" <= c <= "z" for c in seq):
        raise ValueError(f"cache keys must be a non-empty string of a..z, got {seq!r}")
    if len(seq) > support_size - 1:
        raise ValueError(f"{len(seq)} accesses exceed the outcome support 0..{support_size - 1}")
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8).astype(np.int64) - ord("a")


def _check_time(state: State, t: int) -> None:
    if state["t"].size and t < int(state["t"].max()):
        raise ValueError(f"action at tick {t} precedes the mechanism state ({state['t'].max()})")


class _Base:
    name: str = ""

    def __init__(self, support_size: int) -> None:
        if support_size < 2:
            raise ValueError("support_size must be >= 2")
        self.support_size = int(support_size)


@lru_cache(maxsize=8)
def _log_choose(top: int) -> np.ndarray:
    """log C(a, j) for a, j in 0..top (−inf where j > a); read-only."""
    a = np.arange(top + 1, dtype=np.float64)[:, None]
    j = np.arange(top + 1, dtype=np.float64)[None, :]
    with np.errstate(invalid="ignore"):
        out = special.gammaln(a + 1) - special.gammaln(j + 1) - special.gammaln(a - j + 1)
    out = np.where(j <= a, out, -np.inf)
    out.setflags(write=False)
    return out


def _binom_pmf_table(trials: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Bin(trials[g], q[g]) PMF over 0..max(trials), zero beyond trials[g]; shape (G, A+1).
    Direct log-space evaluation with a cached log-binomial table (much cheaper than
    ``scipy.stats.binom`` for small trial counts)."""
    top = int(trials.max()) if trials.size else 0
    j = np.arange(top + 1, dtype=np.float64)[None, :]
    t = trials.astype(np.float64)[:, None]
    logc = _log_choose(top)[trials]
    with np.errstate(divide="ignore", invalid="ignore"):
        logq = np.log(q)[:, None]
        log1mq = np.log1p(-q)[:, None]
        logp = logc + np.where(j > 0, j * logq, 0.0) + np.where(t - j > 0, (t - j) * log1mq, 0.0)
    return np.where(j <= t, np.exp(logp), 0.0)


# --------------------------------------------------------------------------- rate limiters


class _RateMechanism(_Base):
    """Shared predict/advance for deterministic limiters: subclasses implement
    :meth:`_burst`, which returns the admitted counts and the new state without
    touching the old one."""

    def initial_state(self, params: Params) -> State:
        raise NotImplementedError

    def _burst(self, state: State, params: Params, t: int, n: int) -> tuple[np.ndarray, State]:
        raise NotImplementedError

    def noiseless_counts(self, state: State, params: Params, record: Record) -> np.ndarray:
        """Noise-free admitted counts per grid point (shape (G,))."""
        t, n, _ = _burst_of(record, self.support_size)
        _check_time(state, t)
        return self._burst(state, params, t, n)[0]

    def predict(self, state: State, params: Params, record: Record) -> np.ndarray:
        t, n, _ = _burst_of(record, self.support_size)
        _check_time(state, t)
        k, _ = self._burst(state, params, t, n)
        eps, glitch = _noise(params, k.shape[0])
        return report_probs_point(k, n, eps, glitch, self.support_size)

    def advance(self, state: State, params: Params, record: Record) -> State:
        t, n, _ = _burst_of(record, self.support_size)
        _check_time(state, t)
        return self._burst(state, params, t, n)[1]


class TokenBucketMechanism(_RateMechanism):
    name = "systemlab.token_bucket"

    def initial_state(self, params: Params) -> State:
        cap = _int_col(params, "capacity")
        per = _int_col(params, "period")
        return {"level": cap * per, "t": np.zeros(cap.shape[0], dtype=np.int64)}

    def _burst(self, state: State, params: Params, t: int, n: int) -> tuple[np.ndarray, State]:
        cap = _int_col(params, "capacity")
        per = _int_col(params, "period")
        level = np.minimum(cap * per, state["level"] + (t - state["t"]))
        k = np.minimum(n, level // per)
        return k, {"level": level - k * per, "t": np.full(k.shape[0], t, dtype=np.int64)}


class LeakyMeterMechanism(_RateMechanism):
    name = "systemlab.leaky_meter"

    def initial_state(self, params: Params) -> State:
        g = _int_col(params, "capacity").shape[0]
        return {"tat": np.zeros(g, dtype=np.int64), "t": np.zeros(g, dtype=np.int64)}

    def _burst(self, state: State, params: Params, t: int, n: int) -> tuple[np.ndarray, State]:
        cap = _int_col(params, "capacity")
        per = _int_col(params, "period")
        tau = (cap - 1) * per
        tat = state["tat"]
        k = np.zeros(cap.shape[0], dtype=np.int64)
        for _ in range(n):
            ok = tat - t <= tau
            tat = np.where(ok, np.maximum(tat, t) + per, tat)
            k = k + ok
        return k, {"tat": tat, "t": np.full(k.shape[0], t, dtype=np.int64)}


class FixedWindowMechanism(_RateMechanism):
    name = "systemlab.fixed_window"

    @staticmethod
    def _offset(params: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        limit = _int_col(params, "limit")
        window = _int_col(params, "window")
        phase = (
            _int_col(params, "phase", minimum=0)
            if "phase" in params
            else np.zeros_like(limit, dtype=np.int64)
        )
        if np.any(phase > 3):
            raise ValueError("phase is in quarters of the window: 0..3")
        return limit, window, phase * window // 4

    def initial_state(self, params: Params) -> State:
        limit, window, offset = self._offset(params)
        g = limit.shape[0]
        return {
            "win": offset // window,
            "count": np.zeros(g, dtype=np.int64),
            "t": np.zeros(g, dtype=np.int64),
        }

    def _burst(self, state: State, params: Params, t: int, n: int) -> tuple[np.ndarray, State]:
        limit, window, offset = self._offset(params)
        win = (t + offset) // window
        count = np.where(win != state["win"], 0, state["count"])
        k = np.minimum(n, np.maximum(limit - count, 0))
        new = {"win": win, "count": count + k, "t": np.full(k.shape[0], t, dtype=np.int64)}
        return k, new


class SlidingWindowMechanism(_RateMechanism):
    """Sliding log. ``buf`` row g holds the last ``limit[g]`` admit times sorted
    ascending in its first ``limit[g]`` slots (``_NEG`` = never), padded with
    ``_BIG``. Fewer than ``limit`` admits in ``(t − W, t]`` ⇒ admit."""

    name = "systemlab.sliding_window"

    def initial_state(self, params: Params) -> State:
        limit = _int_col(params, "limit")
        _int_col(params, "window")
        lmax = int(limit.max())
        valid = np.arange(lmax)[None, :] < limit[:, None]
        return {
            "buf": np.where(valid, _NEG, _BIG).astype(np.int64),
            "t": np.zeros(limit.shape[0], dtype=np.int64),
        }

    def _burst(self, state: State, params: Params, t: int, n: int) -> tuple[np.ndarray, State]:
        limit = _int_col(params, "limit")
        window = _int_col(params, "window")
        buf = state["buf"]
        if buf.shape != (limit.shape[0], int(limit.max())):
            raise ValueError("sliding-window state does not match the parameter grid")
        slots = np.arange(buf.shape[1])[None, :]
        valid = slots < limit[:, None]
        in_window = valid & (buf > (t - window)[:, None])
        k = np.minimum(n, limit - in_window.sum(axis=1))
        # The k oldest valid entries are out of the window (in-window ones are the
        # newest), so overwrite them with t and re-sort; _BIG padding stays last.
        new_buf = np.sort(np.where(slots < k[:, None], t, buf), axis=1)
        return k, {"buf": new_buf, "t": np.full(k.shape[0], t, dtype=np.int64)}


class AllowanceMechanism(_Base):
    """Residual for rate limiters (see module docstring). State: the times and sizes of
    the last ``_HIST`` bursts (identical across grid points) and the last burst tick."""

    name = "systemlab.allowance"

    def initial_state(self, params: Params) -> State:
        g = _grid_len(params)
        return {
            "t": np.zeros(g, dtype=np.int64),
            "hist_t": np.full((g, _HIST), _NEG, dtype=np.int64),
            "hist_n": np.zeros((g, _HIST), dtype=np.int64),
        }

    def true_count_pmf(self, state: State, params: Params, record: Record) -> np.ndarray:
        """P(K = k), k ∈ 0..n, per grid point: shape (G, n + 1)."""
        t, n, wait = _burst_of(record, self.support_size)
        g = _grid_len(params)
        alpha = _float_col(params, "alpha", g)
        beta = _float_col(params, "beta", g)
        gamma = _float_col(params, "gamma", g, 0.0)
        memory = _int_col(params, "memory") if "memory" in params else np.full(g, 16)
        if np.any(memory > _HIST):
            raise ValueError(f"memory must be <= {_HIST} ticks")
        amax = (
            _int_col(params, "amax", minimum=0)
            if "amax" in params
            else np.full(g, self.support_size - 1, dtype=np.int64)
        )
        recent = state["hist_t"] > (t - memory)[:, None]
        load = np.sum(np.where(recent, state["hist_n"], 0), axis=1)
        q = special.expit(alpha + beta * np.log1p(float(wait)) - gamma * np.log1p(load))
        pa = _binom_pmf_table(amax, q)  # (G, max(amax) + 1): P(A = a)
        width = max(n + 1, pa.shape[1])
        full = np.zeros((g, width), dtype=np.float64)
        full[:, : pa.shape[1]] = pa
        pk = np.empty((g, n + 1), dtype=np.float64)
        pk[:, :n] = full[:, :n]
        pk[:, n] = full[:, n:].sum(axis=1)  # K = min(n, A): the tail A >= n maps to n
        return pk / pk.sum(axis=1, keepdims=True)

    def predict(self, state: State, params: Params, record: Record) -> np.ndarray:
        t, n, _ = _burst_of(record, self.support_size)
        _check_time(state, t)
        pk = self.true_count_pmf(state, params, record)
        eps, glitch = _noise(params, pk.shape[0])
        return report_probs(pk, n, eps, glitch, self.support_size)

    def advance(self, state: State, params: Params, record: Record) -> State:
        t, n, _ = _burst_of(record, self.support_size)
        _check_time(state, t)
        g = _grid_len(params)
        hist_t = np.concatenate([state["hist_t"][:, 1:], np.full((g, 1), t)], axis=1)
        hist_n = np.concatenate([state["hist_n"][:, 1:], np.full((g, 1), n)], axis=1)
        return {"t": np.full(g, t, dtype=np.int64), "hist_t": hist_t, "hist_n": hist_n}


# --------------------------------------------------------------------------- caches


class CacheMechanism(_Base):
    """LRU / FIFO / LFU (ties: least recent) caches, vectorized over grid points.

    State per grid point: resident key codes (``-1`` = empty slot), last-access
    and insertion stamps, in-cache access counts, and the access counter ``t``.
    """

    name = "systemlab.cache"

    @staticmethod
    def _params(params: Params) -> tuple[np.ndarray, np.ndarray]:
        size = _int_col(params, "size")
        try:
            pol = np.asarray(params["policy"]).astype(str)
        except KeyError as e:
            raise KeyError("mechanism needs parameter 'policy'") from e
        unknown = sorted(set(pol.tolist()) - set(_POLICY_CODE))
        if unknown:
            raise ValueError(f"unknown cache policies {unknown}")
        code = np.asarray([_POLICY_CODE[p] for p in pol.tolist()], dtype=np.int64)
        return size, code

    def initial_state(self, params: Params) -> State:
        size, _ = self._params(params)
        shape = (size.shape[0], int(size.max()))
        return {
            "keys": np.full(shape, -1, dtype=np.int64),
            "last": np.zeros(shape, dtype=np.int64),
            "ins": np.zeros(shape, dtype=np.int64),
            "freq": np.zeros(shape, dtype=np.int64),
            "t": np.zeros(size.shape[0], dtype=np.int64),
        }

    def _run(self, state: State, params: Params, seq: np.ndarray) -> tuple[np.ndarray, State]:
        size, code = self._params(params)
        keys = state["keys"].copy()
        last = state["last"].copy()
        ins = state["ins"].copy()
        freq = state["freq"].copy()
        tick = state["t"].copy()
        g, kmax = keys.shape
        if kmax != int(size.max()) or g != size.shape[0]:
            raise ValueError("cache state does not match the parameter grid")
        slots = np.arange(kmax)[None, :]
        valid = slots < size[:, None]
        hits = np.zeros(g, dtype=np.int64)
        rows_all = np.arange(g)
        for x in seq.tolist():
            stamp = tick[:, None]
            match = keys == x
            hit = match.any(axis=1)
            hits += hit
            last = np.where(match, stamp, last)
            freq = freq + match
            miss = ~hit
            if miss.any():
                empty = (keys == -1) & valid
                has_empty = empty.any(axis=1)
                score = np.where(
                    code[:, None] == 0,
                    last,
                    np.where(code[:, None] == 1, ins, freq * (2**32) + last),
                )
                score = np.where(valid, score, np.iinfo(np.int64).max)
                slot = np.where(has_empty, np.argmax(empty, axis=1), np.argmin(score, axis=1))
                r = rows_all[miss]
                s = slot[miss]
                keys[r, s] = x
                last[r, s] = tick[miss]
                ins[r, s] = tick[miss]
                freq[r, s] = 1
            tick = tick + 1
        return hits, {"keys": keys, "last": last, "ins": ins, "freq": freq, "t": tick}

    def noiseless_counts(self, state: State, params: Params, record: Record) -> np.ndarray:
        return self._run(state, params, _keys_of(record, self.support_size))[0]

    def predict(self, state: State, params: Params, record: Record) -> np.ndarray:
        seq = _keys_of(record, self.support_size)
        k, _ = self._run(state, params, seq)
        eps, glitch = _noise(params, k.shape[0])
        return report_probs_point(k, int(seq.shape[0]), eps, glitch, self.support_size)

    def advance(self, state: State, params: Params, record: Record) -> State:
        return self._run(state, params, _keys_of(record, self.support_size))[1]


class ReuseMechanism(_Base):
    """Residual for caches (see module docstring). State: last-access stamp per key
    letter (``-1`` = never), identical across grid points, plus the access counter."""

    name = "systemlab.reuse"

    def initial_state(self, params: Params) -> State:
        g = _grid_len(params)
        return {
            "last": np.full((g, _N_KEYS), -1, dtype=np.int64),
            "t": np.zeros(g, dtype=np.int64),
        }

    @staticmethod
    def stack_distances(last: np.ndarray, t0: int, seq: np.ndarray) -> tuple[list[int], np.ndarray]:
        """LRU stack distance of each access (−1 for a first-ever access) from one
        state row, and the updated row."""
        row = last.copy()
        out: list[int] = []
        t = t0
        for x in seq.tolist():
            if row[x] < 0:
                out.append(-1)
            else:
                out.append(int(np.count_nonzero(row > row[x])))
            row[x] = t
            t += 1
        return out, row

    def true_count_pmf(self, state: State, params: Params, record: Record) -> np.ndarray:
        seq = _keys_of(record, self.support_size)
        g = _grid_len(params)
        q = {b: _float_col(params, f"q_{b}", g) for b in ("near", "mid", "far")}
        dists, _ = self.stack_distances(state["last"][0], int(state["t"][0]), seq)
        pmf = np.zeros((g, len(dists) + 1), dtype=np.float64)
        pmf[:, 0] = 1.0
        for j, d in enumerate(dists):
            if d < 0:
                continue  # compulsory miss
            p = q["near"] if d <= 1 else q["mid"] if d <= 3 else q["far"]
            shifted = pmf[:, : j + 1] * p[:, None]
            pmf[:, : j + 1] *= (1.0 - p)[:, None]
            pmf[:, 1 : j + 2] += shifted
        return pmf

    def predict(self, state: State, params: Params, record: Record) -> np.ndarray:
        pk = self.true_count_pmf(state, params, record)
        n = pk.shape[1] - 1
        eps, glitch = _noise(params, pk.shape[0])
        return report_probs(pk, n, eps, glitch, self.support_size)

    def advance(self, state: State, params: Params, record: Record) -> State:
        seq = _keys_of(record, self.support_size)
        g = _grid_len(params)
        _, row = self.stack_distances(state["last"][0], int(state["t"][0]), seq)
        return {
            "last": np.broadcast_to(row, (g, _N_KEYS)).copy(),
            "t": state["t"] + int(seq.shape[0]),
        }


# --------------------------------------------------------------------------- registry

_CLASSES: tuple[type[Any], ...] = (
    TokenBucketMechanism,
    LeakyMeterMechanism,
    FixedWindowMechanism,
    SlidingWindowMechanism,
    AllowanceMechanism,
    CacheMechanism,
    ReuseMechanism,
)
MECHANISM_NAMES: tuple[str, ...] = tuple(c.name for c in _CLASSES)


def _make_factory(cls: type[Any]) -> Any:
    def factory(support_size: int) -> Mechanism:
        mech: Mechanism = cls(support_size)
        return mech

    factory.__name__ = f"make_{cls.__name__}"
    factory.__qualname__ = factory.__name__
    return factory


for _cls in _CLASSES:
    register_mechanism(_cls.name)(_make_factory(_cls))
