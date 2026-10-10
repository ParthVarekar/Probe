"""Measurement (reporting) noise of the system lab, shared by simulator and mechanisms.

The *service* is deterministic: given the action sequence and its parameters it
admits (or serves from cache) an exact number ``k`` of the ``n`` items in a
pattern. What the probe client *reports* is noisy, and the noise touches only the
report, never the service state. That is what lets a grid mechanism advance its
state from the action alone (:mod:`probe.engine.mechanisms` contract) while the
likelihood of the reported count stays exact.

Noise model for one action with ``n`` items of which ``k`` are truly positive
(accepted requests or cache hits):

1. **Status flips.** Each item's reported status is flipped independently with
   probability ``eps`` (a timed-out success read as a rejection, a stale
   status code, ...). The flipped count is
   ``F = Bin(k, 1 - eps) + Bin(n - k, eps)`` (independent terms), so
   ``P(F = y | k) = Σ_j Bin(j; k, 1-eps) · Bin(y - j; n - k, eps)``.
2. **Glitches.** With probability ``glitch`` the whole report is garbage and the
   count is uniform on ``0..n`` (a heavy-tailed contamination component).

    P(Y = y | k) = (1 - glitch) · P(F = y | k) + glitch / (n + 1),   y ∈ 0..n,

and 0 for ``y > n``. With ``eps > 0`` or ``glitch > 0`` every count in ``0..n``
has positive probability under every hypothesis, so wrong hypotheses are
down-weighted, never eliminated outright. With ``eps = glitch = 0`` (R11) the
report is exact and elimination is exact.

:func:`sample_report` draws from exactly this law (the simulator), and
:func:`report_probs` / :func:`report_probs_point` evaluate it (the mechanisms);
the agreement is tested by Monte Carlo over counter-based seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats

__all__ = [
    "ReportNoise",
    "flip_channel",
    "report_probs",
    "report_probs_point",
    "sample_report",
]


@dataclass(frozen=True)
class ReportNoise:
    """Parameters of the reporting channel (see module docstring)."""

    eps: float = 0.0
    glitch: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.eps < 0.5):
            raise ValueError("eps must be in [0, 0.5)")
        if not (0.0 <= self.glitch < 1.0):
            raise ValueError("glitch must be in [0, 1)")

    def as_dict(self) -> dict[str, float]:
        return {"eps": float(self.eps), "glitch": float(self.glitch)}


@lru_cache(maxsize=512)
def _flip_channel_cached(n: int, eps: float) -> np.ndarray:
    out = np.zeros((n + 1, n + 1), dtype=np.float64)
    if eps == 0.0:
        out[np.arange(n + 1), np.arange(n + 1)] = 1.0
    else:
        for k in range(n + 1):
            kept = stats.binom.pmf(np.arange(k + 1), k, 1.0 - eps)  # positives still positive
            false_pos = stats.binom.pmf(np.arange(n - k + 1), n - k, eps)
            out[k] = np.convolve(kept, false_pos)
        out /= out.sum(axis=1, keepdims=True)  # remove ~1e-16 rounding drift
    out.setflags(write=False)
    return out


def flip_channel(n: int, eps: float) -> np.ndarray:
    """``C[k, y] = P(F = y | K = k)`` for ``k, y ∈ 0..n`` (read-only, cached)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    if not (0.0 <= eps < 0.5):
        raise ValueError("eps must be in [0, 0.5)")
    return _flip_channel_cached(int(n), float(eps))


def _as_grid(x: float | np.ndarray, g: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 0:
        return np.full(g, float(a))
    if a.shape != (g,):
        raise ValueError(f"{name} must be a scalar or shape ({g},), got {a.shape}")
    return a


def report_probs(
    true_pmf: np.ndarray,
    n: int,
    eps: float | np.ndarray,
    glitch: float | np.ndarray,
    support_size: int,
) -> np.ndarray:
    """Reported-count PMF, shape (G, support_size), from a PMF over the true count.

    ``true_pmf`` has shape (G, n + 1): row g is P(K = k) under grid point g (a
    one-hot row for deterministic mechanisms). ``eps`` and ``glitch`` are scalars
    or per-grid-point arrays of shape (G,).
    """
    pk = np.asarray(true_pmf, dtype=np.float64)
    if pk.ndim != 2 or pk.shape[1] != n + 1:
        raise ValueError(f"true_pmf must be (G, {n + 1}), got {pk.shape}")
    if n + 1 > support_size:
        raise ValueError(f"pattern of {n} items exceeds the outcome support 0..{support_size - 1}")
    g = pk.shape[0]
    e = _as_grid(eps, g, "eps")
    gl = _as_grid(glitch, g, "glitch")
    flipped = np.empty((g, n + 1), dtype=np.float64)
    for val in np.unique(e):
        rows = e == val
        flipped[rows] = pk[rows] @ flip_channel(n, float(val))
    out = np.zeros((g, support_size), dtype=np.float64)
    out[:, : n + 1] = (1.0 - gl)[:, None] * flipped + (gl / (n + 1))[:, None]
    return out


def report_probs_point(
    k: np.ndarray,
    n: int,
    eps: float | np.ndarray,
    glitch: float | np.ndarray,
    support_size: int,
) -> np.ndarray:
    """:func:`report_probs` for deterministic true counts ``k`` (shape (G,), ints in 0..n)."""
    kk = np.asarray(k, dtype=np.int64)
    if kk.ndim != 1:
        raise ValueError("k must be 1-D")
    if np.any(kk < 0) or np.any(kk > n):
        raise ValueError("true counts must lie in 0..n")
    pk = np.zeros((kk.shape[0], n + 1), dtype=np.float64)
    pk[np.arange(kk.shape[0]), kk] = 1.0
    return report_probs(pk, n, eps, glitch, support_size)


def sample_report(k: int, n: int, noise: ReportNoise, rng: np.random.Generator) -> int:
    """Draw a reported count from the channel, given the true count ``k`` of ``n``.

    The draw order is fixed (glitch uniform, ``n`` flip uniforms, garbage count),
    and all three are always drawn, so the random stream consumed per action is a
    function of ``n`` only. With a counter-based ``rng`` (keyed by world seed,
    action id and clock) this gives common random numbers across policies.
    """
    if not (0 <= k <= n):
        raise ValueError("true count must lie in 0..n")
    u_glitch = float(rng.random())
    flips = rng.random(n) < noise.eps
    garbage = int(rng.integers(0, n + 1))
    positive = np.arange(n) < k
    reported = int(np.count_nonzero(positive ^ flips))
    if u_glitch < noise.glitch:
        return garbage
    return reported
