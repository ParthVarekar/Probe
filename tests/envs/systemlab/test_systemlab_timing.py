"""Runtime budgets (handover §12 M2: "runtime per run measured"). Bounds are generous
(about 10x the values measured on a 2026 laptop) so CI noise does not flake them;
the measured numbers are reported in the module's docs and the milestone report."""

import time

import numpy as np
import pytest

from probe.engine.ir import GridHypothesis
from probe.engine.mechanisms import get_mechanism
from probe.envs.systemlab import FAMILY, REGIME_IDS


def _median_time(fn, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return float(np.median(out))


@pytest.mark.parametrize("regime", REGIME_IDS)
def test_make_world_is_fast(regime):
    seeds = iter(range(10_000))
    t = _median_time(lambda: FAMILY.make_world(regime, next(seeds)), 25)
    assert t < 0.02  # measured ~0.1-0.3 ms


def test_execute_is_fast():
    env = FAMILY.make_world("R1", 0).env
    cands = env.candidates()
    i = iter(range(10_000))
    t = _median_time(lambda: env.execute(cands[next(i) % len(cands)].action), 200)
    assert t < 0.002  # measured ~35 us


@pytest.mark.parametrize("regime", ["R1", "R8", "R11"])
def test_scoring_all_candidates_is_fast(regime):
    """One step of engine work on the mechanism side: predict every candidate under
    every hypothesis (largest grids: R8)."""
    hs = FAMILY.hypothesis_set(regime)
    s = hs.outcome.max_count + 1
    env = FAMILY.make_world(regime, 0).env
    models = []
    for h in hs.hypotheses:
        assert isinstance(h, GridHypothesis)
        m = get_mechanism(h.mechanism, s)
        p = h.grid_table()
        models.append((m, p, m.initial_state(p)))
    recs = [c.record for c in env.candidates() if c.feasible]

    def step():
        for m, p, st in models:
            for r in recs:
                m.predict(st, p, r)

    assert _median_time(step, 3) < 0.5  # measured ~20-45 ms
