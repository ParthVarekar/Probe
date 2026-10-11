"""The system lab through the real engine (probe.engine.factory/grid/eig): the registered
mechanisms satisfy GridModel's checks, and the engine's evidence, model posterior and
model-indicator EIG agree with the independent reference filter (``systemlab_ref``)
along real environment paths, including exact refutation under zero noise (R11)."""

import math
import time

import numpy as np
import pytest
from systemlab_ref import RefPosterior

from probe.engine.eig import log_prior_weights, model_indicator_eig, posterior_weights
from probe.engine.factory import build_models
from probe.engine.grid import GridModel
from probe.envs.systemlab import FAMILY


def _engine_state(models, hs, records):
    ev = [m.log_evidence() for m in models]
    w = posterior_weights(log_prior_weights(hs), ev)
    u = model_indicator_eig([m.predictive(records) for m in models], w)
    return np.asarray(ev), w, u


@pytest.mark.parametrize(
    ("regime", "seed"), [("R1", 0), ("R3", 1), ("R5", 2), ("R8", 3), ("R11", 2)]
)
def test_engine_matches_reference_filter(regime, seed):
    hs = FAMILY.hypothesis_set(regime)
    models = build_models(hs)
    assert all(isinstance(m, GridModel) for m in models)
    ref = RefPosterior(hs)
    env = FAMILY.make_world(regime, seed).env
    rng = np.random.default_rng(seed)
    prequential = np.zeros(len(models))
    for _ in range(14):
        cands = [c for c in env.candidates() if c.feasible]
        records = [c.record for c in cands]
        ev, w, u = _engine_state(models, hs, records)
        ref_ev = np.asarray([m.log_evidence for m in ref.models])
        np.testing.assert_array_equal(np.isneginf(ev), np.isneginf(ref_ev))
        fin = np.isfinite(ev)
        np.testing.assert_allclose(ev[fin], ref_ev[fin], rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(w, ref.post(), atol=1e-10)
        np.testing.assert_allclose(u.value, ref.eig(records), atol=1e-9)
        assert np.all(u.value >= 0) and np.all(u.value <= u.upper_bound + 1e-12)
        c = cands[int(rng.integers(len(cands)))]
        obs = env.execute(c.action)
        y = int(obs.value)
        for i, m in enumerate(models):
            prequential[i] += m.log_predictive(obs.record, y)
            m.update(obs.record, y)
        ref.update(obs.record, y)
    for i, m in enumerate(models):
        if math.isfinite(m.log_evidence()):
            assert prequential[i] == pytest.approx(m.log_evidence(), abs=1e-9)
        else:
            assert m.refuted and prequential[i] == -math.inf
    if regime == "R11":  # zero noise: some registered policy is refuted exactly
        assert any(m.refuted for m in models if not m.is_residual)
        assert not models[[m.hypothesis_id for m in models].index("residual")].refuted
    if regime == "R5":  # the twins never separate
        ids = [m.hypothesis_id for m in models]
        tb, lm = models[ids.index("token_bucket")], models[ids.index("leaky_meter")]
        assert tb.log_evidence() == lm.log_evidence()


def test_engine_step_time_on_the_largest_grids():
    """One engine step (predictives of all candidates under all hypotheses, then U_M)
    on R8, the largest grids (1680-point residual). Measured ~0.04 s on a 2026 laptop;
    the bound is generous."""
    hs = FAMILY.hypothesis_set("R8")
    models = build_models(hs)
    env = FAMILY.make_world("R8", 0).env
    records = [c.record for c in env.candidates() if c.feasible]
    t0 = time.perf_counter()
    _engine_state(models, hs, records)
    assert time.perf_counter() - t0 < 2.0
