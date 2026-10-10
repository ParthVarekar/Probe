"""build_model / build_models: IR to engine dispatch, in set order."""

import pytest
from engine_testkit import COUNT_OUT, GRID_COVS, grid_h, linear_h, numeric_covs

from probe.engine.factory import build_model, build_models
from probe.engine.grid import GridModel
from probe.engine.ir import HypothesisSet, OutcomeSpec
from probe.engine.models import HypothesisModel
from probe.engine.nig import NIGModel

CONT = OutcomeSpec(name="y", kind="continuous")


def test_build_models_dispatches_by_family_in_set_order():
    hs = HypothesisSet(
        outcome=CONT,
        covariates=numeric_covs(2),
        hypotheses=(linear_h("B", 2), linear_h("A", 1), linear_h("R", 0, residual=True)),
    )
    models = build_models(hs)
    assert [m.hypothesis_id for m in models] == ["B", "A", "R"]
    assert all(isinstance(m, NIGModel) and isinstance(m, HypothesisModel) for m in models)
    assert [m.is_residual for m in models] == [False, False, True]
    assert [m.n_params for m in models] == [3, 2, 1]
    assert all(m.n_obs == 0 and m.log_evidence() == 0.0 for m in models)

    gs = HypothesisSet(
        outcome=COUNT_OUT,
        covariates=GRID_COVS,
        hypotheses=(
            grid_h("G1", "test_binomial", {"p": (0.2, 0.8)}),
            grid_h("G0", "test_binomial", {}, {"p": 0.5}, residual=True),
        ),
    )
    gm = build_models(gs)
    assert all(isinstance(m, GridModel) for m in gm)
    assert [m.grid_size for m in gm] == [2, 1]
    assert gm[0].support_size == COUNT_OUT.max_count + 1


def test_build_model_requires_a_registered_hypothesis():
    hs = HypothesisSet(
        outcome=CONT,
        covariates=numeric_covs(1),
        hypotheses=(linear_h("A", 1), linear_h("R", 0, residual=True)),
    )
    assert isinstance(build_model(hs.hypotheses[0], hs), NIGModel)
    with pytest.raises(ValueError, match="not registered"):
        build_model(linear_h("X", 1), hs)
    with pytest.raises(ValueError, match="differs"):
        build_model(linear_h("A", 1, prior_sd=9.0), hs)


def test_fresh_models_are_independent():
    hs = HypothesisSet(
        outcome=CONT,
        covariates=numeric_covs(1),
        hypotheses=(linear_h("A", 1), linear_h("R", 0, residual=True)),
    )
    a1, _ = build_models(hs)
    a2, _ = build_models(hs)
    a1.update({"x0": 1.0}, 2.0)
    assert a2.n_obs == 0
