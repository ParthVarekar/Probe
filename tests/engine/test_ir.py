import numpy as np
import pytest
from pydantic import ValidationError

from probe.engine.ir import (
    CovariateSpec,
    FourierTerm,
    HypothesisSet,
    IndicatorTerm,
    InteractionTerm,
    InterceptTerm,
    LinearGaussianHypothesis,
    NumericTerm,
    OutcomeSpec,
    RFFInput,
    RFFTerm,
    column_names,
    design_matrix,
    ir_hash,
    prior_arrays,
    term_signature,
)

COVS = (
    CovariateSpec(name="shade", kind="categorical", levels=("open", "tree", "built")),
    CovariateSpec(name="hour", kind="numeric", unit="h"),
    CovariateSpec(name="unshielded", kind="binary"),
)
OUT = OutcomeSpec(name="dT", kind="continuous", unit="degC")


def _h(hid, terms, residual=False):
    return LinearGaussianHypothesis(
        id=hid, label=hid, statement=f"{hid} statement", terms=tuple(terms), is_residual=residual
    )


H1 = _h("H1", [InterceptTerm(), IndicatorTerm(covariate="shade", level="tree")])
H3 = _h(
    "H3",
    [
        InterceptTerm(),
        IndicatorTerm(covariate="shade", level="tree"),
        InteractionTerm(
            factors=(IndicatorTerm(covariate="shade", level="tree"), FourierTerm(harmonics=1))
        ),
    ],
)
H0 = _h(
    "H0",
    [
        InterceptTerm(prior_sd=5.0),
        RFFTerm(inputs=(RFFInput(covariate="shade"), RFFInput(covariate="hour", scale=6.0))),
    ],
    residual=True,
)
RECS = [
    {"shade": "tree", "hour": 13.0, "unshielded": False},
    {"shade": "open", "hour": 9.5, "unshielded": True},
]


def test_hypothesis_set_validates_and_hashes_stably():
    hs = HypothesisSet(outcome=OUT, covariates=COVS, hypotheses=(H1, H3, H0))
    assert hs.residual.id == "H0"
    again = HypothesisSet.model_validate(hs.model_dump(mode="json"))
    assert ir_hash(again) == ir_hash(hs)


def test_design_matrix_shapes_and_values():
    x = design_matrix(H3, RECS, COVS)
    assert x.shape == (2, len(column_names(H3))) == (2, 4)
    np.testing.assert_allclose(x[0, :2], [1.0, 1.0])
    np.testing.assert_allclose(x[1, 1:], 0.0)  # open site: indicator and interaction vanish
    ang = 2 * np.pi * 13.0 / 24.0
    np.testing.assert_allclose(x[0, 2:], [np.sin(ang), np.cos(ang)])


def test_rff_is_seeded_and_deterministic():
    a = design_matrix(H0, RECS, COVS)
    b = design_matrix(H0, RECS, COVS)
    assert a.shape == (2, 21)
    np.testing.assert_array_equal(a, b)


def test_priors_and_signature_nesting():
    mu, var = prior_arrays(H0)
    assert var[0] == 25.0 and np.all(var[1:] == 1.0) and np.all(mu == 0)
    assert term_signature(H1) <= term_signature(H3)


@pytest.mark.parametrize(
    "bad",
    [
        lambda: HypothesisSet(outcome=OUT, covariates=COVS, hypotheses=(H1, H3)),  # no residual
        lambda: _h("X", [IndicatorTerm(covariate="shade", level="nope")]).model_copy()
        and HypothesisSet(
            outcome=OUT,
            covariates=COVS,
            hypotheses=(_h("X", [IndicatorTerm(covariate="shade", level="nope")]), H0),
        ),
        lambda: HypothesisSet(
            outcome=OUT, covariates=COVS, hypotheses=(_h("X", [NumericTerm(covariate="shade")]), H0)
        ),
    ],
)
def test_invalid_sets_rejected(bad):
    with pytest.raises(ValidationError):
        bad()
