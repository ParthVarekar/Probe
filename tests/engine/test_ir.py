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
        lambda: (
            _h("X", [IndicatorTerm(covariate="shade", level="nope")]).model_copy()
            and HypothesisSet(
                outcome=OUT,
                covariates=COVS,
                hypotheses=(_h("X", [IndicatorTerm(covariate="shade", level="nope")]), H0),
            )
        ),
        lambda: HypothesisSet(
            outcome=OUT, covariates=COVS, hypotheses=(_h("X", [NumericTerm(covariate="shade")]), H0)
        ),
    ],
)
def test_invalid_sets_rejected(bad):
    with pytest.raises(ValidationError):
        bad()


def test_design_matrix_rows_are_independent_of_the_batch():
    """Each row depends on its own record only (bitwise). The RFF block used a BLAS
    matrix product, which rounded ~94% of rows of a 300-record batch differently from
    the same record evaluated alone (~1/3 for this map); it must also agree with the
    plain formula."""
    h = _h(
        "R",
        [
            InterceptTerm(),
            FourierTerm(harmonics=2),
            RFFTerm(  # 5 input dimensions: with fewer, this BLAS kernel happened to agree
                inputs=(
                    RFFInput(covariate="hour", scale=6.0),
                    RFFInput(covariate="unshielded"),
                    RFFInput(covariate="shade"),
                ),
                n_features=50,
                lengthscale=1.3,
            ),
        ],
        residual=True,
    )
    rng = np.random.default_rng(3)
    recs = [
        {
            "shade": str(rng.choice(["open", "tree", "built"])),
            "hour": float(rng.uniform(0, 24)),
            "unshielded": bool(rng.integers(0, 2)),
        }
        for _ in range(300)
    ]
    full = design_matrix(h, recs, COVS)
    for i in range(300):
        np.testing.assert_array_equal(design_matrix(h, [recs[i]], COVS)[0], full[i])
    np.testing.assert_array_equal(design_matrix(h, recs[::-1], COVS)[::-1], full)
    # value check against z @ w (agreement to rounding)
    from probe.engine.ir import _rff_inputs, _rff_params

    t = h.terms[2]
    w, b = _rff_params(t, {c.name: c for c in COVS})
    z = _rff_inputs(t, recs, {c.name: c for c in COVS})
    np.testing.assert_allclose(full[:, -50:], np.sqrt(2 / 50) * np.cos(z @ w + b), atol=1e-14)
