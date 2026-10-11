from probe.claims import check_wording, claim_dicts, claims_for_run
from probe.determinism import to_jsonable
from probe.report import render_report
from probe.run import run_world


def _rec():
    return to_jsonable(run_world("systemlab", "R2", 1, "P8")[0])


def test_claims_are_l0_and_lexicon_clean():
    cs = claim_dicts(claims_for_run(_rec(), world_label="systemlab/R2"))
    assert cs and all(c["level"] == "L0" and not c["violations"] for c in cs)


def test_lexicon_rejects_causal_wording():
    assert check_wording("In the simulator, shade reduces temperature", "L2")
    assert check_wording("shade is associated with cooler air", "L0")  # not scoped
    assert not check_wording("In the simulator world, shade is associated with cooler air", "L0")


def test_report_has_watermark_and_hashes():
    r = _rec()
    html = render_report(r, world_label="systemlab/R2")
    assert "SIMULATED" in html and r["decision_hash"] in html and "<svg" in html
