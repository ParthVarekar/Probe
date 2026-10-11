import pytest

from probe.envs.microclimate import FAMILY
from probe.ingest import analyze_import, neutralize_cell, parse_csv

HS = FAMILY.hypothesis_set("R1")
MAP = {c.name: c.name for c in HS.covariates} | {HS.outcome.name: "y"}


def _csv_from_run():
    names = [c.name for c in HS.covariates]
    lines = [",".join([*names, "y"])]
    # records are agent-knowable covariates, taken straight from a simulated world
    w = FAMILY.make_world("R2", 0)
    for c in w.env.candidates()[:40]:
        if not c.feasible:
            continue
        o = w.env.execute(c.action)
        r = o.record
        y = "" if o.value is None else repr(o.value)
        lines.append(
            ",".join(
                [*(str(r[n]).lower() if isinstance(r[n], bool) else str(r[n]) for n in names), y]
            )
        )
        if len(lines) > 25:
            break
    return ("\n".join(lines) + "\n").encode()


def test_import_roundtrip_and_l1_ceiling():
    data = _csv_from_run()
    res = parse_csv(data, HS, MAP)
    assert len(res.records) >= 10 and not res.rejected
    out = analyze_import(res, HS)
    assert out["claim_ceiling"] == "L1" and abs(sum(out["posterior"].values()) - 1) < 1e-9


def test_bad_rows_rejected_with_reasons_and_missing_kept():
    names = [c.name for c in HS.covariates]
    good = parse_csv(_csv_from_run(), HS, MAP)
    r0 = good.records[0]
    row = [str(r0[n]).lower() if isinstance(r0[n], bool) else str(r0[n]) for n in names]
    bad = list(row)
    bad[names.index("shade")] = "=HYPERLINK(evil)"
    text = (
        ",".join([*names, "y"]) + "\n" + ",".join([*row, ""]) + "\n" + ",".join([*bad, "1"]) + "\n"
    )
    res = parse_csv(text.encode(), HS, MAP)
    assert res.missing_outcome == 1 and len(res.records) == 1
    assert len(res.rejected) == 1 and res.formula_cells == 1


def test_neutralize():
    assert neutralize_cell("=1+1") == "'=1+1"
    assert neutralize_cell("-3.5") == "-3.5"
    assert neutralize_cell("abc") == "abc"


def test_mapping_errors():
    with pytest.raises(ValueError):
        parse_csv(b"a,b\n1,2\n", HS, MAP)
