import json
import time

from fastapi.testclient import TestClient

from probe.api import create_app


def _client(tmp_path):
    return TestClient(create_app(tmp_path / "ws"))


def test_meta_and_index(tmp_path):
    c = _client(tmp_path)
    assert c.get("/").status_code == 200
    m = c.get("/api/meta").json()
    assert "microclimate" in m["families"] and "P8" in m["policies"]


def test_foreign_host_rejected(tmp_path):
    c = _client(tmp_path)
    assert c.get("/api/meta", headers={"host": "evil.example"}).status_code == 403


def test_run_stream_verify_report(tmp_path):
    c = _client(tmp_path)
    r = c.post(
        "/api/runs",
        json={"family": "systemlab", "regime": "R2", "seed": 1, "policy": "P8", "budget": "medium"},
    )
    assert r.status_code == 202
    exp = r.json()["experiment_id"]
    for _ in range(200):
        if c.get(f"/api/runs/{exp}/evaluation").status_code == 200:
            break
        time.sleep(0.1)
    ev = c.get(f"/api/runs/{exp}/events").json()
    types = [e["type"] for e in ev]
    assert types[0] == "experiment_created" and "run_record" in types
    first_obs = types.index("observation")
    assert "decision" in types[:first_obs]
    # truth never on agent-facing routes before the evaluation endpoint
    assert "salt" not in json.dumps(ev)
    with c.stream("GET", f"/api/runs/{exp}/stream") as s:
        body = "".join(s.iter_text())
    assert "event: done" in body
    v = c.post(f"/api/runs/{exp}/verify").json()
    assert v["ok"], v["problems"]
    assert "SIMULATED" in c.get(f"/api/runs/{exp}/report").text


def test_bad_request_rejected(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/runs", json={"family": "x", "regime": "R1", "seed": 0, "policy": "P8"})
    assert r.status_code == 422


def test_validate_reports_identifiability(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/validate", json={"family": "microclimate", "regime": "R1"}).json()
    assert r["ok"] and r["identifiability"]["n_designs"] > 0 and "warnings" in r


def test_csv_import_is_l1_and_ledgered(tmp_path):
    from probe.envs.microclimate import FAMILY

    hs = FAMILY.hypothesis_set("R1")
    names = [cv.name for cv in hs.covariates]
    w = FAMILY.make_world("R1", 0)
    lines = [",".join([*names, hs.outcome.name])]
    for cand in [x for x in w.env.candidates() if x.feasible][:8]:
        o = w.env.execute(cand.action)
        vals = [
            str(o.record[n]).lower() if isinstance(o.record[n], bool) else str(o.record[n])
            for n in names
        ]
        lines.append(",".join([*vals, "" if o.value is None else repr(o.value)]))
    c = _client(tmp_path)
    r = c.post(
        "/api/imports",
        json={"family": "microclimate", "regime": "R1", "csv": "\n".join(lines) + "\n"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["claim_ceiling"] == "L1" and body["origin"] == "imported"
    ev = c.get(f"/api/runs/{body['experiment_id']}/events").json()
    assert [e["type"] for e in ev] == ["import_raw", "import_analysis"]


def test_compare_runs_diverge_after_shared_warmup(tmp_path):
    c = _client(tmp_path)
    ids = []
    for pol in ("P8", "P1"):
        r = c.post("/api/runs", json={"family": "systemlab", "regime": "R1", "seed": 2,
                                      "policy": pol, "budget": "medium"}).json()
        ids.append(r["experiment_id"])
    for exp in ids:
        for _ in range(1200):
            if c.get(f"/api/runs/{exp}/evaluation").status_code == 200:
                break
            time.sleep(0.1)
    d = c.get("/api/compare", params={"a": ids[0], "b": ids[1]}).json()
    warm = sum(1 for s in d["steps"] if s["a"] and s["a"]["mode"] == "warmup")
    assert d["first_divergence"] is None or d["first_divergence"] >= warm
    assert all(s["same_decision"] for s in d["steps"][:warm])
