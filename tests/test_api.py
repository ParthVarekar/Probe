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
