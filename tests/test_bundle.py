import json
import sqlite3
import zipfile

import pytest

from probe.bundle import export_bundle, record_run, verify_bundle
from probe.ledger import Ledger


@pytest.fixture
def bundle(tmp_path):
    exp, _ = record_run(tmp_path / "ws", "systemlab", "R2", 1, "P8")
    return export_bundle(tmp_path / "ws", exp, tmp_path / "b.zip"), tmp_path


def test_bundle_verifies_with_replay(bundle):
    path, _ = bundle
    rep = verify_bundle(path)
    assert rep["ok"], rep["problems"]
    assert rep["replay_ok"] is True


def test_ledger_is_append_only(bundle):
    _, tmp = bundle
    db = sqlite3.connect(tmp / "ws" / "probe.db")
    with pytest.raises(sqlite3.DatabaseError):
        db.execute("UPDATE ledger SET type='x' WHERE seq=1")
    with pytest.raises(sqlite3.DatabaseError):
        db.execute("DELETE FROM ledger WHERE seq=1")


def _rewrite(path, out, fn):
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(out, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, fn(n, zin.read(n)))
    return out


def test_tampered_event_detected(bundle, tmp_path):
    path, _ = bundle

    def edit(name, data):
        if name == "ledger.jsonl":
            lines = data.decode().splitlines()
            e = json.loads(lines[3])
            e["type"] = "observation"
            lines[3] = json.dumps(e)
            return ("\n".join(lines) + "\n").encode()
        return data

    rep = verify_bundle(_rewrite(path, tmp_path / "t.zip", edit), replay=False)
    assert not rep["ok"]


def test_tampered_payload_detected(bundle, tmp_path):
    path, _ = bundle
    with zipfile.ZipFile(path) as z:
        ev = json.loads(z.read("ledger.jsonl").decode().splitlines()[-1])
    target = f"blobs/{ev['payload_hash']}"
    rep = verify_bundle(
        _rewrite(
            path, tmp_path / "p.zip", lambda n, d: d.replace(b"0", b"1") if n == target else d
        ),
        replay=False,
    )
    assert not rep["ok"]


def test_ledger_head_links(tmp_path):
    with Ledger(tmp_path / "x.db") as led:
        a = led.append("e", "t", {"a": 1})
        b = led.append("e", "t", {"a": 2})
    assert b.prev_hash == a.hash
