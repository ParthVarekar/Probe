"""Hash-chained, append-only ledger on SQLite plus a content-addressed blob store
(handover §12 Part G ``LedgerEvent`` / ``RawBlob``, Part M tampering row).

* Every event is ``{seq, experiment_id, type, actor, payload_hash, prev_hash}``
  and ``hash = sha256(prev_hash || canonical_json(event))``. The payload itself
  is stored in the CAS under its own hash, so the chain commits to it.
* SQLite triggers forbid ``UPDATE`` and ``DELETE`` on the ledger table.
* :func:`verify_chain` recomputes every hash; any edit to an event, a payload
  blob or the order breaks it.

Deviation from the plan: stdlib ``sqlite3`` instead of SQLAlchemy + Alembic (no
extra dependency for v0; the schema is small and created idempotently).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probe.determinism import canonical_json, sha256_hex

__all__ = ["GENESIS", "BlobStore", "Ledger", "LedgerEvent", "event_hash", "verify_chain"]

GENESIS = "0" * 64
ACTORS = ("engine", "human", "llm", "external", "simulator")

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS ledger (
  seq INTEGER PRIMARY KEY,
  experiment_id TEXT NOT NULL,
  type TEXT NOT NULL,
  actor TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  hash TEXT NOT NULL UNIQUE
) STRICT;
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger
BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END;
"""


class BlobStore:
    """Immutable blobs at ``root/sha256/ab/cd/<hash>``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, h: str) -> Path:
        return self.root / "sha256" / h[:2] / h[2:4] / h

    def put(self, data: bytes) -> str:
        h = sha256_hex(data)
        p = self._path(h)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
        return h

    def put_json(self, obj: Any) -> str:
        return self.put(canonical_json(obj))

    def get(self, h: str) -> bytes:
        data = self._path(h).read_bytes()
        if sha256_hex(data) != h:
            raise ValueError(f"blob {h} is corrupted")
        return data

    def has(self, h: str) -> bool:
        return self._path(h).exists()


@dataclass(frozen=True)
class LedgerEvent:
    seq: int
    experiment_id: str
    type: str
    actor: str
    payload_hash: str
    prev_hash: str
    hash: str

    def header(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "experiment_id": self.experiment_id,
            "type": self.type,
            "actor": self.actor,
            "payload_hash": self.payload_hash,
            "prev_hash": self.prev_hash,
        }


def event_hash(prev_hash: str, header: Mapping[str, Any]) -> str:
    return sha256_hex(prev_hash.encode("ascii") + canonical_json(dict(header)))


class Ledger:
    """One SQLite file + a blob directory next to it."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.blobs = BlobStore(self.path.parent / "blobs")
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def head(self) -> tuple[int, str]:
        row = self._db.execute("SELECT seq, hash FROM ledger ORDER BY seq DESC LIMIT 1").fetchone()
        return (0, GENESIS) if row is None else (int(row[0]), str(row[1]))

    def append(
        self, experiment_id: str, type_: str, payload: Any, actor: str = "engine"
    ) -> LedgerEvent:
        if actor not in ACTORS:
            raise ValueError(f"actor must be one of {ACTORS}")
        ph = self.blobs.put_json(payload)
        self._db.execute("BEGIN IMMEDIATE")
        try:
            seq, prev = self.head()
            header = {
                "seq": seq + 1,
                "experiment_id": experiment_id,
                "type": type_,
                "actor": actor,
                "payload_hash": ph,
                "prev_hash": prev,
            }
            h = event_hash(prev, header)
            self._db.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?,?,?)",
                (seq + 1, experiment_id, type_, actor, ph, prev, h),
            )
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        return LedgerEvent(seq + 1, experiment_id, type_, actor, ph, prev, h)

    def events(self, experiment_id: str | None = None, after: int = 0) -> Iterator[LedgerEvent]:
        q = (
            "SELECT seq, experiment_id, type, actor, payload_hash, prev_hash, hash "
            "FROM ledger WHERE seq > ?"
        )
        args: list[Any] = [after]
        if experiment_id is not None:
            q += " AND experiment_id = ?"
            args.append(experiment_id)
        for r in self._db.execute(q + " ORDER BY seq", args):
            yield LedgerEvent(int(r[0]), r[1], r[2], r[3], r[4], r[5], r[6])

    def payload(self, ev: LedgerEvent) -> Any:
        import json

        return json.loads(self.blobs.get(ev.payload_hash))


def verify_chain(events: list[Mapping[str, Any]], blob_get: Any | None = None) -> list[str]:
    """Problems found in a full chain of exported event dicts (empty list = valid).
    ``blob_get(hash) -> bytes`` re-checks each payload against its hash."""
    problems: list[str] = []
    prev = GENESIS
    for i, e in enumerate(events, start=1):
        header = {
            k: e[k] for k in ("seq", "experiment_id", "type", "actor", "payload_hash", "prev_hash")
        }
        if e["seq"] != i:
            problems.append(f"event {i}: sequence gap (seq={e['seq']})")
        if e["prev_hash"] != prev:
            problems.append(f"event {i}: prev_hash does not link to the previous event")
        if event_hash(e["prev_hash"], header) != e["hash"]:
            problems.append(f"event {i}: hash mismatch (event edited)")
        if blob_get is not None:
            try:
                if sha256_hex(blob_get(e["payload_hash"])) != e["payload_hash"]:
                    problems.append(f"event {i}: payload does not match its hash")
            except (OSError, KeyError, ValueError):
                problems.append(f"event {i}: payload missing or corrupted")
        prev = e["hash"]
    return problems
