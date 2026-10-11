"""Recorded experiments, exportable bundles and ``probe verify`` (milestones M5/M7).

:func:`record_run` runs one simulated experiment and writes every step to the
hash-chained ledger *as it happens*: the protocol and hypothesis set are locked
(hashed) first, the truth commitment is logged at world creation, and each
decision is appended **before** its observation exists (commit-before-execute is
visible in the ledger order).

:func:`export_bundle` writes a zip with the ledger events, every payload blob and
a manifest. :func:`verify_bundle` re-checks it with no trust in the producer:

1. the hash chain and every payload hash;
2. decide-before-execute ordering in the ledger;
3. a deterministic **replay** from the manifest (same code version): every
   decision hash, the decision-sequence hash and the final posterior must match
   (decisions exactly, floats within ``rtol``; handover ADR-006).
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from probe import __version__
from probe.determinism import canonical_json, sha256_hex, to_jsonable
from probe.ledger import Ledger, verify_chain

__all__ = ["export_bundle", "record_run", "verify_bundle"]


def record_run(
    workspace: Path,
    family: str,
    regime: str,
    seed: int,
    policy_id: str,
    budget_key: str = "medium",
    experiment_id: str | None = None,
    stop_requested: Any = None,
) -> tuple[str, dict[str, Any]]:
    """Run and record one experiment; returns ``(experiment_id, metrics)``."""
    from probe.run import get_family, make_protocol, run_world

    exp = experiment_id or f"{family}-{regime}-s{seed}-{policy_id}-{budget_key}"
    fam = get_family(family)
    protocol = make_protocol(family, regime, budget_key)
    hs = fam.hypothesis_set(regime)
    with Ledger(Path(workspace) / "probe.db") as led:
        led.append(
            exp,
            "experiment_created",
            {
                "family": family,
                "regime": regime,
                "seed": seed,
                "policy": policy_id,
                "budget_key": budget_key,
                "probe_version": __version__,
                "origin": "simulated",
            },
            actor="human",
        )
        led.append(
            exp,
            "protocol_locked",
            {
                "protocol": protocol.model_dump(mode="json"),
                "protocol_hash": protocol.protocol_hash(),
                "hypothesis_set": hs.model_dump(mode="json"),
            },
            actor="human",
        )

        def on_event(ev: Mapping[str, Any]) -> None:
            actor = "simulator" if ev["type"] == "observation" else "engine"
            led.append(exp, str(ev["type"]), to_jsonable(dict(ev)), actor=actor)

        record, metrics = run_world(
            family,
            regime,
            seed,
            policy_id,
            budget_key,
            on_event=on_event,
            stop_requested=stop_requested,
        )
        led.append(
            exp, "truth_commitment", {"commitment": record.truth_commitment}, actor="simulator"
        )
        led.append(exp, "run_record", to_jsonable(record))
    return exp, metrics


def export_bundle(workspace: Path, experiment_id: str, out: Path) -> Path:
    with Ledger(Path(workspace) / "probe.db") as led:
        events = list(led.events())  # the full chain: verification needs every link
        mine = [e for e in events if e.experiment_id == experiment_id]
        if not mine:
            raise KeyError(f"no events for experiment {experiment_id!r}")
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            lines = [json.dumps({**e.header(), "hash": e.hash}) for e in events]
            z.writestr("ledger.jsonl", "\n".join(lines) + "\n")
            for e in events:
                z.writestr(f"blobs/{e.payload_hash}", led.blobs.get(e.payload_hash))
            created = led.payload(mine[0])
            manifest = {
                "experiment_id": experiment_id,
                "bundle_format": "probe.bundle/v0",
                "probe_version": __version__,
                "run": created,
                "n_events": len(events),
            }
            z.writestr("manifest.json", canonical_json(manifest))
    return out


def verify_bundle(path: Path, *, replay: bool = True, rtol: float = 1e-9) -> dict[str, Any]:
    problems: list[str] = []
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        manifest = json.loads(z.read("manifest.json"))
        events = [json.loads(x) for x in z.read("ledger.jsonl").decode().splitlines() if x]

        def blob(h: str) -> bytes:
            if f"blobs/{h}" not in names:
                raise KeyError(h)
            return z.read(f"blobs/{h}")

        problems += verify_chain(events, blob)
        exp = manifest["experiment_id"]
        mine = [e for e in events if e["experiment_id"] == exp]
        payloads = {}
        for e in mine:
            try:
                payloads[e["seq"]] = json.loads(blob(e["payload_hash"]))
            except KeyError:
                pass
    # decide-before-execute: decision for step k precedes observation for step k
    seen_decisions: set[int] = set()
    decisions: dict[int, str] = {}
    record: dict[str, Any] | None = None
    for e in mine:
        p = payloads.get(e["seq"], {})
        if e["type"] == "decision":
            seen_decisions.add(p["step"])
            decisions[p["step"]] = p["decision_hash"]
        elif e["type"] == "observation" and p.get("step") not in seen_decisions:
            problems.append(f"observation for step {p.get('step')} has no prior decision")
        elif e["type"] == "run_record":
            record = p
    replay_ok = None
    if replay and record is not None and not problems:
        from probe.run import run_world

        run = manifest["run"]
        if run.get("probe_version") != __version__:
            problems.append(
                f"bundle made by probe {run.get('probe_version')}, "
                f"verifier is {__version__}; replay may differ"
            )
        stop_fn = None
        if record["stop"].get("reason") == "user_stopped":
            # replay the user's stop at the same step boundary (stop_requested is
            # polled exactly once per step, before that step's decision)
            n_steps, calls = len(record["steps"]), [0]

            def stop_fn() -> bool:
                calls[0] += 1
                return calls[0] > n_steps

        rec, _ = run_world(
            run["family"],
            run["regime"],
            run["seed"],
            run["policy"],
            run["budget_key"],
            stop_requested=stop_fn,
        )
        again = to_jsonable(rec)
        replay_ok = True
        if again["decision_hash"] != record["decision_hash"]:
            problems.append("replay: decision-sequence hash differs")
            replay_ok = False
        for s in again["steps"]:
            if decisions.get(s["step"]) != s["decision_hash"]:
                problems.append(f"replay: decision {s['step']} differs from the ledger")
                replay_ok = False
        for h, w in record["final_weights"].items():
            w2 = again["final_weights"].get(h)
            if w2 is None or abs(w2 - w) > rtol * max(1.0, abs(w)):
                problems.append(f"replay: final posterior of {h} differs ({w} vs {w2})")
                replay_ok = False
    return {
        "bundle": str(path),
        "bundle_sha256": sha256_hex(Path(path).read_bytes()),
        "experiment_id": exp,
        "events": len(events),
        "chain_ok": not any("hash" in p or "seq" in p or "payload" in p for p in problems),
        "replay_ok": replay_ok,
        "ok": not problems,
        "problems": problems,
    }
