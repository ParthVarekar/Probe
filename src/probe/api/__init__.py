"""Local REST + SSE API and the bundled single-page UI (milestones M5/M6).

Security posture (handover §12 Part M): bind 127.0.0.1 only; reject requests
whose ``Host`` is not a loopback name (DNS-rebinding defence); no CORS; state
changes are POST only. Truth never appears on any route: runs are recorded via
:func:`probe.bundle.record_run`, which only returns evaluator metrics after the
run has stopped, and those metrics are shown as a separate *evaluation* panel
labelled as such.

Deviation from the plan: the UI is a dependency-free static page
(``static/index.html``) instead of a React/Vite build, so ``probe demo`` needs
no Node toolchain.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from probe import __version__

__all__ = ["create_app"]

STATIC = Path(__file__).with_name("static")
LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "testserver"}


class RunRequest(BaseModel):
    family: str = Field(pattern="^(microclimate|systemlab)$")
    regime: str = Field(pattern="^R[0-9]{1,2}$")
    seed: int = Field(ge=0, le=10**9)
    policy: str = Field(pattern="^P[0-9]{1,2}$")
    budget: str = Field(default="medium", pattern="^(small|medium|large)$")


class ValidateRequest(BaseModel):
    family: str = Field(pattern="^(microclimate|systemlab)$")
    regime: str = Field(pattern="^R[0-9]{1,2}$")
    budget: str = Field(default="medium", pattern="^(small|medium|large)$")


class ImportRequest(BaseModel):
    family: str = Field(pattern="^(microclimate|systemlab)$")
    regime: str = Field(pattern="^R[0-9]{1,2}$")
    csv: str = Field(max_length=20 * 1024 * 1024)
    mapping: dict[str, str] | None = None
    name: str = Field(default="import", pattern="^[A-Za-z0-9_.-]{1,64}$")


def create_app(workspace: Path, results_dir: Path | None = None) -> FastAPI:
    from probe.ledger import Ledger

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="PROBE", version=__version__)
    jobs: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    @app.middleware("http")
    async def host_check(request: Request, call_next: Any) -> Any:
        host = (request.headers.get("host") or "").rsplit(":", 1)[0]
        if host not in LOOPBACK:
            return JSONResponse({"detail": "forbidden host"}, status_code=403)
        return await call_next(request)

    def ledger() -> Ledger:
        return Ledger(workspace / "probe.db")

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        from probe.run import FAMILIES, get_family
        from probe.select import POLICY_IDS, POLICY_NAMES

        fams = {}
        for f in FAMILIES:
            fam = get_family(f)
            fams[f] = {
                r: {"title": s.title, "expected": s.expected, "budgets": dict(s.budgets)}
                for r, s in fam.regimes().items()
            }
        return {
            "version": __version__,
            "families": fams,
            "policies": {p: POLICY_NAMES[p] for p in POLICY_IDS},
        }

    results = Path(results_dir) if results_dir is not None else Path("results")

    @app.get("/api/benchmarks")
    def benchmarks() -> list[dict[str, Any]]:
        out = []
        if results.is_dir():
            for d in sorted(results.iterdir()):
                f = d / "summary.json"
                if f.is_file():
                    summ = json.loads(f.read_text(encoding="utf-8"))
                    out.append({"name": d.name, "meta": summ["meta"], "gate": summ["gate"]})
        return out

    @app.get("/api/benchmarks/{name}/map", response_class=HTMLResponse)
    def benchmark_map(name: str) -> FileResponse:
        if not name.replace("_", "").replace("-", "").isalnum():
            raise HTTPException(404)
        f = results / name / "regime_map.html"
        if not f.is_file():
            raise HTTPException(404)
        return FileResponse(f)

    @app.post("/api/validate")
    def validate(req: ValidateRequest) -> dict[str, Any]:
        from probe.protocol import validate_protocol

        return validate_protocol(req.family, req.regime, req.budget)

    @app.post("/api/imports")
    def import_csv(req: ImportRequest) -> dict[str, Any]:
        from probe.ingest import analyze_import, parse_csv
        from probe.run import get_family

        fam = get_family(req.family)
        if req.regime not in fam.regimes():
            raise HTTPException(422, "unknown regime")
        hs = fam.hypothesis_set(req.regime)
        mapping = req.mapping or {n: n for n in [hs.outcome.name, *(c.name for c in hs.covariates)]}
        data = req.csv.encode("utf-8")
        try:
            res = parse_csv(data, hs, mapping)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        out = analyze_import(res, hs)
        exp = f"import-{req.name}-{res.raw_sha256[:12]}"
        with ledger() as led:
            raw = led.blobs.put(data)  # raw bytes first, immutable
            led.append(
                exp,
                "import_raw",
                {"raw_sha256": raw, "bytes": len(data), "mapping": mapping, "origin": "imported"},
                actor="human",
            )
            led.append(exp, "import_analysis", out)
        return {"experiment_id": exp, **out}

    @app.post("/api/runs", status_code=202)
    def start(req: RunRequest) -> dict[str, Any]:
        from probe.bundle import record_run
        from probe.run import get_family

        if req.regime not in get_family(req.family).regimes():
            raise HTTPException(422, f"unknown regime {req.regime} for {req.family}")
        exp = f"{req.family}-{req.regime}-s{req.seed}-{req.policy}-{req.budget}"
        with lock:
            if exp in jobs and jobs[exp]["status"] == "running":
                return {"experiment_id": exp, "status": "running"}
            with ledger() as led:
                if any(True for _ in led.events(exp)):
                    return {"experiment_id": exp, "status": "completed"}
            jobs[exp] = {"status": "running"}

        def work() -> None:
            try:
                _, metrics = record_run(
                    workspace,
                    req.family,
                    req.regime,
                    req.seed,
                    req.policy,
                    req.budget,
                    experiment_id=exp,
                )
                jobs[exp] = {"status": "completed", "evaluation": metrics}
            except Exception as e:  # recorded, surfaced on the job
                jobs[exp] = {"status": "failed", "error": repr(e)}

        threading.Thread(target=work, daemon=True).start()
        return {"experiment_id": exp, "status": "running"}

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        with ledger() as led:
            ids: dict[str, dict[str, Any]] = {}
            for e in led.events():
                d = ids.setdefault(
                    e.experiment_id,
                    {"experiment_id": e.experiment_id, "events": 0, "status": "running"},
                )
                d["events"] += 1
                if e.type == "run_record":
                    d["status"] = "completed"
        for k, j in jobs.items():
            ids.setdefault(k, {"experiment_id": k, "events": 0})["status"] = j["status"]
        return list(ids.values())

    @app.get("/api/runs/{exp}/events")
    def events(exp: str, after: int = 0) -> list[dict[str, Any]]:
        with ledger() as led:
            return [
                {**e.header(), "hash": e.hash, "payload": led.payload(e)}
                for e in led.events(exp, after)
            ]

    @app.get("/api/runs/{exp}/stream")
    async def stream(exp: str, request: Request) -> StreamingResponse:
        start_after = int(request.headers.get("last-event-id") or 0)

        async def gen() -> Any:
            after = start_after
            idle = 0
            while idle < 600:
                with ledger() as led:
                    evs = [(e, led.payload(e)) for e in led.events(exp, after)]
                for e, p in evs:
                    after = e.seq
                    data = json.dumps({"seq": e.seq, "type": e.type, "payload": p})
                    yield f"id: {e.seq}\nevent: ledger\ndata: {data}\n\n"
                    if e.type == "run_record":
                        yield "event: done\ndata: {}\n\n"
                        return
                idle = 0 if evs else idle + 1
                await asyncio.sleep(0.2)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/runs/{exp}/evaluation")
    def evaluation(exp: str) -> dict[str, Any]:
        j = jobs.get(exp)
        if not j or j["status"] != "completed":
            raise HTTPException(404, "evaluation available only after the run has stopped")
        return {
            "label": "EVALUATOR ONLY: scored against the sealed truth after the run",
            **j["evaluation"],
        }

    @app.get("/api/compare")
    def compare(a: str, b: str) -> dict[str, Any]:
        """Step-by-step diff of two recorded runs (decisions and observations)."""

        def steps(exp: str) -> list[dict[str, Any]]:
            with ledger() as led:
                recs = [led.payload(e) for e in led.events(exp) if e.type == "run_record"]
            if not recs:
                raise HTTPException(404, f"no finished run {exp!r}")
            return list(recs[-1]["steps"])

        sa, sb = steps(a), steps(b)
        rows, first = [], None
        for i in range(max(len(sa), len(sb))):
            x = sa[i] if i < len(sa) else None
            y = sb[i] if i < len(sb) else None
            same = bool(x and y and x["decision_hash"] == y["decision_hash"])
            if not same and first is None:
                first = i
            rows.append(
                {
                    "step": i,
                    "a": None
                    if x is None
                    else {"action": x["action_id"], "mode": x["mode"], "value": x["value"]},
                    "b": None
                    if y is None
                    else {"action": y["action_id"], "mode": y["mode"], "value": y["value"]},
                    "same_decision": same,
                }
            )
        return {"a": a, "b": b, "first_divergence": first, "steps": rows}

    def _bundle(exp: str) -> Path:
        from probe.bundle import export_bundle

        out = workspace / "bundles" / f"{exp}.probe.zip"
        try:
            return export_bundle(workspace, exp, out)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/api/runs/{exp}/verify")
    def verify(exp: str) -> dict[str, Any]:
        from probe.bundle import verify_bundle

        return verify_bundle(_bundle(exp))

    @app.get("/api/runs/{exp}/bundle")
    def bundle(exp: str) -> FileResponse:
        p = _bundle(exp)
        return FileResponse(p, filename=p.name, media_type="application/zip")

    @app.get("/api/runs/{exp}/report", response_class=HTMLResponse)
    def report(exp: str) -> str:
        from probe.bundle import verify_bundle
        from probe.report import render_report

        p = _bundle(exp)
        ver = verify_bundle(p, replay=False)
        with ledger() as led:
            recs = [led.payload(e) for e in led.events(exp) if e.type == "run_record"]
            created = next(
                led.payload(e) for e in led.events(exp) if e.type == "experiment_created"
            )
        if not recs:
            raise HTTPException(409, "run has not finished")
        label = f"{created['family']}/{created['regime']}"
        return render_report(recs[-1], world_label=label, verification=ver)

    return app
