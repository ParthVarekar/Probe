"""Headless benchmark runner and regime-map report (handover §12 Part I, milestone M4).

* The regime matrix (``bench/matrix_v0.yaml``) is hash-locked before any run:
  ``matrix_hash`` covers everything except the operational ``seeds``/``workers``
  overrides, and is written into every result row.
* Every (cell, seed) world is shared by all policies: worlds are a pure function
  of ``(family, regime, seed)`` and observation noise is counter-based, so the
  comparison is paired (common random numbers).
* Metrics are reported separately and never collapsed: correctness, severity
  (confidently-wrong rate, the headline), efficiency (cost, cost to decision),
  calibration (Brier). A policy that is cheaper but more often confidently
  wrong *loses* the cell.
* The positive control must separate the primary policy from the baselines or
  the whole comparison is marked invalid (the M4 go/no-go gate).
"""

from __future__ import annotations

import html
import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from probe.determinism import content_hash, make_rng

__all__ = ["aggregate", "gate", "load_matrix", "matrix_hash", "run_cell_seed", "run_matrix"]


def load_matrix(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("cells", "policies", "primary_policy", "baselines", "positive_control"):
        if key not in data:
            raise ValueError(f"matrix lacks {key!r}")
    return data


def matrix_hash(matrix: Mapping[str, Any]) -> str:
    return content_hash({k: v for k, v in matrix.items() if k not in ("seeds", "workers")})


def run_cell_seed(job: tuple[str, str, int, tuple[str, ...], str]) -> list[dict[str, Any]]:
    """All policies on one world (paired). Runs in a worker process."""
    from probe.run import run_world

    family, regime, seed, policies, budget = job
    rows = []
    for pid in policies:
        t0 = time.perf_counter()
        record, m = run_world(family, regime, seed, pid, budget)
        m["runtime_s"] = time.perf_counter() - t0
        m["decision_hash"] = record.decision_hash
        m["acceptable"] = "|".join(m["acceptable"])
        rows.append(m)
    return rows


def _boot_ci(diff: np.ndarray, rng: np.random.Generator, n: int = 2000) -> tuple[float, float]:
    if len(diff) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(rows: Sequence[Mapping[str, Any]], matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Per cell x policy summaries plus the paired primary-vs-best-baseline contrast."""
    cells: dict[str, Any] = {}
    rng = make_rng(0, "bench", "bootstrap")
    for cell in matrix["cells"]:
        key = f"{cell['family']}/{cell['regime']}"
        sub = [r for r in rows if r["family"] == cell["family"] and r["regime"] == cell["regime"]]
        if not sub:
            continue
        pol: dict[str, Any] = {}
        for pid in matrix["policies"]:
            rs = [r for r in sub if r["policy"] == pid]
            if not rs:
                continue
            briers = [r["brier"] for r in rs if r["brier"] is not None]
            pol[pid] = {
                "n": len(rs),
                "correct": float(np.mean([r["correct"] for r in rs])),
                "confidently_wrong": float(np.mean([r["confidently_wrong"] for r in rs])),
                "top1": float(np.mean([r["top1_correct"] for r in rs])),
                "decided": float(np.mean([r["decided"] for r in rs])),
                "cost": float(np.mean([r["cost"] for r in rs])),
                "brier": float(np.mean(briers)) if briers else None,
                "runtime_s": float(np.mean([r["runtime_s"] for r in rs])),
            }
        primary = matrix["primary_policy"]
        base = [b for b in matrix["baselines"] if b in pol]
        contrast = None
        if primary in pol and base:
            # best baseline = highest correctness, then lowest cost
            best = max(base, key=lambda b: (pol[b]["correct"], -pol[b]["cost"]))

            def score(r: Mapping[str, Any]) -> float:
                # cost-effectiveness: cost if correct, else full budget penalty (x2)
                return float(r["cost"]) if r["correct"] else 2.0 * float(r["cost"])

            seeds = sorted({r["seed"] for r in sub})
            by = {(r["policy"], r["seed"]): r for r in sub}
            d = np.array(
                [
                    score(by[(primary, s)]) - score(by[(best, s)])
                    for s in seeds
                    if (primary, s) in by and (best, s) in by
                ]
            )
            lo, hi = _boot_ci(d, rng)
            ref = float(np.mean([score(by[(best, s)]) for s in seeds if (best, s) in by]))
            rel = float(np.mean(d)) / ref if ref > 0 and len(d) else float("nan")
            sev = pol[primary]["confidently_wrong"] - pol[best]["confidently_wrong"]
            if sev > 0.01:
                outcome = "lose"
            elif hi < 0 and rel <= -0.15:
                outcome = "win"
            elif lo > 0:
                outcome = "lose"
            else:
                outcome = "tie"
            contrast = {
                "primary": primary,
                "best_baseline": best,
                "mean_diff": float(np.mean(d)) if len(d) else None,
                "ci95": [lo, hi],
                "relative": rel,
                "severity_diff": sev,
                "outcome": outcome,
            }
        cells[key] = {"expected": cell.get("expected", ""), "policies": pol, "contrast": contrast}
    return cells


def gate(cells: Mapping[str, Any], matrix: Mapping[str, Any]) -> dict[str, Any]:
    pc = matrix["positive_control"]
    key = f"{pc['family']}/{pc['regime']}"
    c = cells.get(key, {}).get("contrast")
    pc_ok = bool(c and c["outcome"] == "win")
    realistic = [
        k
        for k, v in cells.items()
        if k != key and v.get("contrast") and v["contrast"]["outcome"] == "win"
    ]
    return {
        "positive_control": key,
        "positive_control_separates": pc_ok,
        "realistic_win_cells": realistic,
        "go": pc_ok and bool(realistic),
        "note": "GO requires the positive control to separate and >=1 other win cell at "
        "equal or better severity (handover §12 Part J, M4).",
    }


_COLORS = {"win": "#1b7f3b", "tie": "#7a7a7a", "lose": "#b3261e"}


def write_report(
    cells: Mapping[str, Any], g: Mapping[str, Any], meta: Mapping[str, Any], path: Path
) -> None:
    """Static HTML regime map: one row per cell, primary-vs-best-baseline outcome,
    and every policy's correctness / confidently-wrong / cost (losses included)."""
    pols = list(meta["policies"])
    head = "".join(f"<th>{html.escape(p)}</th>" for p in pols)
    rows = []
    for key, v in cells.items():
        c = v["contrast"] or {}
        oc = c.get("outcome", "n/a")
        badge = f'<span class="b" style="background:{_COLORS.get(oc, "#444")}">{oc}</span>'
        tds = []
        for p in pols:
            s = v["policies"].get(p)
            tds.append(
                "<td>—</td>"
                if not s
                else f"<td>{s['correct']:.2f}<br><small>cw {s['confidently_wrong']:.2f} · "
                f"${s['cost']:.0f}</small></td>"
            )
        ci = c.get("ci95", [float("nan")] * 2)
        rows.append(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(v['expected'])}</td><td>{badge}"
            f"<br><small>vs {html.escape(str(c.get('best_baseline')))}: Δ "
            f"{c.get('mean_diff', float('nan')) or 0:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]</small>"
            f"</td>{''.join(tds)}</tr>"
        )
    verdict = "GO" if g["go"] else "NO-GO"
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PROBE regime map</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--line:#ddd}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121212;--fg:#eee;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin:16px}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}
td,th{{border:1px solid var(--line);padding:4px 6px;vertical-align:top;text-align:left}}
.b{{color:#fff;border-radius:3px;padding:1px 6px}} .sim{{font-weight:700;letter-spacing:.1em}}
code{{font-family:ui-monospace,monospace}}</style></head><body>
<p class="sim">SIMULATED — synthetic worlds only; no real-world claim.</p>
<h1>PROBE regime map ({html.escape(str(meta["version"]))})</h1>
<p>Gate: <b>{verdict}</b>. Positive control {html.escape(g["positive_control"])} separates:
<b>{g["positive_control_separates"]}</b>. Realistic win cells:
{html.escape(", ".join(g["realistic_win_cells"]) or "none")}.</p>
<p>Matrix hash <code>{meta["matrix_hash"]}</code> · seeds per cell {meta["seeds"]} ·
budget {html.escape(str(meta["budget"]))}. Cell entries: correct rate, cw = confidently-wrong
rate, mean cost. Outcome compares {html.escape(str(meta["primary"]))} with the best baseline on
paired cost-effectiveness (cost if correct, 2×cost otherwise) with a 95% bootstrap CI; higher
severity than the baseline is a loss.</p>
<table><thead><tr><th>cell</th><th>pre-registered</th><th>outcome</th>{head}</tr></thead>
<tbody>{"".join(rows)}</tbody></table></body></html>"""
    path.write_text(doc, encoding="utf-8")


def run_matrix(
    matrix_path: Path, out: Path, *, seeds: int | None = None, workers: int = 1
) -> dict[str, Any]:
    matrix = load_matrix(matrix_path)
    mhash = matrix_hash(matrix)
    n = int(seeds if seeds is not None else matrix.get("seeds", 20))
    budget = str(matrix.get("budget", "medium"))
    pols = tuple(matrix["policies"])
    jobs = [(c["family"], c["regime"], s, pols, budget) for c in matrix["cells"] for s in range(n)]
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for batch in ex.map(run_cell_seed, jobs, chunksize=1):
                rows.extend(batch)
    else:
        for j in jobs:
            rows.extend(run_cell_seed(j))
    for row in rows:
        row["matrix_hash"] = mhash
    out.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = sorted({k for r in rows for k in r})
    table = pa.table({k: [r.get(k) for r in rows] for k in cols})
    pq.write_table(table, out / "results.parquet")
    cells = aggregate(rows, matrix)
    g = gate(cells, matrix)
    meta = {
        "version": matrix.get("version"),
        "matrix_hash": mhash,
        "seeds": n,
        "budget": budget,
        "policies": pols,
        "primary": matrix["primary_policy"],
        "runtime_s": time.perf_counter() - t0,
        "n_runs": len(rows),
    }
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "gate": g, "cells": cells}, indent=2, default=str),
        encoding="utf-8",
    )
    write_report(cells, g, meta, out / "regime_map.html")
    return {"meta": meta, "gate": g}
