"""Static HTML research report for one run (handover §12 Part L "Research Report").

Every number comes from the run record; the chart is an inline SVG of the
pre-computed posterior trajectory (charts never compute statistics). The page
carries the SIMULATED watermark, the claim level, downgrades, the decision log
with "why this next", and exact reproduction/verification commands.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

from probe.claims import claim_dicts, claims_for_run

__all__ = ["render_report"]

_PALETTE = ("#3366cc", "#dc7633", "#239b56", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d")


def _svg(record: Mapping[str, Any], width: int = 640, height: int = 220) -> str:
    steps = record["steps"]
    ids = list(record["hypothesis_ids"])
    if not steps:
        return "<p>No observations.</p>"
    n = len(steps)
    px = lambda i: 40 + (width - 60) * (i / max(n - 1, 1))  # noqa: E731
    py = lambda p: 10 + (height - 40) * (1 - p)  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Posterior probability of each hypothesis by step" class="chart">'
    ]
    for p in (0, 0.5, 0.95, 1):
        y = py(p)
        parts.append(
            f'<line x1="40" x2="{width - 20}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
            f'<text x="34" y="{y + 4:.1f}" text-anchor="end">{p:g}</text>'
        )
    for k, h in enumerate(ids):
        pts = " ".join(f"{px(i):.1f},{py(s['weights'][h]):.1f}" for i, s in enumerate(steps))
        dash = ' stroke-dasharray="4 3"' if h == record.get("extra", {}).get("residual_id") else ""
        c = _PALETTE[k % len(_PALETTE)]
        parts.append(f'<polyline fill="none" stroke="{c}" stroke-width="2"{dash} points="{pts}"/>')
        last = steps[-1]["weights"][h]
        parts.append(
            f'<text x="{width - 18}" y="{py(last) + 4:.1f}" fill="{c}">{html.escape(h)}</text>'
        )
    parts.append(f'<text x="{width / 2:.0f}" y="{height - 4}" text-anchor="middle">step</text>')
    parts.append("</svg>")
    return "".join(parts)


def _val(v: Any) -> str:
    return "—" if v is None else f"{v:.3g}"


def render_report(
    record: Mapping[str, Any], *, world_label: str, verification: Mapping[str, Any] | None = None
) -> str:
    claims = claim_dicts(claims_for_run(record, world_label=world_label))
    esc = html.escape
    rows = "".join(
        f"<tr><td>{s['step']}</td><td>{esc(s['mode'])}</td><td><code>{esc(s['action_id'])}"
        f"</code></td><td>{s['chosen_u_model']:.3g}</td><td>{s['chosen_cost']:.3g}</td>"
        f"<td>{_val(s['value'])}</td>"
        f"<td>{s['propensity']:.3g}</td><td>{esc(s['reason'])}</td></tr>"
        for s in record["steps"]
    )
    cl = "".join(
        f"<li><span class='lvl'>{c['level']}</span> {esc(c['text'])}"
        + (
            f"<br><small>Downgrades that would apply to real data: "
            f"{esc('; '.join(c['downgrades']))}</small>"
            if c["downgrades"]
            else ""
        )
        + "</li>"
        for c in claims
    )
    post = "".join(
        f"<tr><td>{esc(h)}</td><td>{w:.4f}</td></tr>" for h, w in record["final_weights"].items()
    )
    ver = ""
    if verification is not None:
        ver = (
            f"<h2>Verification</h2><p>Bundle verified: <b>{verification['ok']}</b> "
            f"(chain + ordering + replay). Problems: "
            f"{esc('; '.join(verification['problems']) or 'none')}</p>"
        )
    stop = record["stop"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PROBE run report</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--acc:#3366cc}}
@media (prefers-color-scheme:dark){{:root{{--bg:#111;--fg:#e8e8e8;--mut:#999;--line:#333}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;
max-width:1000px;padding:16px}}
.sim{{border:2px solid #c0392b;color:#c0392b;font-weight:700;padding:4px 8px;
display:inline-block;letter-spacing:.08em}}
table{{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%;font-size:13px}}
td,th{{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top}}
code{{font-family:ui-monospace,monospace;font-size:12px}} .lvl{{background:var(--acc);color:#fff;
border-radius:3px;padding:0 5px;font-size:12px}} .chart{{width:100%;max-width:640px}}
.chart text{{font-size:11px;fill:var(--mut)}} .chart .grid{{stroke:var(--line)}}
.scroll{{overflow-x:auto}}</style></head><body>
<p class="sim">SIMULATED — no real-world claim</p>
<h1>Run report: {esc(world_label)}</h1>
<p>Policy <b>{esc(record["policy_id"])}</b> · seed {record["seed"]} · stop
<b>{esc(str(stop["reason"]))}</b> · verdict <b>{esc(str(stop["verdict"]))}</b> · cost
{record["spent"]:.1f} · n={record["n_obs"]}</p>
<h2>Claims</h2><ul>{cl}</ul>
<h2>Posterior over hypotheses</h2>{_svg(record)}
<p><small>Dashed: residual hypothesis. 0.95 = decision threshold π*.</small></p>
<table><thead><tr><th>hypothesis</th><th>final posterior</th></tr></thead>
<tbody>{post}</tbody></table>
<h2>Decisions (committed before each observation)</h2><div class="scroll"><table><thead><tr>
<th>step</th><th>mode</th><th>action</th><th>U_M</th><th>cost</th><th>value</th>
<th>propensity</th><th>why this next</th></tr></thead><tbody>{rows}</tbody></table></div>
{ver}
<h2>Provenance</h2><p>Protocol hash <code>{record["protocol_hash"]}</code><br>
Hypothesis-set hash <code>{record["hypothesis_set_hash"]}</code><br>
Truth commitment <code>{record["truth_commitment"]}</code><br>
Decision-sequence hash <code>{record["decision_hash"]}</code></p>
<p>Reproduce: <code>probe run --family {esc(record["family"])} --policy
{esc(record["policy_id"])} --seed {record["seed"]} ...</code> then <code>probe verify
&lt;bundle&gt;</code>.</p></body></html>"""
