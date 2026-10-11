# PROBE

Verifiable adaptive experimentation: choose the next observation that best tells
competing explanations apart, stop honestly, and produce conclusions anyone can
re-check.

> **Status: pre-alpha research software.** All built-in worlds are **SIMULATED**;
> no output makes a real-world claim. Benchmark evidence so far is mixed; see
> [`docs/RESEARCH_HYPOTHESES.md`](docs/RESEARCH_HYPOTHESES.md).

**For coding agents and contributors:** the full research-backed plan (Parts A–Q,
milestones M0–M13, acceptance criteria) is in [`handover.md`](handover.md) §12.
**Current status and next steps are in `handover.md` §0.0, so read it first.**
Design decisions and deviations from the plan are in
[`docs/ARCHITECTURE_DECISIONS.md`](docs/ARCHITECTURE_DECISIONS.md).

## What it does

1. **Hypotheses as data.** Competing explanations are written in a closed IR
   (linear-Gaussian feature models with conjugate priors, or finite-grid mechanisms).
   A flexible *residual* hypothesis always competes.
2. **Adaptive selection.** At each step, every feasible action is scored by exact
   model-indicator expected information gain per unit cost. The guarded policy mixes
   in an audit design with logged propensities.
3. **Honest stopping.** It decides only when the posterior reaches π* **and** a
   universal-inference e-process rejects every rival. Otherwise it stops with
   "inconclusive", "none fits" or "undecided".
4. **Integrity.** Every decision is committed to a hash-chained ledger *before* its
   observation. The hidden truth is sealed, and bundles verify by deterministic replay.

## Quick start

```bash
uv sync
uv run probe demo                       # local web app on http://127.0.0.1:8765
uv run probe run --family systemlab --regime R2 --seed 1 --policy P8
uv run probe record --family microclimate --regime R4 --seed 3 --policy P8
uv run probe export microclimate-R4-s3-P8-medium --out run.probe.zip
uv run probe verify run.probe.zip       # chain + ordering + replay
uv run probe report run.probe.zip       # HTML report with claim levels
uv run probe bench --matrix bench/matrix_v0.yaml --seeds 100 --workers 6
uv run probe calibrate --seeds 40 --workers 6   # misspecification alarm thresholds
uv run pytest -m "not slow" -n 4
```

Optional AI assist (claim paraphrase only, always validated and labelled
*proposed*): `uv sync --extra ai`. Nothing requires a key or network.

## Layout

| Path | What |
|---|---|
| `src/probe/engine/` | IR, NIG and grid engines, exact EIG (Gauss–Kronrod with error bounds) |
| `src/probe/envs/` | Microclimate SCM (R1–R10) and black-box system lab (R1–R11) |
| `src/probe/select/`, `stop/`, `diagnostics/` | Policies P1–P10 (+ P8a), stopping rules and e-processes, PIT/alarms/identifiability |
| `src/probe/run/` | Run loop, evaluator, protocol |
| `src/probe/bench/` | Benchmark runner, regime map, alarm calibration |
| `src/probe/ledger/`, `bundle.py` | Hash-chained ledger, CAS, export/verify |
| `src/probe/claims.py`, `report.py` | Claim-guard (L0–L5) and HTML report |
| `src/probe/ingest.py` | CSV import (claim ceiling L1) |
| `src/probe/api/` | FastAPI REST + SSE and the single-page UI |
| `src/probe/ai/` | Optional provider port (Null/Replay/Recording/Anthropic) |

Licence: Apache-2.0 (see `LICENSE`, `THIRD_PARTY_NOTICES.md`).
