# PROBE

Verifiable adaptive experimentation: choose the next observation that best tells competing
explanations apart, and produce conclusions anyone can re-check.

**For coding agents / contributors:** the full research-backed plan (Parts A–Q, milestones
M0–M13, acceptance criteria) is in [`handover.md`](handover.md) §12. Current implementation
status and the next steps are in `handover.md` §0.0, so read that first.

## Quick start

```bash
uv sync
uv run probe run --family systemlab --regime R2 --seed 1 --policy P8
uv run probe bench --matrix bench/matrix_v0.yaml --seeds 20 --workers 4   # writes results/bench/regime_map.html
uv run pytest -m "not slow" -n 4
```

All worlds are **SIMULATED**; no real-world claim is made by any output.
