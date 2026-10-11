# Architecture decisions (Nygard-style, condensed)

The plan these refine is `handover.md` §12. Status: **Accepted** unless noted.

## ADR-001: S1 base: a deterministic evidence engine with integrity grafts
Lean numpy/scipy model-discrimination core plus a ledger/verify integrity layer.
Rejected: a general PPL core (S2), an LLM-investigator core (S3). See §12 Part E.

## ADR-002: Exact engine families only in v0
`linear_gaussian` (NIG, closed-form evidence and predictive) and `finite_grid`
(exact over a parameter grid). U_M is exact for discrete outcomes and uses
adaptive Gauss–Kronrod with a rigorous error bound for Student-t mixtures.

## ADR-003: SQLite + hash-chained ledger + CAS; no durable-execution engine
**Deviation:** stdlib `sqlite3` instead of SQLAlchemy + Alembic. The schema is
one append-only table guarded by triggers plus a blob directory; an ORM adds
dependency weight without benefit at this size. Revisit if multi-table
relational state (questions, protocol versions, jobs) moves into the DB.

## ADR-004: No generated-code execution
Hypotheses are IR data; grid mechanisms are in-tree reviewed code registered at
import time. Nothing from an LLM or user input can register a mechanism.

## ADR-006: Replay contract
Decisions must match exactly (decision hashes); floats within `rtol=1e-9`.
`argmax_with_ties` uses a relative tie tolerance and a sha256 tie-break so that
ulp-level BLAS differences cannot flip a decision. Golden decision hashes live in
`tests/golden/`.

## ADR-007: Claim-guard levels are computed, never written
`src/probe/claims.py`. Simulated data ⇒ L0; imported CSV ⇒ ceiling L1. A lexicon
validator rejects causal wording below L3.

## ADR-008: Hidden-truth sealing is API discipline, not cryptography
`SealedTruth` is opened only with an `EvaluatorKey`. Agent-facing outputs are
scanned in tests. Benchmark seeds for any official leaderboard must come from a
large private space, because campus layouts are a public function of the seed.

## ADR-013: The residual hypothesis is exempt from the e-process decision requirement
The H.6 decision rule needs E_t(leader : rival) ≥ 1/α against every rival. The
residual is deliberately flexible (often more parameters than observations), so
its maximum likelihood is unbounded and no universal-inference e-process can
ever reject it. Requiring it made every microclimate decision impossible
(benchmark run 1). The residual is instead handled by the posterior threshold
and the inadequacy rule (stop with "none fits" when it leads). Rivals whose
model class contains the leader's (nesting) are also exempt.

## ADR-014: Dependency-free single-page UI instead of React/Vite (for now)
`src/probe/api/static/index.html` covers the core loop: lock and run, the live
posterior via SSE, "why this next", the ledger, verify and the report. `probe
demo` therefore needs no Node toolchain, in line with the P.9 install goal. The full
9-screen product UI (M9) may move to Vite + React with the built `dist` bundled
in the wheel.

## ADR-015: Benchmark matrix budget `large` (pre-lock deviation)
With the `medium` budgets, about 7 microclimate observations fit and no policy,
not even the oracle, could reach a decision. The matrix was changed before the
lock was declared, and the change is recorded in `bench/matrix_v0.yaml`.
