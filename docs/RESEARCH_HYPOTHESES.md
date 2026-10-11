# Research hypotheses and current evidence

The pre-registered hypotheses are from `handover.md` §12 H.10, with falsification
thresholds from Part I. Evidence is recorded here as runs complete. **All results
are SIMULATED**, and no real-world claim is made.

| Id | Hypothesis (short) | Falsified if | Status (2026-10-11) |
|---|---|---|---|
| **A′** | Guarded cost-aware model-indicator EIG (P8) reaches the decision at ≥ 15% lower cost than the best of {P1–P5, P9} in realistic regimes, with confidently-wrong ≤ best baseline + 1 pp in every regime. | No realistic win cell, or the safety condition fails in any cell. | **Mixed.** Win cells with 100 seeds: microclimate R2, R4; systemlab R1 (see `bench_v0_100seeds_summary.json`). The **safety condition fails** in misspecified/nuisance regimes (microclimate R7–R9, systemlab R3/R11). The positive control (R10) does not separate. **Not supported yet.** |
| **A″** | Unguarded EIG (P6) is more often confidently wrong under misspecification than baselines; the guards remove the excess at ≤ 10 pp efficiency cost. | Guards cannot bring the excess within 1 pp, or cost more than they save. | **Not supported so far.** In R3/R7–R9, P6 and P8 have similar or higher declared-wrong rates than the baselines; the guards (audit fraction + alarms) do not remove the excess. Pending: the calibrated-alarm rerun. |
| **B′(i)** | Deterministic inadequacy detection reaches ≥ 70% detection at ≤ 10% false alarms. | Below that. | **Falsified for the current diagnostic set** (residual weight, standardized surprise, PIT KS): 0.36 (microclimate) / 0.18 (systemlab) detection at 0.10 false alarms (`src/probe/diagnostics/calibration.json`). |
| B′(ii–iii) | Enumerative / LLM proposers recover the true structure class. | See H.10. | Not started (M12, gated). |

## What the evidence says so far

1. Where the truth is one of the registered hypotheses and effects are moderate
   (microclimate R2/R4, systemlab R1), cost-aware EIG pays off clearly: much higher
   correctness, or the same correctness at a fraction of the cost.
2. The e-process stopping rule is only valid when the truth is inside the
   rival's model class. Under misspecification, *every* policy declares
   wrong verdicts often, and the adaptive ones do so faster. This is the central
   safety problem and the next research priority (better inadequacy diagnostics,
   an audit-based posterior-predictive check, or tempering).
3. **The positive control separates adaptive EIG from the baselines, but the guard
   is expensive there.** On microclimate R10 (100 seeds), unguarded EIG/cost (P7) is
   0.79 correct vs 0.57 for the best baseline (P9 static maximin) and 0.87 for the oracle,
   while the guarded P8 is 0.63. The audit fraction ε = 0.15 costs 16 pp, above the
   ≤ 10 pp guard budget in A″. Options: a smaller ε, or ε that scales with alarm
   evidence instead of a constant. The gate is defined on P8, so it reports NO-GO.
   Systemlab R10 is a weak positive control (random already reaches 0.89).
4. (Superseded by 3.) The positive control R10 is not yet strong enough to "force" adaptivity: the
   static maximin design (P9) almost matches P8, while the oracle shows headroom
   (0.87 vs 0.63).

## Non-claims
No novelty is claimed for EIG, Bayesian experimental design, e-values or provenance.
