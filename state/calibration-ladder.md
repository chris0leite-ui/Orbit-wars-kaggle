# state/calibration-ladder.md — predicted-rank vs actual-rank

> Code-comp form of the s6e5 OOF/LB anchor table. Each new candidate
> agent gets a predicted-rank-bracket logged BEFORE submission; after
> Kaggle returns the tournament rank, the delta is logged here. Rule
> 26 calibration loop. Drift ≥1 bracket on consecutive submits fires
> the strategy-critic-loop (Rule 14).

| Date       | Agent slug                              | Predicted μ (or Δμ vs prev) | Actual μ | Δ vs predicted | Notes |
|------------|------------------------------------------|------------------------------|----------|----------------|-------|
| 2026-05-10 | shipped-baseline (#52497828)             | starting μ₀=600              | 303.2    | -297 (live ladder is hostile to greedy) | Calibration anchor; aim-at-current-position kills fleets to orbit drift + sun. |
| 2026-05-10 | v1_orbitfix (#52507539)                  | Δμ +200 to +400 vs baseline  | 508.1    | +205 (mid-range of prediction) | Orbit-aware lead + tiebreak randomisation. Δ vs baseline matches the local 40/40 sweep. |
| 2026-05-10 | v1.1_orbitfix_arrival_size (#52509319)   | Δμ +50 to +200 vs v1 (local 85% vs v1) | PENDING  | TBD            | Production-aware sizing for enemy targets; +30% local ablation lift. |
| 2026-05-10 | (LOCAL ONLY) simple/roi                  | Δμ +200 to +500 vs v1 (8-seed local 100% / 16-of-16 vs v1_orbitfix; 97% mean panel WR) | not submitted | TBD | `roi` strategy = argmax `target.production / dist`. Same DEFAULT_MECHANISMS stack as v1.1; only the target-score differs. Submission deferred until (i) 32-seed confirmation pulls Wilson lo ≥0.6 vs v1, AND (ii) v1.1 live μ has settled (rolling-last-2 economy). audit/tournaments/20260510T123059Z.json. |
| 2026-05-10 | (LOCAL ONLY) simple/roi (32-seed confirm) | Δμ +200 to +500 vs v1 (32-seed local 100% / 64-of-64 vs v1_orbitfix; 97.1% mean panel WR) | not submitted | TBD | 32-seed confirmation of the 8-seed result. Wilson lo on 64/64 ≈ 0.94 — submission gate (i) clears. Now blocked only on (ii) v1.1 μ settling. audit/tournaments/20260510T140907Z.json. |
| 2026-05-10 | v1.2 simple/roi (#52518060)              | Δμ +200 to +500 vs v1 (so live μ ≈ 700-1000); reached after v1.1 settled at μ=597.4 | PENDING (validating)  | TBD            | Production/distance ROI targeting; same DEFAULT_MECHANISMS as v1.1. Submitted 14:59 UTC after Kaggle 503-OAuth flapped (one retry). Rolling-last-2: [v1.1 (597.4), v1.2/roi (PENDING)]; v1 (568.0) evicted. |
| 2026-05-10 | (LOCAL ONLY) simple/production           | Δμ +50 to +250 vs v1 (8-seed local 69% / 11-of-16 vs v1_orbitfix; 75% mean panel WR; 32-seed: 67.7% mean WR, 52% / 33-of-64 vs v1)  | not submitted | TBD | argmax `target.production`, tiebreak distance. 32-seed regresses below `roi` decisively; not a submission candidate at this point — `roi` dominates it. Same submission gate as `roi`. |
