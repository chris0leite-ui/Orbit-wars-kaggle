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
