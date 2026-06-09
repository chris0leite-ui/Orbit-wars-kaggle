# Empirical verification — reach-frontier doctrine

Date: 2026-05-27. Team: `reach_frontier`. Sub IDs: 2026-05-27-rf-v2-triage.

Games analysed: 32 (0 wins, 32 losses).

## Primary metric — hold_fraction medians (per capture)

| Outcome | Captures | Median hold_fraction |
|---|---:|---:|
| Wins   | 0 | None |
| Losses | 498 | 0.240 |

## Confirmatory — production-share medians (per game)

| Outcome | Games | Median share_focal | Σp̃·τ_me / (Σp̃·τ_me + Σp̃·τ_opp) |
|---|---:|---:|---:|
| Wins   | 0 | None | — |
| Losses | 32 | 0.151 | — |
| Separation | — | -0.151 | (gate: > 0.15 = confirmatory) |

## By game size

| Size | Wins captures | Wins-median | Loss captures | Loss-median |
|---|---:|---:|---:|---:|
| 2 | 0 | None | 498 | 0.240 |

## By submission id

| Sub | Wins captures | Wins-median | Loss captures | Loss-median |
|---|---:|---:|---:|---:|
| 2026-05-27-rf-v2-triage | 0 | None | 498 | 0.240 |

## Gate verdict

**NO_DATA** — insufficient captures.

Pre-registered thresholds (§9 of `knowledge-base/concepts/reach-frontier-doctrine.md`):

- Strong: wins ≥ 0.70 AND losses ≤ 0.45
- Weak-positive: wins ≥ 0.60 AND losses ≤ 0.50
- Falsified: wins < 0.60 OR losses ≥ wins
