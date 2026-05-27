# Empirical verification — reach-frontier doctrine

Date: 2026-05-27. Team: `ChrisLeiteScha`. Sub IDs: 52744856, 52894340, 52893236.

Games analysed: 92 (29 wins, 63 losses).

## Primary metric — hold_fraction medians (per capture)

| Outcome | Captures | Median hold_fraction |
|---|---:|---:|
| Wins   | 1551 | 0.881 |
| Losses | 1740 | 0.169 |

## Confirmatory — production-share medians (per game)

| Outcome | Games | Median share_focal | Σp̃·τ_me / (Σp̃·τ_me + Σp̃·τ_opp) |
|---|---:|---:|---:|
| Wins   | 29 | 0.677 | — |
| Losses | 63 | 0.190 | — |
| Separation | — | 0.488 | (gate: > 0.15 = confirmatory) |

## By game size

| Size | Wins captures | Wins-median | Loss captures | Loss-median |
|---|---:|---:|---:|---:|
| 2 | 1019 | 1.000 | 826 | 0.235 |
| 4 | 532 | 0.332 | 914 | 0.120 |

## By submission id

| Sub | Wins captures | Wins-median | Loss captures | Loss-median |
|---|---:|---:|---:|---:|
| 52744856 | 574 | 0.330 | 787 | 0.114 |
| 52893236 | 482 | 1.000 | 453 | 0.224 |
| 52894340 | 495 | 0.337 | 500 | 0.223 |

## Gate verdict

**STRONG** — wins_median=0.881 >= 0.70 AND loss_median=0.169 <= 0.45. Doctrine confirmed; proceed to chooser build.

Pre-registered thresholds (§9 of `knowledge-base/concepts/reach-frontier-doctrine.md`):

- Strong: wins ≥ 0.70 AND losses ≤ 0.45
- Weak-positive: wins ≥ 0.60 AND losses ≤ 0.50
- Falsified: wins < 0.60 OR losses ≥ wins

## Diagnosis (written after gate evaluation)

### Headline

On the combined 92-game sample, **hold_fraction discriminates winners
from losers cleanly**: wins-median 0.881, loss-median 0.169, gap 0.71.
Production-share separation between wins and losses is 0.488, well over
the 0.15 confirmatory threshold. The doctrine's central premise (§2.3 of
`knowledge-base/concepts/reach-frontier-doctrine.md`) — that the
production-time integral is the load-bearing structural signal — is
empirically supported on our ladder data.

### The 2P/4P split is the real story

The aggregate STRONG verdict is real but hides an important asymmetry:

| Sample | Wins-median | Loss-median | Verdict if run in isolation |
|---|---:|---:|---|
| 2P only (1845 captures) | **1.000** | 0.235 | STRONG |
| 4P only (1446 captures) | 0.332 | 0.120 | Below weak-positive floor |
| Combined | 0.881 | 0.169 | STRONG |

In 2P the median capture by a winner is held to game end (hf ≈ 1.0); in
4P even a winner holds only ~33% of remaining-game on the median
capture. This matches doctrine §8.3 ("4P kingmaker"): three concurrent
opponents recapture more, so τ_p^me is bounded by *minimum-ρ across
opponents* and that minimum is much smaller in 4P than 2P. Ordering is
correct in both formats (winners > losers), so the doctrine is
directionally valid in 4P, but hold_fraction *alone* is not a clean
discriminator at 4P sample sizes.

### Cross-sub heterogeneity is a composition effect (Rule 41 sweep)

Per-sub wins-medians (52744856 = 0.330, 52893236 = 1.000, 52894340 =
0.337) are explained entirely by 2P/4P mix:

| Sub | 2P games | 4P games | Wins-median (combined) |
|---|---:|---:|---:|
| 52744856 (team peak) | 12 | 20 | 0.330 |
| 52893236 (rolling) | 22 | 8 | 1.000 |
| 52894340 (rolling) | 10 | 20 | 0.337 |

Within each cell (sub × size), the within-cell gap is in the same
direction. No cross-sub quality effect — just composition. Rule 41
confound sweep clears.

### Home-planet segment is not biasing the result

t=0 segments (focal-owned at game start, i.e. home planet) are 92 out
of 3291 total segments — under 3%. Removing them shifts wins-median
trivially (2P wins-median t>0-only = 1.000 vs combined 1.000;
4P wins-median t>0-only = 0.314 vs combined 0.332). The discriminator
is genuinely about captured-then-held vs captured-then-lost.

### Comet expiry is small enough to ignore at this sample

End-reason distribution across all 3291 focal segments:
- flipped_to_opp: 2410 (73%)
- game_end: 768 (23%)
- planet_vanished (comet expired): 113 (3%)

3% planet-vanished is small enough that the artificial hf-deflation
discussed in script comments does not move any verdict. No separate
filter needed at this sample size.

### Implications for the chooser

The doctrine is confirmed STRONG enough to justify the reach-frontier
chooser build (Part C of the original plan), but with one
pre-registered caveat now activated:

1. **2P chooser is the priority.** The closed-form ρ-table + Hungarian
   assignment as sketched in doctrine §5 should land first and be
   evaluated on 2P-heavy panels.
2. **4P needs explicit kingmaker mitigation** before submitting any
   reach-frontier agent. Doctrine §8.3 flagged this; the empirical
   data confirms hold_fraction saturates differently in 4P. Two
   tractable options: (a) opponent-strength-weighted de-rating where
   ρ_opp = min over weaker opponents only; (b) explicit 4P branch that
   prefers contested-band captures when leading the FFA but switches
   to defensive mine-cell holding when trailing. Decision deferred to
   the chooser-design session.
3. **Gate for chooser submit:** the multi-opponent-panel + n≥32 A/B
   gates (Rules 43, 45) apply unchanged. Local 2P-panel performance is
   the strongest expected signal given the empirical asymmetry.

### Sample limitations

- 92 games is sufficient for medians on this n-of-captures (1500+
  per cell). Per-game variance is high; the by-size cells with n < 20
  games (52893236 4P = 8 games) are noisy and a 4P-only re-sample at
  n ≥ 30 would tighten the 4P verdict.
- No top-10 cross-validation in this pass — we measured our own
  ladder play (μ band 900-1150), not top-10's. If top-10's
  hold_fraction is *also* 0.3 in 4P, the metric isn't a discriminator
  *between us and top-10* in 4P specifically; that's a separate
  follow-up.
- The κ_p^me weighted_hold totals are reported per game but the
  median-of-share-focal aggregation may be more interpretable as a
  rank statistic; the 0.488 separation is robust either way.

### Status

Doctrine §9 verdict: **STRONG (combined) with 2P-confirmation + 4P-
caveat.** Doctrine pointer in CLAUDE.md should update from "empirical
verification queued" to "empirical verification STRONG (2P) /
directional (4P) — see audit/2026-05-27-hold-time-empirical.md."
