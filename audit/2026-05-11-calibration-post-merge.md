# 2026-05-11 — post-merge calibration of σ-equivariance fixes

## Setup

Merged `origin/main` (b430b13, v3.4 submitted as #52556866) into branch
`claude/game-theory-strategy-analysis-0oH4N` (commit fb9fbc9). Clean
merge — no conflicts. Three σ-equivariance patches (commits 6c12b9f,
7b60938, 24bae06) integrate orthogonally with main's WorldModel-aware
`arrival_size` (commit 1f459a3) and LEADER_MULTIPLIER 4P spoiler.

Critical gate: v3 vs v3 16-seed self-play, 500 steps each.

  **POST-MERGE: 16/16 draws preserved. Cannot-lose lock holds.**

`pytest tests/`: 215 passed, 1 failure (`test_v3_snipe_frozen_bundle_replay_parity_100pct`)
— EXPECTED, since our σ-equivariance patches intentionally change v3's
tied-target picks, so the live #52544634 bundle no longer reproduces
bit-for-bit. The test serves its purpose (flagging the change); will
need updating when we re-submit.

## Calibration tournament

`v3_snipe` (with σ-equivariance patches) vs 5 opponents × 16 seeds ×
both seats = 32 games each. Tournament JSON at
`audit/tournaments/post-merge-calibration.json`.

| Opponent | P0 W/D/L | P1 W/D/L | Total W/D/L | W/D% |
|---|---|---|---|---:|
| v3 (self) | (16D from prior probe) | — | 0/16/0 | **100.0** |
| v1_orbitfix | 16/0/0 | 16/0/0 | 32/0/0 | 100.0 |
| baseline (Nearest Sniper) | 16/0/0 | 16/0/0 | 32/0/0 | 100.0 |
| random | 16/0/0 | 16/0/0 | 32/0/0 | 100.0 |
| roi (sibling) | 15/0/1 | 15/0/1 | 30/0/2 | 93.8 |
| **v2** | 7/0/9 | 10/0/6 | 17/0/15 | **53.1** ⚠ |

## Read

**Strict cannot-lose property at v3-class: confirmed.** Self-play
locks 100% draws after merge. The empirical verification of the
symmetric-game value theorem.

**Dominance over weaker classes: confirmed.** v3 beats v1/baseline/
random with zero losses (32/32 wins each). roi loses 30/32 — strong
but expected (roi is a simpler strategy in the same family).

**v2 result is the surprise.** 53.1% W/D = barely above the cannot-
lose floor against an OLDER baseline. Audit
`audit/2026-05-10-v2-strategy-mechanism-split.md` documented v3_snipe
vs v2 = 57.8% Wilson [45.6, 69.2] over 64 games. Our 53.1% (32 games)
falls within that confidence interval but at the LOW end. Possible
explanation: the σ-equivariance tie-break is deterministic and picks
a specific σ-paired target; that consistency may align with patterns
v2 happens to defend well. Pre-σ-equiv v3 had insertion-order tie-
breaks, which gave a slightly different (and v2-unfavorable) target
distribution.

This is NOT a regression of the cannot-lose property — v3 still wins
against v2 more than it loses (17 W vs 15 L), and ZERO draws shows
the σ-equiv lock specifically applies to v3-class self-play, not to
non-v3 opponents.

It IS a regression of the v2 *win rate* by a few percentage points
(53.1% vs 57.8%). Worth investigating; the σ-equiv fix may need a
secondary tie-break that doesn't make us this predictable against v2.

## Predictions vs results

From the plan:

| Pairing | Predicted | Observed | Status |
|---|---|---|---|
| v3 vs v3 | 16/16 draws | 16/16 draws | ✓ |
| v3 vs v2 | ≈ 58% (per audit) | 53.1% | ⚠ low end of CI |
| v3 vs v1_orbitfix | ≥ 95% | 100% | ✓ |
| v3 vs baseline | ≥ 98% | 100% | ✓ |
| v3 vs roi | ≥ 80% | 93.8% | ✓ |
| v3 vs random | ≥ 99% | 100% | ✓ |

5 of 6 predictions match. v2 is the one that needs follow-up.

## Not tested

**precision_v3** (origin/merge-precision-to-main). Cross-class test —
would tell us whether the σ-equiv lock generalizes to a different
strategy family. Skipped this iteration (would require cherry-pick of
26 commits including new agents/precision/main.py). Recommended
follow-up: pull precision_v3 binary into our branch as a calibration
opponent only (no merge of its code into v3's path).

**4P FFA**. Calibration matrix is 2P only. Main has 4P-specific
LEADER_MULTIPLIER which we inherit; should test via
`scripts/ffa_panel.py` against a panel of 3 backgrounds.

## Recommendations for next session

1. **Investigate v2 win rate.** Run a longer head-to-head (64 seeds)
   to tighten the confidence interval; if confirmed < 58%, isolate
   which σ-equiv patch causes it (revert each in turn).
2. **Cross-class test vs precision_v3.** Cherry-pick `agents/precision/`
   from `origin/merge-precision-to-main` for benchmark purposes only;
   do NOT merge precision's lib changes (would conflict with
   σ-equivariance).
3. **4P FFA panel test.** Run `scripts/ffa_panel.py --focal v3_snipe
   --background v2 v2 v2` + variants. Verify LEADER_MULTIPLIER
   integrates with our σ-equiv work.
4. **Re-submit v3_snipe**. Frozen-bundle replay parity will continue
   to fail until we ship a new submission with the σ-equiv changes
   bundled. Update test fixture after submission settles.

## Branch state

`claude/game-theory-strategy-analysis-0oH4N`, 19 commits ahead of
origin (1 merge + 18 prior). v3_snipe HEAD is functionally:
v3.4 + σ-equivariance(planner) + sym_hypot(missions) + score_round.

Calibration data at `audit/tournaments/post-merge-calibration.json`.
