# Phase C diagnostic: cands=5 root cause + bench

**Date**: 2026-05-18
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Trigger**: Yesterday's bench showed bundle losing 0-20 to v7_0 regardless
of `BUNDLE_ME_FOLLOWUP` mode. PI asked for diagnosis of WHICH layer is broken
(scoring weights / enumeration / search topology) before further chooser work.

## TL;DR

Default `BUNDLE_OWN_CANDS_PER_SOURCE=2` causes the beam search to converge
on the empty bundle even when better moves exist. Bumping to **5** unlocks
the search: bundle picks productive attacks, survives 60 turns longer
vs v7_0 (eliminated turn 178 instead of 121), and **stays within timing
budget** (0/178 turns over 800ms). Still loses 0-21 to v7_0 — the
chooser-axis fix gets us competitive but not winning. Probable next gap
is endgame play (bundle peaked at 13-7 mid-game, lost the late territorial
fight).

## Diagnostic methodology

`scripts/diag_single_turn.py` replays bundle vs v7_0 to turn 20 (the inflection
point per yesterday's audit), instruments `BundleEvaluator.score` to log every
candidate's score, re-scores v7_0's chosen move under the same `opp_overlays`,
and identifies which of three root causes is responsible:

  (a) SCORING — bundle considers v7_0's move but ranks its own higher
  (b) ENUMERATION — bundle never considers v7_0's move
  (c) SEARCH TOPOLOGY — bundle scores v7_0's higher but beam prunes it

The `--reuse-obs` flag pins the captured obs to disk so subsequent runs
with different knobs re-evaluate the SAME state, avoiding the
game-trajectory divergence that breaks naive cross-config comparison
(without this, the bundle's different moves produce different turn-20 states).

## Sweep results (turn 20, seed 42, mode=off, apples-to-apples scoring)

| horizon | cands | bundle picks | score range across 72 candidates |
|---|---|---|---|
| 15 | 2 (default) | empty | -35 to -23 (saturated) |
| 15 | 3 | empty | -74 to -23 |
| 15 | 5 | **attack** | -74 to +35 |
| 25 | 2 | empty | -295 to -283 |
| 25 | 3 | empty | -334 to -283 |
| 25 | 5 | **attack** | -334 to -95 |
| 30 | 2 | empty | -425 to -413 |
| 30 | 3 | empty | -464 to -413 |
| 30 | 5 | **attack** | -464 to -160 |

Pattern: **the "bundle picks attack vs empty" transition is gated by
`cands`, not `horizon`.** At cands=2 or 3, the search returns empty
regardless of horizon. At cands=5, it finds a positive-EV bundle.

This is root cause (c) SEARCH TOPOLOGY masquerading as (b) ENUMERATION.
v7_0's specific move (src 0 → P17, distance 48) is never enumerated by
bundle even at cands=5 (P17 is #3 closest), but with cands=5 the SEARCH
finds a DIFFERENT productive move from the same source that scores
above empty.

v7_0's actual move (src 0 → P17, 36 ships):
  - distance 48, log-speed 2.87 → 16.7 turn ETA
  - horizon=15 → P17 capture invisible to path integral (score 107 vs empty's 143)
  - horizon=30 → P17 capture visible (score -84 vs empty's -258), gap +174 in v7_0's favor

So bumping horizon ALSO helps (makes long strikes visible to the scorer),
but cands is the bigger blocker — without it the search never finds the
productive corner of the action space.

## Full-game bench (cands=5, default horizon=15, seed 42)

Bundle (cands=5) vs v7_0:
- **Eliminated turn 178** (0-21) — vs lite's 118, off's 121
- Bundle p50 / p95 / max: **752 / 756 / 760 ms** — 0/178 over 800ms
- v7_0 p50 / p95 / max: **529 / 1125 / 1399 ms** — 14/178 over 1000ms
- Bundle peaked at 13-7 mid-game (vs lite's 4-2 then collapse)
- Game ran 178 turns (vs lite's 118, off's 121)

The cands=5 fix is real: bundle survives ~60 more turns, stays within
its own budget, peaks WAY higher mid-game. But v7_0 still wins the
endgame.

## Why bundle still loses

Hypothesis (n=1, low confidence): bundle's mid-game peak (13-7) shows
the score function correctly values aggressive expansion, but late-game
play degrades. Possible causes:

1. **Endgame scoring weights**: planet_weight=5 + elimination_bonus=200
   may not correctly value LATE planet captures vs LATE planet defense.
2. **Opp model staleness**: as the game progresses, the predicted opp
   trajectory diverges more from reality, hurting bundle's late
   decisions.
3. **Beam saturation in territorial endgame**: when the board is full
   of owned planets, every launch has similar score (the saturation
   pattern we saw earlier at turn 20 with cands=2 may re-emerge with
   cands=5 at turn ~100 when state complexity grows).

The diagnostic script can be re-run at a late-game turn (e.g. turn 130
when bundle was 8-12 and slipping) to identify which.

## v7_0 timing finding (incidental)

v7_0 had 14/178 turns > 1000ms during this game. On our local CPU,
v7_0 is at the edge of the live-env actTimeout. Either v7_0 has its
own timing issues we haven't measured, OR our local CPU is materially
slower than Kaggle's compute. Bundle's clean 760ms max likely has
more headroom in production than the raw numbers suggest.

## Recommended next step (PI to ratify)

ONE focused experiment (not a menu):

**Make cands=5 the new bundle default**, re-run n=8 quick A/B vs v7_0
AND vs agents/baseline. ~30 min wallclock. If Wlo > 0.40 on EITHER,
submit. If both null, run the late-game diagnostic to confirm/refute
the endgame hypothesis above.

Per Rule 40 (modeling-correctness over restriction-tuning), bumping
cands isn't a band-aid — it's correcting the model's view of the
action space. The default was systematically myopic.

The me-followup work from Phase B remains valid (the mechanism does
what we designed), but its cost-benefit is still negative without
further optimization. Leave `BUNDLE_ME_FOLLOWUP=off` default.

## Friction logged (proposed addition to audit/friction.md)

`cands-per-source-2-saturates-search` — Bundle's default
`BUNDLE_OWN_CANDS_PER_SOURCE=2` causes the beam search to converge
on empty even when productive moves exist; bumping to 5 unlocks
productive search across all tested horizons. The oracle suite
masked this because oracles set cands=5 in their fixture. Phase A
oracles were testing the BETTER config; production was running
the worse one. **Rule:** when oracle fixture knobs diverge from
production defaults, the oracle results don't transfer to live play.

## Artifacts

- `scripts/diag_single_turn.py` — root-cause discriminator
- `audit/2026-05-18-bundle-cands5-profile.prof` — cands=5 full-game profile
- `audit/diag_turn20_obs.pkl` — pinned obs for reproducible re-scoring
