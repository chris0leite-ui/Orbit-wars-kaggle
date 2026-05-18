# Phase E Phase 1 post-mortem — joint coordination fires but isn't picked

**Date**: 2026-05-18
**Branch**: `claude/ml-competition-strategy-PFhzM`
**Trigger**: Post-fix Phase 1 A/B was NULL (identical to cands=5-only
baseline: 13/16 vs v7_0, 2/16 vs baseline). PI ratified diagnosis of
WHY before deciding next move.

## TL;DR

Joint coordination mechanism (scorer bonus + frontier seeding, fixed for
distinct-source requirement at `9cbcc8f`) **fires often** in real games
(seeds on 26% of turns, detector triggers on 22% of turns) but the
**chooser practically never picks a joint** (1 of 283 turns = 0.4%
pick rate). The joint bonus (`0.5 × value`) isn't enough to outweigh
the alternative — solo capture of a closer / easier target scores
higher on the underlying path-integral and the small bonus doesn't
close the gap.

Phase 1 is structurally a no-op for bundle's current matchups. It
ships as **default-off code** (env vars `BUNDLE_JOINT_BONUS=0.0`,
`BUNDLE_JOINT_SEEDS=0` preserve prior behavior exactly); no revert
needed.

Rule 37 counter: variant #1 NULL on coordinated-ROI axis. Two more
nulls trigger axis escalation.

## Diagnostic procedure

`scripts/diag_joint_firing.py` monkeypatches the bundled file's
`BundleEvaluator._detect_joint_captures`, `BundleSearch._enumerate_joint_seeds`,
and `BundleSearch.search` to count:
1. Seeds produced per turn (search-side joint enumeration output)
2. Detector hits per turn (scorer-side joint detection on any
   candidate bundle)
3. Whether the CHOSEN bundle (what bundle emits as action) contained
   a joint

Run: `BUNDLE_JOINT_BONUS=0.5 BUNDLE_JOINT_SEEDS=10 python
scripts/diag_joint_firing.py --seed 42 --opponent baseline`.

One game: bundle vs baseline, seed 42, 283 turns.

## Results

| Metric | Count | % of turns |
|---|---|---|
| `_enumerate_joint_seeds` produced ≥1 seed | 74 | 26.1% |
| `_detect_joint_captures` found ≥1 joint in some scored bundle | 64 | 22.6% |
| **CHOSEN bundle contained a joint** | **1** | **0.4%** |
| Total seeds generated across all enum calls | 311 | — |
| Total scorer calls | 30,440 | — |
| Scorer calls that detected ≥1 joint | 7,389 | 24.3% |
| Pick rate (chosen-is-joint / detect-fires) | 1 / 64 | **1.6%** |

## Why joints fire but aren't picked

**Joint candidates have inherently lower base score** than the
alternatives the chooser sees:

1. Joint targets (high-defender enemy planets, by `_enumerate_joint_seeds`
   selection criterion: `targets.sort(key=lambda t: -int(t.ships))[:5]`)
   are typically the FAR enemy strongholds. Long travel time → fewer
   turns of post-capture ownership within `horizon=15` → lower
   path-integral production credit.
2. Two-fleet ship cost is large. A 30+30 ship joint commits 60 ships
   while a solo half-ratio launch at a nearby target commits ~10.
   The opp_overlay-applied scorer sees the ship_delta loss
   immediately and penalizes the joint by ~50 ship-equivalents.
3. The joint bonus formula:
   `0.5 × (production × (horizon - arrival_turn) + planet_weight)`
   At horizon=15, prod=2, arrival=10 → bonus = 0.5 × (2×5 + 5) = 7.5.
   That's ~10-15% of the typical score range (40-60 points). Not
   enough to flip the argmax when the solo alternative has a
   30-point base-score advantage.

The mechanism is correctly DETECTING joints. The chooser is correctly
RANKING them. The ranking just doesn't favor joints in this game's
geometry.

## What would change the picture

Hypothetical knob tunes (not pursued — Rule 37):

- `BUNDLE_JOINT_BONUS=2.0`-`5.0` would close the gap, but at the cost
  of over-prioritizing far high-defender targets and likely failing
  to defend / expand correctly. Trades one wrong policy for another.
- Wider `joint_seeds` (top-10 instead of top-5) wouldn't help because
  the detector / chooser logic is unchanged; just more candidates
  scored, same picks.
- Joint enumeration of low-defender targets (where joints would just
  over-commit) is anti-strategy.
- Scope-narrow the joint detection to specific game contexts
  (early-game, dominance, etc.) — would add complexity without
  evidence that contextual joints win.

The data says joint coordination is the wrong mechanism for the
bundle-vs-baseline ceiling. Phase 0's 21.3% bounced_enemy bucket
is better addressed by **preventing** the bounces (Phase 2 bounce
penalty) than by **coordinating** them into joint captures.

## Connection to Phase 0

Phase 0 measured 368 bounced_enemy fleets across 16 games. We
hypothesized many of these were "would-be joint partners." This
diagnostic says NO — the joint enumeration considers them, the
scorer detects the joints, but the chooser ranks solos at other
targets higher every time.

The bounces are not "missed joints." They are "the chooser launched
too-small ships at an over-defended enemy planet, in a state where
the alternative was no-op or smaller solos elsewhere." Phase 2's
bounce penalty (cost-of-launching with insufficient ships) addresses
this directly: it pushes the chooser away from solo bounces toward
either empty bundles or correctly-sized solos.

## Phase 1 disposition

- Commits `bd2219b` + `9cbcc8f` stay in the tree.
- Code is dormant by default (env vars `BUNDLE_JOINT_BONUS=0.0`,
  `BUNDLE_JOINT_SEEDS=0`). Existing 8 bundle oracles continue passing.
- O-J1 + O-J2 oracles remain in `tests/test_bundle_oracles.py` —
  they document the mechanism's correctness for the synthetic state
  it WAS designed for, and guard against future regressions if joint
  coordination is revisited.
- Move to Phase 2.

## Re-evaluation triggers (defer Phase 1 revisit until)

- Phase 2 ships, A/B improves vs baseline. Re-measure joint
  pick-rate to see if a healthier base policy makes joints
  competitive.
- Future opponent class where joints are uniquely valuable
  (e.g. a 4-player FFA scenario with high-defender central
  planets).
- Learned value head (foamy-pondering Direction 1.A) shipped —
  may rank joints differently than the hand-tuned scorer.

## Artifacts

- `scripts/diag_joint_firing.py` — reusable joint-fire-rate probe.
- This audit.
- `audit/tournaments/20260518T173850Z.json` — post-fix Phase 1 A/B
  raw data (matches cands=5-only baseline at 13/16 + 2/16).
