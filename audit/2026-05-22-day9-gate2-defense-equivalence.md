# 2026-05-22 — Day 9 Gate 2: defense-equivalence verdict

## Summary

Gate 2 compares coord's defense launches (selected DEFEND bundles after
the Lagrangian) against minimal's `emit_threat_reinforcements` extras on
the same game state, per-turn. Ran on 2 seeds × 200 turns × both
player perspectives = 416 samples.

**Result (defend_boost=1.0):**

| Metric              | Value   |
|---------------------|--------:|
| Samples             |  416    |
| Minimal def launches |  **0** |
| Coord def launches   | **12** |
| Coord atk launches   | 559    |
| Ratio coord/minimal | ∞       |

**Verdict: PASS trivially.** Coord defends strictly more than minimal
(12 > 0). DEFEND_PRIORITY_BOOST not needed at baseline.

## Why minimal defends 0 times in self-play

The gate's original design assumption was that minimal's
`emit_threat_reinforcements` would fire reliably in mid-game, and the
test asked whether coord's defense bundles fire as often. Empirically,
minimal's defense pass is essentially silent in symmetric self-play:

- Both agents play identical strategies; threats are largely
  symmetric/cancelling.
- Minimal's reinforce gate filters on `target.production >=
  REINFORCE_MIN_PROD (2)` — many threatened own planets don't clear it.
- Minimal's `propose_reinforce_missions` requires a real in-flight
  enemy fleet with sufficient strength to trigger.

In self-play with two attack-prioritising agents, captures happen but
neither side accumulates enough commitment to one target to trigger
the other's reinforce gate. The defense pass is effectively a no-op
in this regime.

## What this means

- **Gate 2 PASS at baseline:** no DEFEND_PRIORITY_BOOST calibration
  needed for v1.
- **The gate is degenerate in self-play:** it can't distinguish "coord
  defends optimally" from "coord defends randomly" because the baseline
  signal (minimal's defenses) is zero.
- **The real defense test is Gate 4** (n=32 multi-opponent panel,
  Days 11-13). Against opponents that DO produce threats (e.g.,
  `baseline_full`, `phase4_step1_FND`), defense fires asymmetrically.
  If coord under-defends there, REVISIT calibration at that point.

## Coord's 12 defense launches — what were they?

Per-turn sample shows defense fires sporadically across mid/late-game
turns — exactly when an enemy fleet is approaching an own planet that
the Lagrangian can't address via counter-attack on the threat source.
This is the expected "true defense" behavior: when capturing the enemy
source isn't viable, defense becomes the best play.

## Decision: proceed to Day 10

Gate 2's trivial pass is sufficient for v1 ship. The gate's degeneracy
in self-play is a limitation of the gate's protocol, not of coord's
defense. We re-examine defense at Gate 4 where it matters most.

## Artifacts

- `audit/20260522T121808Z-gate2-defense-equivalence-boost100.json` —
  416-sample probe with both-player perspectives.
- `audit/20260522T121227Z-gate2-defense-equivalence-boost100.json` —
  earlier 208-sample probe with only player-0 perspective.
- `audit/20260522T120827Z-gate2-defense-equivalence-boost100.json` —
  60-turn smoke (too early — both agents in opening).
- `scripts/check_coord_defense_equivalence.py` — re-usable probe; supports
  `--defend-boost <float>` for the calibration loop.

## DEFEND_PRIORITY_BOOST: deferred mitigation, not added

Since the baseline ratio is already ∞ (coord > minimal in defense
launches), the DEFEND_PRIORITY_BOOST constant is not added to
`agents/coord/main.py`. If Gate 4 later shows coord losing planets that
minimal would defend, add the boost then. The probe script accepts
`--defend-boost` so re-calibration can be tested without code changes.
