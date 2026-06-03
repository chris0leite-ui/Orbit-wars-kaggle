# HANDOVER.md — next-session brief

## Mode

**Observation-driven iteration on a single strategy.** No parallel exploration.
One observation from the PI → one mechanism → one push.

## Strategy

`baseline_adaptive_k` — see `state/STRATEGY.md` for the full spec, the build
script, the smoke procedure, and the iteration protocol.

Read `state/STRATEGY.md` first thing every session.

## Live status

- **Latest submission (#1):** `champ_computeByShips_on.py`, sub **53332500**
  (2026-06-03 15:11 UTC), bundle sha256 `53bf813b...`, 697 927 B. Adaptive K
  + compute_by_ships lever both ON. Predicted μ ≈ 1170 (parity with sibling
  per local n=16 A/B).
- **Backstop (#2):** `champ_adaptiveK_on.py`, sub **53324164**, live
  **μ = 1185.2** (our anchor — stays in the rolling pair).
- **TrueSkill warm-up reminder:** starts at μ ≈ 600 and climbs over ~24 h. Do
  not interpret the first few hours of leaderboard data.
- Read the rolling pair on demand:
  `kaggle competitions submissions orbit-wars | head -5`.

## Next action

**Next mechanism is queued:** large-idle-fleet spend-down — force a launch
from any planet exceeding ~200 ships, K-cap bypassed, nearest opp target.
Spec-only at this point; see `state/STRATEGY.md` § "Next mechanism" for the
design. Wait for compute_by_ships's live μ to settle (~24 h from submit)
before starting implementation, so we have data on whether compute_by_ships
moved the needle.

## Pointers

- `state/STRATEGY.md` — strategy, build, smoke, iteration protocol.
- `CLAUDE.md` — process rules (lean).
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
