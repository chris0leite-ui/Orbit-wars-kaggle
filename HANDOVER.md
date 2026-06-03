# HANDOVER.md — next-session brief

## Mode

**Observation-driven iteration on a single strategy.** No parallel exploration.
One observation from the PI → one mechanism → one push.

## Strategy

`baseline_adaptive_k` — see `state/STRATEGY.md` for the full spec, the build
script, the smoke procedure, and the iteration protocol.

Read `state/STRATEGY.md` first thing every session.

## Live status

- **Latest submission:** `champ_adaptiveK_on.py`, sub **53324164** (2026-06-03
  10:37 UTC), bundle sha256 `6c0419dc20`. Predicted μ ≈ 1170 based on prior
  live settle of the identical agent (sub 53265480, μ = 1170.4).
- **TrueSkill warm-up reminder:** starts at μ ≈ 600 and climbs over ~24 h. Do
  not interpret the first few hours of leaderboard data.
- Read the rolling pair on demand:
  `kaggle competitions submissions orbit-wars | head -5`.

## Next action

Wait for the PI's first observation. Then run the loop in `state/STRATEGY.md` §
"Iteration protocol — observation-driven".

## Pointers

- `state/STRATEGY.md` — strategy, build, smoke, iteration protocol.
- `CLAUDE.md` — process rules (lean).
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
