# 2026-06-21 — PI replay observation: scattered attacks + greedy-grab-no-defense (seed 6013)

> Append-only (Rule 35). Transcribed from PI during replay-watching of the
> lead-then-collapse loss. Plain English (Rule 0).

## The observation (PI's words, lightly cleaned)

Watching seed 6013, 2P vs Producer V2, we are BLUE (the losing side), around step 40
of 117:

> We attack early and scattered in the loss — both in the upper-right corner where
> our attack fails, and in the lower-left corner where we capture all the neutrals
> greedily and are then not able to defend against the obvious attack from the
> opponent.

## What the replay shows at step ~40

- We (blue) have grabbed a spread of small planets across the lower-left and bottom
  (garrisons like 3, 9, 15, 15, 35) — greedy neutral expansion.
- Orange (V2) has massed a concentrated **120-ship fleet** near center plus several
  strongholds up top — a visible, building punch.
- We are dribbling small fleets up the right side at orange's defended planets;
  those attacks bounce (fail to capture/hold).
- Net: our ships are spread thin across many low-garrison planets with no reserve,
  while the opponent concentrates. When the punch lands, the spread folds and the
  production lead (we were 201/76 ahead at step 29) cascades away to a loss.

## Two facets, same root

1. **Scattered failing attacks (upper-right):** ships spent on captures that do not
   hold — the wasteful-attack / non-sticking-capture problem.
2. **Greedy neutral grab with no defense (lower-left):** every neutral taken, every
   source drained, nothing held back to meet the obvious concentrated attack.

Root cause (same as the 2026-06-20 thesis): we optimize captures/production without
pricing **the cost of leaving the source undefended against a concentrated enemy
force.** The leaf's threat model spreads the enemy's mass across all our planets, so
each reads a small hazard → consolidation never wins the comparison → we keep
grabbing and dribbling.

## Mapping to existing (default-OFF) machinery

The prior session built three levers for exactly this and left them OFF/unvalidated:

- `LR_GARRISON_FLOOR` — reserve enough garrison at a threatened source to win the
  defensive fight BEFORE candidates are built; only surplus may attack. (Targets the
  lower-left greedy-drain facet at the generation step.)
- `LR_NATIVE_BUILDER` — score the greedy plan-builder with the hold-aware native
  leaf instead of the producer ship-count scorer, so far thin grabs / exposed drains
  are never built. (Targets the upper-right scattered-attack facet.)
- `LR_NATIVE_THREAT_MAX` — leaf defense sees the worst single enemy stronghold, not
  the diffuse split. (Makes the chooser feel the concentrated punch.)

Next action: reproduce the 6013 + 6007 seat-0 losses, switch these on (alone and
combined), re-render, confirm the collapse is gone WITHOUT breaking the clean wins
(1127764379, 6031, 6007-seat1), send before/after to PI to watch.
