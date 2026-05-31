# Synchronized two-source coalition regime is empty in real games (null)

Date: 2026-05-31
Axis: synchronized-arrival JOINT coalitions (`BASELINE_JOINT_SYNC`)
Verdict: **FALSIFIED for the 2-nearest-source variant.** Surface to PI; do
not proceed to A/B (Rule 12/43 gate not cleared); Rule 37 axis-caution.

## What we built (Phase 4)
The live sync mechanism fired only ~once per 3 games. Hypothesis: the
proposer's `cheap_marginal_value` scores a bouncing single-source launch at
`-0.5·ships`, and `CHEAP_REJECT_THRESHOLD = -10.0` deletes any ~20+ ship
bounce before the chooser sees it — so the building blocks a coalition needs
were starved upstream. Fix: build two-source coalitions DIRECTLY from
`world` geometry inside the chooser (bypass `prerank`), default-OFF behind
`BASELINE_JOINT_SYNC=1`. For each defended target no single source can take,
pull the 2 nearest friendly sources and assemble a minimally-sized
same-arrival strike.

## What the data said
Firing-rate probe (`scripts/joint_sync_probe.py`, 3 games / 357 turns):
**0 sync coalitions emitted** — worse than the ~once-per-3-games baseline.

Gate-survival census (1 game, seed 100, 4516 target-considerations):
- `solo_skip = 4489` (99.4%) — at least one of the two nearest sources can
  already SOLO-capture the target (so no coalition is needed).
- `gate3 = 15` (0.3%) — even the two nearest COMBINED cannot beat the
  garrison (heavily defended enemy planets needing 3+ sources).
- `lt2_src = 12` (0.3%) — fewer than 2 eligible (non-reserved) sources.
- `scored = 0` — zero viable two-source coalitions reached the scorer all game.
  (`size_fail / eta_eq / horizon / gate1 / solve_none` all 0 — nothing even
  reached those later gates.)

## The real lesson
Prerank starvation was **not** the bottleneck. The bypass worked exactly as
designed (4516 targets/game evaluated, no upstream filter), and it revealed
that the addressable regime — "neither of the two nearest sources can solo,
but together they can, arriving the same tick" — is essentially **empty**.
When your nearest planet can't solo a target, your second-nearest almost
never closes the gap: either your nearest CAN solo it (99.4%), or the target
is defended enough that two combined still fall short (0.3%). The "two nearby
sources must combine for a capture neither can make alone" picture does not
occur in practice with this geometry/economy.

This also bears on the broader synchronized-arrival axis (commits bf0f740,
3b97181, bbf0f740-line): the empirical regime for SAME-TICK two-source
stacking looks near-dead, independent of how candidates are generated. That
is a PI-level call on whether to retire the whole line (Rule 37).

## Open question for PI
1. Retire the synchronized-arrival axis entirely (revert the sync mechanism),
   or keep the default-OFF generator as documented-dead scaffolding?
2. If the underlying intuition (small interior fleet compounding a frontier
   capture) is still wanted, the data points away from same-tick 2-source
   stacking and toward either (a) 3+ source coalitions for the 0.3%
   heavily-defended cell, or (b) the originally-falsified sequential-reinforce
   framing — both need explicit PI sign-off before any build (Rule 37/4).

## Repro
`python scripts/joint_sync_probe.py 3` (fire-rate). Gate census was a
throwaway `BASELINE_JOINT_SYNC_DIAG` instrumentation, removed after use.
