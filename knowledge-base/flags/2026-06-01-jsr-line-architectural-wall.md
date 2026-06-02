# FLAG — jsr-line cannot beat champion (architectural wall)

**Date raised:** 2026-06-01
**Severity:** strategic
**Owner:** any branch picking up jsr-line work

## What

Every architectural mod to the jsr agent today loses 0/16 vs the
rolling champion locally, even when the mod improves over jsr itself
(best: addone v5 = +22pp vs jsr, but 0/16 vs champion).

## Implication

**Do not iterate on jsr internals expecting to beat champion.** The
load-bearing component making champion stronger lives outside what
jsr's chooser/aggression-handoff/value-head/opp-model can address.

This is consistent with the May 31 "Tier 2 falsified for chooser-budget
reasons" finding — the jsr stack is wallclock-tight and chooser-bound,
not value-bound. Adding more sophistication INSIDE jsr-line hits the
same wall.

## What to do instead

Three paths (PI to choose):
1. Apply jsr's learnings (add_one handoff, slot_res, joint_sync) to
   the CHAMPION lineage instead. New axis per Rule 37.
2. Public-notebook scan (Rule 22). What's the field doing?
3. Replay analysis: find a seed where champion beats addone-v5
   decisively, identify the load-bearing decision, target it.

## Counter-evidence that would clear this flag

- An ablation showing some jsr-internal component IS load-bearing
  in jsr-vs-champion (currently no component is — every bisect
  removed something that lost μ when removed, but adding combinations
  doesn't recover champion-level play).
- A different baseline ladder where jsr-line agents DO beat
  champion-like agents (would mean we're seeing rock-paper-scissors,
  not a wall).

## Tags

`jsr-line-cannot-beat-champion-axis`,
`bundle-name-collision-overwrites-imported-function`,
`closed-form-roi-myopic-about-cumulative-drain`
