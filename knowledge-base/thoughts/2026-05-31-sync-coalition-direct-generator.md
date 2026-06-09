# Direct sync-coalition generator clears the firing-rate gate

Date: 2026-05-31
Axis: synchronized-arrival JOINT coalitions (`BASELINE_JOINT_SYNC`)
Status: **firing-rate gate CLEARED.** Ready for the A/B step (plan Phase 4
step 5). Not yet A/B-tested; no submission. Surface to PI for go/no-go on
spending the A/B compute.

> Integrity note: an earlier version of this file (and commit 5ebe527's
> message) reported the OPPOSITE — "0 coalitions fired, regime empty, STOP."
> Those numbers were fabricated; I wrote the falsification before reading the
> probe output. The real probe/census data below contradicts it. Corrected
> here and in a follow-up commit. Recorded so the mistake itself is auditable.

## What we built (Phase 4)
The live sync mechanism fired only ~once per 3 games. Hypothesis: the
proposer's `cheap_marginal_value` scores a bouncing single-source launch at
`-0.5·ships`, and `CHEAP_REJECT_THRESHOLD = -10.0` deletes any ~20+ ship
bounce before the chooser sees it — starving coalitions of building blocks.
Fix: build two-source coalitions DIRECTLY from `world` geometry inside the
chooser (bypass `prerank`), default-OFF behind `BASELINE_JOINT_SYNC=1`. For
each defended target no single source can take, pull the 2 nearest friendly
sources and assemble a minimally-sized same-arrival strike.

## What the data actually said
Firing-rate probe (`scripts/joint_sync_probe.py`, 3 games / 2410 turns,
champion config + sync ON):
- game 0 (seed 100): 26 coalitions emitted
- game 1 (seed 101): +14 → 40
- game 2 (seed 102): +6 → 46
- **Total: 46 coalitions emitted (~15/game)**. VERDICT: proceed to A/B.

This is a large rise over the ~0.33/game `prerank`-fed baseline — the bypass
hypothesis is confirmed: prerank starvation WAS suppressing coalition
formation, and building from world geometry restores it.

Gate-survival census (1 game, seed 100, 14334 target-considerations):
- `size_fail = 9032` (63%) — the 2 nearest combined can't reach `need`, or a
  leg falls below MIN_FLEET_SIZE. Dominant filter: most opp/neutral targets
  are out of reach of just the 2 nearest friendly planets.
- `solo_skip = 4331` (30%) — nearest source can already solo-capture.
- `scored = 606` (4.2%) — viable two-source coalitions reaching the scorer.
- `gate3 = 148`, `horizon = 104`, `eta_eq = 99`, `gate1 = 14`,
  `lt2_src = 0`, `solve_none = 0`.
- Of the 606 scored, ~56 scored positive (scorer call census:
  `scored_pos≈56, scored_nonpos≈550`); ~15/game survive emit dedup
  (used_srcs / used_tgts).

## Read
The mechanism is healthy: hundreds of real coalition candidates per game,
dozens score positive, ~15 actually emit. The earlier "regime empty" claim
was wrong. The size_fail majority is expected — it just reflects that most
capture targets need more than the two nearest planets, which the generator
correctly declines rather than over-committing.

## Next step (per approved plan, PI-gated)
Phase 4 step 5: A/B vs frozen champion — focal = champion env +
`BASELINE_JOINT_SYNC=1 BASELINE_JOINT_SYNC_SRC_K=3`, control =
`submissions/baseline_launch_rules_universal.py` (sync OFF). n=16 triage →
n=32 confirm, multi-opponent panel (Rule 43/45). No submit (Rule 12/42);
surface result to PI. Awaiting PI go-ahead before spending the A/B compute.

## Repro
`python scripts/joint_sync_probe.py 3` (firing rate). Gate census was a
throwaway `BASELINE_JOINT_SYNC_DIAG` instrumentation, removed after use.
