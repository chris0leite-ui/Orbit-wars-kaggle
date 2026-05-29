# Is the perf chain neutral, or is the strategy axis genuinely closed?

**Session:** 2026-05-29 PM, claude/game-theory-winning-strategy-SEU7P
**Status:** open question for next session

## The data point

n=48 subprocess-isolated A/B, `baseline_pv_eta_seu7p.py` (SEU7P
perf chain + cherry-picked PV_ETA + preserved wait-grid) vs
`baseline_pv_eta_anchor_1163.py` (frozen µ=1163.5 ladder peak,
pre-perf-chain):

- **Combined: 25/48 = 52.1%, Wilson 95% [0.383, 0.655]**
- Per-seat: P0 = 9/24 = 37.5%, P1 = 16/24 = 66.7% (30-pp asymmetry)

Rule 45 says no submit. The bigger question is what the data
*means* for next session's branch selection.

## Two readings

**Reading A — "perf chain is neutral, axis still has room"**
The CI spans both sides of parity; we cannot distinguish lift from
regression at n=48. The seat asymmetry (P1 dominates) is a clue
that the perf chain's extra compute helps reactive moves more than
opening moves. Maybe the gating is wrong, not the strategy. Next-
step: instrument WHERE the perf chain spends its extra 200 ms and
see if the headroom is going to candidate breadth (good) or
duplicate scoring (wasted).

**Reading B — "axis is genuinely closed"**
Per `audit/friction.md` 2026-05-29 line 129
(`chooser-and-proposer-axes-both-saturated-this-branch`): four
consecutive falsifications today on SEU7P across three axes (H41
floor=50; level-0 perf-chain; level-1 JOINT-expanded; level-2 H44
Phase 3a). My n=48 is the fifth consecutive null/parity on a
strategy-axis attempt. Rule 37 says ≥3 consecutive variants in the
same axis is closure; we're well past that. The right move is to
pivot off SEU7P to hqNVM (MLP filter — live champion µ=1109.9) or
btjeK (chooser-sizing) and not spend another session here.

## The asymmetry as evidence

The 30-pp P0/P1 split is striking and the obvious interpretation
(P1 reacts to P0's commit; more compute helps reaction more than
initiation) is consistent with Reading A. But it could equally well
be seed-specific: 24 seed-seats per side is still small, and the
seeds used (0..23) are not stratified by geometry archetype. Re-
running on the 128-seed geometry-stratified panel
(`data/seed_panel_128.json`, `audit/2026-05-18-seed-panel.md`)
would resolve the seed-variance question — but only after deciding
whether SEU7P is worth that compute.

## What's tractable in one session

- Re-run n=48 on geometry-stratified subset (4 archetypes × 4
  seeds = 16 seeds × 2 seats = 32 games per archetype slice;
  ~40 min per slice). Tells us whether the seat asymmetry is
  archetype-driven.
- Add a per-turn compute trace to the candidate bundle: where do
  the extra 200 ms go? If it's "more candidates scored at the
  same depth," that's lift potential not yet captured. If it's
  "same candidates scored deeper," there's no lift to chase.
- OR: pivot to hqNVM, leave SEU7P at "parity with PV_ETA, no
  submit," and let it cool.

## Recommendation for next session

Treat as undecided. Default to Reading B (pivot) per Rule 37,
unless the per-turn compute trace surfaces a clear "candidate
breadth was bottlenecked, the chain unblocks it" signal. The cost
of pivoting wrong is one session; the cost of grinding a closed
axis is many sessions of nulls.
