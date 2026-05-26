# 2026-05-26 — open questions for next session

## Q1. Submit `4ad192f` to recover floor, or pivot lineage?

**Option A — submit `4ad192f`** (fcaf414-equivalent strategic head):
- Live μ expectation: ~1100-1140 (matches fcaf414 lineage)
- Effect: drops `53024913` (μ=1135.4) from rolling pair. Pair becomes
  {53032723 μ=984, new ~1130}. Floor recovers from 984 → ~984 (no
  change — bad sub still there).
- Net: doesn't help unless we burn two submissions to evict 53032723.

**Option B — pivot to `baseline_ev_per_ship`** lineage:
- Live μ proven at 1135.4 (sub 53024913)
- Effect: same floor problem — submitting it AGAIN drops the older half
  (53024913 itself!). Pair becomes {53032723 μ=984, new ~1135}.
- Iterate trickle-reduction from this base. Per-ship sort already in.

**Option C — submit TWICE today** to fully evict 53032723:
- Burn 2 of remaining 5 daily slots
- Floor recovers fully to whatever the better-of-two settles at
- Risky if either submit is below 984

Which?

## Q2. The trickle-launch problem itself

The original PI question was about small waste fleets. We haven't
solved it; we've moved code around it. Should the next session attack
the trickle directly?

Possible attacks:
- Add `Δ > c · ships_launched` threshold to chooser (per-ship efficiency
  gate) — like `BASELINE_SORT_BY_EV_PER_SHIP=1` but as a HARD gate not
  a sort.
- Tune rollout K-step depth (longer rollout might show waste consequences)
- Add a "concentration term" to the leaf rewarding `max(my_planet_ships)`
  (the user's "bundle the ships" intuition from this session)

## Q3. Should we resubmit baseline_ev_per_ship as a "safety floor"?

If 53024913 is about to drop out (next submission evicts it), do we
RESUBMIT it once to keep its μ=1135.4 in the rolling pair? That'd be
a defensive move costing 1 slot.

## Q4. What's the right N for panel-test-before-merge?

Today's n=2 panels were enough to detect the unified-favor regression
(50% 2P, 12.5% 4P obviously bad) but not enough to distinguish
fcaf414-equivalent (50%/50%) from "maybe better." Should we move to
n=5 or n=8 as the default before merging any value-head change?
Cost: ~50 min for n=8 panel vs ~12 min for n=2.
