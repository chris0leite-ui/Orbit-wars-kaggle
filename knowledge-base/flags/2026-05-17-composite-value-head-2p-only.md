# FLAG — `composite_capture_value` is 2P-only semantics

Date: 2026-05-17
Filed-by: claude/audit-workflow-performance-btjeK
Touches: `agents/baseline/value.py`, `lib/value_heads.py`

## What

`lib.value_heads.composite_capture_value` does not differentiate
opponents. All non-me planets collapse into one "enemy" bucket; the
function asks only `pred_owner == my_id` to decide capture.

Meanwhile `agents/baseline/value.favor` distinguishes:

- 2P → `opp_agg = max(opp_ships, opp_prod)` — there is only one opp
- 4P → `opp_agg = sum(...)` — capturing a *weak* opp's planet still
  produces full credit, because we're below the sum-of-3

These are *different value philosophies*. Composite uses linear time-
remaining; favor uses pv-discounted production.

## Why this is a flag

We wired `favor_composite = composite_capture_value` into the
baseline chooser via env var `BASELINE_VALUE_HEAD=composite`. The
dispatcher returns `favor` by default to preserve the v15-line
behaviour; `composite` is opt-in for A/B testing only.

If anyone defaults the env to `composite` and submits, the 4P games
(~36% of ladder) will lose the sum-of-opps signal that gives
v15 its 4P performance. **Do not submit composite-default without
a 4P-aware variant.**

## What to do before defaulting composite

1. Either: add a 4P branch to `composite_capture_value` that sums
   capture credit across distinct opp seats (mirroring `favor`'s
   `sum-of-opps`).
2. Or: have the chooser route to `composite` only when `num_seats == 2`
   and to `favor` in 4P.
3. Either way: re-run `fast.py eval agents/baseline --vs-panel default
   --require-h2h agents/baseline` with `BASELINE_VALUE_HEAD=composite`
   set in env. PASS required on 4 opponents (champion + panel).

## Prior live evidence

`iter_v1` (sub 52661990, 2026-05-14) was the v7_0 chooser + composite
head + PV_GAMMA=0.99. Ladder μ 1034.7 vs v15 1108.4. Composite head
alone produced a ~74μ regression on v7_0. The hypothesis: works in
2P, loses in 4P. The replay-mine bucket breakdown on iter_v1 would
prove or refute this — currently unexplored.
