# 2026-05-23 — leaf-side value-head axis closed (post capture-fix)

Branch: `claude/strategy-framework-design-OyoYR`
Commits: `28ce9f3` (capture-classifier + projection-transfer fix),
`49c02dd` (today's audit confirming TIE).

## What landed

The `_per_seat_in_flight_credit` function inside
`lib/value_heads.py::_projected_totals` had two interacting bugs:

1. The "capture vs reinforcement" classifier always took the
   reinforcement branch because it checked WorldModel ownership AT eta,
   but WorldModel already resolves combat at eta — so the post-combat
   owner equals the attacker on every successful capture, and the
   capture branch was dead code in production play.
2. When a capture WAS correctly classified, the defender's projected
   production stream still received `λ × P × (T-step)` while the
   attacker also received `capture_weight × P × (T-step-eta)`. The
   `[eta, T]` window was double-counted across two seats.

Both fixed in 28ce9f3. Worked example: step=100, T=500, P=5, eta=8 →
attacker's V_diff goes from 0 (capture credit never fired) to +196.

## Why I re-opened the axis

The 5/19 audit closed leaf-side at "TIE, axis exhausted." 4 days
later, the fix landed and the worked example showed a large V_diff
shift on capture frames. I re-ran the panel expecting the fix to
unlock projected_sum's advantage. Pre-panel spot-check (4 seeds × 1
seat) showed 2 winner flips out of 4 and 80-120 turn divergences per
seed — a strong "this is expressive" signal that justified the
12-minute panel.

## Result

8 games × 2 focals: favor_v2 5/8 (62.5%), projected_sum_v2 5/8 (62.5%).
Identical winner pattern on all 8 seeds. Trajectories diverged
substantially (step counts off by 20-200 per seed) but placements
converged to the same outcome.

## Lesson

Rule 37 (3-variant axis cap) needs a corollary for fix-induced
re-litigation:

> When a fix lands on an already-exhausted axis, the axis stays
> exhausted unless the spot-check shows ≥1 placement flip AND
> PI explicitly re-opens the axis. A new fix is not a fresh slot.

What I should have done instead of the panel: a 2-minute unit-test
asserting that `_per_seat_in_flight_credit` shifts the chooser's
argmax on a synthesised capture frame. The full panel is overkill
because the question I was answering ("does the fix matter at the
chooser level?") could have been answered structurally without
playing 16 games. The panel only tells me "does the chooser-level
shift propagate to placements on this background?" — and the answer
has the same prior as the 5/19 result regardless of the fix.

## Status of the fix

The fix is real, regression-tested, committed (28ce9f3), and stays
on the branch. It does not change the agent's submission value
(favor head is still the v15 baseline; projected_sum head ties
favor). The fix is a permanent modeling improvement: any future
agent that uses `projected_rank_diff*` heads benefits.

## What's actually load-bearing for the next session

The sibling branch (`extract-physics-trajectory-Vjaz9`) shipped
`baseline_joint_aggr_consolidated_orbitfix.py` at μ=1165.4. v15
baseline (this branch's foundation) is at μ≈1119.6 — a 46-μ gap.
The leaf-side axis was never going to close that gap. Today's work
confirmed that with high confidence.

Next session, the productive direction is one of:
- merge / cherry-pick the sibling-branch orbital-safety stack
  (commit 38372f4 + the joint-aggr-consolidated chooser) and rebase
  this branch's value-head work onto that;
- OR work the chooser-side candidates the 5/19 audit listed
  (K-horizon sweep, opp-model strength, proposer dedup) on top of
  the v15 baseline, knowing the ceiling is still below sibling.

The first path is higher EV. The 46 μ gap is the open question, not
the value-head shape.
