# Composite value head clears the v9_scavenge ceiling

Date: 2026-05-17 (afternoon)
Branch: claude/audit-workflow-performance-btjeK
Source data: `audit/replays/composite-ab-2026-05-17.md`

## The number that matters

**Composite head 30/32 = 93.8% (Wlo=0.799) vs v9_scavenge (μ=1119.9).**

v9_scavenge is our team peak — the strongest agent we ever shipped
to the live ladder. The 2026-05-17 fleet-efficiency negative-result
session burned 7 variants (v21/v21_a/v21_ae/v21_solo + v22 + v23 at
two windows) trying to lift past v9_scavenge / v15. All failed,
range 15.6%–31.2%. Durable lesson at the time: *"Next iteration
must be a wholesale architectural change — different value head AND
different proposer AND different chooser — not a fix on top of v15."*

Composite-on-baseline is the wholesale change. Different value head
(per-fleet capture/waste credit, not F1+F2 ship/prod-delta), same
proposer + chooser + opp model as the v15-parity baseline. And it
works. This is the first lift past the v9_scavenge ceiling in a
session.

## Strategic implication

The previous attempts (v17/v18/v20/v21/…) modified the chooser or
proposer while keeping the favor (F1+F2) leaf. All regressed because
v15's chooser is co-tuned to favor. **Replacing only the leaf was
never tried before this session** — earlier composite work (iter_v1,
sub 52661990 at μ=1034.7) was composite on the v7_0_drop_one chooser,
which is a different stack entirely.

The win pattern:
- Other chooser changes (v17-v23) tried to fix v15's behaviour while
  keeping its leaf. Co-tuning broke; agents lost 25-35pp.
- The leaf swap keeps the chooser's calibration but changes the
  reward gradient. The chooser now finds different optima — and
  apparently better ones in 2P.

Open question: does this generalise to 4P, where favor's sum-of-opps
matters? Step 2 of the current plan addresses this by routing 4P games
to favor, keeping composite for 2P only. We have not measured
composite's 4P performance.

## Why this isn't yet a submission

Two blockers:

1. **Wallclock**: max turn-ms 1292 > 1000ms env budget. The chooser
   thinks it has 600ms; the validate cap probe doesn't measure leaf
   cost; composite leaves at ~2-5ms each blow the budget on busy
   turns. **Fix: instrumented probe in `affordable_validate_cap`.**

2. **4P semantics**: composite collapses non-me planets into one
   "enemy" bucket. In 4P that loses the sum-of-opps signal favor uses.
   ~36% of ladder games are 4P. **Fix: 4P→favor dispatch in
   `favor_composite`.**

Both fixes are small (~20 LOC each). After they land + re-verify,
the agent is submission-ready — modulo PI sign-off (Rule 1).

## What this changes about the roadmap

Previous roadmap (`audit/2026-05-17-fleet-efficiency-negative-result.md`
+ `knowledge-base/concepts/v8-analytic-architecture-state.md`):
- "Next iteration must be wholesale architectural change"
- Candidates: portfolio search, IL from top-10, 4P-specific chooser

Today's evidence: **wholesale leaf-only change works.** No portfolio,
no IL, no 4P-specific anything — just a different value function with
explicit waste-attack and capture-success terms.

This doesn't invalidate the IL / portfolio paths — those remain the
next levers if composite-on-baseline plateaus. But it pushes them
out by one iteration: ship composite first, measure live ladder
μ, decide IL/portfolio based on the residual gap to top-5%.

## How PI grades this

When the timing + 4P fixes land:

```bash
python -m pytest tests/test_baseline_*.py \
  tests/test_episode_postmortem_comet.py -q
# expect 31+ green

BASELINE_VALUE_HEAD=composite python fast.py eval agents/baseline \
  --vs /tmp/v15_resurrect/main.py --max-seeds 32 --workers 6
# expect 65-75% point estimate, max turn-ms < 1000
```

If both hold, recommend submission. The rolling-last-2 means a fresh
submit evicts v20 (μ=1094.2); we'd have v15 (1108.4) + composite as
the rolling pair. Live μ predicted in the 1100-1140 range based on
the 67-94% panel; outside chance to clear v9_scavenge's 1119.9 peak.
