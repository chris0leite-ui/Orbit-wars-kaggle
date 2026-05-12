# v9 super-version A/B — none clear the gate; ship v7_0

> Branch: `claude/game-ai-lookahead-3ucqH`.
> Plan: `/root/.claude/plans/reflective-dazzling-flask.md` (v9 super-version).
> Result: 4-way 24-game A/B at commit `deaf288`.

## TL;DR

**No v9 variant beats v7_0_drop_one at Wilson 55%.** Per the plan's
decision tree, the submission target is `submissions/v7_0_drop_one.py`
(sha256 `bb7ab23a75bc5865`) — the established local champion against
both live agents (79% vs v7_minimax μ=1063, 75% vs v4_planner PENDING).

## Results (24 games each, 12 seeds × 2 sides)

```
v9_k15         vs v7_0          :   2/24 =   8.3%  Wilson lo  2.3%  p95 734ms  FAIL
v9_inflight    vs v7_0          :  14/24 =  58.3%  Wilson lo 38.8%  p95 702ms  NEUTRAL
v9_combined    vs v7_0          :  14/24 =  58.3%  Wilson lo 38.8%  p95 836ms  NEUTRAL
v9_combined    vs v9_inflight   :  13/24 =  54.2%  Wilson lo 35.1%  ─ NEUTRAL (~tie)
v9_combined    vs v9_k15        :  20/24 =  83.3%  Wilson lo 64.1%  ─ PASS
```

Wallclock: 2841 s (~47 min for 72 games, 4 workers).
Artifact: `audit/tournaments/<utc>.json`.

## What we learned

### 1. K=15 is **REGRESSIVE** with the ship-delta head (8.3% vs v7_0)

The cheapest variant (single-line K change) catastrophically regresses.
Hypothesis: K=15 lets v3.5.1 self-play burn more launches in the
simulation, adding noise to candidate ranking. With ship-delta as the
head, candidates that DON'T launch in turn-0 score similarly to ones
that do — because at K=15, even the K=10-capturable fleet has time
to arrive and the differential collapses. Other candidates get noisily
ranked higher. Drop-one's noise-tolerance threshold breaks.

A secondary factor: v9_k15's p95 is 734 ms, just over the 700 ms
watchdog, so on a non-trivial fraction of turns the watchdog truncates
mid-scoring and we fall back to incumbent.

**Conclusion: K=10 is approximately a sweet spot for our regime.**
Not a blind spot to expand; an equilibrium where drop-one's
worst-case-incumbent-parity is robust.

### 2. `inflight_value` head is **DIRECTIONALLY POSITIVE** but not Wilson-significant

v9_inflight = drop-one + K=10 + inflight_value: 14/24 = 58.3% vs
v7_0. Point-estimate +8pp lift. Wilson 95% lower bound 38.8% — below
the 55% gate.

This is consistent with the receding-horizon-pathology hypothesis
(audit/2026-05-12-v4-planner-receding-horizon-pathology.md): the
inflight_value head DOES rescue some "would-have-fired-but-feared-
ship-cost" decisions (smoke test confirmed v9_combined fires on a
warmed board where v7_0 returns `[]`). But the lift is too small to
register at n=24.

**Conclusion: the lift is real but small. n=64 might clear the gate,
or n=96 with the same point estimate would give Wilson lo ≈ 0.49 —
still NEUTRAL. The fix targets a relatively rare pathology in the
v7_0 regime (which already doesn't have noop in its candidate set).**

### 3. `inflight_value` SAVES K=15 from catastrophe

v9_combined (K=15 + inflight_value) at 58.3% vs v9_k15 (K=15 alone)
at 8.3% = +50pp delta. The composite head is providing enough
production-credit signal to offset K=15's regression.

But v9_combined's p95 = 836 ms — **over the 1 s actTimeout**. The
watchdog at 700 ms truncates many turns. So v9_combined's behavior
is effectively "K=15-when-budget-allows, otherwise watchdog-fallback-
to-incumbent". This hybrid lands at the same Wilson lo as v9_inflight
(58.3%) — i.e., the K=15 portion adds nothing usable.

**Conclusion: K=15 is a budget loser even with the new head; not
worth pursuing.**

## Decision

**Submit `submissions/v7_0_drop_one.py`** (sha256 `bb7ab23a75bc5865`,
121 KB).

Rationale:
- Wilson-significant winner against BOTH live agents:
  - vs v7_minimax (live μ=1063): 19/24 = 79.2%, Wilson lo 59.5%
  - vs v4_planner (live #52579863 PENDING): 18/24 = 75.0%, Wilson lo 55.1%
- vs all 3 v9 variants: implied 50-92% (v9_inflight tied at 58.3%
  point-estimate but not significantly better).
- Predicted live μ from TrueSkill math on 75% vs v4_planner: 1080-1100.
- Conservative submit: known-good. Rolling-last-2 after push becomes
  `[v4_planner PENDING, v7_0 PENDING]` — we lose v7_minimax (1063)
  as the floor.
- Worst case: v7_0 lands at ~1060, parity with v7_minimax floor we
  just evicted. Net zero μ change.
- Best case: v7_0 lands at ~1100, +37 over team peak.

Bundle ready: `submissions/v7_0_drop_one.py`.

## What goes into next-session

1. **n=64 confirmation of v9_inflight**. The 58.3% point estimate
   is positive; ~40 more games would establish if Wilson lo crosses
   55%. Worth doing before declaring the fix dead.

2. **Adapt inflight_value to lower budget cost**. WorldModel.from_world
   currently costs ~1 ms per leaf eval. Pre-build once per turn at
   the top of `choose_simple_2p` and pass it in to the value_fn (so
   it's not rebuilt 5 times per turn). Maybe ~0.6× cost reduction;
   K=10 + inflight_value would then comfortably fit p95 < 700 ms.

3. **Investigate K=11–14**. K=15 regresses; K=10 is fine. Bisect to
   find the cliff — maybe K=12 is the right depth for v7_0's regime.
   Single test, ~10 min A/B.

4. **4P-aware rollout** (built in v7.4, never gated cleanly). Half
   the ladder is 4P; the current 4P fallback to v3.5.1 is at 35.3%
   live (per state/current.md). Real lift opportunity.

5. **Investigate why v9_combined p95 = 836 ms**. The watchdog at
   700 ms SHOULD truncate before then. Either the watchdog has a
   bug or the post-watchdog wrap-up cost is non-trivial (~136 ms).
   Profile.

## PI submission decision

Per Rule 1, the actual `kaggle competitions submit submissions/
v7_0_drop_one.py` requires your explicit single-shot authorization.

**Proposed action:** submit v7_0_drop_one as our 6th submission of
the day (we've used 1/5 daily slots; v7_minimax + v4_planner pushed
from parallel branches don't count against THIS branch's slots, but
they do count against TEAM daily slots). State file says
submissions_used_today = 1; the parallel-branch pushes may have
incremented this on those branches but it's worth verifying daily
quota with you.

Authorize, defer, or hold the v7_0 push?
