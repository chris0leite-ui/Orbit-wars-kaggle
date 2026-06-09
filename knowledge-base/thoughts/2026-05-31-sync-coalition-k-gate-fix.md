# 2026-05-31 — Sync coalition: the K-gate bug, and why "below 50%" wasn't a null

Branch: `claude/champion-strategy-rules-00JzI`

## What happened

The synchronized two-source team-up ("sync coalition") A/B'd **below 50%**
vs the no-sync champion. The instinct-to-kill reading was "the mechanism
doesn't help — null it." PI pushed back: *it might not be a null, look at
what actually happens.* That was right.

## The bug (a self-sabotage, not a null)

The generator gated coalition arrival on `MAX_HORIZON` (tick 40). But the
champion runs `enforce_launch_rules` with capture-horizon **K=10**, which
deletes any launch arriving after tick 10. Both coalition legs are built to
land on the same synchronized tick `tarr`. When `tarr > K` (92% of the
time — median `tarr`=15), the fire-now **far leg was silently deleted**
downstream, while the **near leg's wait-commit persisted** and fired alone,
undersized, and bounced. We spent two fleets to accomplish *less than
nothing*. That's why it was below 50%: the feature wasn't weak, it was
actively bleeding ships on 92% of its attempts.

## The fix (Rule 40 in action)

The tempting "fix" was to loosen K or bump MAX_HORIZON. Wrong — Rule 40
says align the model, don't bump a constant. The right fix: the generator
must respect the *same* ceiling the downstream filter enforces. Cap sync
arrival at `min(MAX_HORIZON, K)` when launch rules are on. Now both legs
always survive. Census 24/26 deleted → 0/9 deleted. A/B 45.8% → 56.2%.

## Lessons (durable)

1. **A sub-50% A/B is a question, not a verdict.** "Worse than champion"
   can mean "the feature is harmful *as built*" (a bug) rather than "the
   idea is wrong" (a null). The single-game trace is what told them apart —
   I should reach for the trace *before* declaring a null, not after.
2. **Generator/filter ceiling mismatch is a whole bug class.** Any time a
   proposer builds candidates that a downstream enforcer can silently drop,
   check that the proposer is gated on the *same* limit. The half-fire
   here was invisible in aggregate win-rate; only the per-coalition census
   exposed it.
3. **The next ceiling is "hold," not "capture."** Trace showed 2/4
   coalitions capture-but-don't-hold (garrison+1 sizing → recaptured). A
   56% edge is expensive to *confirm*; a size-to-hold mechanism that lifts
   the edge is cheaper to *prove*. That's the argument for building Lever 1
   over grinding confirmation games.

## Open question (logged)

Is a ~56% local edge vs one champion config worth the ~150-game confirm,
or do we skip straight to size-to-hold and re-baseline? Leaning: build the
bigger edge. PI chose confirm-first this session; panel running.
