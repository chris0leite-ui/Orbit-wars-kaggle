# 2026-05-20 — When do we need a faster production surrogate?

Today's sary-class anchor failure showed: **only current production
catches under-emission regressions.** Roi, sary_class, v7_0_drop_one
all lose to the failed ledger. So h2h vs current production is the
gate.

But current production at ~80s/game is SLOW. n=8 h2h takes 5-7 min;
n=64 takes ~25 min on 4 workers. For rapid iteration this becomes
the bottleneck.

**Open question**: at what point is it worth investing 3-5 days in
distilling current production into a fast surrogate (e.g., a small
NN trained on production's per-turn actions, deployed as a 5-10ms
agent)?

Cost-benefit at hand:
- Each candidate iteration through h2h n=8 = 5-7 min.
- Per session we typically test 2-4 candidates = 20-30 min h2h.
- A fast surrogate would cut that to 5 min total.
- Savings per session: ~20 min. Per week: ~2 hours.
- Cost: 3-5 days of one-time distillation work.
- Break-even: 7-10 weeks.

Comp deadline: 2026-06-23 — 33 days from today. Less than 10 weeks
remain. **Not worth distilling at current iteration cadence.**

Alternative cheaper options:
- Run h2h in background while doing other work (already doing this).
- Tighten the candidate-design loop so fewer h2h runs are needed.
- Run smaller-n early (n=4 or n=8) and only escalate if the small
  gate clears.

Defer decision until either (a) iteration cadence increases enough
that h2h becomes the bottleneck or (b) deadline shifts.
