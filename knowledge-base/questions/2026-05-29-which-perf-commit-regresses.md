# 2026-05-29 — question: which of the five perf commits owns the regression?

The post-perf bundle loses ~12pp vs the pre-perf bundle on the same
seeds. Five commits in flight. Sequential A/B bisection would
identify the culprit. Estimated cost: 5 × n=16 sequential ≈ 8h CPU.

Sub-questions to answer along the way:

1. Is the regression monotonic in commit count (each commit costs
   ~2.4pp) or carried by a single commit?
2. Is the KT singleton actually idempotent across game boundaries
   in the A/B harness, or does the singleton survive between
   `env.run` invocations and leak prior-game state?
3. Does the agent_deadline fire on ANY turn in seed 0 (the one
   we previously thought went 1035 ms without it)? If the
   deadline fires on >5% of turns, we're trading recall for
   latency — measure the trade.
4. Does the vec edit change `predict_fleet_fate`'s output on any
   single-game trace vs the scalar path? A test that asserts
   bit-identity over a recorded episode would catch it.

If the answer is "the regression is in KT singleton," the fix is
likely a `kt_table.reset()` call in the harness setup between
games. If it's in agent_deadline, the fix is to widen the
deadline (950 → 970 ms) and accept slightly more Kaggle-cap
risk. If it's in vec, the fix is to introduce an atol-tolerant
path comparison and confirm the differences are at FP-noise level
in non-decision-critical bits.
