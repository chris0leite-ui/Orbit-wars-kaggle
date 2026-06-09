# HANDOVER.md — next-session brief

## Mode

**Observation-driven iteration on a single strategy.** One observation from
the PI → one mechanism → one push.

## Strategy (updated 2026-06-09)

The main agent is **`producer_plus_multi_opp_def`** — Producer's engine
(vendored, MIT) + our multi-size candidate enumeration + Producer-mirror
opponent projection + opp-aware defensive shortlist. Build with:

```
python scripts/bundle_producer_plus.py --variant multi_opp_def
```

`state/STRATEGY.md` has the full picture. The working branch is
`claude/awesome-clarke-ixy57v` (the majestic-storm producer_plus track is
merged into it; main is 166 commits behind that track).

## Live status (2026-06-09 21:30 UTC)

- **Rolling pair:** sub 53450504 `multi_size` (06-07 manual resubmit,
  settled **μ = 1181.1**) and sub 53390700 `multi_tick_recap` (settled
  **μ = 1099.3**). Team rank 446 of ~4130; rank 100 needs ≈ 1261.
- **Best-ever agent** `multi_opp_def` (settled 1263–1287) was **evicted**
  by the 06-07 resubmit. A restore submission is prepared, Rule 42/45/46
  GREEN, **awaiting PI sign-off** — see the top row of
  `state/MULTI_BRANCH.md`.
- Field drift: identical multi_size code settled 1282 on 06-04 but 1181 on
  06-07 — the ladder strengthened ~100 μ in 3 days. Expect restored
  multi_opp_def to settle ≈ 1180–1260, not necessarily 1280.

## What the 2026-06-09 session established

1. **Force-concentration is a null (negative) result.** All three variants
   regressed hard in clean n=32 A/Bs vs vanilla producer: standalone 6/32,
   lean 7/32, multi-tick stack 5/32. The relaxed one-wave-per-target mutex
   as implemented hurts; do not ship, do not re-compose without a new
   mechanism-level diagnosis.
2. **Rebuilt multi_opp_def re-validated:** 24/32 = 75% vs producer,
   Wilson [0.579, 0.867] — identical to its 06-05 measurement.
3. **Harness rot fixed** (current kaggle_environments execs agent files
   without `__file__` and with ±1 rewards): producer_agent wrappers and all
   producer_plus shims got a `sys.path[-1]` fallback; tests clear
   `PRODUCER_PLUS_*` env between in-process games and compare action
   streams, not just rewards. Before these fixes, shim-based test games
   silently played None every turn.

## Next action

1. If the PI signs off: submit `submissions/producer_plus_multi_opp_def_on.py`
   (claim row already on the board).
2. Next mechanism candidates, in order of prior evidence: strategic-value
   bonuses re-tuned to weight ≈ 0.005–0.02 (calibrate against the dumped
   competitive_score distribution first — see 2026-06-05 handover lesson);
   diagnose WHY force-concentration regresses (replay a lost game) before
   any retry.

## Pointers

- `state/STRATEGY.md` — strategy, build, smoke, iteration protocol.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `CLAUDE.md` — process rules.
