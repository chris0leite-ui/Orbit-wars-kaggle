# 2026-06-15 — Baseline loses to producer (0–4 clean); the positional-architecture question is settled

Tail of the positional session. After establishing that our **old baseline IS
the native positional `Φ` agent** (value fn = material-diff + PV-discounted
economy-diff + 4P-weakness; chooser selects by ΔΦ; reach/space term opt-in and
OFF by default), the question was whether to revive it (it's fully ours, no
vendored-producer provenance limit) instead of patching producer.

## The decisive test (PI-designed to avoid cherry-picking)

I first reported "baseline beats producer 2–0" — but that was **two
cherry-picked seeds (7, 42)**, the small-n referee-blind trap. The PI called for
a clean test: **4 games, 4 fresh seeds (101/202/303/404), balanced seats, no
per-seed swap.** Result:

**baseline 0–4 vs producer** (producer won as both P0 and P1 → not a seat
artifact).

And these local games have **no timeout** — baseline ran its full ~530 ms/turn
and still lost every game. So baseline's stale ~1170 < producer's ~1280 is
**real inferiority, not timeout-drag.** The "its μ is just speed" hypothesis is
dead. (Latency profile, for the record: ~530 ms/turn, 73% in the forward-search
chooser running the exact interpreter ~10k sim-steps/turn + Python collision/
orbit hot loops — strength and slowness are the same thing.)

## Architecture question — settled

Every branch explored this session converges:
- **Clean-`Φ` agents (the `phi` instrument, the baseline) both lose to
  producer.** Producer's flow-scorer + tactical engine (orbit-aim, exact combat,
  capture-floor sizing, reachability) beats explicit-`Φ` + forward-search.
- **Producer is the substrate.** Don't revive baseline; don't rewrite a
  clean-room positioner (converges to producer-with-an-economy-knob at higher
  cost; the search-wrapper version is already a dead end).
- **The positional yield is an objective steer, not new code:** economy
  (compounding production) + tempo are the winning levers → push via the live
  **`expand`** (deeper horizon + wider shortlist) and the untested **`hold_value`**
  (post-horizon economy credit gated to holdable captures). Options/reach
  (frontier) and durability/caution (tenure) are refuted — keep gated-OFF.

## What I got wrong (process note)
Reported a 2-seed cherry-picked h2h as if it were signal. The fix is the PI's
rule, now reinforced: **balanced seats + fresh seeds, n ≥ a handful, before any
"X beats Y" claim** — even for a quick local read. Two seeds is an anecdote.

## Net state for the final week (~6 days)
- Champion line: producer_plus seq_strength (~1280, stale); live ladder pair
  `expand` + `wideshortlist` (grounded economic-expansion fixes).
- On-thesis untested lever: `hold_value` (gated economy) — the one principled
  "positional economy" fold left.
- Refuted/parked: frontier (reach), tenure (caution), baseline revival,
  clean-`Φ` rewrite, search-over-producer, RL/IL.
