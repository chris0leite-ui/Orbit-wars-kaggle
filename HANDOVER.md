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

## Live status (2026-06-10 ~08:30 UTC)

- **Rolling pair:** sub **53527125** `ffa_uniform` (07:07 UTC, warming —
  720 at ~1.5 h) and sub **53523036** `multi_opp_def` restore (04:12 UTC,
  1241.9 near its predicted settle). Identical 2P play by construction —
  the settled μ gap is a pure live 4P A/B of the FFA objective fix.

## What the 2026-06-10 session established

1. **4P loss anatomy** (`audit/2026-06-10-4p-loss-anatomy-mining.md`):
   losses are decided in the step-20..80 brawl window (production peaks
   ~40 then declines; rank 1 at step 20 even in losses). NOT separators:
   drained-then-carved rate, neutral expansion, defensive-shortlist
   width. Multi-front carving is the end state, not the cause.
2. **Two more 4P nulls on the 3×producer pool** (baseline 13/32):
   `tick4p` (4P-only multi-tick mirror, K=3) 10/32 — the mirror re-spends
   rival ships across rounds (no budget debit), phantom aggression;
   `reinforce_deficit` (defense candidate sizing fix, default-OFF code in
   producer_plus/main.py, 10/10 unit tests, OFF-path hash-verified)
   9/32. Six seeds win under BOTH variants → the pool is dominated by
   the map/seat draw; treat it as a regression triage, not a verdict
   instrument. Per-seed logs now archived under `audit/pools/`.
3. **Fleet speed RISES with size** (log curve to 1000 ships) — big
   rescue/strike fleets are FASTER. Remember when reasoning about
   timing mechanisms.

## Next action

1. **Read the rolling pair's settled μ after ~2026-06-11 07:00 UTC:**
   sub **53527125** `ffa_uniform` (submitted 06-10 07:07, the 4P objective
   fix — 2P byte-identical to multi_opp_def, predicted 1230-1330 if the 4P
   lift translates) and sub **53523036** `multi_opp_def` restore (06-10
   04:12, backstop). Compare the two settles: identical 2P play means any
   μ gap between them is pure 4P signal — a free live A/B of the FFA
   objective.
2. If ffa_uniform settles ABOVE multi_opp_def: the FFA objective fix is
   live-validated AND the local 3×producer pool is shown non-predictive
   (it said parity) — weight the namespaced self-play pool and live A/Bs
   from then on. Next candidates: strength/uniform weight blend;
   budget-debited multi-tick (fix the re-spend flaw first);
   reinforce_deficit composed with ffa_uniform (its standalone pool was
   within draw-noise, and it's cheap to ride along a future submit after
   a 2P n=32 A/B clears it).
3. If ffa_uniform settles AT/BELOW multi_opp_def: pull its live 4P
   episodes (`scripts/live_episode_summary.py 53527125 --pull`), check
   the 2P/4P winrate split vs the 29% baseline, and diagnose from the
   replays before touching the mechanism. The brawl-window finding
   (audit doc) is the lens: what did the FFA fix change between steps
   20-80, if anything?

## Pointers

- `state/STRATEGY.md` — strategy, build, smoke, iteration protocol.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `CLAUDE.md` — process rules.
