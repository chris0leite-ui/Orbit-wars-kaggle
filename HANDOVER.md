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

## Live status (2026-06-10 ~14:00 UTC)

- **Rolling pair:** sub **53529884** `mass2p_ffa` (08:53 UTC, the
  mass-concentration pivot — see STRATEGY.md) and sub **53527125**
  `ffa_uniform` (07:07 UTC, 1170 at ~6.5 h and climbing). The pair
  differs ONLY in 2P behavior → settled gap = clean live 2P A/B of the
  mass mechanisms.
- Evicted: sub 53523036 `multi_opp_def` restore froze at ~1214 (9.5 h).
  The 4P objective-fix verdict ≈ (ffa_uniform settle) vs that frozen
  1214, read after ~2026-06-11 07:00 UTC.
- Submissions used today: 3 of 5.

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

1. **Read settles after ~2026-06-11 09:00 UTC — DRIFT-ADJUSTED (PI
   instruction 2026-06-10).** The field strengthens ~100 μ / 3 days
   (~30-35 μ/day; identical code settled 1282 on 06-04 but 1181 on
   06-07), so absolute μ comparisons across days are biased AGAINST the
   later submission. Three readings:
   (a) 4P objective fix: ffa_uniform (53527125) settle vs frozen
   multi_opp_def 1214 **minus ~30-65 μ of drift for the 1-2 day gap** —
   i.e. ffa_uniform settling anywhere above ~1150-1180 is already
   parity-or-better in relative terms;
   (b) mass mechanisms: mass2p_ffa (53529884) settle vs ffa_uniform
   settle — both submitted within 2 h of each other, so this pair is
   drift-FREE and is the clean reading;
   (c) absolute level vs the same-day rank-100 μ (re-query it; do not
   reuse the 1261 snapshot).
2. **If mass lifts:** push the mass axis further — the top agents are
   at fleet p50 83 vs our 44 after these changes: sweep
   REGROUP_MIN_SEND (40?), OVERKILL_FACTOR (3), and attack the second
   mined gap (expansion: 8 planets by step 40 vs our 6 — they spend
   their early stockpile; we sit on 46 ships at step 20, top sits on
   33). Measure 2P head-to-head vs `_ns_multi_opp_def` AND a new
   namespaced mass partner; vanilla-producer A/B is a non-regression
   check only, NOT a verdict instrument (it steered us into the dribble
   meta for a week).
3. **If mass flat/down:** the 64-game head-to-head said parity-or-
   better, so a live regression means the 1200-1400 band punishes mass
   differently than our champion does — pull mass2p_ffa's live 2P
   losses (`scripts/live_episode_summary.py 53529884 --pull`) and
   profile the opponents that beat it (behavior_profile.py works on any
   corpus).
4. Tools added today: `scripts/crawl_top_replays.py` (walk the public
   episode graph to any rating level; top-3 team corpora in
   `audit/top-replays/`, gitignored) and `scripts/behavior_profile.py`
   (behavioral fingerprint of any team from replays).

## Pointers

- `state/STRATEGY.md` — strategy, build, smoke, iteration protocol.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `CLAUDE.md` — process rules.
