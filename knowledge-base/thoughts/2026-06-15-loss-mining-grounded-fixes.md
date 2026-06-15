# 2026-06-15 — Loss-mining → grounded fixes (the session that found a method)

Long session. Started chasing a search "moonshot," spent hours hitting walls,
then a single PI replay observation flipped us onto the method that actually
works: **mine real losses → diagnose the specific bug → verify a fix →
ladder-A/B it.** This is the durable record.

## The pivotal discovery (read this first)

**`agents/producer/main.py` on `claude/festive-knuth-roggck` is the BARE
`orbit_lite` engine — 0 `PRODUCER_PLUS_*` flags. It is NOT the 1280 ladder
agent.** The real agent is `agents/producer_plus/main.py` — 70 flag-gated
behaviours (veto, reactive-floor, FFA-leader, multi-size, opening-search,
neutral-shortlist…) — and it + its matching `orbit_lite` + its variant-bundler
lived on `claude/awesome-clarke-ixy57v`. **Every local A/B before this discovery
used the wrong, weaker base — treat them as void.** I brought the lineage onto
this branch (commit `1e2e747`); the real agent runs here now (seq_strength flags
→ 55 ms median, budget-safe).

## The dead-end map (all failed — locally AND on the real ladder)

The first ~two-thirds of the session, chasing "use the 98% idle headroom":
- **Search wrapper** (producer-in-`fast_sim` 1-ply candidate search): tied
  **28%** (lite_greedy opp model) and **25%** (producer opp model). Search over
  a strong base adds nothing, with weak *or* strong opponent models.
- **Deeper internal planning** (bare engine, 3× horizon): *hurt* (19%).
- **Distillation / hand-condensed fast rollout policy:** failed (38% / 29% vs
  lite_greedy). Root cause: a strong policy's strength **is** its expensive
  forward simulation (orbit-aware, combat-resolving). No ~25 µs copy preserves
  it → no fast-strong rollout policy → search is stuck with weak policies → the
  documented μ~1120 ceiling.
- **The real ladder already ran this experiment:** `oracle_rw` (imitation
  learning) settled **1018**, `rl_v7` (RL, 95M env-steps) **938** — both far
  below the producer heuristic line (**1280**). Models/compute don't beat the
  heuristic here.
- **Defensive direction** (`garval` = garrison-value + source-safety): ladder
  **1181 < 1280**.

Conclusion that held: **producer sits at a tuned local optimum; the ceiling is
*strategy*, not compute.** And: **local self-play is referee-blind** — it can't
reward fixing a flaw the self-opponent shares (deep & aggressive both came back
inconclusive for this). **The ladder is the only honest A/B**, and submissions
are free to spend on it until the deadline (only the kept-2 at 06-23 matter).

## The method that worked (the whole point)

PI watched a live replay (2P loss to CPMP, seed 641308308) and said: *"the
opponent captures the upper-right corner; we don't capture the symmetric
lower-left one."* One observation → a confirmed, reproduced, fixed bug. The loop:
1. Reproduce on `info.seed` in the **producer_plus mirror** (two
   `ProducerLiteRuntime` instances; flags are global env so only symmetric games
   in-process).
2. Find the specific bad behaviour (corner left neutral @step95).
3. Sweep flags for what fixes it; **verify on-seed** (`NEUTRAL_SHORTLIST=20`
   grabbed the corners; fc/opening/overkill/hold/denial did NOT).
4. Bundle `seq_strength_<name>`, Rule-46 smoke, fire to ladder.

This beat *everything* systematic. Replays are in
`audit/live-episodes/53564198/episode-*-replay.json` (45 losses).

## The loss landscape (systematic analysis of all 46 loss replays)

1. **Under-expansion — #1 driver (~76% of losses).** We trail on planets by
   **median step 30**; **5–6 planets vs winners' 8 by step 60**; winner holds
   more far high-value planets in 27/46. Far planets fall outside the nearest-K
   neutral shortlist → never candidates. **Verified fix (6→8 planets@60 = winner
   rate): `NEUTRAL_SHORTLIST=20` + a deeper horizon together** — shortlist
   surfaces far targets, horizon values them. Either alone is partial (→7 / →6).
2. **Collapse — long-game cluster (12/15 long losses).** We peak at 6–10 planets
   then lose them; 6 were even/ahead at step 60 then lost everything
   (ep79522517 vs CPMP: peak 10 → 0). The trace shows we held *even* for 80
   steps, reinforcing, then a higher-rated opponent broke through — **partly a
   skill gap, not a clean flag bug.** Reserve: `GARRISON_VALUE_FROM_STEP`
   (late-game-only defence). **Tension with the expansion fix** (one says expand
   more, the other says don't over-extend) — watch long-game results.

## What's on the ladder (fired this session, read ~2026-06-16)

A/Bs are **SERIAL** (only the rolling-2 ladder; evicted subs freeze →
king-of-the-hill, ~7 rounds left). Current pair, both *grounded* fixes:
- `53714433 seq_strength_expand` — `NEUTRAL_SHORTLIST=20 + HORIZON_2P=30 +
  HORIZON_4P=18` (the #1-driver fix; h30 not h45 for latency safety).
- `53711823 seq_strength_wideshortlist` — `NEUTRAL_SHORTLIST=20` (the corner fix).
- (evicted: `seq_strength_opening` — opening beam-search, tracked ≈ baseline;
  speculative, didn't help.)

Warm-up at ~4 h was noisy (~1130, below the ~1220 field) — Rule 12, wait ~24 h.

## Banked, not a champion lever
The **~12× `lib/world_model` ledger speedup** (ring-crossing reframe for orbital
fleet attribution, commit `262a4e3`, bit-exact 136/136, median 10 ms → 0.7 ms):
real and committed, but `producer_plus` uses its own `orbit_lite`, so it helps
the lib/v3.x lineage + tools, not the champion.

## Open questions / flags
- **Q:** do the grounded fixes beat the field on the ladder? (~24 h). If both
  ≈ field, the diagnoses are right but the fixes don't convert → mine the 4P
  losses (worst format, 46%) next.
- **FLAG:** HORIZON in 2P spikes latency (one-off ~880 ms; midgame ~464 ms).
  Chose h30. Watch for ladder timeouts.
- **FLAG:** producer_plus lineage now on two branches (festive-knuth +
  awesome-clarke) — consolidate.
- **FLAG:** 1280 → 1600+ (prize zone) is a large gap; manage expectations.
