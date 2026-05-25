# 2026-05-25 — momentum_strike repositioned as calibration probe

## The reframe

PI's wrap-up message: *"we will use this strategy to benchmark other of
our strategies."*

This recontextualises everything this session produced. I'd been
treating momentum_strike as a failed ladder submission (0/8 vs
`agents/baseline`, blocked under Rule 42). The PI's reframe says: the
27/32 early-elim against simple opponents IS the artifact. We have a
**known-quantity, deterministic, fast-converging opponent** that sits
cleanly between trivial (random/nearest) and production (baseline) on
the difficulty curve.

For evaluating future agents, that's a useful tier to have. Beating
momentum_strike says "the new agent is past the simple-strategy class."
Losing to momentum_strike says "the new agent isn't there yet."

## Why momentum_strike makes a good benchmark

- **Predictable.** Greedy 1-step proposer, no rollout randomness, no
  K-step opp modeling. Same seed produces same game.
- **Honest middle tier.** Beats `nearest`/`weakest`/`enemy_first` by
  elimination ≤250 turns ~7-8/8. Loses ~0/8 to baseline. Not noise —
  a real difficulty floor.
- **Single-file simple.** `agents/momentum_strike/{main.py, proposer.py}`
  + `lib/polar.py` is ~350 LOC total. Easy to read; easy to explain
  what behaviour the new agent is beating.
- **Mechanism-pipeline native.** Uses `realize(intents, obs,
  mechanisms=DEFAULT_MECHANISMS)` — same safety stack the simple
  agents use. So losses to momentum_strike are strategy losses, not
  safety-stack artifacts.
- **No cross-turn state in default config.** (V4 salvo + ledger
  exists, env-gated off.) That means no hidden behaviour the new
  agent has to deconvolve.

## What the V1→V2→V3→V4 sequence actually proved

Beyond "which agent wins", the iteration trajectory clarified the
shape of the simple-agent ceiling:

1. **V1 (hand-rolled aim + fate gate): 1/8 vs nearest.** Reimplementing
   the mechanism safety stack from scratch with `predict_fleet_fate`
   alone misses lead-aim and production-during-flight inflation. Lost
   to the simplest opponent.

2. **V2 (production-first + defense + DEFAULT_MECHANISMS): 7/8 vs
   nearest, 0/8 vs baseline.** The "use the existing pipeline" pivot
   captured all of the simple-strategy bar in one move.

3. **V3 (added ENEMY_MULTIPLIER when behind): +1 simple-panel win
   over V2, 0/8 vs baseline.** Single-knob improvements over V2 give
   marginal lift against simple opponents. Don't close the baseline
   gap.

4. **V4 (synchronized salvo + cross-turn ledger): 0/8 vs baseline,
   simple-panel early-elim regresses 27→26.** Adding a "novel
   mechanism" (the synchronized arrival) doesn't lift either —
   long-wait commits starve expansion velocity more than the
   coordinated landing wins back. Gated OFF; code retained for
   future re-evaluation.

The throughline: **the gap from `realize(DEFAULT_MECHANISMS)+greedy`
to `realize(DEFAULT_MECHANISMS)+K-step-rollout` is structural, not
constant-tunable.** No single-knob or single-mechanism addition
crossed the gap in 3 iterations. The next class of improvement
requires either porting baseline's chooser or pivoting to a wrapper
strategy.

## Methodology lessons (already in friction.md)

- Diagnose ≥2 failure modes before picking a knob.
- `play_one`/`env.run` is the authoritative A/B path; sequential
  `env.step` traces diverged from A/B results twice this session.
- After 2 failed knobs on the same dominant metric, the gap is
  structural — stop iterating constants.

## Implications for future agent evaluation

When the next agent gets built:

1. **Run `scripts/momentum_strike_ab.py --focal <new_agent> --vs
   agents/momentum_strike --seeds 8 --workers 4`.**
2. Interpret:
   - ≥ 6/8 wins → past simple-strategy class; worth comparing
     against baseline.
   - 3-5/8 wins → roughly momentum_strike class; check what
     mechanism it has that momentum_strike lacks.
   - ≤ 2/8 wins → below momentum_strike; debug.
3. The 250-turn elim cap is the eliminationonsuccess metric — note
   how many wins are by elimination vs by ship-count-at-turn-500.
   Faster eliminations = stronger strategy.

The agent isn't a ladder candidate. It's a measuring stick.
