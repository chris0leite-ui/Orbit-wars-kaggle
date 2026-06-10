# 2026-06-10 — Ledger agent: from-first-principles rebuild, one night

## What happened

PI directive at session start: "Forget everything we have done so far.
What would be the simplest, yet feasible most powerful solution to this
game? Think it through and implement it. You have all night, iterate as
fast as you can." Later: "Iterate quickly, a win 4 seeds clearly then
move on."

## The thesis that drove the design

Re-deriving the game from the engine source produced three load-bearing
facts:

1. **All combat trades ships exactly one-for-one** (largest attacker vs
   second largest, difference vs garrison, flip on strict negative). So
   fighting never moves the final-score differential directly. The
   differential is moved only by: production flow (planets held × turns
   held), neutral garrisons paid to expand (a pure sink), and waste (sun,
   out-of-bounds, comet departures, tie annihilations). This was later
   verified empirically in a full game: my measured combat losses matched
   the opponent's to within the neutral-sink difference.
2. **The future is exactly computable.** Deterministic physics + full
   observability mean every in-flight fleet's landing (planet, tick) and
   every planet's (owner, garrison) timeline is knowable, absent new
   launches.
3. **Planet positions are action-independent**, so fleet landings are
   decided at launch. A "rollout" therefore needs no per-tick physics —
   it is a joint ledger walk with arrivals as events.

The agent ("ledger", `agents/ledger/main.py`, single self-contained
file = the submission) computes the exact future every turn and buys the
best actions priced off it.

## The night's iteration log (observation → mechanism)

Each fix came from reading a real losing game, not from tuning:

1. Lost to the trivial nearest-sniper: zero-garrison play + no
   multi-source attacks → **reserves** (single-strongest-strike threat
   with assigned helpers) + **coalitions** (joint arrival schedules
   re-walked exactly through the target timeline).
2. Dribbling tiny slow fleets at far targets (fleet speed grows with
   size — small = slow) → **banking** (the best unaffordable plan
   freezes its funding sources) + **forecast decay** (value × 0.97 per
   tick of flight: the no-new-launches ledger loses validity with depth).
3. Opponent (v7_0) took high-production planets first → buy order by
   **value, not value-per-ship**: production is the compounding resource.
4. Captures near enemy mass re-sniped within my old binary "hold 8
   ticks" check (their response launches after seeing my fleet, lands
   inside the window, invisible at decision time) → **price expected
   flow duration** against the enemy's feasible response curve; require
   only the capture, never the hold.
5. Mid-game paralysis: the hard hold-requirement and the soft response
   pricing jointly rejected everything → unified into capture-only
   requirement + flow pricing + **rollout veto**: an event-driven
   18-tick simulation with a reactive opponent (just-in-time defense,
   re-snipes, both sides) compares {all buys, drop-one, defense-only}.
6. Coalition shares undershot (smaller shares fly slower than the
   full-spare estimate, land later, face regrown garrisons) → **iterate
   share allocation to arrival-tick consistency**.
7. Late-game passivity from two modeling errors: the in-flight liquidity
   tax (pressure-scaled) must apply **only when defending a production
   lead** (when production-behind, converting bank to production is
   mandatory — the bank is the only non-producing resource); and the
   analytic response curve charges each target against the enemy's
   whole army although their response budget is **shared across my
   simultaneous attacks** → admit slightly-negative plans and let the
   rollout veto decide. This was the breakthrough: 2/6 → 5/6 vs v7_0,
   all wins by total elimination.
8. 4-player: fighting any one opponent is negative-sum for both relative
   to bystanders → with ≥2 live opponents, discount player-owned targets
   (except snipe-cheap), no negative admit floor.

Failed directions (kept honest): synchronized-wave reserves (froze all
capital, lost everything — passive defense cannot beat a coordinated
attacker; the answer was response-priced offense, not bigger garrisons);
rollout-MARGINAL selection as the primary chooser (the reactive model is
not good enough to steer buys; as a veto its bias differences out).

## Measured results (this container, kaggle_environments 1.29.1)

| Opponent | Result | Wilson-lo | Note |
|---|---|---|---|
| random | 32/32 | 0.89 | smoke |
| nearest sniper | 31/32 | 0.84 | smoke |
| **v7_0 (production baseline)** | **28/32 (87.5%)** | **0.719** | balanced seats, n=32 |
| **Producer (public; beats live champ 81%)** | **32/32 (100%)** | **0.893** | balanced seats, n=32 |
| v4_planner | 29/32 (91%) | 0.758 | balanced seats, n=32 |
| v3.5.1 | 27/32 (84%) | 0.682 | balanced seats, n=32 |
| v7_0, FRESH seeds 100-115 | 14/16 (87.5%) | 0.640 | seeds never used in iteration |
| Producer, FRESH seeds 100-115 | 16/16 (100%) | 0.806 | Producer total tonight: 48/48 |

Timing: p50 13 ms, p95 46 ms, max 74 ms (budget 1000 ms). Self-play
validation episode clean (500 steps). Forecast parity test green
(`tests/test_ledger_forecast.py`: agent's predicted owner+ships per
planet per tick match the engine exactly, >500 checks, 2 seeds — caught
a real comet-expiry off-by-one during development).

## Caveats / open risks

- ~~Iteration-seed overlap~~ CLOSED: a fresh-seed sweep (seeds 100-115,
  never used during iteration) reproduced the strength (v7_0 14/16,
  Producer 16/16). No overfitting to the iteration seeds.
- Live-ladder mix measured from our champion's replays: 47 two-player vs
  22 four-player episodes (~32% are 4P). The engine's 4P reward is
  BINARY (one winner +1, the other three -1, no placement credit), so 4P
  is win-or-nothing; the ledger agent spot-checks at 1/4 wins vs three
  strong family agents = the 25% parity baseline. Acceptable for v1.
- Simultaneous-launch races on neutrals (both players launch the same
  turn) are handled only by a blunt race discount.
- The Producer sweep says nothing about unseen top-ladder archetypes;
  live μ is the only true test.

## Status

`submissions/ledger_v1.py` is the ready-to-submit artifact (byte-copy of
`agents/ledger/main.py`). NOT submitted — Rule 1 requires explicit PI
sign-off per submission. STRATEGY.md intentionally not rewritten: that
is a PI decision (this doc + HANDOVER.md carry the candidate).
