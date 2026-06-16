# 2026-06-16 — "run the producer once, simulate the best reply" agent

**PI request.** Build an agent on our fast, reliable engine that simulates what
the Producer does as the opponent, then backwards-searches the best response
over a sparse set of actions — running the Producer **once**, then applying the
simulation over the 18 steps the Producer has converged to.

**What I built.** `agents/best_response/` (committed `6bc0704`). Per turn:
1. Run the Producer **once per seat** — our own plan (a strong default + the
   seed for candidates) and each opponent's predicted move (the opponent
   model). The Producer never runs inside the rollout loop.
2. Hand the cheap, parity-tested engine (`lib.fast_sim`) the lookahead: for a
   sparse set of candidate first-moves (the Producer's plan, its greedy
   prefixes, drop-one defensive variants, and more-aggressive expansion
   variants), forward-simulate the turn (us = candidate, opponents = their
   predicted Producer move) and let it settle over an 18-step horizon.
3. Keep the candidate whose simulated position is best (ship lead + a
   planet-count term); fall back to the Producer's own move if nothing beats it.

This is the clean, latency-unbound redo of the prior search wrappers
(`scripts/producer_opp_wrapper.py`, which tied ~25%). The prior ran the
Producer at every rollout step for every seat and was forced to a 3-step
horizon. Running the Producer once frees the whole budget: measured per-turn
**p95 ≈ 126 ms, max 278 ms** (Producer ≈ 23 ms/call), so the 18-step horizon
and a 24-candidate search are nowhere near the 1000 ms cap.

## ⚠️ RETRACTION — the "50% tie" and the whole "seat asymmetry" were a HARNESS BUG

My first-pass A/B and the entire seat-asymmetry thread below were **contaminated
by a bug in my hand-rolled `/tmp` harnesses**, found 2026-06-16 by single-game
tracing. They are RETRACTED. (The original numbers were: best_response vs
producer 16/32 with BR 16/16 as P0 and 0/16 as P1; producer-vs-producer 15/0
P0; producer-vs-nearest 100%/20%. All artifacts — do not cite.)

> **The pitfall (reusable, costly).** `env.state[i].observation` for seats
> **i ≥ 1 is missing the shared `step` field** (9 keys vs 10). kaggle_environments
> only merges `step` in when it invokes the agent via `env.run`. Every manual
> `env.step` loop that fed an agent `env.state[seat].observation` therefore gave
> the non-zero seat **`step = 0` forever** → the Producer's orbit model froze at
> the t=0 planet layout → mis-aim → failed captures → an opening **stall**. So
> "whoever plays seat 1" looked crippled, which masqueraded as a seat asymmetry.
> **Always feed a complete obs:** use `env.run`, or build
> `{**dict(env.state[0].observation), "player": seat}` per seat.

**Verified by reproduce-then-fix (Rule 38), seed 0, producer vs idle:**
```
BROKEN feed (env.state[seat].obs):  P0 first-launch t11, 3 planets@40
                                    P1 first-launch t25, 1 planet@40  (STALL)
FIXED feed  (complete obs):         P0 t11, 3 planets ; P1 t11, 3 planets  (IDENTICAL)
```

So **the Producer is seat-symmetric — there is NO producer / `producer_plus`
seat bug.** Code review agreed independently (engine combat/score ties resolve
symmetrically; the producer is fully `player_id`-parameterised; all geometry is
center-relative). The PI's instinct that "something is buggy" was right — the
bug was in the measurement, not the agent.

**Real head-to-head (correct harness, 24 games, max_steps 300):**
```
best_response vs producer:  7/24 = 29.2%   Wilson-lo 0.149   draws 0
  by seat:  as P0 4/12 (33%) , as P1 3/12 (25%)   <- balanced; seat-lock gone
  search deviated from the producer's plan on 1490/3796 turns (39%)
```
So best_response is **worse** than the bare producer, not equal — the search
**actively hurts**. It overrides the producer's move 39% of the time, and
re-ranking with a weaker 18-step rollout evaluator picks worse moves than the
producer's own scorer. (Sanity: producer-vs-producer on the fixed harness was
P0 0 / P1 4 / draws 2 over 6 seeds — no P0 sweep.) This is a *stronger* form of
the documented dead-end: search over the producer doesn't just add nothing, it
subtracts.

## Seat-bias investigation — how the harness bug was caught (don't re-walk)

The PI flagged "this is a bug, if it is so seat-depending." The chase (and three
wrong intermediate conclusions of mine — "real producer bug" → "float32
degeneracy" → "real asymmetry") all came from the contaminated harness above.
What finally localised it, in order, was code review + one game:
- **Engine is seat-fair** (`lib/game/interpreter.py`): combat exact ties →
  mutual destruction → neutral; final-score ties → both rewarded; the only
  ordering is "P0 acts first," which never converts to a win.
- **Producer is seat-symmetric** (obs parse, `competitive_score`, forecaster,
  flow projector, aiming): no hardcoded seat; opponent fleets attributed from
  observed owner; geometry center-relative.
- **The single-game trace** (`/tmp/trace_game.py`, `/tmp/compare_seats.py`)
  showed the producer *stalling* as the non-zero seat vs idle — which pointed at
  the obs, not the logic. `/tmp/obs_check.py` then proved `env.run` passes a
  complete obs to both seats while `env.state[1].observation` is missing `step`,
  and `/tmp/compare_seats_fixed.py` confirmed the fix. → harness bug (see the
  RETRACTION box above). **There is no seat fix to make in the producer.**

## Why the search underperforms the Producer (the diagnosis)

The measured ~29% is the *worse* end of what this analysis predicts: a search
that overrides a strong policy using a weaker evaluator can only match it (when
it correctly defers) or hurt it (when the weak evaluator misranks) — never beat
it. Both mechanisms below push toward "hurt."

1. **My leaf evaluator is weaker than the Producer's own scorer.** The Producer
   is strong *because* of its expensive internal forward projection. My
   18-step rollout with a *cheap* tail policy is a coarser position-evaluator.
   Re-ranking the Producer's candidates by a coarser evaluator can't reliably
   beat the Producer's own ranking → deviations come out value-neutral. This is
   the HANDOVER's "a strong policy's strength IS its expensive sim; no cheap
   copy preserves it," seen from the evaluation side.
2. **The 18-step horizon is the Producer's own blind-spot horizon.** The
   Producer plans at horizon 18 and under-expands *because* far planet captures
   take longer than 18 turns to land — they never show their value inside an
   18-step window. My sim runs at the **same** 18 steps, so a far-capture
   ("expansion") candidate looks like ships-still-in-flight at the leaf (no
   planet-count gain) — value-neutral, never chosen. **The sim inherits the
   Producer's horizon blindness, so it cannot discover the expansion moves the
   Producer misses.** This is the single most important finding.

## The lever, if we push further

The expansion fix needs the captures to *land inside the sim*. Latency allows
it — a 30–40 step rollout is still only a few ms with the cheap tail. So the
most promising single change is to **decouple the sim horizon from the
Producer's 18** (e.g. 30–40 steps) so far captures resolve and the planet-count
term rewards them. This is the one way the search could find value the
Producer's own scorer is structurally blind to. (Trade-off: a longer cheap-tail
rollout is a noisier opponent model; watch the collapse/over-extension failure
mode the loss-mining flagged.) Other levers, weaker: a stronger leaf value
(learned, or the Producer's own internal scorer as the evaluator), or
best-responding to a *weaker* opponent than the one supplying the value signal.

## Submission caveat

best_response embeds the vendored Producer (opponent model + candidate source),
whose PROVENANCE says "local evaluation only." It is a **local research build**,
not a submit candidate, until the licensing question is settled. (Our ladder
line `producer_plus` is itself built on the Producer's `orbit_lite` engine, so
the team may already have a position — confirm with the PI.)

## Artifacts
- `agents/best_response/` (+ PROVENANCE), `tests/test_best_response.py`.
- **Correct** A/B harness pattern (`/tmp/ab_fixed.py`): feeds every agent
  `{**dict(env.state[0].observation), "player": seat}`. Reuse this shape; do NOT
  reuse the broken `/tmp/ab_br.py` / `ab_pp.py` / `prod_vs_near.py` /
  `compare_seats.py` (they read `env.state[seat].observation` → missing `step`
  for seat 1 → contaminated). Or just use `fast.py` (it goes through `env.run`).
- Bug-hunt trail (kept for the lesson): `/tmp/obs_check.py` (proves the
  missing-`step` pitfall), `/tmp/compare_seats_fixed.py` (reproduce-then-fix).
