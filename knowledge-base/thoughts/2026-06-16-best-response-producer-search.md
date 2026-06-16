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

## Result — it ties the Producer (clean negative)

Balanced both-seats A/B, 1-thread torch (16 seeds × 2 seats):

```
best_response vs producer: 16/32 = 50.0%   (draws=0)
  BR as P0: 16/16 WINS     BR as P1: 0/16 WINS
search deviated from the Producer's plan on 936/3404 turns (27%)
```

The search is genuinely **active** (deviates on 27% of turns — it is not a
Producer clone), yet the outcome is **100% decided by seat**.

**Control — producer vs producer, same seeds:** P0 wins 15/15, P1 wins 0/15.
So "P0 always wins" is a **pure 2P game artifact** between two mirror-strength
deterministic agents, not anything about best_response. BR reproduces the
Producer's seat pattern exactly → **BR is precisely Producer strength**; the
27% deviation is net-neutral (it changes which moves are played, not who wins).

> **Methodology flag (reusable).** A 2P head-to-head between two near-identical
> strong deterministic agents is **degenerate** — the seat (P0) advantage
> saturates the signal, every game is seat-locked, and a balanced design nets
> exactly 50% for *equal-strength* agents. To detect a *small* edge you must
> use an arena where outcomes aren't seat-locked (4P FFA, or non-mirror
> opponents). Do not read a 2P-mirror 50% as "no difference in behaviour" — read
> it as "equal strength."

## Seat-bias investigation (PI: "this is a bug, if it is so seat-depending")

Localized it (probe `/tmp/seat_probe.py`):
- **Not the harness.** `nearest`-vs-`nearest` gives identical outcomes via the
  official `env.run` and via a manual `env.step` loop.
- **Not a universal engine seat bias.** `nearest`-vs-`nearest` on a symmetric
  board mostly **draws** (seeds 0/3/4/5 exactly equal, e.g. 874=874) and splits
  the rest (seed1→P1, seed2→P0). The engine resolves symmetric play fairly.
- **The board is symmetric** (seed 0: P0 home (73.4,73.7) ↔ P1 home (26.6,26.3),
  point-mirror around center).
- **So the P0 sweep is producer-specific.** Two identical *producer* policies on
  a symmetric board diverge deterministically in P0's favour and one side
  dominates by ~step 100. Likely cause: **float32 mirror-breaking** — P0
  (positive offsets from center) and P1 (negative offsets) compute mirrored
  decisions whose `atan2`/`floor`/sort-tie rounding isn't sign-symmetric, so the
  tiny asymmetry compounds. (The bare producer is fully `player_id`-parameterised
  — no hardcoded seat — so it's not a friend/foe mislabel.)

**Open (decisive) test — does it generalise past mirror self-play?** producer
& best_response vs `nearest`, both seats (`/tmp/seat_vs_nearest.py`):
- If each wins ~equally from P0 and P1 → the sweep is **only** a mirror artifact,
  harmless on the ladder (different opponents = no mirror to break); the lesson
  is just "don't A/B in a 2P mirror."
- If P1 winrate << P0 winrate vs `nearest` too → a genuine "plays seat 1 worse"
  bug in the producer **and `producer_plus`** (same engine) that would cost
  real ladder games — worth a fix. _RESULT PENDING — fill in._

## Why the search adds nothing over its base (the real diagnosis)

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
- Probes (in `/tmp`, transcribe if we revisit): `ab_br.py` (2P balanced A/B +
  deviation), `ab_pp.py` (producer seat-split control), `ab_br_4p.py` (4P FFA).
- 4P FFA vs 3 producers (the prior dead-end's arena, beat => >25%): _RESULT
  PENDING — fill in._
