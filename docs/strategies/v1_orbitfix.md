# v1 orbitfix — leading the orbit + breaking the mirror

> File: `agents/v1_orbitfix/main.py`.
> Submitted as `submissions/v1_orbitfix.py` (bundle of `lib/{geometry,fleet,orbit}` + agent) on 2026-05-10. Live μ TBD pending validation episode + ladder play.

## One-liner

Same nearest-planet greedy as the shipped baseline, but aim at where
the target **will be when our fleet arrives**, and randomise tie-breaks
so identical seed → identical strategy mirroring no longer dominates.

## Mechanism

Two surgical changes vs `data/main.py`:

### 1. Orbit-aware aim (closes ISSUES.md::B.2)

For each candidate target:

1. Decide if it's an *orbiting non-comet*: `is_orbiting(target)` (true iff
   `orbital_radius + planet_radius < ROTATION_RADIUS_LIMIT = 50`) AND
   `target.id ∉ obs["comet_planet_ids"]`.
2. If yes, **lead the prediction**:
   - `ships = target.ships + 1` (exactly what we're sending).
   - `v = lib.fleet.speed(ships)` — fleet speed scales with size:
     `1 + 5*(log(ships)/log(1000))**1.5`, clamped at 6 for ≥1000 ships.
   - Iterate the fixed point twice:
     - `t = dist(mine, current_target_xy) / v`
     - `lead_xy = lib.orbit.predict_relative(target, omega, t)` —
       project the target forward in its orbit by `omega * t` radians,
       reading the current angle from the obs (no step-counter needed,
       no off-by-one trap).
     - Repeat once with the leaded distance.
3. Aim at `lead_xy` instead of `current_xy`.

For static planets and comets, this collapses to the shipped-baseline
aim (no projection). Comets are deferred to v2 — they move on
`obs["comets"][].paths`, not via the rotation formula.

### 2. Tie-break randomisation (closes ISSUES.md::A.6)

Per owned planet, score every target by 2D distance, then break ties
with `random.Random(step ^ (player + 1) * 1009).random()`. The seed
is per-turn and player-specific; the same step gives both players
the same rng draws but the player-id-mixed salt makes the two players
prefer different equidistant targets when ties occur.

Net effect: in 20-seed self-play, the previous 4/6 P1 / 1/6 P0 / 1/6 tie
asymmetry collapsed to **5 P0 / 4 P1 / 11 draws** — within 5% of
50/50, well under the ±15% gate.

## Why it works

Two compounding wins:

- **Orbiting planets actually get hit.** At default angular velocity
  (~0.04 rad/turn) and inner-orbit radius (~30 units), a planet drifts
  ~1.2 units per step. A 100-ship fleet at speed ~3 takes 5-15 turns to
  cross the board → without lead, the fleet arrives 6-18 units behind
  the target and the fleet is lost (out-of-bounds or sun) or hits a
  different planet.
- **The asymmetry mirror is broken.** Self-play games no longer collapse
  to the same neutral-target races. Both sides get a fair distribution
  of "won the race" outcomes; the tournament fixture's win-rate
  estimates stop being a fictional measurement of who-launched-first.

## Gotchas (same-day fixes if they fire)

- **Single fixed-point iteration is enough at default omega.** If the
  comp publishes faster rotations on a future game variant, the
  iteration count may need to grow. Smell test: residual aim error
  > 1 board unit on inner planets.
- **Comet aim still wrong.** v1 aims at current position for comets.
  At cometSpeed=4 and 5-50-step trajectories, this is wasteful.
  v2 should add a comet-path lead.
- **No combat forecasting.** v1 sends `target.ships + 1` based on the
  *current* garrison; if the target receives a friendly reinforcement
  before our fleet arrives, our fleet loses combat. v2 closes this with
  an arrival ledger.
- **No sun-avoidance.** The geometry primitive `path_clears_sun`
  exists but v1 doesn't call it. If `mine` is on the opposite side of
  the sun from `nearest`, the fleet dies in flight. Local 40/40 win
  rate suggests this rarely matters against the shipped baseline (it
  doesn't avoid the sun either) but live opponents at μ=1000+ exploit it.

## Evidence

- 20 seeds × 4 ordered pairs (random / baseline / v1 / self-play),
  `audit/tournaments/20260510T080307Z.json`:
  - **v1 vs baseline aggregate: 40/40 = 100%** (Wilson 95% 0.84..1.00 each side).
  - v1 self-play: 5 P0 / 4 P1 / 11 draws → A.6 closed.
  - p95 turn = 0.3 ms (1-second budget = ample headroom).
- Bundled vs unbundled parity: 4/4 seeds matched rewards.
- E.2 self-vs-self validation gate: 10/10 reached `DONE`.
- Test suite: 54 green in 46 s; bundler regression test now pins
  the alias-rebind contract that bit us during build.

## What it does NOT do (explicit gap → motivates v2)

- **No arrival-time ownership forecasting.** Sends a 100-ship fleet
  to a 50-ship neutral that's about to be reinforced by an enemy
  500-ship arrival → our fleet dies on contact.
- **No same-turn combat order simulation** (rule 4: two-way ties
  destroy all attackers — exploitable by sending the *exact* tying
  ship count to neutralise an enemy assault for free).
- **No fleet coordination.** Each owned planet plans independently.
  Two of our planets may both target the same neutral with `target.ships+1`
  ships → wasted overlap.
- **No defence.** Garrison stays put. An incoming enemy 200-ship fleet
  walks into our 50-ship home undefended.
- **No mission-classification.** Roman 1224 distinguishes snipe /
  rescue / recapture / reinforce / crash-exploit / gang-up / elimination
  and scores them against a single solver. v1 only does "snipe."

These gaps are the v2 / v3 build agenda — see `roadmap.md`.
