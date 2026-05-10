# v1 / v1.1 orbitfix — leading the orbit + breaking the mirror + production-aware sizing

> File: `agents/v1_orbitfix/main.py`.
> Submission history (same `agents/v1_orbitfix/` directory; rebuilt + re-bundled at each version bump):
> - v1 (commit `17fb9aa`, submission `52507539` 2026-05-10 08:11 UTC) — orbit-aware aim + tiebreak randomisation. **Live μ = 508.1** (started μ₀=600).
> - v1.1 (commit `9108c2a` lib + `1aa7547` strategy/mechanism refactor, submission `52509319` 2026-05-10 09:28 UTC) — same strategy + the `arrival_size` mechanism (production-aware sizing). Live μ pending.

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

## v1.1 — what arrival_size adds

After Step 3.5's strategy/mechanism refactor, the same agent file gained the
`arrival_size` mechanism in its `realize()` pipeline. For enemy-owned targets,
fleet size is bumped from `target.ships + 1` to `target.ships + production *
eta_turns + 1` so that the production growth during fleet flight is covered.
Neutrals stay at `+1` (they don't produce). The bump is monotonic — strategies
asking for an over-sized swarm aren't cut down.

The pipeline order is `[validate, arrival_size, lead_aim]`: arrival_size runs
BEFORE lead_aim because the lead-time estimate depends on fleet speed, which
arrival_size revises. If even our full garrison can't cover the
production-grown target, the intent is dropped (sending under-sized = waste).

## Evidence

### v1 (orbit-aware aim + tiebreak randomisation only)

- 20 seeds × 4 ordered pairs (random / baseline / v1 / self-play),
  `audit/tournaments/20260510T080307Z.json`:
  - **v1 vs baseline aggregate: 40/40 = 100%** (Wilson 95% 0.84..1.00 each side).
  - v1 self-play: 5 P0 / 4 P1 / 11 draws → A.6 closed.
  - p95 turn = 0.3 ms (1-second budget = ample headroom).
- **Live: μ=508.1** (live ladder, submission 52507539). Predicted Δμ +200-400; landed +205. Mid-range of prediction.

### v1.1 (above + arrival_size)

- 20 seeds × 4 ordered pairs, `audit/tournaments/20260510T085929Z.json`
  (the arrival_size ablation):
  - **v1.1 vs v1 aggregate: 32/40 = 80%** (Wilson 95% 0.58..0.92 each side).
  - Both still 40/40 vs shipped baseline; mechanism doesn't degrade
    upstream wins.
- 10 seeds × both sides head-to-head against the actually-submitted-v1
  bundle: **17/20 = 85%** (Step 3.5.E final gate; Rule 27 code-comp variant
  ≥55% threshold cleared 30 percentage points over).
- Bundled-vs-unbundled parity: 4/4 seeds.
- E.2 self-vs-self: 10/10 DONE.
- Live: PENDING (submission 52509319).
- Suite: 111 tests green / 67 s; bundler regression test extended for
  the new `intent`/`mechanism` lib modules; v1 parity gate pinned to
  the `[validate, lead_aim]` subset so DEFAULT_MECHANISMS can grow
  without invalidating the regression catch.

## What v1.1 does NOT do (explicit gap → motivates v2)

- **No arrival-time ownership forecasting.** v1.1 sizes fleets correctly
  for the target's *own* production growth, but doesn't know that an enemy
  500-ship fleet is also inbound to that target — our fleet still dies on
  contact at the combined garrison.
- **No same-turn combat order simulation** (rule 4: two-way ties
  destroy all attackers — exploitable by sending the *exact* tying
  ship count to neutralise an enemy assault for free).
- **No fleet coordination.** Each owned planet plans independently.
  Two of our planets may still both target the same neutral with
  `target.ships+1` ships each → wasted overlap. (Public top notebooks
  don't dedupe explicitly either; they let the world-model combat
  resolver handle it. We defer this to v2.)
- **No defence.** Garrison stays put. An incoming enemy 200-ship fleet
  walks into our 50-ship home undefended.
- **No comet-path leading.** v1.1's pipeline aims comets at their current
  position via `lead_aim`'s atan2 fallback. We have a `comet_aim`
  mechanism implemented + tested but excluded from `DEFAULT_MECHANISMS`
  (Step 3.5.C ablation: -22.5% — needs `search_safe_intercept` fallback).
- **No sun-avoidance.** `sun_avoid` mechanism implemented + tested but
  excluded (Step 3.5.D ablation: -32.5% on v1 — drop-only locks the
  agent on a sun-blocked nearest-target. Becomes positive at v2.)
- **No mission-classification.** Roman 1224 distinguishes snipe /
  rescue / recapture / reinforce / crash-exploit / gang-up / elimination
  and scores them against a single solver. v1.1 only does "snipe."

These gaps are the v2 / v3 build agenda — see `roadmap.md`.
