# 2026-05-10 — Capture-success probe (roi self-play, 32 seeds)

> Diagnostic for PI direction §Half 1 (physics correctness). Pre-registered
> in the strategic-direction plan as the gate that orders Half-1 vs Half-2
> investment. Source: `scripts/capture_probe.py`. Raw output:
> `audit/2026-05-10-capture-success-probe.json`.

## Method

Run `agents/simple/roi.py` against itself for 32 seeds (1..32) of
`make("orbit_wars", configuration={"seed": ...})`. Instrument
`propose_intents → DEFAULT_MECHANISMS → realize()` to log every
emitted launch with its declared `target_id` (the env strips this
to `[from_id, angle, ships]`, so the only way to keep target identity
is to hook the pipeline before the env sees it).

Post-game, match each logged launch to the env fleet it produced via
`(owner, from_planet_id, ships)` at the birth step, then classify
each fleet's outcome using the env's own collision rules
(`swept_pair_hit` in `orbit_wars.py` lines 46-67; planet-then-OOB-
then-sun precedence as in lines 568-609).

## Results

**24,431 launched fleets across 32 games, both players.**

| Outcome             | Count  | %     | Lever |
|---------------------|--------|-------|-------|
| reached             | 18,867 | 77.2% | (already correct)          |
| **collided_other**  | 2,608  | **10.7%** | **biggest physics loss**   |
| **oob**             | 1,858  | **7.6%**  | cheap pre-launch guard     |
| alive_at_end        |   578  | 2.4%  | endgame burn-through heuristic |
| **sun**             |   520  | **2.1%**  | punch #7 target — smallest |

158 of 24,589 logged launches (0.6%) failed to match an env fleet — most
likely simultaneous same-source launches with identical ship count.
Acceptable noise floor.

## Interpretation

### What "reached" actually means

The probe defines `reached` as "the fleet's swept segment intersected
the *declared* target planet's swept segment, accounting for orbital
motion." It does NOT measure combat success — a fleet that arrives
and loses the combat at the target still counts as "reached" here.
Combat-outcome is a separate physics layer (out of scope for this
probe). The 77.2% number is "did we deliver the ships we intended
to deliver."

### What flips the priority order

Before this probe, the open punch-list (HANDOVER §evening) ranked:

1. Punch #7 — sun-avoid arrival-aware
2. Punch #8 — 3-iter lead_aim with ETA fix
3. Capture-success probe (intended as the diagnostic)

This probe inverts that ordering on EV grounds.

- **Punch #7 ceiling = +2.1pp fleet success.** Sun deaths are only 2.1%
  of launches. Even a perfect sun-avoid implementation can only
  recover those 520 fleets. The mechanism is still worth fixing — but
  it is *not* the biggest physics lever.
- **collided_other (10.7%) is the largest physics loss.** These fleets
  hit a non-target planet via swept-pair collision — meaning either
  (i) the target was orbiting and the planet swept through the fleet
  path mid-flight, or (ii) a *different* orbiting planet swept into
  our fleet's path. lead_aim addresses (i) for the target but does
  nothing about (ii) — a third planet entering the path between our
  source and our target. **This is a new mechanism opportunity**:
  `path_clears_planets(fleet_old, fleet_new, omega, eta)` analogous
  to `path_clears_sun`, projecting third-party orbits and rejecting
  intents whose path is contested.
- **OOB (7.6%) is the cheap one.** Aim past the board edge → fleet
  vanishes. Trivial guard: after `lead_aim` sets `aim_angle`,
  forward-project to `fleet_new`; if outside [0, BOARD_SIZE], drop
  the intent (or re-aim toward a closer target).
- **alive_at_end (2.4%)** is the endgame-burn-through tail. Per
  research-note §G.10: in the last ~30 steps, ships in flight count
  toward the launcher's final score. Some of these 578 are "good"
  (intentional burn-through) and some are "bad" (launches placed too
  late to land). Separating those requires a step-aware accounting:
  out of scope for this probe; track in a later refinement.

### Strategic reframing

PI's Half-1 (physics correctness) direction is **validated** — 23% of
launched fleets are wasted before combat, which is a meaningful
ceiling on our μ. But the *order* of work inside Half-1 changes:

1. **Path-clears-other-planets mechanism (new)** — biggest single
   lever at ~10.7pp ceiling.
2. **OOB pre-launch guard** — cheap, ~7.6pp ceiling.
3. **Endgame burn-through** — accounting refinement, ~2.4pp ceiling
   but partly already-intentional.
4. **Punch #7 sun-avoid arrival-aware** — ~2.1pp ceiling. Still
   worth fixing (cheap, shares the arrival-point helper), but its
   priority drops.
5. **Punch #8 3-iter lead_aim** — sub-2pp ceiling on the orbit-
   prediction edge cases (target collisions are already 77.2%
   correct; the gap is tiny).

The combined ceiling if all four physics fixes land cleanly is roughly
+23pp on fleet-success rate. Whether that translates linearly to μ
depends on how many of those wasted ships would have changed game
outcomes — but +23pp is a real lever.

### Half-2 vs Half-1 budget call

The probe also informs Half-2 priority. The arrival-ledger work is
still required (don't-double-commit, intercept-enemy use cases) and
the path-clears-other-planets mechanism naturally consumes the same
orbital-prediction primitive (`predict_relative` from `lib/orbit.py`)
that the arrival-ledger needs. So Half-1 fixes #1 and #2 above
**share substrate** with the v2 roadmap deliverable.

Recommendation: build the path-clears-other-planets mechanism + OOB
guard as part of v2's `arrival_ledger` work, not as separate punches.
That collapses three roadmap items into one substrate build.

## Implications for the strategic-direction plan

Update `/root/.claude/plans/you-are-a-champion-sprightly-sunset.md`
Priority 2 sequence:

- **Step 3 (was "Punch #7+#8")** becomes **"Path-clears-other-planets
  mechanism + OOB guard"** (biggest ceiling, shares orbital-prediction
  with the next item).
- **Step 4 (was arrival-ledger)** keeps its name — the data structure
  is the same.
- **Punch #7 / #8** demote to a polish pass after the bigger fixes
  land; or fold #7 into the new mechanism since they share the
  arrival-point helper.

## Open questions for follow-up

- **Combat success at the target.** Of the 77.2% that reach, how many
  actually flip the planet? Probe extension needed: cross-reference
  arrival step with planet ownership change.
- **Endgame burn-through accounting.** Split `alive_at_end` into
  "good burn" (last 30 steps, launched from a low-prod source) vs
  "wasted launch" (mid-game, never landed). Cheap extension.
- **Is the 10.7% collision symmetric across players?** Seeds 2, 3, 5,
  20+ show symmetric outcomes (roi-vs-roi self-play is deterministic
  per-seed). Seeds where the game ends early (1, 4) have asymmetric
  outcomes — these are the ones worth eyeballing for surprises.
- **Does the same probe on v1.1 / nearest show different numbers?**
  Comparison panel could distinguish "physics losses inherent to the
  scoring rule" from "physics losses we could engineer out."
