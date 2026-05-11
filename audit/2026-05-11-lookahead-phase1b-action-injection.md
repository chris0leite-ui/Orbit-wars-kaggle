# Lookahead Phase 1b — one-turn action injection

> Date: 2026-05-11
> Branch: `claude/bootstrap-agentic-systems-lqnm6`
> Hypothesis: adding this-turn launches as phantom fleets to the
> WorldModel might close part of the Phase 1a oracle-vs-static gap.

## TL;DR

Hypothesis falsified. Injecting one turn of action (ours or both
players') into the WorldModel does NOT improve AUC — within ~0.005
across every (step, horizon) cell tested. The Phase 1a oracle gap
(~32pp at H=50 from step 50) lives in the **sequence** of K future
turns of actions, not in the single boundary turn. Cheap intermediate
fix is dead; Phase 2 (K-step forward simulation) is the load-bearing
experiment.

## Probe extension

Two new predictor families added to `scripts/lookahead_probe.py`:

- **`Hours<H>`** — WorldModel built from obs that has been augmented
  with our (POV) this-turn launches converted to phantom fleets.
  Spawn rule matches the env: `(src.x + cos(angle) · (radius+0.1),
  src.y + sin(angle) · (radius+0.1))` with the launch's angle, ships,
  source planet. Same horizon, same scoring.
- **`Hall<H>`** — same idea but injects BOTH players' this-turn
  launches. Counterfactual ceiling — we wouldn't normally know the
  opponent's launch in real-time, but if the LIFT from action-injection
  is real, Hall should show it strongest.

Same setup as Phase 1a: 32 seeds × {v2 (P0), roi_baseline (P1)}, probe
steps {25, 50, 75, 100, 150}, horizons {50, 100, 200}. 156 samples after
edge-drops. Artifact: `audit/lookahead/20260511T062658Z.json`.

## Results

```
  --- horizon = 50 ---
      step     naive       H50   Hours50    Hall50       O50
        25     0.466     0.502     0.502     0.495     0.864
        50     0.595     0.641     0.639     0.639     0.916
        75     0.864     0.841     0.836     0.832     0.986
       100     0.916     0.850     0.845     0.841     1.000
       150     1.000     1.000     1.000     1.000     1.000

  --- horizon = 100 ---
      step     naive      H100  Hours100   Hall100      O100
        25     0.466     0.493     0.493     0.486     0.986
        50     0.595     0.625     0.625     0.625     1.000
        75     0.864     0.832     0.827     0.827     1.000
       100     0.916     0.850     0.841     0.841     1.000
       150     1.000     1.000     1.000     1.000     1.000

  --- horizon = 200 ---
      step     naive      H200  Hours200   Hall200      O200
        25     0.466     0.493     0.493     0.486     1.000
        50     0.595     0.598     0.598     0.598     1.000
        75     0.864     0.836     0.832     0.832     1.000
       100     0.916     0.836     0.832     0.832     1.000
       150     1.000     1.000     1.000     1.000     1.000
```

(AUC 0.5 = no signal; 1.000 = perfect. n ≈ 32 per row.)

**`Hours<H>` and `Hall<H>` are statistically indistinguishable from
plain `H<H>`** across every cell. The Phase 1a oracle gap is unchanged.

## Diagnosis

Spot-checked sample data confirms the injection is wired correctly:
seed 42 / step 25 / 1 opp launch produces `H50 = 14`, `Hours50 = 14`
(no our action), `Hall50 = 8` — the opp's launch shrank our predicted
50-step delta by 6 ships, as expected.

But the AUC-level effect is null. The cause is structural, not bugged:

- **55% of probe-sample turns have ZERO `n_our_launches`** (86/156
  samples). v2 is selective — it doesn't launch every turn. The action
  injection literally has nothing to inject for the majority of rows.
- When launches DO happen, it's 1-5 fleets per turn. Even at H=50
  that's at most 5 phantom fleets adding to an arrival ledger that's
  already integrating, say, 200+ fleets across both sides' upcoming
  50 turns.
- The 32pp signal gap lives in the FULL SEQUENCE of future actions,
  not the boundary turn. One turn captures ≈ 1/50 of the gap; the
  data is consistent with that (~0pp lift detected against 32pp
  available).

## What this rules in and out

**Ruled out** — cheap one-turn action-injection extensions. The
existing `arrival_ledger` mechanism (excluded from DEFAULT for the
reason in `audit/friction.md::arrival-ledger-mechanism-without-
planner-regresses`) does the same thing for in-flight fleets and is
at the same ceiling. There's no cheap intermediate between snapshot
and full forward sim.

**Ruled in** — Phase 2: full K-step policy-aware forward simulation.
The Phase 1b null is strong enough evidence that any intermediate
probe (e.g. inject 2-3 turns of actions) will also fail; the only
question is how big K needs to be to close most of the oracle gap.

## Suggested Phase 2 design (for PI approval)

```
def project_us_minus_them(world, K=50, policy=v2_policy) -> float:
    sim = clone(world)              # deep-copy env state from world snapshot
    for _ in range(K):
        actions = [policy(sim.obs(0)), policy(sim.obs(1))]
        sim.step(actions)
    return ship_total(sim, P0) - ship_total(sim, P1)
```

Cost: ~1.35 ms (v2 turn) × 2 players × 50 simulated turns ≈ **135 ms
per evaluation**. Budget 1000 ms / per-turn = room for ~5-8 candidate
this-turn intent sets per real turn.

Validation gate: drop `project_us_minus_them` into the probe as a
new predictor family (`Sim50` etc.) and measure AUC vs the oracle. If
`Sim50` at step 50 lifts from H50=0.641 toward O50=0.916, the lookahead
substrate works. If it's still ~0.64, the cheap policy roll-forward
isn't fidelity-enough either and we have to look at iterative
re-planning or learned value heads.

## Implementation prerequisite

`kaggle_environments` doesn't expose a clean way to "clone the env
state and step forward from an arbitrary World snapshot." Building one
needs either:
- A pure-Python re-implementation of the env's step function that
  operates on a `World` snapshot (the existing `WorldModel.
  simulate_planet_timeline` is half of this; missing: fleet launching,
  fleet movement, comet handling, sun collision).
- A monkey-patched `make("orbit_wars")` with state reconstruction from
  a serialised obs (likely fragile against env upgrades).

The pure-Python step is the cleaner long-term path; ~1 day of work to
get to 80% parity (no comets / static planets first, orbiting + comets
deferred to a second pass). Roman's kernel likely has this re-impl;
worth pulling it for reference (currently gitignored / not on disk).
