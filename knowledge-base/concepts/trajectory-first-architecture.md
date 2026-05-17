# Trajectory-first proposer — the sparse-and-accurate world-model reframe

> Filed 2026-05-17 PM after PI critique: "I don't understand how there
> can be any sun accidents. The set of possible trajectories is finite.
> To filter the set of admissible trajectories should be easy and fast.
> That should be deterministic. Our model of the world is not efficient,
> it should be much sparser 100% accurate. We should be thinking in
> fleet trajectories."

Companion to `worldmodel-reuse-options.md` (perf-only refactor for the
current composite head). This is the structural refactor: replace
"approximate value function over K-step rollout" with "deterministic
trajectory analysis + sparse combat prediction at the single arrival
tick."

## What's deterministic given launch params

A candidate launch is `(src_xy, angle, ships)`. Everything below is
closed-form computable in O(1)-ish per candidate per tick:

- **Speed** — `lib.fleet.speed(ships)`. 1-call.
- **Position at tick t** — `(src.x + cos(angle)·spd·t,
  src.y + sin(angle)·spd·t)`. 2-mul.
- **Sun crossing** — `point_to_segment_distance((CENTER, CENTER),
  pos_t, pos_{t+1}) < SUN_RADIUS`. Closed-form per tick. Both
  primitives already in `lib/game/interpreter.py:175-244`.
- **OOB** — bounds check on `pos_t`. 1 comparison.
- **Target acquisition** — first planet whose segment-radius the fleet
  crosses. Same `fleet_target_planet` ray-cast we have today
  (`lib/world_model.py:42-82`).
- **ETA** — `hit_distance / speed`. Already returned.
- **Comet expiry vs ETA** — `comet.path_index + eta >= len(path)`.
  Already wrapped as `comet_remaining_lifetime`
  (`lib/world_model.py:297-314`).
- **Comet collision** — `swept_pair_hit(fleet_seg, comet_seg,
  COMET_RADIUS)` per tick along the trajectory. Both primitives in
  `lib/game/interpreter.py`.

A trajectory is **admissible** iff: no sun, no OOB, no comet collision
en route, and the target planet exists at ETA (lifetime check).

**Admissibility filter cost**: O(eta × (planets + comets)) per
candidate. For typical eta=15, planets=20, comets=4 → ~360 ops.
~50µs per candidate in Python. At 60 candidates per turn: 3 ms total.
Effectively free vs the current chooser's 200-600 ms.

## What's NOT trajectory-deterministic

- **Combat outcome at target** — depends on:
  - Defender garrison at ETA (production accumulated + any combat).
  - Other friendly fleets arriving same tick (combine forces).
  - Enemy fleets arriving by ETA (subtract).
- **Enemy AI choices** — what will they launch in the next eta ticks?
  Still need an opponent model.

For combat, the right answer is **sparse single-tick prediction**, not
the current per-planet horizon-step timeline:

```
defender_at_eta = current_garrison + production × eta
                  + Σ enemy_arrivals[t < eta]
                  - Σ friendly_arrivals[t < eta]
```

This is ONE arithmetic per planet, not 30 simulation steps. The arrival
ledger is already built (`lib.world_model.build_arrival_ledger`). The
expensive thing today is `simulate_planet_timeline` re-running combat
at every horizon step — we don't need that resolution; we need the
state at exactly `eta`.

## Architecture sketch

### Layer 1 — Per-turn enumeration + admissibility filter

`agents/baseline/proposer.py`:

```python
def enumerate_candidates(world, my_id, ...) -> list[Candidate]:
    """Build the FULL candidate space then filter for admissibility.

    For each (src, tgt, wait_N, ships) combination:
      - Compute angle (atan2 + lead-aim for orbiting/comet targets).
      - Run the trajectory through `is_admissible(...)` — drop if
        sun-cross, OOB, comet-collision-en-route, target-expired-by-eta.
    Returns only viable candidates.
    """

def is_admissible(src_xy, angle, ships, world) -> tuple[bool, str | None]:
    """Pure trajectory check. Returns (ok, reason_if_rejected)."""
```

This subsumes the current cheap-rank pre-filter. No approximate
scoring at this layer.

### Layer 2 — Sparse arrival prediction

`lib/world_model.py` (new function alongside `simulate_planet_timeline`):

```python
def predict_garrison_at(planet_id: int, eta: int, world: World,
                        arrival_ledger: dict) -> tuple[int, float]:
    """Return (predicted_owner, predicted_garrison) at exactly `eta`
    ticks from now, computed in O(1) from the arrival ledger.

    No per-step simulation. The arrival order at eta determines who
    holds it; production accumulates linearly until the first flip,
    then resets.
    """
```

For each admissible candidate, call `predict_garrison_at(target.id,
candidate.eta + 1, world, ledger)` to score the engagement. ~10 µs
per call.

### Layer 3 — Direct scoring

Per candidate score (no value-head approximation):

```
if pred_owner_pre_arrival == my_id:
    # We already own it — reinforcement. Skip; the chooser handles
    # defense via threat-aware reinforce missions.
    continue
if ships > pred_garrison_at_arrival:
    # Capture. Score = production × (episode_remaining - eta),
    # capped at comet_remaining_lifetime - eta if applicable.
    score = production × min(time_remaining, comet_life_cap)
else:
    # Bounce. Score = -ships (deterministic cost).
    score = -ships
```

No `composite_capture_value`, no `favor`, no fast_sim rollout — direct
arithmetic. PI's "sparse and 100% accurate" target.

### Layer 4 — Selection

Same as today: greedy non-dogpile by score, 1 launch per source per
target per turn, wait_N>0 winners reserve.

## What this solves vs what it doesn't

| failure mode | trajectory-first fixes? | mechanism |
|---|---|---|
| Sun-death | **0%** (deterministic prefilter) | trajectory cross-check |
| OOB | **0%** | trajectory bounds check |
| Comet-expired-by-arrival | **0%** | `comet_remaining_lifetime` gate |
| Comet-collision-en-route | **0%** | swept-pair per tick |
| Lead-aim error for orbiting target | **0%** if lead-aim is on | reuse `lib.mech.lead_aim` |
| Combat bounce (under-sized attack) | **better** | sparse prediction is accurate at exactly eta |
| Combat loss to multi-fleet arrival | **better** | arrival ledger captures combined forces |
| Opponent counter-launches | **same** | still need opponent model |

The four "0%" rows are the failure modes that motivated this concept.
The "better" rows recover some of what's lost when we ditch the K-step
rollout (we trade rollout-induced approximation for single-tick
exactness, and we get back full rollout depth via the arrival ledger
which has horizon-independent reach).

## Cost reduction

- **Current**: `build_idle_baseline` (40 favor calls × 2-5ms composite
  leaf each) + `score_action` for ~30 candidates (each running K-step
  rollout + 1 composite leaf) = **~200-700 ms per turn**.
- **Trajectory-first**: enumerate (~100 candidates × 50 µs admissibility
  + 10 µs scoring) = **~6 ms per turn**.

30-100× speedup. Wallclock issue (max 1196-1580 ms) goes away.

## Migration plan

1. Build the admissibility filter as a standalone function with
   unit tests against synthetic obs (sun cases, OOB cases, comet
   cases). ~2 h.
2. Build `predict_garrison_at` against the existing arrival ledger;
   parity-test against `simulate_planet_timeline` for the
   single-tick case. ~3 h.
3. Add a new chooser variant `chooser_trajectory_first` that uses
   layers 1-4 directly. Keep `chooser` (current modular baseline) for
   A/B comparison. ~4 h.
4. Smoke + A/B vs current baseline at n=32. ~2 h compute.
5. If A/B beats current baseline → make trajectory-first the default;
   deprecate composite/favor leaf value head.

**Total estimate**: 1-1.5 focused days. Half-day of code + half-day
of A/B + iteration buffer.

## What this plan deliberately does NOT do

- Does not delete `composite_capture_value`. Keep as fallback / for the
  "explore broader candidate spaces" case where exact prediction
  isn't available.
- Does not change the opp model. `lite_greedy_policy` still drives
  reactive rollouts if we want them in a hybrid mode.
- Does not address 4P FFA-specific dynamics (gang-ups, second-place
  optimization). Those are orthogonal.

## Cross-references

- `lib/game/interpreter.py:175-244` — sun + OOB + swept-pair primitives.
- `lib/fleet.py:speed` — speed formula.
- `lib/world_model.py:42-82,265-314` — fleet_target_planet,
  comet helpers, build_arrival_ledger, simulate_planet_timeline.
- `lib/mechanism.py:446-509` — comet_aim (lead-aim for moving targets).
- `agents/baseline/proposer.py` — current candidate enumerator
  (where layer-1 admissibility goes).
- `agents/baseline/chooser.py` — current K-step rollout (replaced by
  layers 2-4).
- `audit/2026-05-17-sun-death-investigation.md` — observation that
  motivated this concept (PI: "we should be thinking in fleet
  trajectories").
- `knowledge-base/concepts/worldmodel-reuse-options.md` — companion
  perf-only refactor for the current composite head; trajectory-first
  is the architecturally cleaner alternative.

## Open question

Does the **chooser still need rollout depth** for non-leaf inspection
(e.g. "does my action provoke an enemy counter-launch that changes my
NEXT turn's options")? Probably yes for tournament play, but the
horizon for that should be 1-2 turns of ENEMY ACTION simulation, not
40 ticks of static-world drift. That's a different kind of model.
