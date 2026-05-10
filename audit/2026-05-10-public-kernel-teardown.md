# 2026-05-10 — Public kernel teardown (Roman 1224 + Pilkwang + sigmaborov + sun-dodging)

> Pulled via `kaggle kernels pull` into `external/kernels/`. Each notebook
> contains one or more code cells; extracted to `<kernel>/agent.py` for
> reading. This audit answers PI plan Priority 1 ("mine Roman").

## Files

| Kernel | Lines | Author | Date | Claimed μ |
|---|---|---|---|---|
| roman-1224 | 3,307 | Roman Tamrazov | 2026-04-20 | 1224 |
| structured-baseline | 3,469 | Pilkwang Kim | 2026-04-21 | (baseline) |
| physics-accurate-planner | 1,995 | sigmaborov | 2026-04-18 | 928.7 |
| sun-dodging | 2,173 | Natsu Yamaguchi | 2026-04-24 | (baseline) |

## Calibration recalc — the field has moved

| Reference μ | Source | Notes |
|---|---|---|
| **1663.4** | LB #1 (bowwowforeach, 2026-05-10) | Private; new today. |
| **1641.9** | LB #2 (flg, 2026-05-07) | Plausibly Roman post-iteration (initials match a Russian name). |
| **1577.3** | LB #3 (Ebi) | Private. |
| **1460.0** | LB #10 (sash) | **The prize cliff.** |
| **1224** | Roman's published kernel | The strongest public-known agent. |
| **978.7** | Us (v1.2/roi) | TODAY — settled lower than the evening 1105 reading. |

**Top-10 cutoff is μ ≈ 1460, NOT 1100.** Our gap to the cliff is +481 μ.
Roman's published μ=1224 is 236 below the cliff. **Adopting Roman's full
architecture is necessary but not sufficient for a prize spot.** There is
~240 μ of "what the top players know that Roman did not publish" still
to close.

## Architecture — Roman and Pilkwang share the same skeleton

Both kernels have the same six-section structure with near-identical
function names. Pilkwang published one day after Roman; one is forked
from the other, or both descend from a common ancestor. Function-level
overlap:

```
Shared Configuration   →  SUN_SAFETY=1.5, HORIZON=110, COMET_MAX_CHASE_TURNS=10
Shared Types           →  ShotOption, Mission dataclasses
Physics                →  dist, fleet_speed, point_to_segment_distance,
                          segment_hits_sun, predict_planet_position,
                          predict_comet_position, estimate_arrival,
                          safe_angle_and_distance,
                          search_safe_intercept,    ← orbital intercept search
                          aim_with_prediction       ← 5-iter fixed-point
World Model            →  build_arrival_ledger,    ← our v2 substrate
                          resolve_arrival_event,   ← combat resolver
                          simulate_planet_timeline,← per-planet timeline
                          state_at_timeline, count_players,
                          detect_enemy_crashes,
                          WorldModel class         ← central state
Strategy               →  build_modes, target_value, plan_moves
                          + mission builders (see below)
Agent Entry            →  agent(obs)
```

**Roman adds over Pilkwang** (the extra ~150 μ on the public LB):

- Mission builders Pilkwang lacks: `rescue`, `recapture`, `reinforce`,
  `gang_up`, `elimination` (Pilkwang has only snipe / reinforcement /
  crash_exploit).
- Policy-state helpers: `min_legal_reaction_time`,
  `policy_reaction_times`, `stacked_enemy_proactive_keep`,
  `swarm_eta_tolerance`.
- Extra detection: `detect_enemy_planet_battles` (predict where enemy
  agents are fighting each other so we can exploit).
- Many more value multipliers — `HOSTILE_TARGET_VALUE_MULT`,
  `EXPOSED_PLANET_VALUE_MULT`, `FINISHING_HOSTILE_VALUE_MULT`,
  `BEHIND_ROTATING_NEUTRAL_VALUE_MULT`, etc. Roman is **heavily
  parameter-tuned**.

## Things Roman does that we don't (the punch list)

Ordered by my read on μ-lift potential:

1. **Arrival ledger + planet-timeline simulator** (build_arrival_ledger /
   simulate_planet_timeline / state_at_timeline). The "don't double-
   commit," "intercept enemy arrivals," and "predict planet ownership
   at arrival" features all live here. Our v2 substrate.

2. **Mission framework with global settle_plan solver.** Instead of
   per-source greedy (our current architecture), Roman builds candidate
   Missions across all (source, target, mission_class) triples, scores
   them via `target_value`, then runs `settle_plan` — a global solver
   that allocates source-planet garrisons to mission proposals under
   no-overlap constraints. This is the Hungarian-assignment idea (H8
   in our hypothesis board) — but extended with mission classes.

3. **`search_safe_intercept`** (lines 354-397). For orbital / comet
   targets that lead-aim cannot converge on (or where the direct path
   crosses the sun), this iterates candidate arrival turns over the
   60-step search horizon and finds a self-consistent intercept point
   (where predicted ETA matches the candidate turn within ±1). This
   is **a stronger sun-avoid + a fix for the orbital lead-aim edge
   cases our probe attributed to "collided_other."** Roman calls
   this from `aim_with_prediction` as a fallback when the 5-iter
   fixed point fails to converge.

4. **5-iter `aim_with_prediction`** (lines 399-436). Our v1 uses 1-iter
   fixed point; punch #8 was queued as 3-iter. Roman uses **5 iter
   AND** an explicit convergence check (`abs(ntx - tx) < 0.3 and
   abs(nty - ty) < 0.3`). When iteration doesn't converge, he falls
   back to `search_safe_intercept`. This is the right pattern.

5. **Per-mission scoring with class-specific multipliers.** Different
   missions (snipe / recapture / reinforce / gang_up / elimination)
   use different value formulas. snipe has `SNIPE_VALUE_MULT=1.12`,
   reinforce has `REINFORCE_VALUE_MULT=1.35`, etc. Our single `roi =
   prod/dist` score collapses all this.

6. **Mode / phase logic** (`build_modes`). Roman selects modes like
   `EARLY_TURN_LIMIT=40`, `LATE_REMAINING_TURNS=70`, and explicit
   `OPENING_HOSTILE_TARGET_VALUE_MULT` for phase-conditional scoring.
   This is our research-note §D.6 phase segmentation — operationalised.

7. **4-player-specific tuning.** Roman has
   `FOUR_PLAYER_ROTATING_SEND_RATIO=0.55` (recently lowered from 0.62 —
   "less overcommit in 4P"). **This confirms PI decision #4: 4P games
   ARE meaningful on the ladder.** Top players tune separately for it.

8. **Sun-avoid `safe_angle_and_distance`** with `SUN_SAFETY=1.5` margin
   (vs the env's 0). The non-zero safety margin is risk-aware — a
   fleet aimed exactly at SUN_RADIUS distance is one float error from
   dying.

9. **Comet chase gate.** `COMET_MAX_CHASE_TURNS=10` — bounded comet
   chasing matches our G.14 intuition. Currently we have `comet_aim`
   disabled because the ablation regressed; this gate is the missing
   piece.

10. **`detect_enemy_crashes`** — when enemy fleets are about to die
    (sun / OOB / collision), opportunistically capture the targets
    those fleets were going for. A free win on opponent mistakes.

## Strategic implications — what changes in our plan

### Calibration update (load-bearing)

- v1.2/roi settled at **μ=978.7** (not 1105). We are well below top-5%
  and **the gap to the prize cliff is +481 μ**. The plan's μ-floor of
  1400-1500 is now genuinely ambitious — it requires not just Roman's
  architecture but ~240 μ of work *on top* of Roman.
- The 44-day budget is the same. The required work per week is higher.

### Architecture decision (PI plan Q6): **adapt Roman's structures into our pipeline**

Confirmed after reading the kernels. Roman's Strategy/Mission/Mechanism
split maps cleanly onto our existing `Strategy → Intent → realize()`:

| Roman | Us (current) | Us (v3 proposed) |
|---|---|---|
| `aim_with_prediction` / `search_safe_intercept` | `lead_aim` mechanism | `lead_aim` (extend to 5-iter + safe-intercept fallback) |
| `safe_angle_and_distance` | (none) | new `sun_safe` mechanism (punch #7) |
| `build_arrival_ledger` + `WorldModel` | (none) | new `lib/world_model.py` (v2 substrate) |
| `target_value` + `build_*_missions` | `Strategy.propose_intents` | new `lib/missions.py` — one builder per class; each emits Mission proposals |
| `settle_plan` | (none — per-source greedy) | new `lib/solver.py` — global bipartite assignment + same-step-arrival timing |
| `realize()` | `realize()` | unchanged |

The Mechanism layer survives intact. The Strategy layer evolves from
"per-source argmax" to "build candidate Missions across all
(source, target, class) triples". A new Planner/Solver layer sits
above Strategy and does the global allocation.

This is **more invasive than v2 → v3 in the existing roadmap suggests**
because the per-source greedy is being replaced, not extended. But the
mechanism layer (validate / arrival_size / lead_aim / sun_avoid) keeps
its API and gets reused by every mission class. So our existing tests
(151+ green) remain mostly valid.

### Revised priorities

The capture-success probe's reframing (sun=2.1%, collided_other=10.7%,
oob=7.6%) and Roman's existence change the order:

- **P1 — Replicate Roman's physics module into our `lib/` layer.**
  Specifically: 5-iter `aim_with_prediction` + `search_safe_intercept`
  fallback + `safe_angle_and_distance` sun-safe routing + `predict_planet_position`. This addresses:
  - punch #7 (sun-avoid) — included.
  - punch #8 (3-iter → 5-iter) — included.
  - much of the 10.7% collided_other — `search_safe_intercept` finds
    self-consistent intercepts that lead_aim's 1-iter misses.
  - the OOB 7.6% — `estimate_arrival` returns None on impossible
    intercepts; we drop those intents.
  **Cost: ~1 day. EV: large fraction of the +23pp physics-loss bucket.**

- **P2 — Build `lib/world_model.py` (arrival ledger + timeline
  simulator + WorldModel class).** This is the v2 substrate. Reuse
  Roman's `build_arrival_ledger`, `resolve_arrival_event`,
  `simulate_planet_timeline` as the starting blueprint; adapt to our
  module layout. **Cost: ~2-3 days.**

- **P3 — Mission framework.** Start with `snipe` only as the proof
  that the pipeline composes; then add `reinforce`, `recapture`,
  `gang_up`. Defer `rescue`, `crash_exploit`, `elimination` to v3.5.
  **Cost: ~5-7 days for the snipe pipeline + first 3 missions.**

- **P4 — `settle_plan` solver.** Global assignment + same-step-arrival
  timing. The piece that wins the +120 μ Roman has over Pilkwang.
  **Cost: ~2-3 days.**

- **P5 — Closing the +240 μ on top of Roman.** Open question. Hypotheses:
  - Better tuning of value multipliers (Roman's are hand-tuned).
  - A learned value function on top of his rule structure (IL on the
    top public replays we can capture).
  - 4P-FFA-specific play (Roman has the constants but not necessarily
    the optimum).
  - Look-ahead search (depth-2 MCTS over the top-K mission
    combinations from settle_plan).

Total v2→v3 budget: ~12-15 days; v3 in by ~2026-05-25. Leaves ~14
days for P5 R&D + submission cadence + the final-window lock.

### Risks

- **License.** Roman's notebook is Apache 2.0 (Kaggle default) — we can
  read and adapt patterns, but we should not copy verbatim
  function-by-function. The teardown approach (map his ideas onto our
  module layout, re-implement) is the right legal posture.
- **Tooling debt.** Replicating Roman's tuning constants (40+
  multipliers) without his tuning context is a tarpit. Plan: take his
  defaults as starting points, run our own ablation tournaments on
  each multiplier as we go.
- **Wallclock.** Roman's WorldModel does a lot per turn. We need to
  profile against the 1s `actTimeout` early — fragmenting into 40+
  source planets × 40 targets × 7 mission classes is O(8000)
  proposals/turn before the solver. Memoise aggressively.

## Verification — quick sanity checks done

- `external/kernels/{roman-1224,structured-baseline,physics-accurate-planner,sun-dodging}/agent.py`
  all present and >50KB each.
- Roman and Pilkwang's function signatures match for the shared
  skeleton; their `build_arrival_ledger` is byte-identical (modulo
  whitespace).
- Roman's `agent` entry point reads `obs.player`, `obs.planets`,
  `obs.fleets`, `obs.angular_velocity`, `obs.initial_planets`,
  `obs.comets`, `obs.comet_planet_ids`, `obs.step` — all fields we
  already destructure in `lib/intent.py::World.from_obs`. No new IO
  contract.
- 4P-specific constants present in Roman; matches the env's 2-or-4
  player spec.
