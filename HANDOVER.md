# HANDOVER.md — next-session brief

> Last written: 2026-05-13 EVE by `claude/consolidate-fast-simulation-ysd9M`.
> Format budget ≤ 160 lines. Prior wraps archived under
> `audit/archive-2026-05-1*-handover-*.md` (including
> `archive-2026-05-13-handover-pre-jax-sprint.md` for the Phase 2/3a
> content this rewrite replaces).

## Where we are

- **Comp:** Orbit Wars (slug `orbit-wars`). Deadline 2026-06-23 23:59 UTC →
  **41 days remaining.**
- **Live submitted agent:** `v7_0_drop_one`, **submission #52588156**,
  publicScore **μ=1094.9** (team peak). Public-LB rank **109 / 2587**
  (top 4.2 %). Gap to top-10 prize: +336 μ.
- **Rolling-last-2:** `[v4_planner #52579863, v7_0_drop_one #52588156]`.
- **JAX sprint complete.** 9 sub-phases (1-8f), ~30 commits.
  117 / 118 JAX tests green. The Kaggle T4 vmap A/B harness
  (`chrisleitescha/orbit-wars-jax-a-b`) runs 64 games × 500 turns
  in **1.5 min wallclock** (vs ~30 min scalar on this 4-core box).
  All OFFLINE-ONLY — the live submission stays the pure-Python
  `v7_0_drop_one` bundle.

## This session — JAX sprint sub-phase 8e + 8f

- **8e:** closed the pure-JAX path parity gaps (sun_avoid +
  path_clears + oob_guard + search_safe_intercept fallback). Angle
  parity tightened to 0.02 rad. (commit `b67c868`, kernel v5-v7)
- **8f:** resolved findings from a competitive-programmer code
  review:
  - **C1:** `fleet_launch` slot-allocation off-by-one was silently
    wasting half the fleet capacity. Fixed.
  - **C2:** lead-aim convergence tolerance 0.5 → 0.3 (matches scalar).
  - **C4:** terminate now masks reward writes by `num_agents` (no
    more `-1` on phantom seats 2-3 in 2P games).
  - **B/G/H/J:** rollout-pure 2P assert; `A_AGGRESSIVE` knob now
    works; harness verdict uses `state.rewards` (engine win/lose)
    not ship-delta proxy; `cometSpeed` plumbed through.
  - **P1/P4/D:** orbit-table build hoisted; comet-schedule
    `lru_cache`'d (-25 % setup); removed per-K host transfer.
  - **T1/T2/T3:** new scalar-vs-JAX rollout parity test + C1 sanity
    probe + tightened existing tolerances.

Kaggle T4 post-8f: init 11.8 s, compile 67.2 s, hot 13.0 s. A=25 %
winrate vs B at Wilson lo 0.160 (matches scalar reference).

## Falsified or dead

- `v7_wide_deep` (kitchen-sink: combined enumerator + K=25 + maximin
  + composite value): failed 4-seed A/B (2/8 wins, Wilson lo 0.071).
  See `audit/2026-05-12-v7-wide-deep-32-seed.md`.
- `v7_1_minimax`..`v7_6_no_recapture` ablations: all under v7_0
  baseline. Falsified pre-merge; bundles removed.

## Next-session first-action — rapid-iteration platform

PI request: build infrastructure for fast experimentation across
several ideas. **Gated on a code review of the platform plan before
implementation** (the JAX sprint hit two correctness bugs that a
review caught; same discipline applies to the next refactor).

### Proposed platform refactor (~1-2 days, ~800 lines)

Three foundation pieces that unlock all the experiments below:

1. **Precomputed planet trajectories on GameState.** Move
   `_build_planet_orbits_jax` from per-turn → per-game; store
   `planet_trajectory: f32[P_max, 500, 2]` in `GameState`. Simplifies
   decision code (reachability + distance fields become gathers).
   ~50-100 lines.
2. **`AgentState` Pytree threaded through `lax.scan` carry.**
   Today the rollout scan carry is just `GameState`; extend to
   `(GameState, AgentState)`. Unlocks turn-to-turn memory inside the
   vmap'd kernel (recapture, opponent classifier, plan cache).
   ~150-200 lines.
3. **`StrategyConfig` dataclass + `build_policy_from_config()`
   factory.** Drives mission classes, value head, chooser, opp tier,
   `AgentState` shape. Kernel A/B takes two named configs (YAML/dict)
   instead of code edits per experiment. ~300-500 lines.

With those three, the per-experiment iteration cost collapses from
~half-day to ~30 min for non-trivial strategy changes.

### Five experiments queued (post-platform)

| Idea | Feasibility | Code scope | Notes |
|---|---|---|---|
| **A. Simplified rep via precomputed trajectories** | High | folded into platform piece 1 | Distance / reachability fields become gathers; ~5-10 ms/turn savings |
| **B. Turn-to-turn state carry** | High | folded into platform piece 2 | Enables recapture inside vmap rollouts, opp behaviour model, plan cache |
| **C. Multi-launch from one planet per turn** | **Env already supports this.** Same `src_id` can appear multiple times in actions; scalar `fleet_launch` subtracts sequentially. | ~100 lines (settle_plan N-per-source variant or "split" candidate enumerator) | Strategic lever currently unused |
| **D. Strategy library + meta-selector** | High but biggest unknown | ~500 lines | Strategy registry + `lax.switch` for vmap; meta-controller is its own ML problem |
| **E. Geometry-based opening classifier (small ML)** | High — Kaggle T4 harness is the data engine | ~300 lines + offline training | InitialFeatures extractor + ~200-float logistic/MLP; embed in bundle |

### Recommended order

1. Read pre-build code review (PI gate).
2. Platform piece 1 (precomputed trajectories) — ~2 hrs.
3. Platform piece 2 (AgentState scan carry) — ~half day.
4. Platform piece 3 (StrategyConfig + YAML kernel) — ~1 day.
5. Experiment C (multi-launch) — ~half day; lowest cost, high lever.
6. Experiment E (opening classifier) — ~1-2 days; ML plug-in.
7. Experiment D (strategy library) — ~1 week; biggest payoff.

(A and B are absorbed into the platform; "do" them by building the
platform piece they belong to.)

## Out of scope for next session

- **No new submission.** The JAX path is offline-only by design.
  Live bundle stays `v7_0_drop_one` until a new agent passes the 32-
  seed scalar A/B and PI authorises.
- **Deferred from 8f:** P3 (kill numpy-mixed agent path via candidate-
  override) — ~150-line refactor; Q1 (dedup numpy/JAX mirror in
  `jax_mechanisms.py`) — ~250 lines; Q6 (CI bounds probe for
  MAX_PLANETS/MAX_FLEETS).
- **PR to main:** opening this session per PI direction; future
  branch work merges via standard PR flow.

## Pointers — JAX sprint deliverables

- `lib/game/jax/jax_interpreter.py` — 11-phase engine port.
- `lib/game/jax/jax_world_model.py` — vectorised timeline forecast.
- `lib/game/jax/jax_missions.py` — snipe/reinforce/recapture score
  matrices + `settle_plan_jax` via `lax.scan`.
- `lib/game/jax/jax_mechanisms.py` — validate / arrival_size /
  lead_aim_v2 / sun_avoid / path_clears / oob_guard (numpy mirror +
  vmap'd JAX form).
- `lib/game/jax/jax_score.py` — `score_candidate_jax` (agent path) +
  `score_candidate_jax_pure` (vmap'd K-step rollout).
- `agents/jax_v7_0/main.py` — drop-one wrapper that takes obs →
  GameState → JAX rollout.
- `scripts/kaggle_ab_kernel/run_jax_ab.py` — vmap'd 64-game A/B
  harness; deployed as kernel `chrisleitescha/orbit-wars-jax-a-b`.
- `tests/test_jax_*_parity.py` — 117 / 118 green parity tests.
- `audit/archive-2026-05-13-handover-pre-jax-sprint.md` — full
  Phase 2 / 3a pre-sprint handover content.
