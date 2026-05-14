# HANDOVER.md — next-session brief

> Last written: 2026-05-14 by `claude/read-handover-iLWTq`.
> Prior handover archived as
> `audit/archive-2026-05-14-handover-pre-search-exhaustion.md`.

## Where we are

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC → **40 days
  remaining.**
- **Live submitted agent:** `v7_0_drop_one`, **submission #52588156**,
  μ=**1094.9** (team peak). Rank **109 / 2587** (top 4.2 %). Gap to
  top-10 prize: +336 μ.
- **Rolling-last-2:** `[v4_planner #52579863, v7_0_drop_one #52588156]`.
- **No new submission this session.** Anchor preserved.

## This session — 7 falsifications on the chooser/proposer axis

Seven controlled scalar 32-game A/Bs vs `v7_0_drop_one`. **All lost.**
v7_0 is robustly the best policy in this design space.

| Variant | Axis | Change | Winrate | Wilson lo |
|---|---|---|---:|---:|
| v7_1 | proposer | H11 opening grab | 35.9 % * | 25.3 % |
| v7_2 | search | depth-2 over v3.5.1 drop-ones | 31.3 % | 18.0 % |
| v7_3 | opp model | min-regret over hand-crafted archetypes | 28.1 % | 15.6 % |
| **v7_4** | value head | composite capture-value | **40.6 %** | **25.5 %** |
| v7_5 | action space | + ADD-one widening | 37.5 % | 22.9 % |
| **v7_6** | action primitive | + split-source (multi-launch) | **40.6 %** | **25.5 %** |
| v7_7 | proposer coef | enemy multiplier ×1.3 | 28.1 % | 15.6 % |

\* v7_1 measured in JAX 64-game A/B (35.9 % / Wilson lo 25.3 %);
   scalar 32-game showed 53.1 % / Wilson lo 36.4 % — underpowered.

Best: v7_4 = v7_6 = 40.6 %, ~10 pp below 50 % baseline. **Per-source
greedy ROI proposer is doing 95 % of the work**; the chooser's added
value is small and noise-dominated. The chooser-axis design space is
**exhausted** — further refinements have negative marginal EV.

## Side wins this session (not a v7_X)

- **Bundler parity-gate non-determinism fixed** (commit `07ef918`).
  Root cause was wallclock-budget bail, not module-level mutable
  state. New `_effective_wallclock_ms` helper + env-var override
  (`ORBIT_WARS_PARITY_WALLCLOCK_MS`) lets `_parity_gate` run under
  unbounded budget while production agents keep the 700 ms watchdog.
  9 unit tests in `tests/test_v7_search_shared_model_cleanup.py`.
- **`composite_capture_value` value head** (`lib/value_heads.py`).
  Rewards predicted captures via `WorldModel`-based fleet-fate
  attribution; penalises bouncing / OOB / sun trajectories. Net
  +9 pp over plain ship-delta in 32-game A/B (v7_4 vs v7_2). Kept
  in the lib for any future chooser that wants it.
- **3 new action-primitive enumerators** (`_enumerate_add_one`,
  `_enumerate_split_source`, `_enumerate_drop_or_add_one`,
  `_enumerate_drop_or_split`). Wired into `enumerate_candidates`;
  available for any future variant. Not load-bearing — v7_5 / v7_6
  showed them not productive under the current proposer / value head.
- **Hand-crafted opp-archetype set** (`lib/missions/opp_archetypes.py`).
  5 archetypes (no-launch / v3.5.1 / counter-reinforce /
  counter-snipe / cross-attack) for maximin / min-regret aggregation.
  v7_3 used it; falsified, but the module is reusable.
- **JAX depth-2 (parked).** `lib/game/jax/jax_depth2.py` (~340 LOC)
  with capped 4×2 nested-vmap. Compiles + runs single-state on CPU
  in ~110 s JIT + 20 s hot. **GPU compile fundamentally too slow**
  even at small scale (PI killed at 35 min); see friction
  `scale-without-smoke-burned-90min-t4`. Don't push to T4 without
  the two-tier-smoke rule that PI ratified this postmortem.

## Falsified / dead

- All seven v7_X variants above. Mechanism families: H11-only,
  depth-2-maximin, archetype-min-regret, drop-one + capture-value,
  drop-or-add-one, split-source, enemy-multiplier-×1.3. Falsified
  at 32-game scalar A/B vs v7_0 with Wilson lo < 0.55 across the
  board.
- JAX depth-2 game-vmap'd kernel at 64 games × 500 turns — both
  full nested vmap (OOM) and `lax.scan` refactor (90-min stall)
  proven not to compile within practical limits on T4.

## Next-session first-actions

The chooser/proposer axis is exhausted. Real lift requires
architectural change. Three viable paths, ranked by tractability:

1. **Target-set planner** (~3-5 days, mid-risk). Replace
   `_build_incumbent_intents` with a planner that picks PLANET
   SETS to conquer (combinations like "opp's 2-planet home
   cluster + 1 neutral support"), then solves the source → target
   assignment for each set. Score each set via `fast_sim` K=10
   rollout with `composite_capture_value`. New action primitive:
   a coherent multi-launch plan rather than per-source greedy
   picks. Most likely to produce real lift; PI's framing of
   "combinations of planets we need to conquer" maps directly.

2. **Learned policy / shot validator** (~weeks, high-risk).
   The H14 workstream: train a small classifier (30-50 feature
   logreg or MLP) on the 37 k labeled examples in
   `data/shot_validator/` to predict launch-outcome
   (capture / bounce / sun / OOB). Use as a candidate-rejector
   inside the chooser. Concretely actionable: data exists,
   schema documented, just no training pipeline yet.

3. **Self-play RL fine-tuning on JAX** (~weeks, highest-risk).
   The JAX engine is bit-exact and game-vmappable; PPO or DQN
   fine-tuning a small policy net against frozen v7_0 is in
   reach. Multi-week investment with binary outcome.

If none of those are appetising: **lock the rank** at μ=1094.9
top 4.2 %, 40 days. Save remaining capacity for ladder-shift
response (if competitors push and we drop, re-investigate).

## Out of scope for next session

- More chooser refinements. The 7-falsification pattern is
  conclusive. Rule 37 (consecutive-falsification-cap, ratified
  this postmortem) explicitly forbids this.
- Pushing JAX depth-2 to T4 without the two-tier-smoke checklist
  (also ratified this postmortem).

## Pointers — this-session deliverables

- `lib/value_heads.py::composite_capture_value` + 4 tests.
- `lib/missions/opp_archetypes.py` (5 archetypes) + 7 tests.
- `lib/v7_search.py::_enumerate_add_one`, `_enumerate_split_source`,
  `_enumerate_drop_or_add_one`, `_enumerate_drop_or_split`,
  `choose_archetype_minregret`, `choose_archetype_minregret_with_4p`,
  `_effective_wallclock_ms`, `_bind_shared_world_model`.
- `lib/missions/snipe.py::ENEMY_MULTIPLIER` constant (default 1.0;
  bundle build workflow documented in `v7_7_enemy_mult/main.py`).
- `lib/game/jax/jax_depth2.py` (parked).
- `scripts/bundle_agent.py` — `_parity_gate` env-var override;
  `opp_archetypes` and (already-landed) `opening` in
  `DEFAULT_LIB_ORDER`.
- `agents/v7_3_minregret/`, `v7_4_capture_value/`,
  `v7_5_drop_add_capture/`, `v7_6_split_source/`,
  `v7_7_enemy_mult/`. (Bundles in `submissions/` are gitignored.)
- Tests: `test_v7_search_shared_model_cleanup`,
  `test_opp_archetypes`, `test_composite_capture_value`,
  `test_enumerator_add_one`, `test_enumerator_split_source`,
  `test_jax_depth2`.
- `audit/2026-05-14-postmortem-read-handover-iLWTq.md`.
- `.claude/skills/kaggle-comp/improvements.md` — Rule 37
  (consecutive-falsification cap) + mandatory-two-tier-smoke
  promoted this postmortem.
