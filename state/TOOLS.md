# state/TOOLS.md — tools registry across all branches

> **Written:** 2026-05-20 by `claude/review-skills-improvements-moKOR`.
> Scannable catalog so future agents REUSE instead of REBUILD. Reach
> for the right tool by the question being asked.

## `fast.py` CLI — single-file iteration harness

```
python fast.py smoke <agent>              # triage vs random + nearest
python fast.py eval  <agent>              # adaptive Wilson-gated A/B vs v7_0
python fast.py eval  <agent> --vs <opp>   # h2h vs specific opponent
python fast.py eval  <agent> --vs-panel   # 3-agent panel (v7_0, v4_planner, v3.5.1)
python fast.py eval  <agent> --geometry-panel  # 128 seeds × 32 archetypes
python fast.py play  <agent> --seed <N>   # one game, verbose
python fast.py bench <agent>              # per-turn ms vs 1000ms budget
```

Agent resolution: baseline names (`v7_0`, `v4_planner`, `v3.5.1`), path to `.py`, or `agents/<name>/` directory. Bundled submissions preferred (parity-safer).

---

## A/B testing tools (pick by question being asked)

| Tool | Question it answers | Sample mode | CRN | Use when |
|---|---|---|---|---|
| `fast.py eval <agent>` | Beats v7_0 baseline? | adaptive (Wilson-gated) | yes | First-pass triage |
| `fast.py eval <agent> --vs <opp>` | Beats a SPECIFIC opp? | adaptive | yes | h2h vs current rolling champion (Rule 43) |
| `fast.py eval <agent> --vs-panel` | Beats the 3-agent panel? | adaptive | yes | Pre-submit minimum (Rule 43) |
| `fast.py eval <agent> --geometry-panel` | Consistent across 32 archetypes? | 128 seeds | yes | Catch flavor-dependent regressions |
| `scripts/tournament.py` | Pairwise winrate matrix across N agents | configurable | yes | Round-robin, per-side splits |
| `scripts/ffa_tournament.py` | First-place rate in 4P FFA | configurable | yes | 4P-specific seat dynamics |
| `scripts/ab_variants.py` | Which hyperparameter wins? | Wilson-gated | yes | Parameter sweep |
| `scripts/ablation.py` | Which mechanism contributes the lift? | configurable | yes | Mechanism ablation (strategy fixed) |
| `scripts/strategy_panel.py` | Which strategy wins across mechanisms? | both-sides | yes | Strategy ablation (mechanism fixed) |
| `scripts/ffa_panel.py` | Focal-vs-focal against fixed background | configurable | yes | 4P with controlled opponents |
| `scripts/manifold_check.py` | Does behaviour diverge from baseline across panel? | per-archetype | yes | Behaviour divergence detector |
| `scripts/_ledger_ab_driver.py` *(branch: btjeK)* | Per-launch what-if A/B sweep | per-launch | yes | Ledger-instrumented A/B. **Caveat:** what-if alone is misleading (Rule 41) |

---

## Single-game diagnostic tools (use when A/B is surprising)

A/B answers "whether," single-game tracing answers "why." When an A/B disagrees with intuition, run `fast.py play --seed <losing-seed>` then drill in.

| Tool | What it shows | Input |
|---|---|---|
| `fast.py play <agent> --seed <N>` | One game, verbose action stream | seed |
| `scripts/play4p.py` | One 4P game, four-perspective trace | seeds + 4 agents |
| `scripts/episode_postmortem.py` (685 LOC) | Per-turn missions, arrivals, captures, ROI accounting | episode (live or local) |
| `scripts/diag_v8.py` | Per-turn chooser candidate ledger + action emit | seed + agent |
| `scripts/diag_outcomes.py` | Episode-level outcome classification | episode |
| `scripts/instrument_v12_chooser.py` *(WIP)* | Per-turn chooser instrumentation | seed + chooser variant |
| `scripts/lookahead_probe.py` (511 LOC) | Does WorldModel horizon forecast beat current-ship-delta? | seeds |
| `scripts/capture_probe.py` (378 LOC) | Capture success vs arrival-size / lead-aim | seeds |
| `scripts/orbit_prediction_check.py` | Orbit-prediction parity vs `env.steps[N]` | seed |
| `scripts/probe_emits_via_fate.py` *(branch: PFhzM)* | Per-emit physics-waste classification (sun / OOB / comet) | replay |
| `scripts/inspect_goal_planner_game.py` *(branch: PFhzM)* | Turn-by-turn goal_planner state tracing | seed |
| `scripts/h44_landing_capture_diagnostic.py` *(branch: btjeK, 360 LOC)* | Fleet-fate tracing for landing-capture failures. Source of the 65% fleet-destroyed-in-flight finding | episode |
| `scripts/one-game-close-read-for-chain-bonus-mechanism.py` *(branch: phase7)* | Chain-relay trace (negative-result archived) | seed |
| `scripts/replay_mine.py` (370 LOC) | Extract event streams from replay JSON | replay file |
| `scripts/classify_losses.py` (337 LOC) | Root-cause tag per loss (comet snipe / planet overrun) | replays |
| `scripts/live_episode_summary.py` (343 LOC) | Aggregate ladder episodes per submission (WR, seat dist) | submission ID |
| `scripts/label_shot_outcomes.py` | Per-departure tag (capture / snipe / miss) | replays + shot_validator |
| `scripts/archetype_action_audit.py` (594 LOC) | Per-archetype turn-by-turn action mix (% snipe / defend / settle) | seeds |
| `scripts/fingerprint_external.py` | Top-ladder submissions' move fingerprints | external replays |
| `scripts/bench_fast_sim.py` | `fast_sim.step` vs `env.clone+step` speedup ratio | microbench |
| `scripts/jax_path_search.py` (333 LOC, WIP) | Trajectory pathfinding diagnostic | seed |

The btjeK H44 finding (65% fleet-destroyed-in-flight) came from exactly this loop — A/B suggested a chooser change should help, single-game tracing showed the failures were physics, not strategy.

---

## Validation + testing tools (the trust layer)

Every code consolidation merge MUST clear the gates marked **CRITICAL** below.

| Category | Test / tool | What it gates | Status |
|---|---|---|---|
| **Parity (substrate trust — CRITICAL)** | `tests/test_fast_sim_parity.py` | `fast_sim.step` ↔ `env.clone()+step()` byte-exact | gate |
| | `tests/test_game_parity.py` | `lib/game/interpreter.py` ↔ kaggle env; zero tolerance | gate |
| | `tests/test_jax_*_parity.py` (5 files) | Batch JAX interpreter ↔ scalar paths | gate |
| | `tests/test_jax_world_model_parity.py` | Batch world-model parity | gate |
| | `tests/test_jax_full_step_parity.py` | JAX full step ↔ scalar | gate |
| | `tests/test_batch_interpreter_parity.py` | Batched interpreter ↔ sequential | gate |
| | `tests/test_replay_parity.py` | Replay deserialization ↔ live games | gate |
| | `tests/test_trajectory_layer_positions.py` *(branch: PFhzM)* | `lib/trajectory_layer.py` ↔ `lib/game/interpreter.py` | gate (branch-only — promote with primitive) |
| | `agents/precision/tests/test_intercept_landing.py` *(branch: precision)* | Every `find_shot()` Shot actually lands in kaggle env | gate (branch-only) |
| **Physics primitives (Tier 1)** | `tests/test_geometry.py` | Board constants, point math | stable |
| | `tests/test_orbit.py` | Orbit prediction offset (N-1 correction) | stable |
| | `tests/test_trajectory.py` | Full-trajectory ray-cast safety (sun/planet/OOB) | stable |
| | `tests/test_combat.py` | Arrival-order combat rules 1-4 | stable |
| | `tests/test_fleet.py` | Fleet ETA + arrival sequencing | stable |
| | `tests/test_comet_lifetime.py` | Comet entry/exit + ROI time-integration | stable |
| **Strategic layer (Tier 2)** | `tests/test_mission_*.py` (5: opening, snipe, gang_up, reinforce, drain) | Per-framework mission selection | stable |
| | `tests/test_mech_*.py` (7 files) | Individual mechanism branches | stable |
| | `tests/test_scoring.py` | ROI aggregation + comet penalties | stable |
| | `tests/test_lookahead.py` | `score_action` rollout interface | stable |
| | `tests/test_value_heads.py` | `composite_capture_value` + horizon signals | stable |
| | `tests/test_world_model.py` | Opponent prediction + comet trajectory | stable |
| | `tests/test_intent.py` | World snapshot accessors | stable |
| **Search / chooser** | `tests/test_jax_brute_search.py` | JAX depth-2 brute search | stable |
| | `tests/test_jax_depth2.py` | JAX depth-2 variants | stable |
| | `tests/test_v7_search.py` | v7_search pipeline | stable |
| | `tests/test_enumerator_*.py` (2 files) | Candidate enumeration (add-one, split-source) | stable |
| **Oracles / scenarios (synthetic coordination — CRITICAL for consolidation)** | `tests/test_planner_oracles.py` (14 scenarios) | Chooser passes hand-crafted "should obviously do X" scenarios. 13/14 pass on ROI (Tier-2 broke `solo_capture_but_loses_source`) | **CRITICAL** |
| | `tests/test_baseline_replay_regression.py` *(branch: EpMVP, 364 LOC)* | linrock + Claws fixtures don't regress | branch (merge-up) |
| | `tests/test_migration_solver.py` *(branch: EpMVP, 336 LOC)* | Own→own ship repositioning correctness | branch (merge-up) |
| **Baseline & gate (h2h vs production — CRITICAL pre-submit)** | `tests/test_baseline_smoke.py` | Baseline beats random + nearest (floor) | **CRITICAL** |
| | `tests/test_baseline_h2h.py` | Baseline vs v7_0, n=16 (opt-in `BASELINE_RUN_H2H=1`) | **CRITICAL** |
| | `tests/test_baseline_chooser.py` / `proposer.py` / `value.py` | Per-module baseline unit tests | stable |
| | `tests/test_fixture_smoke.py` | Standard 4-seed fixture smoke | stable |
| **Bundle deployment (CRITICAL pre-submit — Rule 46)** | `tests/test_bundle.py` | Bundler silent-fail guards (import rebinding, `__future__`, multi-line, aliases) | **CRITICAL** |
| | `tests/test_v1_parity.py` | v1 bundle ↔ v1 source (legacy parity template) | stable |
| **Archetype + strategy** | `tests/test_archetype_strategies.py` | Focal agent matches EXPECTED_BEHAVIOR per archetype | stable |
| | `tests/test_opp_archetypes.py` | Archetype classification accuracy | stable |
| | `tests/test_simple_strategies.py` | Trivial baselines (ROI / snipe / defend) | stable |
| **Data assets / harness** | `tests/test_label_shot_outcomes.py` | Shot outcome labeling oracle | stable |
| | `tests/test_ab_variants_gate.py` | Wilson-CI statistical logic of A/B gate | stable |
| | `tests/test_ffa_tournament.py` | FFA tournament result aggregation | stable |
| | `scripts/validate_seed_panel.py` | `seed_panel_128.json` geometry parity vs live | stable |
| | `scripts/validate_panel_vs_replays.py` | Cross-check seed_panel vs replay records | stable |
| **Analytics verification (PFhzM Phase A — partial)** | `scripts/verify_analytics.py` (draft) | 5-check suite: projection-vs-reality, determinism, self-play, vs-random, capture-math | draft (merge-up) |
| | `tests/test_analytics.py` *(branch: PFhzM, ~150 LOC target)* | Pytest-form of the 5-check suite. Test 3 PASS confirmed | draft |
| **Performance regression** | `scripts/bench_fast_sim.py` | `fast_sim.step` vs `env.clone+step` speedup ratio | stable |
| | `fast.py bench <agent>` | Per-turn ms vs 1000ms budget | stable. **Caveat:** wallclock only, not focal-win |

---

## Consolidation-merge gate (Rule-driven)

Any branch's code merges to main if and only if these 6 steps GREEN, in order:

1. **Substrate parity** — `pytest tests/test_*_parity.py` GREEN (no exceptions).
2. **Tier 1 + Tier 2 unit tests** — `pytest tests/test_geometry.py tests/test_orbit.py tests/test_trajectory.py tests/test_combat.py tests/test_fleet.py tests/test_comet_lifetime.py tests/test_fast_sim_parity.py` GREEN.
3. **Planner-oracles** — `pytest tests/test_planner_oracles.py` ≥13/14 pass on chooser being shipped; record which scenario is xfailed and why. Don't ship a chooser that regresses additional oracles vs. the current default.
4. **Baseline gate** — `BASELINE_RUN_H2H=1 pytest tests/test_baseline_h2h.py` Wilson-lo ≥0.45 for additions; ≥0.55 for replacements.
5. **Bundle smoke** — `python scripts/bundle_agent.py <agent> && pytest tests/test_bundle.py && python fast.py play <bundled_agent>` (single-game cold-load works).
6. **Full baseline** — `python -m pytest tests/ -q --tb=line` (12 min). Treat any new failure vs. last GREEN baseline as a blocker. (Audit-noted: ~16 pre-existing failures on main as of 2026-05-14; baseline is "no NEW failures," not "100% green.")

---

## `lib/` — organised by substrate tier

### Tier 1 — closed-form

- `lib/geometry.py` (20 L header) — board constants, sun/planet geometry, Point type.
- `lib/orbit.py` (20 L header) — orbit-prediction (absolute/relative modes); planet rotation offset.
- `lib/trajectory.py` (20 L header) — full-trajectory ray-cast for fleet path safety.
- `lib/aim.py` — lead-aim geometry + trajectory interception.
- `lib/geometry_features.py` — extract features from board geometry (for panel stratification).
- **Branch-only:** `lib/trajectory_layer.py` (PFhzM), `agents/precision/sim.py` + `intercept.py` (precision-physics-engine-ymJkA).

### Tier 2 — simulation

- `lib/fast_sim.py` (20 L header) — ~20× speedup; direct interpreter call on Snapshot.
- `lib/game/interpreter.py` — byte-exact pure-Python port of the kaggle engine.
- `lib/fleet.py` — Fleet dataclass + arrival-time logic.
- `lib/combat.py` — arrival-order combat resolution (rules 1-4 stacking).

### Strategic layers

- `lib/mission.py` — Mission (target selection + fleet alloc) primitives.
- `lib/planner.py` — `settle_plan` greedy per-source with same-turn arrival ledger (v3.1 solver).
- `lib/mechanism.py` — individual action mechanisms (sun_avoid, path_clears, oob_guard).
- `lib/scoring.py` — ROI scoring + comet-lifetime integration.
- `lib/lookahead.py` — `score_action` wrapper for forward-sim rollouts.
- `lib/lookahead_planner.py` — full pipeline (world-model + lookahead + plan).
- `lib/intent.py` — World snapshot + accessors.
- `lib/value_heads.py` — `composite_capture_value` + horizon-integrated ship deltas.
- `lib/world_model.py` — opponent state + comet trajectory prediction.
- `lib/opp_model.py` (20 L header) — pluggable opponent policy.

### Frameworks

- `lib/archetype_strategy.py` — EXPECTED_BEHAVIOR per panel archetype (32 slots).
- `lib/archetype_binning.py` — classify opponent geometry + opening into archetype slot.
- `lib/candidate_portfolios.py` — Portfolio chooser (multi-objective Pareto).
- `lib/fingerprint.py` — agent move signature extraction.
- `lib/seed_panel.py` — `seed_panel_128.json` loader + accessors.
- `lib/mirror.py` — mirror self vs rollout policy (superseded by opp_model).

### Subdirectories

- `lib/game/` — JAX-accelerated batch interpreter (parity-tested vs `fast_sim`).
- `lib/geo/` — closed-form geometry solvers.
- `lib/missions/` — Mission framework (snipe, opening, reinforce, gang_up).

---

## `agents/` — 23 directories

**Baseline + simple:** `agents/baseline/` (modular framework — STABLE), `agents/simple/` (nearest.py, roi.py).
**Versioned production:** `agents/v7_0` → `agents/v7_7`, `agents/v3.5.1`, `agents/v3_snipe`, `agents/v3_lookahead`, `agents/v4_planner`, `agents/v1_orbitfix`, `agents/v2`.
**Ablations:** `agents/abl_K15`, `agents/abl_lite`, `agents/abl_maximin`, `agents/abl_value`, `agents/abl_combined`, `agents/v7_ablations`, `agents/v7_wide_deep`, `agents/v7_1_open_drop_comets`.
**Specialty (branch-specific):** `agents/geo/`, `agents/geo_recap/`, `agents/jax_v7_0/`, `agents/_ledger_on/`, `agents/_ledger_off/`, `agents/_ledger_hard/`, `agents/_mpc/`, `agents/sary_class/`, `agents/precision/` (branch only).

---

## Data assets

| Asset | Size | Purpose |
|---|---|---|
| `data/seed_panel_128.json` | 113 KB | 128 geometry-diverse eval seeds, stratified by production / rotation / size |
| `data/shot_validator/` | dir | Shot-outcome oracle data (supports `label_shot_outcomes.py`) |
| `data/README.md` | 8 KB | Configuration constants (board, sun, rules) sourced from kaggle_environments |

---

## Bundler — `scripts/bundle_agent.py`

```
python scripts/bundle_agent.py agents/v3_snipe --lib geometry fleet orbit aim trajectory
# outputs: submissions/v3_snipe.py
```

**Capabilities:**
- Concatenates ordered `lib/` modules + agent `main.py` into a single `.py`.
- AST walk to strip intra-lib imports (`from lib.x import`, `from .x import`).
- Handles `__future__` imports.

**Known issues (5 silent-fail modes — Rule 46):**
- Multi-line imports (backslash continuation) may be mis-parsed.
- Aliased imports (`from lib.x import y as z`).
- Cross-agent imports.
- Float tie-breaking divergence in some bundles.
- No circular-dependency validation.

**EpMVP upgrade (2026-05-20, branch only):** "inline agent submodules + explicit-name imports" — addresses 3 of the 5 modes. **Merge-up candidate.**

---

## Pipeline files

- `Makefile` — not present.
- `requirements.txt` — kaggle, kaggle-environments, numpy, pandas, ipykernel, pytest. JAX optional.
- `.claude/settings.json` — Claude Code harness config (permissions, hooks).
- `bootstrap.sh` — session-start environment setup (data fetch, dependency install, smoke).

---

## Recommended entry points

| Task | Use | Command |
|---|---|---|
| Triage new idea | `fast.py smoke` | `python fast.py smoke agents/my_idea` |
| Full A/B test | `fast.py eval` | `python fast.py eval agents/my_idea` |
| Multi-agent panel | `fast.py eval --vs-panel` | `python fast.py eval agents/my_idea --vs-panel` |
| Geometry stratification | `fast.py eval --geometry-panel` | `python fast.py eval agents/my_idea --geometry-panel --by-archetype` |
| Round-robin tournament | `tournament.py` | `python scripts/tournament.py <agents.json>` |
| 4P FFA comparison | `ffa_panel.py` | `python -m scripts.ffa_panel --focals <list> --background <list>` |
| Mechanism A/B | `ablation.py` | `python scripts/ablation.py <mechanism_name>` |
| Variant sweep | `ab_variants.py` | `python -m scripts.ab_variants --variant ... PARAM=1.0 ...` |
| Benchmark speed | `fast.py bench` | `python fast.py bench agents/my_idea` |
| Parity gate | tests/ | `pytest tests/test_*_parity.py` |
| Bundle for submit | `bundle_agent.py` | `python scripts/bundle_agent.py agents/v7_0 --lib geometry orbit trajectory ...` |
| Single-game trace | `fast.py play` | `python fast.py play agents/my_idea --seed <N>` |
| Episode postmortem | `episode_postmortem.py` | `python scripts/episode_postmortem.py <episode>` |
| Live ladder summary | `live_episode_summary.py` | `python scripts/live_episode_summary.py <sub_id>` |
