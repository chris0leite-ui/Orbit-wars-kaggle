# state/MULTI_BRANCH.md — single source of truth across all dev branches

> **Written:** 2026-05-20 by `claude/review-skills-improvements-moKOR`.
> Replaces the per-branch `state/current.md` divergence. Read this
> file FIRST every session (Rule 44).

## How to use this doc

1. Read the **Live Kaggle** section first; refresh from `kaggle competitions submissions orbit-wars` if any timestamp here is >24h old.
2. Read the **Track registry** section to find which work-track owns the subsystem you're about to touch.
3. Read the **Closed tracks** list — if your idea lives there, STOP, the axis is falsified.
4. Before any `kaggle competitions submit`, fill out the **Push claim board** row (Rule 42).

---

## Live Kaggle (snapshot 2026-05-25 — refreshed from `kaggle competitions submissions orbit-wars`)

Pulled fresh; refresh via `kaggle competitions submissions orbit-wars` at session start.

| Sub ID | Date (UTC) | Agent | μ | Role |
|---|---|---|---:|---|
| **53024913** | 2026-05-25 15:35 | **baseline_ev_per_ship** (THIS branch, EV-per-ship sort variant, commit 0a8308f) | **PENDING** (predicted 1100-1180 band) | **Rolling pair (most recent)** |
| **53018599** | 2026-05-25 11:54 | K1+Z v2 (sibling Q0q9T, kinematic-table priming + Z v2 effective-landing prune) | **1086.8** | **Rolling pair (older half)** |
| 53013786 | 2026-05-25 08:40 | baseline_joint_aggr_consolidated_orbitfix RESUBMIT (THIS branch, bundle SHA 9ec3af83 = original 5/22 peak) | 1144.6 | EVICTED (by 53024913) |
| 53001857 | 2026-05-24 23:51 | baseline_wave v3.1 (THIS branch — orbitfix peak stack + multi-source wave proposer) | 1130.6 | EVICTED |
| 53000996 | 2026-05-24 22:38 | buildup_planner_phi1_only (sibling Q0q9T — Phi-1 leaf swap only) | 1115.2 | EVICTED |
| 52993021 | 2026-05-24 16:10 | concentration A+B (alpha=1.5, two-call orbital) | 1117.9 | EVICTED |
| 52968889 | 2026-05-23 23:59 | buildup_planner (BUILDUP MILP + CONSOLIDATION + FINISHER) | 1142.4 | EVICTED |
| 52966655 | 2026-05-23 21:18 | wave V3 (leaf-Δ gate + planet_positions cache) | 1141.0 | EVICTED |
| **52912707** | 2026-05-22 04:56 | **baseline_joint_aggr_consolidated_orbitfix** (B1-B7 orbital-safety modeling fix) | **1165.4** | **TEAM PEAK** — EVICTED |
| 52744856 | 2026-05-17 14:17 | composite_a2_hybrid (composite head 2P + A2 4P) | 1149.2 | EVICTED |
| 52754310 | 2026-05-17 22:06 | trajectory chooser v4 + wait_N + wallclock budget | 1143.7 | EVICTED |
| 52784853 | 2026-05-18 17:42 | PV off + bug #3/#4/#12 fixes | 1130.4 | EVICTED |
| 52874528 | 2026-05-21 06:00 | baseline_joint_aggr (JOINT structural lift) | 1128.8 | EVICTED |
| 52811320 | 2026-05-19 12:54 | hold-feasibility solo | 1135.1 | EVICTED |
| 52882014 | 2026-05-21 10:26 | baseline_joint_aggr_consolidated | 1124.0 | EVICTED |
| 52827111 | 2026-05-19 19:52 | comet-aim + reactor-aware | 1122.0 | EVICTED |
| 52766596 | 2026-05-18 07:12 | Direction B v3 joint candidate evaluation | 1118.3 | EVICTED |
| 52894340 | 2026-05-21 14:33 | _phase4_step1_FND (endgame predicate + orbital safety) | 1092.3 | EVICTED |

- **Rolling pair floor:** μ = 1086.8 (K1+Z v2, sibling Q0q9T).
- **Rolling pair ceiling:** sub **53024913** PENDING (baseline_ev_per_ship, predicted 1100-1180 band). Note: sub 53013786 orbitfix RESUBMIT settled μ=1144.6 (-20.8 vs original 5/22 peak 1165.4, normal TrueSkill noise on byte-identical resubmit) — now EVICTED.
- **Team peak (evicted):** μ = **1165.4** (sub **52912707** baseline_joint_aggr_consolidated_orbitfix, 2026-05-22, branch `claude/review-skills-improvements-moKOR`). Beat prior peak (composite_a2_hybrid 1149.2) by **+16.2 μ** via B1-B7 orbital-safety modeling fix (Rule 40 — modeling fix, not restriction-tuning).
- **baseline_wave v3.1 outcome (THIS branch):** settled at μ=**1144.1**, FAR above the predicted 1000-1100 band. The pre-submit hypothesis "regression vs orbitfix peak μ=1165.4; PI learning submit" was wrong by ~140 μ in our favour — the multi-source wave proposer is the real lift, not just a calibration probe. n=8 local A/B vs orbitfix (3/8 win-by-reward) under-predicted live result.
- **Daily submission budget:** 5/day. Today (2026-05-25 UTC) used: **2** (sub 53013786 RESUBMIT @ 08:40, sub 53024913 baseline_ev_per_ship @ 15:35). 3 remaining. Last submit was 2026-05-25 15:35 UTC.
- **Deadline:** 2026-06-23 23:59 UTC. **~29 days remain.**

---

## Substrate tier split (foundational — adopted from `OyoYR-rebased` HANDOVER 2026-05-20)

Every track sits on top of one of these tiers. Tools registry (`state/TOOLS.md`) mirrors this split.

### Tier 1 — closed-form calculations (no simulation)

- `lib/geometry.py`, `lib/orbit.py`, `lib/trajectory.py`, `lib/aim.py` — on main; foundational.
- `lib/trajectory_layer.py` — **branch-only on `claude/ml-competition-strategy-PFhzM`**. Sparse closed-form O(1) "where is every entity at relative turn t" oracle. Parity-pinned by `tests/test_trajectory_layer_positions.py`. Encodes the env's rotate-before-step off-by-one once, in one place. **Merge-up candidate.**
- `agents/precision/sim.py` + `agents/precision/intercept.find_shot()` — **branch-only on `claude/precision-physics-engine-ymJkA`**. Guaranteed-landing inverse-intercept solver, closed-form physics primitives (`segment_oob`, `segment_crosses_sun`, `swept_pair_hit`, `predict_planet_pos`, `combat_resolve`, `ships_for_speed`). End-to-end parity test at `agents/precision/tests/test_intercept_landing.py`. **Merge-up candidate.**

### Tier 2 — exact simulation (when a query truly needs a forward step)

- `lib/fast_sim.py` — thin wrapper over `lib/game/interpreter.py` (byte-exact pure-Python port of the kaggle engine; zero-tolerance parity via `tests/test_game_parity.py`). Entry points: `step(snap, actions)`, `rollout(snap, K, policies)`, `from_obs(obs, configuration, episode_seed)`, `clone(snap)`. ~20× speedup vs `env.clone()+step()`.

---

## Track registry — three parallel work-tracks

### Track A — Analytical chooser

- **Lead branch:** `claude/strategy-framework-design-OyoYR-rebased`. Last commit 2026-05-20 17:00.
- **Sibling (paused):** `claude/strategy-framework-design-OyoYR` (pre-rebase value-head exploration).
- **Lineage:** closed-form ROI / LP / MILP / opening planner / wait-N trajectory validation. 10 analytical slices (Slices 4-10).
- **Status:** knowledge-base 2026-05-20 — *"analytical-chooser axis closed (10 slices, 0 lift)."* Two live pushes regressed (806.5, 829.1). Architectural bind from `audit/2026-05-20-postmortem-strategy-framework-design-OyoYR-rebased.md`: stacking analytical layers on a rollout chooser is noise — analytical needs multi-turn glue OR must replace rollout entirely.
- **Best-evidenced asset:** `agents/baseline/chooser_roi.py` (~750 LOC) — closed-form ROI with N-way coalitions, defensive post-pass, Tier-2 rollout posterior. 13/14 oracles pass on `tests/test_planner_oracles.py`. Opt-in via `BASELINE_CHOOSER=roi`.
- **Open question:** multi-turn analytical glue (DP / rolling LP / receding horizon) vs analytical-leaf-inside-rollout (analytical as ACCELERATION of rollout, not replacement of it).

### Track B — Hybrid-simulation production lineage

- **Lead branches:**
  - `claude/audit-workflow-performance-btjeK` — production line + audit infra.
  - `claude/analyze-game-strategy-EpMVP` — modular phase ports (Phases 0-6 gated off).
- **Sibling (closed):** `claude/phase7-btjek-chain-bonus` — chain-bonus axis exhausted (Rule 37) on 2026-05-20.
- **Lineage:** v8_scavenge → v15_banded → trajectory v4 → composite head 2P + A2 4P → comet-aim+reactor → hold-feasibility solo → **baseline_joint_aggr_consolidated_orbitfix (μ=1165.4, TEAM PEAK)** → baseline_wave v3.1 (μ=1144.1, current rolling-pair ceiling). Lineage peak EVICTED.
- **Status:** live champion lineage. v16-v20 chooser axis falsified (Rule 37). Chain-bonus axis exhausted (Rule 37). Current pivot to physics-driven mechanisms.
- **Latest finding (btjeK, 2026-05-20 PM):** H44 corrected — landing-capture failures are **65% fleet-destroyed-in-flight** (`audit/2026-05-21-h44-phase1-CORRECTED.md`). Substantive new physics-driven mechanism candidate.
- **EpMVP latest:** Phase 4 (defensive migrations, gated off) + Phase 6 (chain-capture bonus, Claws relay pattern) landed 2026-05-20. Phases 0-6 individually env-var gated. Bundler upgraded for inline agent submodules + explicit-name imports.
- **Next moves:**
  1. Hold-feasibility solo validation (btjeK Phase B in HANDOVER).
  2. H44 physics-driven defensive mechanism design (btjeK).
  3. EpMVP individual phase commissioning behind env vars.
- **Best-evidenced asset:** `agents/baseline/` modular re-impl + `agents/baseline/chooser.py` trajectory chooser (production default).

### Track C — Verify-first + Goal-directed planning

- **Lead branch:** `claude/ml-competition-strategy-PFhzM`. Last commit 2026-05-19 PM.
- **Substrate sibling:** `claude/precision-physics-engine-ymJkA` (9 days old; primitives reusable).
- **Lineage:** physics-validation gate → analytics verification suite (5 checks) → goal_planner MVP → greedy_expand MVP → portfolio. Methodologically distinct from Tracks A and B: verify primitives empirically BEFORE chooser design; build from verified substrate.
- **Status (Day-19 PM2 verdict):** *"chooser axis confirmed neutral"* — greedy_expand (60 LOC) tied goal_planner (500 LOC) at 14/32 each. Phase A passed Test 3 (n=16 self-play confirmation). Wrap-baseline asymmetry: 12/32 = 37.5% (the only positive signal vs production across 10+ iterations).
- **Critical finding:** entire trajectory_roi line missed `lib.trajectory.predict_fleet_fate` (6.8% physics waste); this becomes the basis for Rule 47 below.
- **Best-evidenced assets:**
  - `lib/trajectory_layer.py` (closed-form O(1) position oracle).
  - `lib/goal_planner/` (Phases 1-4: winning-state predicate, backwards sequencer, feasibility filter, preemptive defense).
  - `scripts/verify_analytics.py` (draft) + `tests/test_analytics.py` (~150 LOC, Test 3 PASS confirmed).
  - `scripts/probe_emits_via_fate.py` (per-emit physics-waste classification).
  - `scripts/inspect_goal_planner_game.py` (turn-by-turn tracing).
- **Open question:** does the wrap-baseline asymmetry indicate "augment baseline with a portfolio veto layer" rather than "replace chooser entirely"?

---

## Closed tracks — falsified knowledge, do NOT iterate

| Axis | Branch | Verdict date | Evidence |
|---|---|---|---|
| Chain-capture bonus | `claude/phase7-btjek-chain-bonus` | 2026-05-20 | Rule 37 exhausted (Phases 7-9 all tested, impact ≤ noise floor) |
| Value-head aggregators (projected_sum / favor) | `claude/strategy-framework-design-OyoYR` | 2026-05-19 | 4P A/B projected_sum vs favor = TIE; axis exhausted |
| Pure trajectory_roi (no physics primitives) | `claude/ml-competition-strategy-PFhzM` v1-v3 | 2026-05-19 | 0-1/32 vs baseline; dominated by wrap-baseline (12/32). Root cause: never imported `lib.trajectory.predict_fleet_fate` (6.8% physics waste) |
| Cluster-conditional opening overlay (H40 pre-EDA) | `claude/game-strategy-eda-roatN` | 2026-05-14 | Falsified by geometry EDA |
| Closed-form ROI as full chooser replacement | `claude/audit-workflow-performance-btjeK` | 2026-05-19 | `chooser_roi.py` Tier 1+2: 0/32 vs v7_0 / v4_planner / v3.5.1 panel. Keep code as opt-in research (`BASELINE_CHOOSER=roi`); don't ship as default |
| Asymmetric Tier-1 baseline chooser | `claude/reverse-engineer-seat-geometry-BPJKs` (PR #31, merged to main) | 2026-05-18 | CRN-violating, 0/32 panel, reverted in commit `f28c9fc` |
| v9-v15 chooser saturation iteration | recover-main-foundations | 2026-05-16 | Rule 37 (3-variant cap) hit at v16-v20; chooser-axis structural ceiling ~μ=1120 |

---

## Push claim board (Rule 42 — fill before every `kaggle competitions submit`)

Empty rows below mean no pending submission claim. Most recent claim at top.

| Timestamp (UTC) | Branch | Agent | Predicted μ | Will evict (sub_id, μ) | PI signoff |
|---|---|---|---:|---|---|
| 2026-05-26 21:33 | competitive-programming-strategy-ESwSv | **sa_online_v5** (sub **53063161** PENDING) — cascade-aware SA + warm-start from `ctx.admissible` ranked by closed-form `_capture_value`, `SA_REFINE_OPP_POLICY=noop` default, `SA_WARM_START_MAX=16`, commit `147d6aa`. Local 200-step smoke vs baseline_joint_aggr_consolidated_orbitfix seed=7542: 69 emits / 50 active turns, eliminated step 134, mean per-turn 284ms, max 593ms. Rule 42 NOT pre-filled (post-hoc entry per wrap-up); Rule 43 panel NOT run (the agent doesn't beat the peak baseline locally). | 600-1000 broad band (agent now plays but loses to peak by production deficit; live ladder mix unknown) | **53061384** (TIMEOUT, ERROR-equivalent) — rolling pair was {53061384 ERROR, 53062327 sa_v4 PENDING}, v5 push evicts 53061384 | ✅ PI explicit "Ship v5 as-is now" (2026-05-26 PM) — Rule 42 GREEN (evicted μ ≈ 0 from ERROR) |
| 2026-05-26 20:36 | competitive-programming-strategy-ESwSv | **sa_online_v4** (sub **53062327** PENDING / likely ERROR per PI in-session report "submission failed") — cascade-aware SA + fate cache keyed by `(src,tgt,t_dep,ships_bucket)` + snap-cache, commit `6147dcb`. Local 500-step smoke vs `random` opponent reported 57s/60s overage in pocket; live inspection vs peak baseline later showed 5 emits in 79 turns and elimination step 79. Smoke-vs-random was a false positive — random's idleness masked our agent's idleness. | unknown | **53059642** (TIMEOUT, ERROR-equivalent) | ✅ PI explicit "Ship v4" (2026-05-26 PM) — Rule 42 GREEN (evicted μ ≈ 0 from ERROR) |
| 2026-05-26 19:00 | competitive-programming-strategy-ESwSv | **baseline_unified** (PENDING) — unified favor_strategic (commit `3a054c7`): one leaf for both 2P and 4P. F1 = my_ships − max_o(opp_ships); F2 = (my_prod_disc − max_o(opp_prod_raw)) × pv (asymmetric Term A — Phase F's calibrated defensive "fear" gradient on my-side only, raw opp_prod max-of-opps); Term B with capture-feasibility gate (only credit reach when launch ≥ capture-size); Term C continuous-through-elimination with dead-slot credit over `num_seats − 1` expected opps; discrete 4P+ ELIMINATION_BONUS when FINISH_BONUS=0. No `if num_seats` aggregation switch — max-of-opps controls F2 scale across modes. Local evidence is mixed and below Rule 45 floor: 4P n=2 panel at HEAD in progress (orbitfix 0/2, wave 1/2 partial); prior code paths showed 4P n=5 = 7/20 = 35% (vs Phase F 10%), 2P n=2 = 5/8 = 62.5% (Phase F 75%). Rule 46: bundle compile + test_bundle.py 10/10 GREEN; smoke 2P seed=0 100-step vs v7_0 = WIN. Bundle SHA: see prologue line 4 (source commit `3a054c7`). | ~900-1100 (broad band — n=2 evidence on the unified code, mixed signals; predicted_mu lower bound deliberately below 1113.2 since this is a learning submit, not a peak-chasing submit) | **53018599** (μ=1113.2 K1+Z v2, older half of rolling pair; 53024913 EV-per-ship μ=1135.4 is the newer half, stays in pair) | ✅ **PI explicit learning-submit 2026-05-26 PM**: "I want you to submit. I want to observe what actually happens. We need to gather data so it's not about winning with the next submission. It's about learning." — Rule 42 YELLOW (predicted-μ lower bound 900 < evicted-μ 1113.2); PI signoff explicit, Rule 45 explicitly overridden for learning purpose. |
| 2026-05-25 15:35 | competitive-programming-strategy-ESwSv | **baseline_ev_per_ship** (sub **53024913** PENDING) — orbitfix env stack + new `BASELINE_SORT_BY_EV_PER_SHIP=1` flag (commit 0a8308f). Chooser sorts positive-EV candidates by `score/ships` instead of `score`, prioritising per-ship efficiency. Diagnostic: 4P launches/turn 0.23→1.68 (7×), owned planet-turns 243→3154 (13×), ranked_out 28.4%→3.4% (per-ship sort converts wait-N commits into fire-now). Panel A/B (5 games × 4 opps, 250-step cap, no seat switch — new standard procedure per commit b36ac7a): vs orbitfix 4/5, vs baseline_wave 3/5, vs v7_0 4/5, vs v4_planner 4/5; pooled 15/20 = 75% Wilson [0.541, 0.886]. Rule 46: bundle compile 10/10 GREEN, smoke vs v7_0 = WIN p0_win max=899ms, 3-seed bench p50=559ms p95=834ms max=1018ms `over_1000ms=3/641` = 0.47% (~54ms total overage vs 60s/game Kaggle allowance — verdict WATCH, not FAIL). | ~1100-1180 (anchored on 80% panel vs orbitfix μ=1144.6; widened DOWNWARD to acknowledge 2P→4P translation risk) | **53013786** (orbitfix RESUBMIT, μ=1144.6 — older half of rolling pair; 53018599 K1+Z v2 μ=1086.8 is the newer half) | ✅ PI explicit "submit then run 4P" (2026-05-25 PM) — Rule 42 YELLOW (predicted-μ lower bound 1100 < evicted-μ 1144.6); PI signoff stands |
| 2026-05-25 08:40 | competitive-programming-strategy-ESwSv | **baseline_joint_aggr_consolidated_orbitfix RESUBMIT** (sub **53013786** settled μ=1144.6; existing bundle SHA `9ec3af83`, originally sub 52912707 commit 458f663, settled μ=1165.4 on 2026-05-22). Rule 46: bundle 10/10 GREEN; single-game smoke vs v7_0 = WIN; 3-seed bench p50=427ms p95=814ms max=953ms `over_1000ms=0`. Post-5/22 modeling fixes (predict_relative static-planet 1ad6cfa, comet-aware 4c80932, 289d8ed) deliberately NOT bundled — every subsequent submission containing them regressed live μ (52968889 μ=1142.4, 52966655 μ=1141.0, 53000996 μ=1109.8, 53001857 μ=1144.1). Same lesson as wave V3.1: local A/B doesn't predict live μ. | ~1140-1190 (TrueSkill re-eval noise band around the 1165.4 peak settle) | **53000996** (Phi-1, μ=1109.8 — older half of rolling pair) | ✅ PI explicit "Yes, submit unchanged" — Rule 42 GREEN (predicted μ ≫ evicted μ; floor moves 1109.8 → 1144.1) |
| 2026-05-24 23:51 | competitive-programming-strategy-ESwSv | **baseline_wave v3.1** (sub **53001857** settled μ=1144.1) — orbitfix peak stack (JOINT_AGGR + ORBITAL_SAFETY + NEUTRAL_BONUS + REINFORCE) + new multi-source wave proposer (`enumerate_wave_candidates`); bleed/stockpile DROPPED after diagnosis showed they starved early-game expansion. Commit `ff08752`. | ~1000–1100 (n=8 vs orbitfix 3/8 win-by-reward, 0/8 elim, Wilson [0.137, 0.694]); learning submit — PI explicit observation. | **52993021** (concentration, μ=1117.9) — rolling pair had shifted; Q0q9T's sub 53000996 Phi-1 μ=1141.6 had already evicted 52968889 between my check and push | ✅ PI explicit "submit baseline wave so i can see it and learn what to improve" — Rule 42 acknowledged (evicted μ ≈ predicted μ band; max-of-pair stays 1141.6 if baseline_wave settles ≤ Phi-1, drops to baseline_wave μ if higher) |
| 2026-05-23 21:00 | review-skills-improvements-moKOR | baseline_joint_aggr_consolidated_orbitfix (PENDING SUBMIT) — consolidated + full B1-B7 orbital safety modeling fix | ~1110-1130 (4/4 vs baseline_full @ μ=1078, 2/4 parity vs consolidated @ μ=1124, 4/4 vs phase4_step1_FND @ μ=1118 — all clean_ab subprocess-isolated) | 52893236 (baseline_full, μ=1078) | ⏳ PENDING |
| 2026-05-21 13:48 | review-skills-improvements-moKOR | baseline_full (sub **52893236**, settled μ=1078) — consolidated + orbital safety + stagnant drain + combat stack + sniper | ~1100-1300 (n=4 = 2/4 vs consolidated AND vs v3.5.1; Wilson [0.150, 0.850] both, point estimate +25pp over symmetric 25% baseline) | 52874528 (μ=1134.9, baseline_joint_aggr) | ✅ PI explicit "submit baseline full now anyway" |
| 2026-05-21 10:26 | review-skills-improvements-moKOR | baseline_joint_aggr_consolidated (sub **52882014**, settled μ=1124, EVICTED) | ~1100-1250 (n=4 1/4 + seed=5 trace WIN 40 planets) | 52872093 (μ=1052.1, analytical_phase_c) | ✅ PI explicit "submit so I can observe" |
| 2026-05-21 06:00 | review-skills-improvements-moKOR | baseline_joint_aggr (sub **52874528**, settled μ=1134.9) | ~1180 (n=16 vs phase_c 87.5%) | 52865089 (μ=805.9, evicted) | ✅ Rule 45 overridden |
| 2026-05-20 17:00 | review-skills-improvements-moKOR | — | — | — | — *(no submission planned this session)* |

**Required fields:** all 6 columns. The PI-signoff column is mandatory if the evicted-μ > predicted-μ (Rule 42). For the 2026-05-23 row: evicted μ=1078 < predicted ≥1100 → Rule 42 GREEN (no PI signoff required by the gate, but Rule 1 still requires explicit "submit" confirmation).

---

## Per-branch sync table (drift detection)

Refresh weekly. Run `git fetch origin && git log -1 --format='%ar' origin/<branch>` for each row.

| Branch | Last commit (rel) | state/ dated | Believed rolling pair | Drift |
|---|---|---|---|---|
| `OyoYR-rebased` | ~2h ago | 2026-05-20 | 52857903 + 52854094 | none — pushed both |
| `audit-workflow-performance-btjeK` | ~6h ago | 2026-05-19 AM | 52784853 + 52766596 | **STALE** — was correct at write, evicted since |
| `phase7-btjek-chain-bonus` | ~6h ago | inherits btjeK | 52827111 (inherited) | **STALE** — evicted |
| `analyze-game-strategy-EpMVP` | ~10h ago | 2026-05-17 | v20_dogpile (52721807) | **VERY STALE** |
| `ml-competition-strategy-PFhzM` | ~23h ago | 2026-05-17 | composite_a2 (52744856) | **VERY STALE** |
| `strategy-framework-design-OyoYR` (pre-rebase) | ~32h ago | 2026-05-17 | v20_dogpile | **VERY STALE** (paused branch) |
| `kaggle-baseline-strategy-lO4mm` | ~3d ago | 2026-05-17 | v20 + v15 | **VERY STALE** (foundation snapshot) |
| `precision-physics-engine-ymJkA` | ~9d ago | n/a | precision_v3 (μ₀=600) | **STALE** (parked) |

**Drift signal:** if your branch's `state/current.md` is more than 24h behind the live Kaggle pull, refresh it from THIS file (`state/MULTI_BRANCH.md`) before any subsystem edits (Rule 44).

---

## Pointers

- `state/TOOLS.md` — tools registry (A/B harnesses, diagnostics, validation).
- `state/mechanism-ledger.md` — every agent family tried, with status.
- `state/hypothesis-board.md` — open ideas, killed list.
- `HANDOVER.md` — next-session brief (Rule 15).
- `CLAUDE.md` — rules (1-47 as of 2026-05-20).
- `audit/friction.md` — current friction summary.
- `.claude/skills/kaggle-comp/improvements.md` — promotion queue.
