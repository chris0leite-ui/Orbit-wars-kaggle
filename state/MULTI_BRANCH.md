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

## Live Kaggle (snapshot 2026-06-01 07:35 UTC)

| Sub ID | Date (UTC) | Agent | μ | Role |
|---|---|---|---:|---|
| **53243763** | 2026-06-01 06:51 | baseline_pv_eta_vh_dist_slotres (composite + per-class slot reservation) | **815.8** | **Rolling pair (newest)** |
| **53239342** | 2026-06-01 04:08 | baseline_pv_eta_vh_dist_composite (dist + B.3 head λ=1.0) | **537.7** | **Rolling pair (oldest)** |
| 53227546 | 2026-05-31 17:46 | baseline_pv_eta_vh_dist (distilled-Tier-2 opp only) | 801.1 | EVICTED |
| 53223160 | 2026-05-31 15:17 | baseline_joint_sync_submit (00JzI joint coalitions) | ~1147 | EVICTED — 00JzI's best |
| 53212044 | 2026-05-31 09:21 | baseline_pv_eta_vh_b3smoke (B.3 head λ=1.0) | ~1142 | EVICTED |
| 53197142 | 2026-05-30 22:17 | composite_universal_submit | 1083.0 | EVICTED |
| 53182323 | 2026-05-30 11:26 | baseline_launch_rules_universal | 1183.7 | EVICTED — peak this week |

- **Rolling-pair floor:** μ = 537.7 (composite oldest).
- **Rolling-pair ceiling:** μ = 815.8 (slotres newest).
- **Historical peak (recent week):** μ = 1183.7 (launch_rules_universal).
- **Floor-at-risk:** **TRUE** — rolling pair sits 330+ μ below the
  evicted-pair peak.
- **Daily submission budget:** 5/day. 2026-06-01 used: 3 (composite + slotres + the bridging probes). 2 remaining.
- **Deadline:** 2026-06-23 23:59 UTC. **22 days remain.**
- **Next-session priority:** execute `audit/2026-06-01-integration-plan/README.md` — integrate slot reservation (ours) with joint_sync + size_balance (00JzI).

### Historical (pre-2026-06-01)

Pulled fresh; refresh via `kaggle competitions submissions orbit-wars` at session start.

| Sub ID | Date (UTC) | Agent | μ | Role |
|---|---|---|---:|---|
| **53131296** | 2026-05-28 23:22 | baseline_validated.py (PM5 25-d MLP filter on baseline) | **1081.3** | **Rolling pair (most recent)** |
| **53117942** | 2026-05-28 13:55 | baseline_leaf_pv_2p.py (PV_ETA + 2P leaf re-enable) | **1084.5** | **Rolling pair (older half)** |
| 53111837 | 2026-05-28 09:42 | baseline_pv_eta.py (PV_ETA=1 PV-gamma discount) | ~1154 (historical peak) | **EVICTED — foundation lock 2026-05-29** |
| 52894340 | 2026-05-21 14:33 | _phase4_step1_FND | 1117.9 | EVICTED |
| 52893236 | 2026-05-21 13:52 | baseline_full | 1078.0 | EVICTED |
| 52882014 | 2026-05-21 10:26 | baseline_joint_aggr_consolidated | 1124 | EVICTED |
| 52874528 | 2026-05-21 06:00 | baseline_joint_aggr | 1134.9 | EVICTED |
| 52744856 | 2026-05-17 14:17 | composite_a2_hybrid | 1149.2 | EVICTED |

- **Rolling pair floor:** μ = 1081.3 (baseline_validated, PM5 25-d MLP).
- **Rolling pair ceiling:** μ = 1084.5 (baseline_leaf_pv_2p).
- **Historical peak (EVICTED, kept as foundation):** μ ≈ 1154.8 (sub 53111837, baseline_pv_eta.py).
- **PI direction 2026-05-29:** "We were going to build really on our latest champion on the latest successful submission pv_eta." pv_eta is the inner-agent foundation for all future wrappers. A/B opponent for submit gates is bare pv_eta at n ≥ 32 (Rule 45).
- **Daily submission budget:** 5/day. Today (2026-05-29 UTC) used: 0. 5 remaining. Last submit was 5/28 23:22.
- **Deadline:** 2026-06-23 23:59 UTC. **~25 days remain.**

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
- **Lineage:** v8_scavenge → v15_banded → trajectory v4 → composite head 2P + A2 4P → comet-aim+reactor → hold-feasibility solo. Peak μ=1149.2 (sub 52744856, EVICTED).
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
| Phase 2 v2 LightGBM Booster per-shot validator FILTER | `claude/competition-objective-alignment-hqNVM` | 2026-05-29 | Threshold sweep 0.30 / 0.10 / 0.05 × corpora 100 ms / 1000 ms × inner = bare baseline OR pv_eta. On pv_eta inner: 1/115 drop rate (no-op + 150 ms overhead). Head-to-head validator-on-baseline vs bare pv_eta pooled n=64 = 24/64 = 37.5 %. Rule 37 axis cap. Booster weights (`data/shot_validator/validator_booster.txt`, val_acc 0.83) preserved as INPUT for the Reframe A chooser-input experiment (see HANDOVER.md "Day-N session 2026-05-29"). |
| Reframe A — Booster P_success as additive chooser term on pv_eta | `claude/competition-objective-alignment-hqNVM` | 2026-05-29 | λ ∈ {4.5, 0.5, −0.5} all regressed catastrophically vs bare pv_eta. Per-shot binary supervision doesn't transfer through chooser argmax. Wrap-pv_eta architecture proven clean at λ=0 (byte-equivalent). Audit `audit/2026-05-29-reframe-a-falsified.md`. |
| Reframe B.2 — observational K=10 ship-delta value head, additive term on pv_eta | `claude/competition-objective-alignment-hqNVM` | 2026-05-30 | LightGBM regressor on per-candidate features (14-d + leaf_delta), trained on 33.8k labelled rows from 95-game pv_eta self-play. Training-time Spearman ρ=+0.359 on held-out val (3× the mandatory 0.10 gate), walker parity 0.000e+00. A/B vs bare pv_eta at λ=1.0 and λ=0.1: **0/32 BOTH** (Wilson 95% CI [0.000, 0.107]). Latency p95=691-714ms inside the 1000ms cap. Diagnosis: selection bias — `trace_accepted` only labels candidates pv_eta accepted; head's predictions on rejected candidates are OOD LightGBM extrapolation. At any non-zero λ the bad OOD predictions flip the argmax in losing directions. Rule 37 axis cap on label semantics, NOT λ — do not re-iterate with different λ, K, or feature subsets. Full audit `audit/2026-05-29-reframe-b2-value-head.md`. Next: B.3 with CRN-paired advantage labels (`audit/2026-05-30-reframe-b3-crn-advantage-plan.md`). |

---

## Push claim board (Rule 42 — fill before every `kaggle competitions submit`)

Empty rows below mean no pending submission claim. Most recent claim at top.

| Timestamp (UTC) | Branch | Agent | Predicted μ | Will evict (sub_id, μ) | PI signoff |
|---|---|---|---:|---|---|
| 2026-06-01 06:51 | competition-objective-alignment-hqNVM | baseline_pv_eta_vh_dist_slotres (sub **53243763**, SUBMITTED 2026-06-01 06:51 UTC; commit cbb862d; bundle 1.1 MB). Composite (pv_eta + distilled-Tier-2 opp + B.3 head λ=1.0) + per-class slot reservation `BASELINE_SLOT_RESERVATION=3/2/2`. Diagnoses sub 53239342 (composite μ=545.4): proposer surfaced 63 candidates at the critical state but chooser only scored top 5 by cheap_delta — ALL defenses with leaf_delta=0. Slot reservation partitions prerank by target class (3 attack + 2 expansion + 2 defense slots) so attacks/expansions get scored. Probe at ep 78367540 step 100: attack leaf_delta=+40.3 NOW scored; expansions +29.4 and +20.7 NOW scored. Wallclock at default 600ms (smoke with 800ms went over the 1000ms env cap). Rule 46 GREEN: pytest tests/test_bundle.py 10/10 (default-off parity preserved); fast.py play vs v7_0 max=860ms inside cap. **Action distribution full game**: defense 47.0% (was 87.3%), expansion 26.8% (was 6.6%), attack **26.3% (was 6.1%)** — class balance restored. n=1 vs v7_0 WIN 106 steps. Rule 37: chooser-budgeter is new axis. | ~700-1100 (structural fix evidence is probe + action-distribution; no live A/B). | Evicts 53227546 dist μ=**801.8** (oldest in rolling pair). μ-LOSS predicted unless slotres ≥ 802. Position-1 backstop after submit: 53239342 composite μ=545.4. | ✅ PI explicit "submit" 2026-06-01 06:51 UTC |
| 2026-06-01 04:08 | competition-objective-alignment-hqNVM | baseline_pv_eta_vh_dist_composite (sub **53239342**, SUBMITTED 2026-06-01 04:08 UTC; commit cc8c101; bundle 1.1 MB). pv_eta + DISTILLED Tier-2 opp model (VH_LAMBDA=1.0 — composes B.3 head ON TOP OF distilled opp). Hypothesis: B.3 head's action-vs-idle residual breaks the idle-equilibrium that distilled-dist (sub 53227546 μ=823.5) got stuck in. **Probe evidence (commit 72ab99a)**: single-state diagnostic on the two losing replays' dead-zone states confirms bare dist returns 0 moves; composite λ=1 returns 2 launches in both cases. ep 78324838 step 120 (own=10/garr=33/inflight=0 vs opp=14/garr=31/inflight=0): bare dist 0 moves → composite 2 launches (25 ships ea). ep 78324483 step 375 (own=7/garr=28/inflight=0 vs opp=22/garr=56/inflight=121): bare dist 0 moves → composite 2 launches (123, 128 ships). **Rule 46 GREEN**: pytest tests/test_bundle.py 10/10 in 53s; fast.py play vs v7_0 completed 251 steps inside cap (p50=738ms, p95=870ms, max=986ms). Single-game outcome was LOSS vs v7_0 but n=1 uninformative (Rule 45). **Local A/B**: n=8 composite vs bare dist TIMED OUT at 50 min wallclock (heavy-vs-heavy each game ~8 min serial). NO local A/B evidence; structural argument is probe-based. **Rule 37 status**: distilled-opp axis is at axis-cap (3 variants tried, all regressed) — pivoting to our-decision side via head composite is the correct axis change. | ~1000-1200 (composite + distilled-Tier-2; structural lift estimate from probe-fix-mechanism) | Evicts 53223160 baseline_joint_sync μ=**1146.9** (position-2 backstop) — **μ-LOSS predicted, status PENDING**. Position 1 backstop after submit: 53227546 dist μ=807.6. | ✅ PI explicit "submit" 2026-06-01 04:07 UTC |
| 2026-05-31 17:46 | competition-objective-alignment-hqNVM | baseline_pv_eta_vh_dist (sub **53227546**, SUBMITTED 2026-05-31 17:46:01 UTC) — pv_eta + distilled-ladder Tier-2 opp model (lib/opp_model.py:trained_logreg_policy v2 replaces falsified filter-on-Tier-1 design); 30-d lite feature encoder via lib/opp_features_lite.encode_lite_batch (parity-verified ≤0.056 abs diff vs slow encoder); booster trained on 50,482 positives from 775 top-10 Kaggle ladder 2P replays (2026-05-30 daily dataset); val_acc 0.952 / Brier 0.035 / separation 0.21 / threshold 0.15. Speed: 1.6ms median policy (Tier 1 was 3.4ms; slow encoder 24ms). KINEMATIC_TABLE_ENABLED hard-disabled in lib/trajectory.py (PI directive d8dad76). Local: n=32 vs launch_rules TIMED OUT at 60min serial (~7min/game); n=1 vs v7_0 WIN seed=0; n=1 vs launch_rules WIN seed=0; n=1 vs v7_0 LOSS seed=0. Rule 46 GREEN: test_bundle.py 10/10; fast.py play completed full game p95=842ms/max=995ms inside 1000ms cap. VH_LAMBDA=0 (head OFF for clean Tier-2 attribution). | unknown — n=1 evidence too noisy; posterior range ~950-1200 | 53212044 (baseline_pv_eta_vh_b3smoke, μ=**1147.0**) — **μ-LOSS eviction confirmed** | ✅ PI explicit "submit" 2026-05-31 17:35 |
| 2026-05-31 09:21 | competition-objective-alignment-hqNVM | baseline_pv_eta_vh_b3smoke (sub **53212044**, SUBMITTED 2026-05-31 09:21:28 UTC; CRN-paired advantage value head, λ=1.0). Local A/B: vs bare pv_eta 32/32 = 100% Wilson-lo 0.893 (panel PASS); vs `baseline_launch_rules_universal` 18/32 = 56.2% Wilson-lo 0.393 (FAILS Rule 43b 0.50 gate — explicit PI override). Bundle 772 KB, parity-tested 2026-05-30. Rule 46 GREEN per audit/2026-05-30-reframe-b3-results.md. | ~1140-1180 (against current ladder μ=1183 champion) | 53182323 (baseline_launch_rules_universal, μ=**1183.7**) — **μ-LOSS eviction** | ✅ PI explicit "submit, so we can learn and observe" 2026-05-31 |
| 2026-05-28 23:20 | competition-objective-alignment-hqNVM | baseline_validated (PM5 25-d MLP shot-validator filter on baseline) — first cut of the konbu17-style per-shot validator (PENDING SUBMIT). Rule 46: bundle 774KB; test_bundle.py 10/10 + test_validator_smoke.py 6/6 GREEN; fast.py play vs v7_0 ran 368 turns, p0_win, p95=887ms / max=994ms (inside the 1000ms env cap) | ~1100-1170 (calibration probe). Local A/B: 60.9% n=64 vs baseline Wilson-lo 0.487; 53.1% n=32 vs pv_eta Wilson-lo 0.364 (both below the Rule 43 0.50 gate — PI override) | 53111837 (baseline_pv_eta, μ=1154.8 — **LIVE CHAMPION**) | ✅ PI explicit "submit the previous version so I can see it playing on kaggle While we develop the new version" 2026-05-28 23:09 |
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
