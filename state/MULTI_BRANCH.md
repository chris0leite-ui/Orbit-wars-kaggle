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

## Live Kaggle — READ IT DIRECTLY, never transcribed here

**Scores (μ) and the rolling pair are NEVER written into this file** — they go
stale within hours and have repeatedly misled sessions into acting on dead
numbers. The leaderboard is the single source of truth. Read it on demand:

```
kaggle competitions submissions orbit-wars
```

- **Rolling pair** = the **two most-recent** submissions in that list (Kaggle
  auto-keeps exactly these two for final evaluation — not PI-selected; Rule 12).
- **Daily submission budget:** 5/day. Check today's usage from the same list.
- **Deadline:** 2026-06-23 23:59 UTC (fixed).

Before any submit, fill the **Push claim board** below (Rule 42) using the
freshly-read pair — do not rely on any number cached in prose.

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

## Open track — reach-frontier doctrine (2026-05-27)

| Branch | State | Evidence | Next |
|---|---|---|---|
| (closed 2026-05-28 — moved to Closed tracks below) | — | — | — |

---

## Open work — sync coalition (2026-05-31, `champion-strategy-rules-00JzI`)

Synchronized two-source team-up on `agents/baseline/chooser_trajectory.py`
(env `BASELINE_JOINT_SYNC=1`, default OFF). **Owns:** the sync-coalition
generator inside the trajectory chooser — coordinate before touching it.

| State | Evidence | Next |
|---|---|---|
| **K-gate bug FIXED** (commit `8d94989`) → **panel CONFIRMED** → **SUBMITTED** (sub `53223160`, calibration probe). Sync coalition is a broad panel winner; **Lever 1 size-to-hold is NULL** (7/7 tie, shelved, default-OFF). | Panel (Rule 43a PASS): v7_0 90.6% / v4_planner 93.8% / v3.5.1 87.5%, Wilson-lo ≥0.72. Champion h2h 44–56% Wilson-lo 0.39 (Rule 43b FAIL → probe). Lever 1 A/B (`_run_hold_ab.sh`): hold-off 7/16 = hold-on 7/16 = 43.8% vs champion, symmetric 2W/2L flip. Trace: recapture leak is opponent-specific (champion 40%→0% with hold; v7_0 0% either way). | **(1) READ sub 53223160 settled μ** — is sync a ladder gain or only a panel-beater? **(2)** Deferred control: run *champion* vs the same 3-opp panel (~30 min) to disambiguate. **(3)** If sync flat vs field → pivot to new mechanism (H44 fleet-survival defense or 2-hop redeploy). Size-to-hold = closed (only un-tried variant: probabilistic/less-pessimistic counter model). See HANDOVER. |

## Closed tracks — falsified knowledge, do NOT iterate

| Axis | Branch | Verdict date | Evidence |
|---|---|---|---|
| Reach-frontier doctrine (closed-form ρ as prescription) | `claude/game-theory-winning-strategy-SEU7P` | 2026-05-28 | Rule 37 exhausted across 3 operationalisations: v1 chooser 0/20 (`audit/2026-05-27-rf-v1-b5-triage.md`), v2 chooser hold-floor+gang-up 0/32 (`audit/2026-05-27-rf-v2-b5-triage.md`), 4P delayed-launch cushion 4/32 vs baseline's 26/32 (`audit/2026-05-28-4p-cushion-falsified.md`). Doctrine MATH (production-time integral as score, n=92 share-separation 0.488) remains sound and is durable knowledge; doctrine PRESCRIPTIONS empirically falsified on our μ-band. Durable artefacts that ship despite the null: `fast.py --save-replays`, `scripts/measure_hold_times.py --replay-dir` (Rule 48 measurement substrate), bundler DEFAULT_LIB_ORDER fix (`kinematic_table`, `joint_solver/columns`, `joint_solver/lp`), `lib/joint_solver/lp.py` greedy-fallback unbreak. Doctrine + chooser-design + evaluation-metrics docs stay in tree as reference for what's been tried. |
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
| 2026-06-01 09:30 | champion-strategy-rules-00JzI | baseline_size_balance — champion full config (pv_eta / orbital_safety / launch_rules K=10 / neutral_bonus / joint_aggr, trajectory chooser) + **size-balance fix (A+D)** baked ON (`BASELINE_SIZE_BALANCE=1`): per-launch, send enough to win on arrival (D: under-delivery) + don't drain a source below its threat-keep floor (A: over-drain) + skip unwinnable launches. Built from current branch (kinematic_table absent — dead code in champion, behaviorally neutral). Rule 46 GREEN (bundle ✓ / test_bundle 15-15 ✓ / cold-load 309-step game ✓). | ~1150 **HIGH UNCERTAINTY** — evidence is **n=16 single-opponent triage only** (fix-ON 75% vs OFF 44%, +31pp). **Rule 43 FAIL** (no multi-opp panel) + **Rule 45 FAIL** (n=16 < 32). Champion base ~1183; whether the +31pp triage lift survives the live multi-opp ladder is unmeasured. Calibration probe. | 53239342 (composite, μ=526.3 — older half of rolling pair). Backstop position-2: 53243763 (slotres, μ=678.9) stays. Evicted-μ 526.3 ≪ predicted → **Rule 42 GREEN**. | ✅ PI explicit "go" (2026-06-01), informed override of Rules 43/45 (n=16 single-opp evidence surfaced via AskUserQuestion + μ=711 precedent cited) |
| 2026-05-31 15:30 | champion-strategy-rules-00JzI | baseline_joint_sync_submit — champion full config (pv_eta / orbital_safety / launch_rules K=10 / neutral_bonus / joint_aggr, trajectory chooser) + **synchronized two-source coalition** (`BASELINE_JOINT_SYNC=1`, src_K=3). Size-to-hold **OFF** (locally null, 7/7 tie vs champion). Config baked into a top-of-file setdefault header (Kaggle has no env). | ~1100–1180 (panel **88–94%** vs v7_0 / v4_planner / v3.5.1, Wilson-lo 0.72–0.80 → **Rule 43a PASS**; champion head-to-head 44–56% Wilson-lo 0.39 → **Rule 43b FAIL** — calibration probe). | 53197142 (composite_universal, μ=1086.9 — our weakest recent half of the rolling pair). Backstop position-2: 53212044 (baseline_pv_eta_vh, μ=1139.6) stays. Evicted-μ 1086.9 < predicted → **Rule 42 GREEN**. | ✅ PI explicit "submit our solution to see how it performs" (2026-05-31) |
| 2026-05-30 11:25 | champion-strategy-rules-00JzI | baseline_launch_rules_universal — champion full config + **universal** K=10 ceiling: EVERY launch (neutral / opponent / own-reinforcement / comet) arriving after K=10 is dropped post-emit, not just opponent captures. Strict superset of the live Rule-A/B build (53175658). | ~1100-1130 (n=64 A/B universal-validator-ON vs no-validator = 44-20 = 0.688, Wilson-lo 0.566, clears Rule 45; drops the 168 wasteful small/far launches the Rule-A/B build keeps. NOT directly measured vs the live 1099.5 — expected small-positive.) | 53175658 (baseline_launch_rules_k10, μ=1099.5 — older half of rolling pair). Backstop position 2: 53177486 (baseline_redeploy_gangup, 971.1, sibling SEU7P push). | ✅ PI explicit "Submit universal ceiling" (AskUserQuestion 2026-05-30, informed of the 1099.5 eviction + weak-971.1 backstop) |
| 2026-05-30 07:15 | champion-strategy-rules-00JzI | baseline_launch_rules_k10 — champion full config (JOINT_AGGR/NEUTRAL_BONUS/ORBITAL_SAFETY/PV_ETA) + post-emit launch-discipline validator (Rule A neutral-capture, Rule B opponent arrival ≤ K=10) | ~1100-1160 (n=4 isolated rules-on-vs-off triage 3/4, both seats; n=32 A/B launched in parallel). Calibration probe. | 53131296 (baseline_validated, μ=1100.8 — older half of rolling pair; keeps 53163774 baseline_pv_eta μ=1110.3 as backstop) | ✅ PI explicit "submit to kaggle so I can observe and we measure" |
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
