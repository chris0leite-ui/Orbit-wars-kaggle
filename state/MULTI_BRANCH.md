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

## Live Kaggle (snapshot 2026-05-25 11:55 UTC)

Pulled fresh; refresh via `kaggle competitions submissions orbit-wars` at session start.

| Sub ID | Date (UTC) | Agent | μ | Role |
|---|---|---|---:|---|
| **53018599** | 2026-05-25 11:54 | buildup_planner K1+Z v2 (commit 603f45f, this branch) | **PENDING** | **Rolling pair (most recent)** |
| **53013786** | 2026-05-25 08:40 | baseline_joint_aggr_consolidated_orbitfix RESUBMIT (sibling ESwSv, commit 458f663) | **1120.1 (adapting)** | **Rolling pair (older half / strong half)** |
| 53001857 | 2026-05-24 23:51 | baseline_wave v3.1 (sibling ESwSv) | 1126.8 | EVICTED 2026-05-25 by sub 53018599 |
| 53000996 | 2026-05-24 22:38 | buildup_planner_phi1_only (Phi-1 leaf swap, this branch) | 1115.2 | EVICTED 2026-05-25 by sub 53013786 |
| 52993021 | 2026-05-24 16:10 | buildup_planner_concentration (A+B α=1.5, this branch) | 1117.9 | EVICTED 2026-05-25 by sub 53001857 |
| 52968889 | 2026-05-23 23:59 | buildup_planner (this branch, bundler-trailer fix) | 1142.4 | EVICTED 2026-05-24 by sub 53000996 |
| 52966655 | 2026-05-23 21:18 | baseline (wave V3, sibling moKOR) | 1130.9 | EVICTED 2026-05-24 by sub 52993021 |
| 52968305 | 2026-05-23 23:17 | buildup_planner (bundler ERROR) | — | EVICTED — ERROR |
| 52965748 | 2026-05-23 20:26 | orbitfix_kt_p23 v5 | 1002.7 | EVICTED |
| 52963659 | 2026-05-23 18:41 | orbitfix_kt_p23 v4 (ERROR) | — | EVICTED |
| 52959167 | 2026-05-23 15:28 | orbitfix_kt_p23 v3 | 973.0 | EVICTED |
| 52894340 | 2026-05-21 14:33 | _phase4_step1_FND (sibling) | 1117.9 | EVICTED |
| 52893236 | 2026-05-21 13:52 | baseline_full (this branch, kitchen-sink) | 1078.0 | EVICTED |
| 52882014 | 2026-05-21 10:26 | baseline_joint_aggr_consolidated | 1124 | EVICTED |
| 52874528 | 2026-05-21 06:00 | baseline_joint_aggr | 1134.9 | EVICTED |
| **52744856** | 2026-05-17 14:17 | composite_a2_hybrid (composite head 2P + A2 4P) | **1149.2** | **TEAM PEAK** — EVICTED |

- **Rolling pair floor:** 1120.1 (sub 53013786 strong half; adapting). Sub 53018599 starts at 600 and will adapt.
- **Team peak (evicted):** μ = 1149.2 (sub 52744856).
- **Daily submission budget:** 5/day. Today (2026-05-25 UTC) used: 1 (sub 53018599). 4 remaining; HOLD per Rule 48 until sub 53018599 settles.
- **Floor risk:** Sub 53018599 evicts sub 53001857 (μ=1126.8). If K1+Z v2 settles below ~1120 the rolling-pair floor drops. K1 is bit-parity by construction; Z v2 is the only behavior change. n=64 local A/B vs phi1_only was +10.9pp parity-band; live signal pending.

## Push claim board (Rule 42)

| Timestamp (UTC) | Branch | Agent | Predicted μ | Evicting (sub_id, μ) | Verdict |
|---|---|---|---|---|---|
| 2026-05-24 16:10 | claude/agent-design-exploration-Q0q9T | buildup_planner_concentration (commit 2878bfd) | 1100-1180 (parity-or-lift band) | sub 52966655 μ=1130.9 | GREEN (predicted band ≥ evicted μ; better-half μ=1144.5 stays in rolling pair) |
| 2026-05-24 22:38 | claude/agent-design-exploration-Q0q9T | buildup_planner_phi1_only (commit 1f020e1 + buildup_planner main hard-set) | 1050-1200 (n=8 parity 4/8, Wilson [0.22, 0.79]) | sub 52968889 μ=1144.5 | MARGINAL (PI explicit override; predicted band overlaps evicted but central tendency below). |
| 2026-05-25 11:54 | claude/agent-design-exploration-Q0q9T | buildup_planner K1+Z v2 (commit 603f45f) | 1100-1180 (n=64 vs phi1_only 56.2% Wilson [0.441,0.677]; +10.9pp over K1-alone) | sub 53001857 μ=1126.8 | MARGINAL (PI explicit override on Rule 45 Wlo<0.50; K1 is bit-parity by construction, Z v2 is the behavior change). Sub 53013786 μ=1120.1 anchors strong half. |
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
| Proposer pre-filter tightening (Z v2 + opp-model floor + holdability floor) | `claude/agent-design-exploration-Q0q9T` | 2026-05-25 | Rule 37: three consecutive falsifications vs joint_aggr (Z v2 parity at n=64; Fix A+B 20% at n=5; Fix A alone 0% at n=5). Wins against quiet opp (phi1_only 80%) but over-restricts against aggressive opp. Reverted Fix A+B; Z v2 stays shipped (sub 53018599). Next axis = chooser-side opp model, not proposer-side thresholds. See audit/2026-05-25-consolidation-review.md + knowledge-base/thoughts/2026-05-25-k1-zv2-ship-and-axis-exhaust.md |

---

## Push claim board (Rule 42 — fill before every `kaggle competitions submit`)

Empty rows below mean no pending submission claim. Most recent claim at top.

| Timestamp (UTC) | Branch | Agent | Predicted μ | Will evict (sub_id, μ) | PI signoff |
|---|---|---|---:|---|---|
| 2026-05-23 23:17 | agent-design-exploration-Q0q9T | buildup_planner (sub **52968305**, PENDING) — phase-dispatch BUILDUP MILP + CONSOLIDATION (v15-functional chooser_trajectory) + FINISHER endgame full-board capture wave; STRIKE/DOGPILE flags default OFF | ~1080-1120 (local n=16 vs agents/baseline 8/16 elim = 50% parity with v15-functional; calibration target μ ≥ baseline_full's 1078) | 52965748 (orbitfix_kt_p23 v5, μ=984.0) | ✅ PI explicit "submit to kaggle for calibration" / "submit now" |
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
