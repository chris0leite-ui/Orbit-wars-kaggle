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

## Live Kaggle (snapshot 2026-05-20 17:00 UTC)

Pulled fresh; refresh via `kaggle competitions submissions orbit-wars` at session start.

| Sub ID | Date (UTC) | Agent | μ | Role |
|---|---|---|---:|---|
| **52857903** | 2026-05-20 16:12 | analytical_wait_N_traj_plus_endgame_play | **806.5** | **Rolling pair (most recent)** |
| **52854094** | 2026-05-20 13:59 | analytical (earlier) | **829.1** | **Rolling pair (older half)** |
| 52845073 | 2026-05-20 09:01 | phase1_chooser_emit_fix_old_lineage | 1051.3 | EVICTED |
| 52827111 | 2026-05-19 19:52 | comet-aim + reactor-aware | 1122.0 | EVICTED |
| 52811320 | 2026-05-19 12:54 | hold-feasibility solo | 1135.1 | EVICTED |
| 52784853 | 2026-05-18 17:42 | baseline_pv_off_bugfix | 1130.4 | EVICTED |
| 52766596 | 2026-05-18 07:12 | joint_candidate_v3 | 1118.3 | EVICTED |
| 52754310 | 2026-05-17 22:06 | trajectory v4 + wait_N + wallclock | 1143.7 | EVICTED |
| **52744856** | 2026-05-17 14:17 | composite_a2_hybrid (composite head 2P + A2 4P) | **1149.2** | **TEAM PEAK** — EVICTED |

- **Rolling pair floor:** μ = 829.1.
- **Team peak (evicted):** μ = 1149.2 (sub 52744856).
- **Floor lost in last 24h:** ~320 μ. Five sequential pushes from `claude/strategy-framework-design-OyoYR-rebased` (analytical track) evicted strong agents from sibling branches without coordination. This is the friction `cross-agent-push-coordination-gap` (btjeK 2026-05-21) — addressed by Rule 42 below.
- **FLOOR AT RISK FLAG: ✅ TRUE** — rolling pair is 320 μ below team peak.
- **Daily submission budget:** 5/day. 5/20 used: 2 (52854094, 52857903). 3 remaining.
- **Deadline:** 2026-06-23 23:59 UTC. **34 days remain.**

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

---

## Push claim board (Rule 42 — fill before every `kaggle competitions submit`)

Empty rows below mean no pending submission claim. Most recent claim at top.

| Timestamp (UTC) | Branch | Agent | Predicted μ | Will evict (sub_id, μ) | PI signoff |
|---|---|---|---:|---|---|
| 2026-05-21 06:00 | review-skills-improvements-moKOR | baseline_joint_aggr (sub **52874528**, PENDING) | ~1180 (n=16 vs phase_c 87.5%) | 52865089 (μ=805.9, evicted) | ✅ Rule 45 overridden |
| 2026-05-20 17:00 | review-skills-improvements-moKOR | — | — | — | — *(no submission planned this session)* |

**Required fields:** all 6 columns. The PI-signoff column is mandatory if the evicted-μ > predicted-μ (Rule 42).

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
