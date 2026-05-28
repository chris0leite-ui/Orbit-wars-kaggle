# state/MULTI_BRANCH.md — single source of truth across all dev branches

> **Written:** 2026-05-20 by `claude/review-skills-improvements-moKOR`.
> Replaces the per-branch `state/current.md` divergence. Read this
> file FIRST every session (Rule 44).

> **2026-05-28 update.** All "build on top of the baseline" work MUST
> start from [`state/PEAK_BASELINE.md`](PEAK_BASELINE.md) — the canonical
> reference for the strongest historical bundle (sub 52912707 μ=1165.4 /
> resubmit 53013786 μ=1144.6, git tag `peak-1165`, bundle SHA `9ec3af83`,
> frozen anchor at `submissions/baseline_peak_1165_anchor.py`). That
> file documents the active vs dormant env-var stack, the mandatory
> build-on-top protocol, and the anti-patterns from two days of regressions.

## How to use this doc

1. Read the **Live Kaggle** section first; refresh from `kaggle competitions submissions orbit-wars` if any timestamp here is >24h old.
2. Read the **Track registry** section to find which work-track owns the subsystem you're about to touch.
3. Read the **Closed tracks** list — if your idea lives there, STOP, the axis is falsified.
4. Before any `kaggle competitions submit`, fill out the **Push claim board** row (Rule 42).
5. Before any "build on top of baseline" change, read [`PEAK_BASELINE.md`](PEAK_BASELINE.md) and follow the 7-step protocol there.

---

## Live Kaggle (snapshot 2026-05-23 21:00 UTC)

Pulled fresh; refresh via `kaggle competitions submissions orbit-wars` at session start.

| Sub ID | Date (UTC) | Agent | μ | Role |
|---|---|---|---:|---|
| **52894340** | 2026-05-21 14:33 | _phase4_step1_FND (sibling, endgame predicate + f1774a7 orbital safety) | **1117.9** | **Rolling pair (most recent)** |
| **52893236** | 2026-05-21 13:52 | baseline_full (this branch, kitchen-sink) | **1078.0** | **Rolling pair (older half)** |
| 52882014 | 2026-05-21 10:26 | baseline_joint_aggr_consolidated | 1124 | EVICTED — best on this branch |
| 52874528 | 2026-05-21 06:00 | baseline_joint_aggr | 1134.9 | EVICTED |
| 52857903 | 2026-05-20 16:12 | analytical_wait_N_traj_plus_endgame_play | 806.5 | EVICTED |
| 52854094 | 2026-05-20 13:59 | analytical (earlier) | 829.1 | EVICTED |
| 52827111 | 2026-05-19 19:52 | comet-aim + reactor-aware | 1122.0 | EVICTED |
| 52811320 | 2026-05-19 12:54 | hold-feasibility solo | 1135.1 | EVICTED |
| **52744856** | 2026-05-17 14:17 | composite_a2_hybrid (composite head 2P + A2 4P) | **1149.2** | **TEAM PEAK** — EVICTED |

- **Rolling pair floor:** μ = 1078.0 (baseline_full).
- **Rolling pair ceiling:** μ = 1117.9 (_phase4_step1_FND).
- **Team peak (evicted):** μ = 1149.2 (sub 52744856).
- **Floor recovered from 5/20:** previous rolling pair was [829, 806]; current floor 1078 is +249 μ over that.
- **Daily submission budget:** 5/day. Today (2026-05-23 UTC) used: 0. 5 remaining. Last submit was 5/21.
- **Deadline:** 2026-06-23 23:59 UTC. **~31 days remain.**

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
| 2026-05-28 13:55 | kaggle-submission-review-gZsCu | **LEAF_PV_2P=1** — `submissions/baseline_leaf_pv_2p.py` (sub **53117942**, bundle SHA `e830691c`, commit `8742170`). Built on top of PV_ETA wrapper + new env-var alias `BASELINE_LEAF_PV_2P=1` that re-enables the `_COMPOSITE_PV_ENABLED` block in the 2P leaf (lib/value_heads.py:181, disabled since 2026-05-18 PV-recalibration-debt). Fixes the silent-turns failure mode documented in `knowledge-base/thoughts/2026-05-28-silent-turns-pre-existing-weakness.md`: peak's 2P leaf was bare ship-delta, captures earned zero credit for future production, candidate Δ collapsed → 13-14 silent turns in mid-game (smoking-gun seed=2 vs v4_planner). 4/4 unit tests green; pre-existing `test_baseline_value` / `test_value_heads` / `test_chooser_pv_eta` / `test_bundle.py` siblings all green; smoke vs v7_0 seed=7 p0_win 159 steps max-turn 736ms. **Local A/B vs `baseline_peak_1165_anchor.py` (PI override of Rule 45): n=5 → 3-2 (60%); n=10 → 7-3 (70%, Wilson 95% lo ≈ 0.40)**. Mechanism check vs `submissions/v4_planner.py`: **5-0 (100%)**, including seed=2 flipped L→W (the known silent-turns failure seed). Caveats acknowledged: Wilson-lo 0.40 below Rule 45 0.50 submit gate (PI override — "submit then investigate"); 3/7 wins are 250-step cap-hits (cap-bias risk axis from Rule 47); A/B step-count drift on seed=3 across runs flagged as non-determinism investigation TODO (post-submit). | ~1050-1300 (mode ~1180). PV_ETA backstop at position 2 (μ=1154.8 settled). Expected lift over evicted PEAK RESTORE (μ=1114.5) by point estimate +65μ; downside band includes ~1050 regression. | 53099429 (PEAK RESTORE, μ=1114.5) | ✅ PI explicit 2026-05-28 13:50: "submit our solution and then do the investigation" + after eviction-trade surfacing: "There's already a score available for our last submission. submission before that did not perform so well. So it's okay to evict it." Evicted-μ (1114.5) < predicted mode (1180) → Rule 42 GREEN by point estimate; PV_ETA μ=1154.8 backstops worst-case downside. |
| 2026-05-28 09:42 | kaggle-submission-review-gZsCu | **PV_ETA=1** — `submissions/baseline_pv_eta.py` (sub **53111837**, bundle SHA `6d839443`, commit `0d71aa6`). Peak-orbitfix preamble + `BASELINE_PV_ETA=1`. Single env-var-gated change behind the peak `peak-1165`: multiply each candidate's final Δ by `γ^(wait_N + eta)` in `score_candidate_v4` and `score_candidate_v4_joint` — modeling-correct PV pull-back replacing the SHIP_TURN_KAPPA band-aid (sub 53099001 disaster). 5/5 unit tests green; 55/55 broader chooser/value/scoring/bundle sweep; wrapper smoke vs v7_0 seed=7 p0_win 164 steps max-turn 774ms. **Local A/B (NEW protocol: n=5 per opp, step-250 cap, focal-as-P0 no swap):** vs peak anchor 4-1 (80%); vs v7_0 4-1; vs v4_planner 4-1; vs v3.5.1 4-1. Pooled 16-4 / 20 = 80%, Wilson-lo 0.58 (clears Rule 43 0.55 gate). 4/4 opponents identical 4/5 → not an A>B>C>A loop; under fair-coin null the 4×(4/5 or 5/5) joint probability is 0.12%. Caveats: n=5 below Rule 45 floor of 32 (PI explicit override); step-250 isn't live metric (PI directive). **OUTCOME: sub 53111837 settled μ=1154.8 — clean lift over PEAK RESTORE (1114.5), within historical peak band [1144-1165].** | ~1100-1170 (mode 1140). Peak anchor currently μ=1116.2 (drifted from 1144-1165 historical band per kaggle CLI 2026-05-28 09:25 pull); our agent beats it 4-1 on panel + at single-opp run; lift band 0-50μ over current anchor reading. Lower bound 1100 set conservatively for step-250 metric tilt. | 53099001 (κ=0.02 disaster, recovered to μ=1109.9 from initial 680 read) | ✅ PI explicit 2026-05-28 09:32: "Submit." after Rule 26 devil's-advocate ritual surfaced step-250 truncation bias + n=5 < Rule 45 floor + recent two-disaster pattern; PI chose "Wait for full panel" then "Submit" after panel landed 4-1 × 4 opps. Evicted-μ (1109.9) > predicted lower-band (1100) → Rule 42 MARGINAL → PI signoff covers it. |
| 2026-05-28 00:30 | kaggle-submission-review-gZsCu | **PEAK RESTORE** — submissions/baseline_joint_aggr_consolidated_orbitfix.py byte-identical to commit 458f663 (sub 52912707 μ=1165.4 / resubmit 53013786 μ=1144.6). Bundle SHA 9ec3af83. Removes the NEUTRAL_BONUS-into-v4 plumbing change (which the REVERT carried forward) AND the Step 2B SHIP_TURN_KAPPA penalty (which just landed at μ=680.0 — confirmed disaster). Pure rollback to the strongest historical bundle. | ~1130-1170 (historical band: 1165.4 original, 1144.6 byte-identical resubmit — rolling-pair noise ~20μ envelope). **OUTCOME: sub 53099429 settled μ=1116.2 — ladder drift in last 24h. Peak bundle byte-identical now reads ~30μ below historical band.** | 53088099 (REVERT, μ=1125.2) | ✅ PI explicit 2026-05-28: "I want you to restore exactly the submission that we had that scored on top." Evicted-μ (1125.2) < predicted lower-band (1130) → Rule 42 GREEN. |
| 2026-05-27 23:55 | kaggle-submission-review-gZsCu | baseline_joint_aggr_consolidated_orbitfix + **BASELINE_SHIP_TURN_KAPPA=0.02** (commit 3c6c6dc) — Step 2B ship-turn opportunity-cost penalty plumbed into `score_candidate_v4` + `_joint` via `delta -= κ × ships × (wait_N + eta)`. Fixes the `favor.pv_horizon(leaf_step, 0) ≈ 99` plateau that scored slow captures (eta=40) at ~88% of fast captures (eta=10) despite tying up ships 4× longer. Trace at κ=1.0 showed 97.7% of positive-Δ candidates killed (over-aggressive); κ=0.02 calibrated to observed Δ scale (3-50 vs plan's 300). 5 unit tests green. | ~1080-1170 (anchor μ=1125.2). Smoke A/B vs head_anchor at κ=0.02 (n=8 across seeds {7,11,23,41} × 2 seats, 215-step cap): **6W-2L (75%, Wilson-lo 0.349)**. Below Rule 45 n=32 floor — submitting on PI explicit signoff for calibration. **OUTCOME: sub 53099001 settled μ=680.0 — -445μ disaster vs REVERT (1125.2). 5% worst-case tail realized. Hypothesis: κ=0.02 was calibrated on n=8 vs anchor only; ladder distribution differs sharply (same shape as sub 53083109 panel divergence). Lesson: Rule 45 n=32 floor + Rule 43 panel exist for cases exactly like this.** | 53083109 (POST-FIX regression, μ=921.1) | ✅ PI explicit 2026-05-27: "Submit to Kaggle" + "Bundle + submit at κ=0.02 now" (chose this over Rule-43-compliant Phase 1 n=32 panel option). Evicted-μ (921.1) < predicted-μ (1080-1170) → Rule 42 GREEN. |
| 2026-05-27 14:35 | kaggle-submission-review-gZsCu | baseline_joint_aggr_consolidated_orbitfix REVERT (sub **53088099**, commit 70ba870) — Fix 2a (holdability v2) + Fix 3 (larger→smaller drain) now opt-in via `BASELINE_HOLDABILITY_V2` / `BASELINE_DRAIN_HARDEN` env vars; wrapper does NOT set them → legacy paths. Fix 1 (`BASELINE_NEUTRAL_BONUS=2.0` plumbing) still active. Diagnostic submit: μ ≈ peer-anchor (1144-1165) → Fix 2a/3 carried the regression; μ stays ~840 → NEUTRAL_BONUS=2.0 was also harmful. | ~1100-1170 (peer anchor: 52912707 μ=1165.4, byte-identical resubmit 53013786 μ=1144.6). Panel n=32 confirmed sub 53083109 lost 2/32 vs peer anchor — this revert removes the two suspected killers. | 53082192 (baseline relay v2, μ=823.6) | ✅ PI explicit 2026-05-27: "submit the reverted version to see if we can recover the score" |
| 2026-05-27 11:50 | kaggle-submission-review-gZsCu | baseline_joint_aggr_consolidated_orbitfix (commit 8133410) — NEUTRAL_BONUS plumbed into score_candidate_v4/_joint, _target_holdable_after_capture iterates ALL opps with recapture-cost fixed-point + B7 orbital, source-drain hardened for larger→smaller (`src.production > tgt.production`) via stockpile floor + SAFETY_MARGIN_DRAIN=1.3 + potential-launch coverage, opt-in BASELINE_FOLLOWON_BONUS wires B3/B4 into live scoring (default OFF this submit) | ~1100-1200 (anchor: peak orbitfix 52912707 μ=1165.4, byte-identical resubmit 53013786 μ=1144.6). No local n≥32 A/B — PI explicit "submit now for early feedback" learning push. **OUTCOME: sub 53083109 settled μ=947.1 (later read; initially 842.8). Panel n=32 vs peer: 2/32 wins (Wlo=0.017) FAIL. Panel n=64 vs v7_0: 48/64 (75%, Wlo=0.632) PASS. Regression is anchor-specific, not uniform.** | 53067354 (baseline relay v1, μ=905.0) | ✅ PI explicit 2026-05-27: "Submit now and then run the pen" |
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
