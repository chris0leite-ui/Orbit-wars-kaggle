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

## Open work — adaptive horizon K + loss-mode re-diagnosis (2026-06-01, `champion-strategy-rules-00JzI`)

**Owns:** `launch_rules.capture_horizon_k()` and its three readers (launch
gate, proposer prune, sync cap) — coordinate before touching the K horizon.

| State | Evidence | Next |
|---|---|---|
| **Loss mode re-diagnosed: NOT ship-hoarding.** We out-*launch* but under-*capture* (conversion gap, not volume). **Adaptive-K v1 built** (step-decay, default OFF, commit `9985e98`); A/B in flight. PI redirected K to **state-driven**. | `audit/2026-06-01-loss-mode-diagnosis.md`: champion's 121 live games — opening planet-gap≈0, midgame capture rate halves in losses (25→12, 2P), defense ~equal; higher in-flight fraction in losses. PI corrections: selection bias (win/loss confounded by opp strength); **fleets don't die in flight** (H44 lever dropped); opening tempo real. Paired positioning = NULL (US 39.2 vs OPP 39.2 dist-to-enemy). v1 A/B (ON vs immune `baseline_champion_nokt.py`, CRN): interim 12W/6L. | **(1)** Read v1 A/B verdict. If non-negative → build **state-driven K** (per-target predictability horizon, K doc §8). **(2)** Build **contest-urgency** (timing/race half of conversion — sizing alone already failed). **(3)** Recover soft rolling pair via champion KT-OFF push(es). |

### Prior (sync coalition — closed/parked)
Synchronized two-source team-up (`BASELINE_JOINT_SYNC=1`, default OFF):
panel winner (v7_0 90.6% / v4_planner 93.8% / v3.5.1 87.5%) but live
**μ=1150.2** (sub 53223160) — below the 1183 champion, so a panel-beater not
a ladder gain. Size-to-hold lever = NULL (7/7 tie). Code stays default-OFF.

## Closed tracks — falsified knowledge, do NOT iterate

| Axis | Branch | Verdict date | Evidence |
|---|---|---|---|
| **VH on state-driven-K (additive AND rerank wirings)** | `claude/champion-ml-graft-majestic-storm` | **2026-06-03** | Rule 37 — **6 falsifications**: additive λ=1.0 raw / solo-only / λ=0.1 BIAS=102 / λ=0.1 BIAS=75.5 on the broken Jun-2 model (12.5%, 0%, 12.5%, 3.1%), additive λ=1.0 on the freshly retrained model (0/32 = 0%, ρ=+0.386), rerank top-K=10 on the freshly retrained model (2/32 = 6.2%). Reference un-VH state-K = 62.5%. Smoke vs random = 16/16 = 100% on the rerank wrapper, so wiring is sound; the failure is structural to the K=10 ship-delta head + chooser-score-augmentation pairing. Hypotheses (any sufficient): target mismatch (K=10 ship-delta vs chooser's PV-discounted 10-30 turn integral); pre-filtered training distribution; emit-loop coupling sensitive to top-K reorder; self-play single-seat distribution shift. See `audit/2026-06-03-vh-axis-closure.md`. Durable artefacts kept: corpus-gen pipeline (`scripts/gen_b2_corpus.py`, `scripts/probe_pveta_selfplay.py` ported from hqNVM 9d32066), solo-only trace hook (`agents/baseline/_trace_hook.py`), rerank infra (`BASELINE_VH_RERANK_K` env var, default OFF), VH bundler (`scripts/_build_champion_stateDrivenK_vh_bundle.sh`). **Does NOT close**: VH model itself (ρ=+0.386 is real rank signal), or learned-head approaches with different targets / consumption surfaces. |
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
| 2026-06-03 10:37 | champion-strategy-rules-00JzI (built via worktree at commit `9985e98` from `claude/champion-ml-graft-majestic-storm` session) | baseline_adaptive_k **RESUBMIT** (sub **53324164**, bundle sha256 `6c0419dc20`, 608844 B) — champion (launch_rules_universal) full config + **adaptive horizon K** baked ON (`BASELINE_ADAPTIVE_K=1`, K_OPEN=20 → floor 10 by step 30). Same agent as evicted sub 53265480 which **settled at live μ=1170.4** — calibration re-anchor of the rolling-pair floor. Built from worktree at 9985e98 using the size_balance script pattern (LIB set matches 9985e98 era; no kinematic_table / no opp_features_lite — those are post-9985e98 additions). | **~1170** based on prior live settle of identical agent (sub 53265480 μ=1170.4). High confidence (same code, same opponent pool dynamic; only stochastic ladder noise). | **53304016** baseline_launch_rules_universal **μ=1131.1** (older half of rolling pair; itself a 2026-06-02 21:09 resubmit of the all-time-champion bundle that did NOT reach its historic 1183.7 peak this time — settled 52 μ below). Backstop position-2: **53316984** baseline_state_k_orbital_lead μ=1109.0 (still settling, ~4h on ladder). Evicted-μ 1131.1 < predicted 1170 → **Rule 42 GREEN**: rolling-pair floor rises ~40 μ. | ✅ PI explicit "research submit this solution: baseline_adaptive_k …" (2026-06-03). Prior identical push was also PI-signed-off (2026-06-01 18:50 AskUserQuestion). Rule 43b technical miss (single-opp h2h Wilson-lo 0.483 < 0.50) but prior live μ=1170.4 is stronger evidence than any local panel. Rule 46 GREEN: bundle 608844 B / test_bundle.py **15/15** / fast.py play vs v7_0 seed=7 281 steps p0_win, max turn 639 ms (<<1000 ms cap). |
| 2026-06-02 14:55 | champion-ml-graft-majestic-storm | baseline_champion_tier2 — **first graft of hqNVM ML stack onto 00JzI champion base.** Distilled Tier-2 opponent model only (`lib.opp_model.trained_logreg_policy` v2 swap); value head dormant (`BASELINE_VH_LAMBDA=0`) for clean single-variable A/B. Champion config = `launch_rules_universal` (historical peak μ=1183.7) + `BASELINE_OPP_TIER=2` + `BASELINE_OPP_FILTER_THRESHOLD=0.15`. Block A + Block B (solo path) wired in `chooser_trajectory.py`; joint-path VH patch deferred (no-op at VH_LAMBDA=0, documented TODO). Bundle commit a1ee64d, sha256-prefix a27c7d9. | ~1130-1180 **HIGH UNCERTAINTY (calibration probe).** Local CRN h2h vs un-ported champion at n=32 = **21/32=65.6%**, Wilson [0.483, 0.796], 1770s elapsed → **Rule 43b MISS** (Wilson-lo 0.483 < 0.50 INCONCLUSIVE). Hypothesis under test: +14 μ Tier-2 lift from hqNVM-isolated A/Bs (where ML sat on `baseline_pv_eta` and tanked at μ=526-688) transfers when sitting on the joint-aggressive champion stack. TLE check: 1/~8000 turns at 1308ms (0.012% rate; ~0.06 dropped moves/Kaggle game, negligible). Cold-load 0.9ms opp + 21ms VH. Rule 46 GREEN: bundle / test_bundle 15-15 / fast.py play 275 steps WIN vs v7_0 max 878ms. | 53277693 (launch_rules_universal, **μ=1102.9** — older half of rolling pair, historic peak μ=1183.7). Backstop position-2: 53280733 (state_driven_k, **still settling — μ unknown at submit time**). Evicted-μ 1102.9 ≤ predicted-μ ~1130 → **Rule 42 BLOCKED-by-default; PI explicit override**. If μ < 1102 we lose the launch_rules anchor for ~24h. | ✅ PI explicit "Submit now with PI sign-off" (AskUserQuestion 2026-06-02 14:55), informed: Wilson-lo 0.483 short of Rule 43b 0.50 gate; backstop state_driven_k still settling (worst case both rolling slots uncalibrated); TLE concern shown to be 1-in-8000 outlier. |
| 2026-06-02 06:30 | champion-strategy-rules-00JzI | baseline_state_driven_k — "**latest advances with kinematics on**": current champion config + **state-driven horizon K** (per-target predictability horizon, ceil 30 / floor 10) + the **de-singletonized kinematic position-cache** (`world._kt`, fresh per seat per turn, bit-identical to inline, prevents dense-late-game timeouts). Strict upgrade over the table-OFF adaptive-K it evicts (better K lever **and** the cache). | **HIGH UNCERTAINTY (uncalibrated live).** Lever A/B (subprocess CRN, n=32) state-K+table vs table-only = **24/32=75.0%, Wilson[0.579,0.867], 0 errs** → clears Rule 45 + Rule 43b. Table win independently verified: live-peak provenance + dense-timing (table-OFF 1003ms timeout vs table-ON 916/926ms under cap) + position bit-parity 0/32 mismatches. Rule 46 GREEN (build / test_bundle 15-15 / cold-load 253-step game max 926ms). **Rule 43a multi-opp panel was IN-FLIGHT at submit time** (PI override). Warm-up: starts ~600, converges upward. | 53265480 (champ_adaptiveK_on, μ=1170.4 — older half of rolling pair, table-OFF). Backstop position-2: 53277693 (launch_rules_universal, warming up ~994→~1180). Evicted 1170.4 > uncalibrated candidate μ → **Rule 42 BLOCKED-by-default; PI explicit override**. | ✅ PI explicit "submit already" (2026-06-02), informed: evicts adaptive-K 1170.4 (current best live), candidate uncalibrated, Rule 43a panel in-flight; strict-upgrade rationale + TrueSkill warm-up dynamics confirmed by PI. |
| 2026-06-02 05:35 | champion-strategy-rules-00JzI | baseline_launch_rules_universal — the **live-peak μ=1183.7 table-ON champion**, resubmitted ("champion with kinematics on"). Full champion config WITH the kinematic_table position-cache primed (`table.begin_turn(world)` per turn); this bundle is the best-ever live artifact and nothing table-free has matched it since the 5/30 removal. PI-directed re-anchor of the rolling-pair floor. Track A 4P timing (this session): table-ON **faster** than table-OFF in 4P (p95 377 vs 405 ms), reversing the 2P penalty — though neither was time-bound in the roi games (0 turns >1000ms, max ~570ms). Rule 46c smoke GREEN (368-step game, max 880ms). | ~1183.7 (its own prior live settle; identical agent) | 53259633 (expand_credit, μ=1133.5 — older half of current rolling pair). Backstop position-2: 53265480 (champ_adaptiveK_on, μ=1165.9) survives. Evicted 1133.5 < predicted 1183.7 → **Rule 42 GREEN** (floor rises 1133.5→1165.9). | ✅ PI explicit "submit" (2026-06-02), table-ON champion re-anchor of best-ever live result. |
| 2026-06-01 18:50 | champion-strategy-rules-00JzI | baseline_adaptive_k — champion full config (KT removed, behaviorally neutral) + **adaptive horizon K** (`BASELINE_ADAPTIVE_K=1`, K_OPEN=20→floor 10 by step 30, step-decay v1): the launch-discipline ceiling K is large in the predictable opening (unlocks far/2nd-ring neutrals — median opening neutral ETA 22 vs static K=10) and decays to the champion's disciplined 10 by midgame. Single lever `capture_horizon_k(step)` read by gate+proposer+sync. | ~1180 **HIGH UNCERTAINTY** — local CRN h2h vs champion **21/32=65.6%**, Wilson [0.483, 0.796]. **Rule 43b MISS** (Wilson-lo 0.483<0.50) + **Rule 43a not run** (single-opp only) → PI-directed calibration probe. Beats every recent probe (which regressed); local h2h→live μ is noisy (sync 88-94% local→1150 live). | 53248277 (size_balance, μ=1139.5 — older half of rolling pair). Backstop position-2: 53259633 (expand_credit, μ=1077.3) survives. Evicted 1139.5 < predicted ~1180 → **Rule 42 GREEN**. | ✅ PI explicit "Submit adaptive-K now as probe" (AskUserQuestion 2026-06-01), informed: Rule 43 not cleared (Wilson-lo 0.483, single-opp) + bounded downside (backstop 1077) surfaced. |
| 2026-06-01 15:30 | champion-strategy-rules-00JzI | baseline_expand_credit — champion full config (kinematic_table removed = behaviorally neutral, 66/66 move-parity vs champion bundle) + size-balance (waste reduction, kept per PI) + **EXPANSION CREDIT** (`BASELINE_EXPAND_CREDIT=1.0`): restores the held-production capture term v4 dropped, so the value function credits territory and the agent spends idle ships instead of hoarding. Targets the dominant live loss mode (ship-hoarding/under-expansion — `audit/2026-06-01-live-replay-diagnosis.md`; step-39 istinetz trace: +916-prod neutral scored −0.99 → 0 launches on 115 idle ships). | **UNKNOWN — no completed winrate A/B** (probe killed twice by container restarts). Verified: flips hoarding→spending at the failure state (step 39: 0→94 ships launched); latency fine (+10ms, 0/32 turns >1000ms); Rule 46 green (cold-load 221-step game). **Rule 43/45 NOT cleared.** PI-directed calibration probe to read the expansion idea on the live ladder. | 53243763 (μ=689.8 — dead-weight older half of rolling pair). Backstop position-2: 53248277 (size_balance, μ=1141.5) stays. Evicted 689.8 ≪ predicted → **Rule 42 GREEN**. | ✅ PI explicit "add your suggestion, then submit" (2026-06-01), informed: no winrate A/B + bounded downside (evicts 690, 1141 backstops) surfaced. |
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
