# HANDOVER.md — next-session brief

> Last written: 2026-05-30 by `claude/game-theory-winning-strategy-SEU7P`
> (additions-axis isolation + best-of-both-worlds plan queued).
> Older 2026-05-20 brief preserved below the "Plan queued" section.

---

## 2026-05-30 — PLAN QUEUED: "best of both worlds" composite

**Read this section first. Execute starting at Step 1 once Pre-condition P1 is confirmed.**

### Context at end of 2026-05-30 session

**Live rolling pair (per kaggle competitions submissions orbit-wars at session end):**
- Pos 1: **sub 53182323 `baseline_launch_rules_universal.py`** μ **≈1209.7** — NEW CHAMPION (sibling branch, NOT ours). Author description: "champion full config (JOINT_AGGR/NEUTRAL_BONUS/ORBITAL_SAFETY/PV_ETA, trajectory chooser) + UNIVERSAL K=10 ceiling. Extends launch discipline from opponent-captures-only to ALL launches — every fleet arriving after K=10 turns dropped post-emit, incl neutral captures, own-planet reinforcements, comet-sourced." Their A/B universal-validator ON vs OFF: n=64 = 44-20 = 68.8% Wilson-lo 0.566 (clears Rule 45).
- Pos 2: **sub 53177486 `baseline_redeploy_gangup.py`** μ **≈1004** (ours, this branch, settling around 970-1004 — net regression vs PV_ETA peak).

**PV_ETA peak reference (frozen anchor):** `submissions/baseline_pv_eta_anchor_1163.py` = bundle of sub 53111837, commit `0d71aa6` on branch `claude/kaggle-submission-review-gZsCu`, settled live μ **1163.5**. Local A/B vs this anchor is our calibration baseline.

**Our current bundle (this branch, commit c097471) contains:**
- PV_ETA mechanism (ported from gZsCu, n=48 parity claim from wrap commit 2623b49 = 52.1% — wide CI)
- New forward-redeploy generator (default ON, env `PROPOSER_REDEPLOY`)
- New gang-up-support generator (default ON, env `PROPOSER_GANG_UP_SUPPORT`)
- New hybrid_spatial value head (default ON, env `BASELINE_VALUE_HEAD=hybrid_spatial`)

### Isolation A/B results (n=16 each vs PV_ETA anchor, this session)

| Arm | Win rate | Wilson 95% | True-seed signal | Verdict |
|---|---|---|---|---|
| Full bundle (all 3 ON) | 19/32 = 59.4% | [0.42, 0.74] | 67% | fails Rule 45 |
| redeploy_only | 11/16 = 68.8% | [0.44, 0.86] | 4/5 = 80% | **strong positive** |
| **gangup_only** | **7/16 = 43.8%** | [0.23, 0.67] | 3/7 = 43% | **REGRESSION — drop** |
| hybrid_spatial_only | 11/16 = 68.8% | [0.44, 0.86] | 3/3 = 100% | strong positive (n small) |
| isolation_none (foundation parity) | **PENDING — A/B 4 running** | — | — | answers P1 |

A/B 4 isolation files: `submissions/isolation_{redeploy_only,gangup_only,hybrid_spatial_only,none}.py` (built this session via sed-edits on `setdefault` lines of `baseline_redeploy_gangup.py`). Results log: `/tmp/ab_logs/isolation_3way.log`. A/B 4 was running as background task `bkdjbpl2q` at session end; expected finish ~40 min after session pause. If the container reclaimed the work, re-run: `python scripts/clean_ab.py submissions/isolation_none.py submissions/baseline_pv_eta_anchor_1163.py --seeds 8 --workers 2`.

### Strategic read

- gang_up is the load-bearing regressor in our additions axis — drop it.
- redeploy + hybrid_spatial show clean local lift but live ladder says ~−200pp vs new champion regardless.
- The universal-launch-rules mechanism (sibling-branch lift) is a MUCH bigger lever (+~50pp above PV_ETA peak) than our additions axis (+10-20pp locally, unproven live).
- "Best of both worlds" hypothesis: PV_ETA + universal-rules (foundation lift) + redeploy + hybrid_spatial (composition lift) → predicted live μ 1180-1260. UNKNOWN composition behavior.

### Pre-conditions (verify BEFORE starting Step 1)

**P1. Foundation parity confirmed.** Read final result of A/B 4 (`isolation_none` vs PV_ETA anchor). If 50% ± Wilson [0.40, 0.60], proceed. If meaningfully <50% (e.g., <40% with Wilson-lo <0.30), foundation has drifted negative since wrap commit 2623b49 — the isolation arm rates are inflated and PV_ETA needs re-validation before adding new code.

**P2. Env-flag inventory of the live champion.** Sub 53182323's description says "JOINT_AGGR/NEUTRAL_BONUS/ORBITAL_SAFETY/PV_ETA". Confirm all four are ON in our SEU7P bundle (check `grep -E "setdefault.*JOINT_AGGR|NEUTRAL_BONUS|ORBITAL_SAFETY|BASELINE_PV_ETA" submissions/baseline_redeploy_gangup.py` and `agents/baseline/main.py`). If one is missing, reconcile via separate parity A/B BEFORE the composite work — don't conflate env-flag port with universal-rules port.

**P3. Universal-rules code is locatable.** Find the author's branch and commit by inspecting sub 53182323's description (or by looking at the live champion's bundle file if downloadable from Kaggle). May require reading sibling branch source via `git log --all` or GitHub MCP tools.

### Sequencing (execute in order; gates between steps)

**Step 1 — Audit the live champion's source** (~1-2 h)
- Locate the universal-rules code on the sibling branch.
- Read the post-emit validator: where it hooks in (chooser commit point? joint-LP output?), what exactly it drops, the K value, any co-changes bundled with it.
- Cross-check env-flag inventory vs our SEU7P bundle (P2).
- **GATE 1A:** if env flags differ, reconcile via separate parity A/B first.

**Step 2 — Surgical port of universal-rules** (~50 LOC, ~1 h)
- Add the post-emit validator at the same hook point on this branch.
- Env-gated: `BASELINE_LAUNCH_VALIDATOR=on` (or whatever the author's env name is — match it for cross-branch parity).
- Default OFF in source; flip default ON via `os.environ.setdefault` in `agents/baseline/main.py`.
- Unit test (Rule 38 fix-verification): feed synthetic launches with explicit etas, prove validator drops eta>K and keeps eta≤K.

**Step 3 — Validate universal-rules port at-parity vs live champion** (n=32 A/B, ~75 min)
- Build bundle with: PV_ETA + universal-rules ON, redeploy/hybrid_spatial OFF.
- A/B vs the live champion's bundle (downloaded from Kaggle's `submissions/53182323` artifact, OR rebuilt from sibling-branch source).
- **Target:** ≥45% (parity). If clean parity, port is correct.
- **GATE 3A:** if step 3 doesn't clear parity, STOP. The port has a bug or hidden env-flag drift. Do NOT add redeploy/hybrid_spatial on top of an unverified port.

**Step 4 — A/B the full composite** (n=32 A/B, ~75 min)
- Composite bundle: PV_ETA + universal-rules + redeploy + hybrid_spatial, gang_up OFF.
- A/B vs PV_ETA anchor: target ≥60% with Wilson-lo ≥ 0.50 (clears Rule 45).
- A/B vs live champion's bundle: target ≥50% (at-parity or lift).
- **GATE 4A:** if composite vs champion <50%, composite REGRESSES vs just-universal-rules. Submit step-3's bundle instead.

**Step 5 — Multi-opponent panel** (Rule 43, ~1 h)
- `python fast.py eval <composite> --vs-panel` → Wilson-lo ≥ 0.55 per opponent.
- `python fast.py eval <composite> --vs <current_rolling_champion>` n≥32, Wilson-lo ≥ 0.50.
- **GATE 5A:** if any opponent fails Wilson-lo 0.55, surface which one + decide.

**Step 6 — Submit** (one slot, PI sign-off)
- Bundle, Rule 46 trio (`scripts/bundle_agent.py`; `pytest tests/test_bundle.py`; `python fast.py play` one full game).
- Rule 42 rolling-pair re-check IMMEDIATELY before `kaggle competitions submit` (this morning's friction: 9h-stale check evicted a sibling's strong submission).
- Rule 39: NO Claude session URLs in the commit / submission description.

### Submission sequencing decision (rolling-pair-of-2 constraint)

Two candidate bundles emerge: (A) step-3's just-universal-rules port (predicted live ≈1210, matches champion), and (B) step-4's composite (predicted live 1180-1260, uncertain).

**Conservative order (recommended):** ship (A) first → settle 1-2h → if μ confirms ≈1210, ship (B). If (B) regresses below 1180, abort (B), keep (A) as our pos-2 backstop.

**Why conservative wins:** rolling-pair-of-2 makes a strong known submission (~1210) recoverable for ~24h, but a weak composite locks us out for 24h. Pos-2 with strong (A) is meaningfully better than pos-2 with unknown composite.

### Known unknowns (cannot answer until measured)

- **U1.** Universal-rules hook point — post-LP-solve or pre-LP-solve? Affects how redeploy composes with the validator. Resolve in Step 1.
- **U2.** Whether the "K" in universal-rules is tied to the rollout K. If same number, may have cascading effects. Resolve in Step 1.
- **U3.** Composition redundancy. hybrid_spatial biases against far-d_min ship-mass; universal-rules indirectly does the same by dropping far launches. They may overlap → hybrid_spatial adds 0pp on top. Step 4 is the test.
- **U4.** Why gang_up regressed locally. Deferred (not on critical path). Worth a +1 A/B arm later: universal-rules MAY indirectly fix gang_up by dropping out-of-range stacks. Don't gate the plan on this.
- **U5.** Sibling-branch adversarial state. The live ladder will shift during this work. Re-check rolling pair at every submit gate.

### Risk register

- **R1.** Foundation parity FAILS (A/B 4 lands <40%): plan changes. Investigate SEU7P PV_ETA port regression before any composite work.
- **R2.** Universal-rules port has subtle bug: gate 3A catches it. Without unit test in Step 2, would silently no-op.
- **R3.** Composition is sub-additive or anti-additive: gate 4A catches it. Fallback is step-3's clean port.
- **R4.** Submit timing: a sibling submit lands while we're A/B-ing, the eviction floor shifts. Rule 42 re-check immediately pre-submit catches this.
- **R5.** Bundler silent-fail mode (Rule 46 has 5 known): keep `pytest tests/test_bundle.py` + cold-load `fast.py play` in the gate.

### Pointers for next session

- Current bundle: `submissions/baseline_redeploy_gangup.py` (sub 53177486 source).
- Frozen PV_ETA anchor: `submissions/baseline_pv_eta_anchor_1163.py`.
- Isolation files (drop after the plan completes): `submissions/isolation_{redeploy_only,gangup_only,hybrid_spatial_only,none}.py`.
- A/B 4 results: `/tmp/ab_logs/isolation_3way.log` (likely lost on container reclaim — re-run if needed).
- Bundle template: `agents/baseline/main.py` setdefault lines (where to flip new `BASELINE_LAUNCH_VALIDATOR` default).
- Live champion's submission ID for download: **53182323** (search Kaggle for `baseline_launch_rules_universal.py`).
- Related sibling-branch commit reference: see sub 53182323's description for branch identification.

### Frictions logged this session

- **Rule 42 stale rolling-pair check.** Checked rolling pair at session start (23:43 UTC), submitted 9h later (08:23 UTC) without re-checking. Sibling branch landed sub 53175658 in between. Cost: ~11pp on the eviction floor (evicted PV_ETA at μ=1111 vs claimed-evict baseline_validated at μ=1100). Fix: re-run `kaggle competitions submissions orbit-wars | head -5` IMMEDIATELY before `kaggle competitions submit`. Friction class same as Rule 42's origin (2026-05-20).

- **Bash polling background job killed when Monitor timed out.** Used Monitor tool to chain A/B 4 after the 3-way chain; Monitor's hidden 30-min timeout killed its child process group including the clean_ab subprocess after 6 games. Restarted as direct Bash `run_in_background` (single-completion notification). Lesson: Monitor is for streaming events, not for long-running child processes. Bash `run_in_background` for "tell me when done."

---

## Read order (Rule 44 — mandatory)

1. **`state/MULTI_BRANCH.md`** — live Kaggle rolling pair, three-track
   registry (Analytical / Hybrid-Sim / Verify-first), closed tracks,
   push claim board.
2. **`state/TOOLS.md`** — A/B harnesses, single-game diagnostics,
   validation suite, consolidation-merge gate.
3. **`CLAUDE.md`** — rules 1-47 (rules 41-47 added 2026-05-20).
4. **This file** — session-start prompt below.
5. `audit/friction.md` if you're about to touch a fragile path.

## Where we are (2026-05-20 17:00 UTC)

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. **34 days remain.**
- **Rolling-last-2 (Kaggle auto-keeps these two):**
  - 52857903 (μ 806.5) — analytical_wait_N_traj_plus_endgame_play (2026-05-20 16:12)
  - 52854094 (μ 829.1) — analytical (2026-05-20 13:59)
- **Team peak (EVICTED):** μ 1149.2 (sub 52744856, composite_a2_hybrid, 2026-05-17).
- **Floor lost in 24 h:** ~320 μ. The five-step eviction chain that
  caused this is documented in `state/MULTI_BRANCH.md` and is the
  origin of new Rule 42 (pre-submit cross-branch coordination gate).
- **Daily submission budget:** 5/day. 5/20 used: 2. 3 slots remain.
- **Floor-at-risk flag:** **TRUE** — rolling pair is 320 μ below team peak.

## Day-N PM extract-physics-trajectory-Vjaz9 (2026-05-22)

**Session shape:** surgical, additive extraction of physics substrate
from the sibling Phase η branch (`claude/strategy-axis-decision-3437`).
No strategy/agent code copied; no experiments; no submissions.

**What landed (sole commit `72fe45a`):**

- `lib/kinematic_table.py` (NEW, 436 lines) — per-turn precompute of
  planet positions (static / orbital / comet). Bit-identical to
  `predict_relative` by construction. Singleton + fingerprint rebuild.
- `lib/orbit.py` (+37) — `predict_relative_cached(planet, ω, lead, *,
  table=None)` lookup wrapper; falls through on any miss.
- `lib/trajectory.py` (+47) — gated behind `KINEMATIC_TABLE_ENABLED=1`
  env var. When primed AND the table covers the needed window, one
  `table.window()` replaces the per-step inline build. Default OFF;
  existing call sites unchanged.
- `tests/test_kinematic_table_parity.py` (NEW, 621 lines) — `==`
  parity pins (no tolerance) for every cache type.

**Deliberately skipped:** `lib/joint_solver/trajectory_matrix.py`
(Phase η.1 opening matrix — couples to `agents.baseline.proposer`,
not pure physics) and the full-game parity test (imports specific
agents). Strategy / chooser / pipeline / missions / value heads from
the sibling branch all left where they are.

**Verification:** 39/39 unit tests green; 80/80 in the wider
geometry+orbital-safety+proposer+snipe sweep; end-to-end parity smoke
on a 2-planet world identical cold vs primed.

**Next-session first action:** build a fresh agent on top of this
substrate. Opt-in protocol + usage example in
`audit/2026-05-22-extract-physics-trajectory.md`. Default-OFF means
no existing agent regressed by this commit.

---

## Day-N PM review-skills-improvements-moKOR (2026-05-20 evening)

**Session shape:** n=8-capped A/B iteration loop attempting to beat sub
52827111 ("comet-aim + reactor-aware", μ=1122). PI directive: no
submission until a candidate shows significant lift at n=8 (gate
≥14/16 = Wilson-lo 0.524). Result: no candidate found. Pivot
direction surfaced at end of session.

### What landed (code + docs)

- **Setup (3 commits + bundler fix):** targeted `git checkout` of sub
  52827111's mechanism source from `claude/audit-workflow-performance-btjeK`
  onto this branch (`d642593`). Imported files:
  `agents/baseline/{proposer,chooser,value,main,chooser_trajectory,chooser_roi}.py`,
  `lib/{world_model,trajectory,aim,opp_model,value_heads}.py`,
  matching tests, and `scripts/bundle_agent.py` (btjeK upgrade with
  parity-gate cache).
  - Bundler indent-preservation fix (`9a45fea`): bundler was breaking
    function-local intra-package imports by hoisting alias rebinds to
    column 0 inside function bodies → IndentationError. Fixed at
    `scripts/bundle_agent.py:268-275`.
  - `.gitignore` for `audit/bundle-parity-cache.json` (`3f123c3`).
- **Pinned baseline:** `submissions/iter_baseline.py` = clean re-build
  of the deployed sub-52827111 bundle (parity-gate green).
- **Iter 1 audit:** `audit/2026-05-21-n8-iter1-reactor-ablation.md`
  (filename off by one day vs UTC; content correct). Documents the
  parallel-vs-serial discrepancy that invalidated the original Iter 1
  diagnostic.

### Load-bearing findings

1. **CPU-contention contaminates n=8 A/Bs.** Three parallel `fast.py
   eval` instances (24 worker processes) produced focal p95=1248ms (over
   the 1000ms env actTimeout). Variant 1b reported 12/16 (75%) under
   contention; same bundle re-tested serially gave **6/16 (37.5%)**.
   Variant 1a similarly fell from 11/16 to 7/16 serial. **Mandatory
   convention going forward: all n=8 A/Bs run serially, no parallel
   fast.py invocations.**

2. **No env-var ablation produces ≥14/16 lift over the deployed
   baseline.** Four serial n=8 runs (all clean wallclock):

   | Variant | Δ vs deployed | Wins | Wlo |
   |---|---|---:|---:|
   | A1 — comet-aim solo (reactor-aware OFF) | 7/16 (43.8%) | 0.231 |
   | A2 — Part B (reactor candidates) OFF | 7/16 (43.8%) | 0.231 |
   | A3 — BASELINE_COMET_AIM=off | 9/16 (56.2%) | 0.332 |
   | A4 | killed before completion (PI directive — see #3) |

   Three runs all landed at 7/16, A3 at 9/16. All INCONCLUSIVE; no
   candidate cleared the gate.

3. **PI verdict mid-loop ("your tests are meaningless, we need a big
   lift"):** env-var ablations tap out at ±5pp which is invisible at
   n=8 (Wilson CI ~±20pp). To produce a ≥14/16 lift over a near-optimal
   bundle requires a STRUCTURAL change, not a knob flip. Loop halted
   at A3 result.

4. **Structural-change candidates that are NOT yet new code on this
   branch:**
   - **`used_tgts` lock removal in `chooser_trajectory.py:898`.**
     Currently blocks multi-source-same-target SOLO emits even when
     JOINT is on; JOINT only fires for pre-paired candidates (capped
     JOINT_TOP_K_PER_TARGET=3, JOINT_MAX_PAIRS=20).
   - **JOINT expansion** — raise the per-target / global pair caps by
     5-10×; remove the lock-checks at `chooser_trajectory.py:885-888`.
   - **Composite value head + A2 restoration** (the μ=1149 team-peak
     architecture). `value.py` has `BASELINE_VALUE_HEAD=composite` opt-in;
     A2 4P-weakness logic also imported.
   - **New chooser** (MCTS / beam search over candidate set) — 1+
     day build.
   - **Increase N_VALIDATE / WALLCLOCK budget** — squeezes the existing
     chooser only marginally; unlikely to be a "big lift."

5. **Confirmed already-implemented (not new work):** `BASELINE_LEDGER=on`
   (wait-N inter-turn commitment memory, the original Iter 4 idea —
   already in chooser_trajectory.py lines 904-915, gated by env var
   defaulting to "off"). `BASELINE_JOINT=1` multi-source coalitions
   (already ON by default, just capped low).

### Verified gaps in the current chooser

- **`agents/baseline/proposer.py:926-928`**: wait_N>0 candidates bypass
  the trajectory filter (`predict_fleet_fate` returns wrong results
  because it doesn't pre-rotate src/tgt to launch time). This is real
  H44 surface: filter has zero coverage for the multi-wait grid.
  Iter 3 (planned, not yet implemented) would extend
  `predict_fleet_fate` with a `launch_step` arg.
- **`predict_fleet_fate` does NOT check enemy-fleet intercepts.** This
  is correct behavior — game rules confirm fleet-vs-fleet collision
  doesn't exist. Original Iter 3 framing ("add enemy fleet ray-cast")
  was based on a misread of the game spec.

### Falsified or weakened this session

- **"Part A (cost-parity filter) is the regressor."** Iter 1's
  parallel-run 12/16 was CPU-contention noise; clean serial gives
  parity-or-loss (6/16). Cannot blame Part A based on this data.
- **"Comet-aim is the key lift in sub 52827111."** A3 turned comet-aim
  OFF and got 9/16 (better than 7/16 from other ablations).
  Directional signal that comet-aim itself may be neutral-or-mildly-
  harmful, not the value-add of the push.
- **Floor-recovery via rebundle of `iter_baseline.py` (== sub 52827111).**
  PI rejected: "we can learn nothing from that." Path is OFF.

## Next-session first action (this session's pivot)

**Priority 1 — Pick one structural change from the list above and ship
it (~few hundred LOC, single axis).** Recommend `used_tgts` lock
removal + JOINT cap expansion in `chooser_trajectory.py` as the
cheapest structural-shape change: combat rule 1 (same-owner same-step
arrivals stack) is well-understood; the existing lock literally
forbids the most powerful combat pattern. Risk: Plan agent flagged
this as needing n=32 minimum (prior asymmetric chooser attempts
0/32). Run n=8 serial first; if directional, escalate to n=32.

**Priority 2 — if Priority 1 doesn't clear:** Composite value head +
A2 restoration (μ=1149 architecture). Code already imported; needs
the right env-var combo + bundle bake. Significant ladder evidence
(sub 52744856 live μ=1149).

**Priority 3 — out-of-session-scope:** Konbu17 shot-validator MLP
(~1 week build, but the only ML attack with empirical precedent
+19pp panel lift).

**Reading order for the next agent:** this section first, then
`audit/2026-05-21-n8-iter1-reactor-ablation.md`, then
`/root/.claude/plans/go-effervescent-mochi.md` for the full
iteration ladder context.

## What just landed (2026-05-20, this session)

This session was a **doc-only consolidation pass** across 8 active
branches. No code changed. New / edited docs:

| File | Change |
|---|---|
| `state/MULTI_BRANCH.md` | **NEW.** Single source of truth across branches. |
| `state/TOOLS.md` | **NEW.** Tools registry (A/B + diag + validation). |
| `CLAUDE.md` | Rules 41-47 appended. Pointers section adds MULTI_BRANCH + TOOLS. |
| `.claude/skills/kaggle-comp/SKILL.md` | Step 0 "load MULTI_BRANCH + TOOLS first" preamble. |
| `.claude/skills/kaggle-comp/day-loop.md` | Step 1 amendment for code-comp branch coordination. |
| `.claude/skills/kaggle-comp/improvements.md` | Rotated: 7 items promoted to rules; 2 superseded. |
| `.claude/skills/kaggle-comp/improvements-archive-2026-05-20.md` | **NEW.** Rotation archive. |
| `state/current.md` | Deprecated to pointer-only banner. |
| `state/mechanism-ledger.md` | Appended 2026-05-18 → 5-20 entries. |
| `HANDOVER.md` | Rewritten (this file). |

**Rules 41-47 summary (read CLAUDE.md for full text):**

- **41.** Confound-sweep before correlational conclusion (btjeK origin).
- **42.** Pre-submit cross-branch coordination gate (the ~320 μ loss origin).
- **43.** Multi-opponent panel mandatory pre-submit (supersedes `--vs-panel` pending item).
- **44.** State-of-truth read before subsystem edits (supersedes "read state docs" pending item).
- **45.** n ≥ 32 minimum for A/B lift claims.
- **46.** Bundle + parity smoke before any submission.
- **47.** Physics-primitive verification before agent design (PFhzM origin).

## Three parallel tracks — current state

| Track | Lead branch | Best result | Status | Next action |
|---|---|---|---|---|
| **A — Analytical chooser** | `strategy-framework-design-OyoYR-rebased` | μ 829.1 (sub 52854094) — both live pushes regressed | knowledge-base 5/20: "axis closed (10 slices, 0 lift)"; architectural bind: analytical needs multi-turn glue OR must replace rollout entirely | Decide: park, or pivot to analytical-leaf-inside-rollout |
| **B — Hybrid-sim production** | `audit-workflow-performance-btjeK` (production) + `analyze-game-strategy-EpMVP` (phases) | μ 1149.2 (EVICTED) | Live champion lineage. H44 finding 5/20: 65% fleet-destroyed-in-flight — new physics-driven mechanism candidate | (i) hold-feasibility solo validation (btjeK Phase B); (ii) H44 defensive mechanism design; (iii) EpMVP Phase 4/6 commissioning |
| **C — Verify-first + Goal-directed** | `ml-competition-strategy-PFhzM` (+ `precision-physics-engine-ymJkA` substrate) | Phase A Test 3 PASS; wrap-baseline 12/32 = 37.5% (only positive signal vs production) | greedy_expand (60 LOC) tied goal_planner (500 LOC); chooser axis confirmed neutral | Decide: is wrap-baseline-as-veto a viable design? Or is Track-C work substrate-only? |

## Next-session first actions (ranked by EV / cost)

### Priority 1 — code consolidation pass (start small, parity-tested)

Following the 6-step consolidation-merge gate in `state/TOOLS.md`:

1. **Substrate primitives** (no chooser changes piggy-backed):
   - Merge `lib/trajectory_layer.py` + `tests/test_trajectory_layer_positions.py` from PFhzM.
   - Merge `agents/precision/sim.py` + `agents/precision/intercept.py` + `agents/precision/tests/` from precision-physics branch.
   - Run consolidation gate; expected: GREEN.
2. **Bundler upgrade** from EpMVP — "inline agent submodules + explicit-name imports."
3. **Diagnostic scripts** (zero-risk leaf merges):
   - `scripts/h44_landing_capture_diagnostic.py` (btjeK)
   - `scripts/probe_emits_via_fate.py` (PFhzM)
   - `scripts/inspect_goal_planner_game.py` (PFhzM)
4. **Oracle tests** (test-only, zero-risk):
   - `tests/test_baseline_replay_regression.py` (EpMVP)
   - `tests/test_migration_solver.py` (EpMVP)

### Priority 2 — recovery submission planning

The rolling-last-2 is 320 μ below team peak. Three sub-IDs have evidence
of being strong:

- **52744856** (μ 1149.2, composite_a2_hybrid 2P + A2 4P)
- **52754310** (μ 1143.7, trajectory v4 + wait_N + wallclock)
- **52811320** (μ 1135.1, hold-feasibility solo)

**Open PI question:** which lineage to rebundle and push? The push will
itself need to clear Rule 42 (claim board) and Rule 43 (panel + h2h)
before submit. Do NOT push without explicit PI sign-off.

### Priority 3 — Track-B physics mechanism design

H44 finding: 65% of landing-capture failures are fleet-destroyed-in-flight.
This is a substrate-level signal that the trajectory chooser's
fleet-safety filter is not catching. Design a defensive mechanism
(NOT a restriction-tuning constant bump — Rule 40) that emerges from
the underlying physics.

## Pointers

- `state/MULTI_BRANCH.md` — cross-branch state-of-truth.
- `state/TOOLS.md` — tools registry + consolidation-merge gate.
- `state/mechanism-ledger.md` — every agent family tried.
- `state/hypothesis-board.md` — open ideas, killed list.
- `CLAUDE.md` — rules 1-47 + R-defaults.
- `audit/friction.md` — current friction summary.
- `.claude/skills/kaggle-comp/` — skill (now multi-branch-aware).
- `audit/2026-05-21-h44-phase1-CORRECTED.md` (btjeK) — physics-failure analysis.
- `audit/2026-05-20-postmortem-strategy-framework-design-OyoYR-rebased.md` — analytical axis closure.
- `audit/2026-05-19-postmortem-PFhzM-physics-gate-and-mvp-confirmation.md` — Track-C verdict.
- `audit/2026-05-21-n8-iter1-reactor-ablation.md` (this branch, filename off by one UTC day) — Iter 1 ablation results + the parallel/serial contention finding.
- `audit/2026-05-22-extract-physics-trajectory.md` (this session) — physics substrate extraction.
- `/root/.claude/plans/go-effervescent-mochi.md` — full iteration-loop plan including the structural-change pivot list.

## Rule reminders (most relevant this session)

- **Rule 1:** submissions PI-approved, single-shot, no retry loops.
- **Rule 12:** rolling-last-2; weak late submits unrecoverable for ~24h.
- **Rule 32:** session-start git fetch; verify rolling pair via Kaggle CLI.
- **Rule 35-36:** PI thoughts append-only; session-end second-brain update.
- **Rule 37:** 3-variant axis cap. v9-v15 chooser hit it; chain-bonus hit it; analytical-slice hit it (10/3+).
- **Rule 40:** prefer modeling-correctness over restriction-tuning.
- **Rules 41-47 (new today):** see CLAUDE.md.

## Open questions for PI

1. Track A (analytical) — park or pivot to analytical-leaf-inside-rollout?
2. Track C — wrap-baseline-as-veto, or substrate-only contribution?
3. Recovery submission — which lineage to rebundle for the next push?
4. Should the SessionStart hook implementation (improvements.md TOP
   PRIORITY) get priority over the code consolidation pass?
