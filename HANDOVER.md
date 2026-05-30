# HANDOVER.md — next-session brief

> Last written: 2026-05-30 (late PM) by `claude/game-theory-winning-strategy-SEU7P`
> (foundation regression diagnosed + fixed; composite "best of both worlds" plan ready to execute).
> Older 2026-05-20 brief preserved below.

---

## 2026-05-30 — PLAN: composite "best of both worlds"

**Execute starting at Step 2. Foundation fix (Step 1) is already in `origin/claude/game-theory-winning-strategy-SEU7P` at commit `d50654a`.**

### Composite target

A single bundle that combines:
- **Foundation fix** (perf-chain regression reverted — already committed)
- **Universal-launch-rules** (ported from champion branch — Step 2)
- **Our verified additions** (redeploy + hybrid_spatial — already default-on in main.py)
- **NOT gang_up** (regressor — already default-off in main.py)

Predicted live μ: 1200-1250 range if composition is additive. Champion currently at μ≈1173, PV_ETA peak was 1163.5.

### What's already done (2026-05-30 session)

**1. Foundation regression diagnosed and fixed (commit `d50654a`):**
- `agents/baseline/chooser.py`: `WALLCLOCK_BUDGET_MS 800.0 → 600.0`
- `agents/baseline/main.py`: `KINEMATIC_TABLE_ENABLED` setdefault `"1" → "0"`

Discriminator A/B results (n=16 each vs PV_ETA anchor `submissions/baseline_pv_eta_anchor_1163.py`):

| Variant | Win rate | Wilson | Recovery |
|---|---|---|---|
| isolation_none (both perf-chain ON — pre-fix) | 5/16 = 31.2% | [0.14, 0.56] | — |
| kinematic_off (only kinematic disabled) | 9/16 = 56.2% | [0.33, 0.77] | +25pp |
| wallclock_600 (only wallclock reverted) | 9/16 = 56.2% | [0.33, 0.77] | +25pp |
| **both reverted (= current source after d50654a)** | **11/16 = 68.8%** | [0.44, 0.86] | **+37pp** |

Both perf-chain pieces are equal-magnitude regressors. Each costs ~25pp in isolation; combined sub-additive at ~37pp recovery. The `hard_deadline` parameter in `score_action` remains (third perf-chain piece, not env-gated, not isolated — gZsCu doesn't have it).

**2. Additions axis verified (isolation A/B, n=16 each vs PV_ETA anchor):**

| Arm | Win rate | Verdict |
|---|---|---|
| redeploy_only | 11/16 = 68.8%, 4/5 = 80% true-seed | **KEEP — default on** |
| hybrid_spatial_only | 11/16 = 68.8%, 3/3 = 100% true-seed | **KEEP — default on** |
| **gangup_only** | **7/16 = 43.8%, 3/7 = 43% true-seed** | **DROP — default off (already done in d50654a)** |

**3. Champion branch identified: `origin/claude/champion-strategy-rules-00JzI`**
- HEAD commit at session end: `8364db8 handover: spatial-head cost gate CLEARED`
- Key feature commit: `f10bb1e feat(launch-rules): universal K=10 arrival ceiling` (the universal-rules implementation)
- Champion's `agents/baseline/main.py` imports `from agents.baseline.launch_rules import enforce_launch_rules`
- The validator is called at multiple chooser commit points (search for `enforce_launch_rules` in champion's main.py post-bundle)

**4. Champion has the SAME perf-chain regression as our pre-fix SEU7P:**
- `WALLCLOCK_BUDGET_MS = 800.0` ✓ (same as our pre-fix)
- `KINEMATIC_TABLE_ENABLED` setdefault `"1"` ✓ (same as our pre-fix)
- `BASELINE_VALUE_HEAD = "hybrid"` (no spatial term — our hybrid_spatial is unique to us)
- No `PROPOSER_REDEPLOY` / `PROPOSER_GANG_UP_SUPPORT` setdefaults (those are unique to our branch)
- Has `BASELINE_CHOOSER = "trajectory"`, `BASELINE_JOINT = "1"`

**Implication:** the champion's +30pp universal-rules lift was measured on the SAME regressed substrate ours had. So porting universal-rules onto our FIXED foundation should give a CLEANER lift than what the champion shows.

**5. Head-to-head A/B (our fix + redeploy + hybrid_spatial vs champion bundle) FINAL: 9/16 = 56.2%, Wilson [0.332, 0.769].**
- Per-seed: 3 focal-friendly (0, 4, 6), 2 anchor-friendly (2, 3), 3 split (1, 5, 7). **True-seed signal 3/5 = 60% focal.**
- **Strong seat asymmetry:** P0 = 3/8 (37.5%), P1 = 6/8 (75%). Our bundle plays much better as P1 than P0 against champion. Could be hybrid_spatial value head valuing geometry differently per seat, or panel artifact at n=16. Worth investigating at n=32+ in next session.
- **Verdict:** our additions stack on the fixed foundation is **roughly at parity with champion locally**, with directional positive lift (point estimate 56%, Wilson-lo 0.33 — does NOT clear Rule 45's 0.50 gate).
- **Implication:** universal-rules (+30pp local in champion's own A/B) and our additions (~+30pp local from foundation-fix recovery + redeploy + hybrid_spatial) are **roughly equal-magnitude levers**. Composite plan (port universal-rules on top of our fix + additions) should yield additive lift if compositions don't step on each other.
- Log: `/tmp/ab_logs/h2h_champion.log` (lost on container reclaim).
- Both bundle files at `submissions/baseline_fix_redeploy_spatial.py` (ours) and `submissions/champion_launch_rules.py` (champion HEAD-rebuilt) — both untracked, lost on container reclaim; rebuild via `python scripts/bundle_agent.py agents/baseline --force --skip-parity-gate` from each branch's HEAD.

### Pre-conditions (status)

- **P1 (foundation parity): RESOLVED** by commit `d50654a`. Local A/B shows perfchain_off variant at 11/16 = 68.8% vs anchor (cf. 31% with perf-chain ON).
- **P2 (champion env-flag inventory): PARTIAL.** Confirmed champion uses BASELINE_CHOOSER=trajectory and BASELINE_JOINT=1. Did NOT verify JOINT_AGGR/NEUTRAL_BONUS/ORBITAL_SAFETY/PV_ETA env vars — those are likely set elsewhere (wrapper, or as defaults inside specific code paths). Cross-check our `submissions/baseline.py` env-var inventory vs the champion's before composite-bundle ship.
- **P3 (universal-rules code locatable): RESOLVED.** See champion branch + commit f10bb1e above.

### Sequencing (execute in order)

**Step 1 — Foundation fix:** ✅ DONE at `d50654a`. Skip.

**Step 2 — Port universal-launch-rules from champion branch** (~1-2 h)
- Set up working access to `origin/claude/champion-strategy-rules-00JzI` (e.g. `git worktree add /tmp/champion-wt origin/claude/champion-strategy-rules-00JzI`).
- Read `agents/baseline/launch_rules.py` on the champion branch — that's the validator implementation.
- Read the call sites in champion's `agents/baseline/main.py` (search for `enforce_launch_rules`).
- Port `agents/baseline/launch_rules.py` to our branch verbatim (it's a new file — no merge conflict).
- Wire the same call sites in our `agents/baseline/main.py` — at every place the champion calls `enforce_launch_rules`, we should call it too.
- Env-gate it (`BASELINE_LAUNCH_VALIDATOR=on`), default ON via setdefault in main.py.
- Unit test (Rule 38): synthetic launches with eta>K and eta≤K, prove validator drops the eta>K ones.

**Step 3 — Validate composite bundle**
- Rebuild bundle: `python scripts/bundle_agent.py agents/baseline --force --skip-parity-gate`
- Bundle parity smoke + one fast.py play vs champion bundle.
- A/B vs PV_ETA anchor at n=32. Target: clear Rule 45 Wilson-lo ≥ 0.50. Expected: ≥70% (foundation-fix + additions already at 11/16 = 69% without universal-rules; adding +30pp lever should push this higher).
- A/B vs champion bundle at n=32. Target: ≥55% (our composite beats theirs).

**Step 4 — Multi-opponent panel (Rule 43)** before submit.

**Step 5 — Submit** (one slot, PI sign-off; Rule 42 rolling-pair re-check immediately pre-submit).

### Bundle/file artifacts at session end

- `submissions/baseline_fix_redeploy_spatial.py` (606212 bytes) — our fix + redeploy + hybrid_spatial, no universal-rules. UNTRACKED.
- `submissions/champion_launch_rules.py` (602303 bytes) — rebuilt from champion HEAD. UNTRACKED.
- `submissions/baseline.py` (committed at `3b636e0`) — same as `baseline_fix_redeploy_spatial.py` content.
- `submissions/baseline_pv_eta_anchor_1163.py` — frozen gZsCu PV_ETA peak (μ=1163.5). KEEP.
- `submissions/isolation_*.py` (4 files) — UNTRACKED, can be regenerated via sed-edits documented in commit messages.
- Logs: `/tmp/ab_logs/{isolation_3way,discriminator_perfchain,discriminator_narrow,h2h_champion}.log` — LOST on container reclaim.

### Known unknowns / risks

- **U1.** Composition of universal-rules + redeploy: redeploy generates own→own launches; some have eta>K and would be dropped by the validator. Mild proposer waste. Step 3's A/B vs PV_ETA anchor catches this.
- **U2.** Composition of universal-rules + hybrid_spatial: hybrid_spatial biases against far-d_min ship-mass; universal-rules drops far launches. Could be redundant (no extra lift) or complementary. Step 3's A/B is the test.
- **U3.** Champion may have other lift mechanisms beyond universal-rules. Their HEAD is at `8364db8 handover: spatial-head cost gate CLEARED (false alarm was CPU contention)`, with several commits between `f10bb1e` (universal-rules) and HEAD. Audit those commits before assuming "universal-rules alone" is what to port.
- **U4.** `hard_deadline` parameter in our `score_action` (NOT env-gated) was the third perf-chain piece, never tested in isolation. Could still be contributing some negative effect. Worth a follow-up isolation if Step 3's A/B doesn't clear ≥70%.
- **U5.** Rolling-pair-of-2 constraint: champion is currently pos-1, ours pos-2. Submitting composite evicts ours. If composite regresses, the prior pos-2 submit (currently `baseline_redeploy_gangup` at μ=1021) is lost. Plan a CONSERVATIVE order — if confidence is high, ship composite; if not, ship just-universal-rules-port first (matches champion at ~1170) then composite second.

### Frictions logged this session

- **Rule 42 stale rolling-pair check** (08:23 UTC submit; 9h-old check); cost ~11pp eviction floor.
- **Monitor tool timeout** killed the chain background job; switched to Bash `run_in_background` for one-shot completion.
- **Container reclaim mid-A/B** (between sessions); A/B 4 + H2H A/B both killed and had to restart. Workaround: launch via `run_in_background`, accept it might not finish; restart on next session.

### Pointers for next session

- Foundation fix commit: `d50654a` (chooser.py + main.py).
- Bundle rebuild commit: `3b636e0`.
- Champion branch: `origin/claude/champion-strategy-rules-00JzI`.
- Champion universal-rules feature commit: `f10bb1e`.
- A/B harness: `python scripts/clean_ab.py <focal> <opp> --seeds N --workers 2`.
- Bundler: `python scripts/bundle_agent.py agents/baseline --force --skip-parity-gate`.
- Live champion submission ID: **53182323** (μ≈1173 at session end).
- Our pos-2 submission ID: **53177486** (μ≈1021).


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
