# HANDOVER.md — next-session brief

> Last written: 2026-05-28 PM by `claude/competition-objective-alignment-hqNVM`
> (Phase A learned-value-head distillation cycle PASSED;
> Phase B greenlit with roadmap below).
> Prior writers preserved: `extract-physics-trajectory-Vjaz9` (5/22),
> `review-skills-improvements-moKOR` (5/20), and the cross-branch
> consolidation pass under "What just landed (2026-05-20, this session)".
> Older per-branch writers (now superseded): `kaggle-baseline-strategy-lO4mm`,
> `audit-workflow-performance-btjeK`, `strategy-framework-design-OyoYR-rebased`,
> `ml-competition-strategy-PFhzM`, `analyze-game-strategy-EpMVP`.

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

## Day-N PM competition-objective-alignment-hqNVM (2026-05-28)

**Session shape:** Phase A of the learned-value-head cycle. Goal was a
binary diagnostic — does the chooser-with-learned-head wiring work at
all, or was the previous failure (v1) doomed by architecture?
**Method:** distill a known-strong scalar value function (`favor_hybrid`,
the head behind the EVICTED team peak at μ=1149) into the 21889-param
MLP from 40 hand-crafted features, then A/B vs `favor_hybrid` itself.

### What landed (5 commits this branch line)

| Commit | Change |
|---|---|
| `9008010` | MVP learned value head infra + GPU training kernel |
| `0157bf0` | A/B variant maker + end-to-end cycle script |
| `132fa2b` | embed first trained value-head weights + canonical bundle |
| `26138b8` | Phase A distillation infra (favor_hybrid label mode) |
| `fb74d22` | Phase A distillation cycle — wiring verified |

Phase A artifacts:
- `data/value_head_distill/training.npz` — distillation corpus.
- `data/value_head_distill/value_head_weights.npz` — trained weights.
- `data/value_head_distill/training_history.json` — `val_rmse 48.4`
  vs `y_std ≈ 1029` ⇒ ~99.8 % variance explained.
- `submissions/baseline.py` — re-bundled with distilled head embedded.

### Load-bearing findings

1. **WIRING IS SOUND.** `baseline_learned` (chooser + distilled head)
   vs `baseline_hybrid` (chooser + `favor_hybrid`), `n=32` (harness
   auto-bumped from 16), `BASELINE_WALLCLOCK_MS=100`:
   **14/32 = 43.8 % wins, Wilson 95 % CI [0.282, 0.607]** — near-parity,
   formally INCONCLUSIVE. v1 was 2/32 = 6.2 % vs plain `favor` on the
   same harness.
2. **40-dim feature set is mostly sufficient.** Distillation R² is
   high (~99.8 %); no major feature-insufficiency diagnostic fired.
   The ~6 pp gap to 50 % parity is consistent with normal distillation
   loss — RMSE 48 is small absolute but ~10× the typical close-call
   action-Δ, so a small fraction of close decisions flip.
3. **Latency budget holds under the chooser hot path.** p50 = 164 ms,
   p95 = 240 ms, max = 459 ms per turn (env actTimeout 1000 ms).
4. **v1's failure was target + data, not wiring.** Margin-on-
   lite_greedy-self-play (the v1 setup) produced 43 % val variance
   explained; favor_hybrid distillation produced 99.8 %. The signal
   is just better when the teacher is competent.

### Falsified or weakened this session

- **"v1 architecture is broken."** Falsified — same architecture
  with a competent teacher near-parities the teacher. The proposer +
  chooser + 40-feature MLP substrate is fine.
- **"Distillation will fall to 70-80 % R² because the features can't
  recover favor_hybrid."** Falsified — 99.8 % R².

### NOT yet known (open against Phase B)

- **Live ladder calibration.** A/B was vs the EVICTED μ=1149 agent,
  not vs the current rolling pair (μ=806 / μ=829). We have not
  measured whether `baseline_learned` would beat the current floor.
  Rule 43 (multi-opponent panel) + Rule 45 (n≥32 vs rolling champion)
  must clear before any submission.
- **Does the distilled head add anything new?** A learned head that
  faithfully mimics a hand-coded head is an inference-cost regression
  with no upside. Phase A is a substrate test, not a candidate. The
  upside has to come from Phase B's richer training signal.

## Roadmap — Phase B and beyond

The learned-value-head program. Sequenced so each phase is its own
diagnostic; a phase only ships if its predecessor cleared.

### Phase B — richer training signal (next session, greenlit)

The Phase A result frees us to invest in the parts that actually
matter for headroom. Five changes, expected order of impact:

1. **Advantage head with Common Random Numbers (CRN).**
   - Train `A(s, a) = margin_action − margin_idle` against the SAME
     opp-model RNG seed for both legs (so the opp noise cancels in
     the difference).
   - Expected: 50–95 % variance reduction on the Δ signal that the
     chooser actually uses. The chooser doesn't care about V(s)
     accuracy — it cares about argmax_a (V(s,a) − V(s,a')).
   - Direct fix for the "RMSE 48 ≫ action-Δ" failure mode that caps
     Phase A at parity.
2. **Multi-horizon target (KataGo-style auxiliary heads).**
   - Outputs: final-margin, K-turn margin (K = 10), win-probability.
   - Weighted loss; auxiliaries regularise. Won't change rank order
     much on its own but stabilises training.
3. **Strong heterogeneous opponent pool.**
   - Pool: `composite_a2_hybrid` (μ=1149), `trajectory_v4_wait_N`
     (μ=1143), `hold_feasibility_solo` (μ=1135), `favor_hybrid`,
     `favor`. ~200 μ spread.
   - Why: v1's single-opponent (`lite_greedy`) self-play gave the
     head no signal for what beats a competent opponent. The Phase A
     fix was a better TARGET (favor_hybrid scalar); the Phase B fix
     is better DATA (decisions that matter against strong opp).
4. **Geometry-archetype-stratified self-play generation.**
   - Use the 32-archetype taxonomy already defined in
     `data/seed_panel_128.json` (audit/2026-05-18-seed-panel.md) —
     32 archetypes × 4 seeds = 128 reference geometries.
   - Generate the training corpus stratified by archetype: M games
     per archetype × 32 archetypes, rather than M total games
     sampled from whatever distribution the default seed generator
     happens to land in.
   - Optionally inverse-frequency-weight the loss so rare archetypes
     (3-planet sparse, comet-heavy, tight-orbit clusters) get
     proportional gradient — otherwise the head fits the modal
     archetype and miscalibrates at the edges.
   - Compositional with step 3: each (opponent × archetype) cell
     gets ≥ ceil(M/32) games. With 5 opponents × 32 archetypes that
     is 160 cells; a 25 600-game corpus is 160 games/cell.
   - Why: the same logic that justified the 128-seed eval panel
     applies to training data. The chooser is asked to handle wildly
     different geometries; a head trained on a non-stratified
     distribution will be miscalibrated on archetype edges that
     happen to come up on the live ladder.
   - Diagnostic to add: per-archetype val loss breakdown. Catches
     "all loss is from one archetype" failure modes before they
     reach the A/B.
5. **Kaggle GPU training.**
   - Local 5-fold > 1 h on this corpus size ⇒ GPU per Rule 13.
   - Use existing kernel template (`machine_shape: GpuT4x2`, Rule 30).
   - Two-tier smoke before production push (Rule 2 GPU clause):
     (i) local CPU single-state with JIT compile + memory recorded,
     (ii) small-scale GPU ≤4 games × ≤50 turns inside 10 min.

### Plan refinements from 2026-05-28 PM lens-critique pass

Three-lens review (mathematician / senior-ML-engineer / sim-game)
against the 5-step roadmap above. The ladder shape is unchanged;
five concrete modifications:

1. **B-1 explicitly reuses Phase A's data distribution.** "CRN
   advantage only" means: same games / opp as Phase A
   (`favor_hybrid` self-play, single opp), only the LABEL changes
   to a CRN-paired advantage `A(s, a) = margin_action − margin_idle`
   with the same opp-RNG seed on both legs. Each Phase A game already
   yields many (s, idle, a) triples; cost is hours not days. Earns
   or kills the line cheaply before any strong-opp / archetype
   corpus is built. Train/val seed-disjointness mandatory to avoid
   val leakage.
2. **Pre-B-3 compute-budget gate (back-of-envelope BEFORE the loop).**
   Before generating the strong-opp + archetype corpus, compute
   `games × turns × per-turn-ms × N_opps × N_archetypes`. If projected
   wallclock exceeds the planned compute window (Rule 2 1h CPU cap →
   Kaggle GPU per Rule 13), revise corpus shape (subsample turns,
   cache rollouts to harvest many (s, a) labels per game) BEFORE
   writing the data-gen loop, not after.
3. **Player-count branching is a roadmap decision, not an
   afterthought.** Team peak μ=1149.2 (sub 52744856) was
   `composite_a2_hybrid` — a 2P/4P branched architecture. The one-head
   Phase B plan must explicitly pick: (a) train two heads (proven by
   team peak, doubles param count), or (b) one head with
   `player_count` as a 41st feature AND ensured 4P coverage in the
   corpus. Decide before B-3 data-gen.
4. **Latency engineering lands with B-1, not deferred.** Phase A's
   p50 = 164 ms already exceeds the 100 ms chooser wallclock budget.
   Per-candidate MLP calls don't scale; the right shape is a single
   batched MLP forward over the chooser's full candidate set per
   turn. Deliver alongside B-1.
5. **Falsification clause needs a chooser-ceiling escape.** Current
   doc: "if Phase B underperforms, blame data/target, not features."
   Add a third candidate: *or chooser is the ceiling*. Concrete probe
   at the end of any failing phase — swap `favor_hybrid` back in as
   the value function while holding proposer + chooser + bundle
   constant; if that doesn't beat the learned head either, the head
   isn't the bottleneck and the head-headroom line should be paused.

**Comparison-baseline decision (2026-05-28 PI):** keep chooser
unchanged (no PV_ETA layered) during the B-1 / B-2 diagnostic phases,
so Δ-attribution stays clean. PV_ETA adoption re-opens only at the
submission gate, after the head itself has been A/B-validated in
isolation against `favor_hybrid`.

### Live-ladder state correction (2026-05-28 PM)

The "Where we are" section above is from 2026-05-20 and is **stale**.
Refreshed snapshot for context (do not edit history — use this for
Phase B baseline-choice only):

| Sub ID | Date (UTC) | Agent | μ | Role |
|---|---|---|---:|---|
| 53111837 | 2026-05-28 09:42 | `baseline_pv_eta` (sibling) | **1154.8** | rolling pair (top) |
| 53099429 | 2026-05-28 00:13 | peak-restore (orbital safety) | 1114.5 | rolling pair (bottom) |

Floor-at-risk flag is **FALSE** (was TRUE in the 5/20 snapshot — the
five-day intervening work on sibling branches recovered the floor).
Phase B's submission bar is now **~μ=1155**, not the evicted ~μ=1149
referenced in the older Phase A debrief. The diagnostic A/B target
(`favor_hybrid`) is still the right comparison for B-1 / B-2; the
submission gate must be re-checked against the live rolling top.

**Phase B decision rule.** Each addition is gated by an A/B vs
favor_hybrid at `n ≥ 32` with `BASELINE_WALLCLOCK_MS=100`:
- B-1 (CRN advantage only): need Wilson-lo ≥ 0.50 (parity-or-better).
  If we don't beat parity here, decompose CRN failure before piling on.
- B-2 (+ multi-horizon): need ≥ B-1 with delta within noise (Wilson
  CIs overlap) OR clearer lift.
- B-3 (+ strong opp pool): need ≥ B-2 with Wilson-lo ≥ 0.50.
- B-4 (+ archetype-stratified gen): the candidate move. Need
  Wilson-lo ≥ 0.55 vs favor_hybrid AND Wilson-lo ≥ 0.50 vs the
  current rolling-pair champion (Rule 43 + Rule 45). Plus the
  per-archetype A/B from the seed panel
  (`--vs-panel --by-archetype`) showing no archetype regresses
  > 10 pp vs the B-3 baseline — catches "we lifted the average by
  tanking one archetype" failure modes.

### Pre-submit checklist when Phase B clears

Apply in this exact order to avoid Rule 42 / 43 / 46 violations:

1. `kaggle competitions submissions orbit-wars | head -5` — read
   rolling-last-2 state.
2. `python scripts/bundle_agent.py agents/baseline` — bundle.
3. `pytest tests/test_bundle.py` + `python fast.py play <bundle>` —
   parity + crash-free game (Rule 46).
4. `fast.py eval <bundle> --vs-panel` — Wilson-lo ≥ 0.55 per opponent.
5. `fast.py eval <bundle> --vs <rolling_champion>` at n ≥ 32 —
   Wilson-lo ≥ 0.50 (Rule 45).
6. Append claim row to `state/MULTI_BRANCH.md` push board (Rule 42);
   verify evicted-μ < predicted candidate μ.
7. PI sign-off (Rule 1).

### Phase C — only if Phase B clears (speculative)

- **Population-based self-play.** Train multiple heads against a
  shifting opponent league (each Phase B agent enters the pool).
  Risk: many-day compute investment for a marginal lift; not
  scheduled until Phase B is on the ladder.
- **Search over the chooser's candidate set.** Replace the scalar
  ranking with a 1-ply beam search using the advantage head's
  variance estimate to prune. Touches the chooser, not just the
  head — higher integration risk.

### Falsified-or-dead so this isn't re-explored

- **Margin-on-lite_greedy-self-play as the value-head target.** v1
  result: 2/32 = 6 %. Target was too noisy AND the opp was too weak.
  Do NOT revisit this combination.
- **40-feature insufficiency.** Falsified by Phase A's 99.8 % R²
  distillation result. If Phase B underperforms, blame the data /
  target, not the feature pipeline. Expanding feature count is NOT
  the move.

---

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
- `audit/2026-05-22-extract-physics-trajectory.md` (Vjaz9) — physics substrate extraction.
- `audit/2026-05-28-postmortem-competition-objective-alignment-hqNVM.md` — Phase A wrap.
- `knowledge-base/thoughts/2026-05-28-value-head-phase-a-distillation-passes.md` — Phase A debrief + Phase B framing.
- `/root/.claude/plans/go-effervescent-mochi.md` — full iteration-loop plan including the structural-change pivot list.
- `/root/.claude/plans/let-s-do-it-um-cozy-peach.md` — original value-head Phase A/B plan (this branch).

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
