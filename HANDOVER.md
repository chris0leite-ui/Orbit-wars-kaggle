# HANDOVER.md — next-session brief

> Last written: 2026-05-31 PM by `claude/champion-strategy-rules-00JzI`
> (sync-coalition panel CONFIRMED; Lever 1 size-to-hold NULL; sync-only
> agent SUBMITTED as calibration probe). Prior 2026-05-31 AM + 2026-05-30
> spatial-head sections demoted below.

## 🟢 ACTIVE — 2026-05-31 PM (champion-strategy-rules-00JzI) — sync agent SUBMITTED (probe); size-to-hold is NULL; next session = READ THE LADDER, then branch

**One-line state:** the synchronized two-source team-up ("sync coalition")
is confirmed strong vs the calibration panel (88–94%), so we **submitted it
to Kaggle as a calibration probe** (sub `53223160`, 2026-05-31 15:17).
The "size-to-hold" refinement (Lever 1) we built this session is a **NULL**
(7/7 tie) — shelved, code stays default-OFF. The single most important
next action is to **read sub 53223160's settled μ** — that answers the open
question "is sync actually a ladder gain over the champion, or only vs our
panel?"

### What we did this session (verified, with data)

- **Confirmation panel PASSED (Rule 43a).** Sync focal vs the three
  calibration opponents: **v7_0 90.6%, v4_planner 93.8%, v3.5.1 87.5%**,
  every Wilson-lo ≥ 0.72 (well above the 0.55 gate). No A>B>C>A loop —
  sync is a broad, decisive winner vs the panel.
- **Built Lever 1 "size-to-hold"** (commit `69755b1`, default-OFF env
  `BASELINE_JOINT_SYNC_HOLD`): size each coalition to survive the
  predicted counter-attack, not just to flip. Reused the existing
  counter-attack estimator in `proposer.py` (factored into a shared
  `hold_need` helper, byte-identical refactor — old filter tests
  unchanged), so it's a modeling fix not a constant bump (Rule 40).
  3 new tests green incl. the key one (a hold-sized stack survives the
  counter that recaptures a garrison+1 stack).
- **Capture-stickiness trace (`scripts/sync_hold_trace.py`, NEW):** the
  recapture leak is **opponent-specific**. Vs weak `v7_0`: 0% recaptured
  either way (no leak; hold-on merely declined a held-able capture →
  mild regression). Vs the **champion** (strong counter-attacker): hold-off
  recaptured **40%** (2/5), hold-on **0%** (2/2) — so the mechanism *does*
  fix the leak, but it's conservative (made 2 coalitions vs 6).
- **Lever 1 A/B (`scripts/_run_hold_ab.sh`, NEW; isolation = our agent vs
  champion, same 8 seeds, only hold off↔on differs because the champion
  has no sync code → immune to the env):** hold-off **7/16 = 43.8%**,
  hold-on **7/16 = 43.8%** — exact TIE. 4 games flipped but symmetrically
  (2 to wins, 2 to losses). **Verdict: NULL.** The stickiness gain and the
  added conservatism cancel. Not promoted to n=32 (a 7/7 tie is not a
  triage-positive). Size-to-hold axis = explored, neutral (Rule 37: one
  variant; a *less pessimistic* counter model is the only un-tried variant).

### What we SUBMITTED (sub 53223160 — calibration probe, PI-authorized)

- **Artifact:** `submissions/baseline_joint_sync_submit.py` (bundle sha256
  `dc82f6d4`, 600792 bytes). = fresh bundle of `agents/baseline` (src commit
  `3e547aa`) **+ a baked top-of-file config header**. Sync-only (hold OFF).
- **⚠️ THE BUNDLE-BAKING GOTCHA (load-bearing lesson):** Kaggle runs the
  agent with **NO environment variables**, so every `BASELINE_*` toggle
  falls back to its code DEFAULT — and sync, pv_eta, orbital_safety,
  launch_rules all default OFF. The local-A/B "focal" bundles
  (`baseline_joint_sync_focal.py` etc.) do **NOT** bake config → submitting
  one would have run a near-vanilla agent. The fix: prepend an
  `os.environ.setdefault(...)` header (the full tested env block) **above
  the first inlined module**, because modules read their constants at
  import. Verified with a clean-env smoke (scrub all `BASELINE_*`, import
  the bundle, assert the baked values took, run one full game). Always do
  this before any code-comp submit.
- **Rule 43b is FAILED** (champion h2h 44–56%, Wilson-lo 0.39 < 0.50) —
  this is explicitly a "submit to observe" calibration probe (PI override),
  same profile as the last several submits (incl. today's AM 53212044).
- **Rolling pair now:** `53223160` (sync, **settling**) + `53212044`
  (baseline_pv_eta_vh, μ=1139.6 backstop). Evicted `53197142`
  (composite_universal, 1086.9 — our weakest recent; Rule 42 GREEN).

### RESUME PLAN — fresh session, ranked

1. **READ THE LADDER FIRST.** `kaggle competitions submissions orbit-wars`
   — get sub **53223160**'s settled μ. This is the calibration data point
   the whole submit was for. Interpret:
   - **μ ≥ ~1140 (near/above the 1183 evicted champion):** sync IS a real
     ladder asset, not just a panel-beater → make sync the production base;
     decide whether to lock it (Rule 12 rolling-pair) and/or iterate.
   - **μ ≤ ~1100:** sync wins the panel but NOT the live field → the panel
     opponents are unrepresentative of our μ-band. Pivot (step 3).
   - **In between / still settling:** run the deferred cheap disambiguator
     (step 2).
2. **Deferred cheap check (~30 min) — is sync > champion vs the FIELD?**
   Run the *champion* (`baseline_launch_rules_universal`) against the SAME
   three panel opponents sync scored 88–94% on. If the champion also scores
   ~90%, sync is not a gain over what's already strong; if clearly lower,
   sync is the upgrade. (We never ran this — it's the missing control.)
3. **If pivoting to a NEW mechanism** (sync flat vs field): two scoped
   candidates, pick by trace evidence of the agent's actual weakness —
   (a) **fleet-survival defense** — H44 finding: **65% of failed captures
   are fleets destroyed in-flight** (`audit/2026-05-21-h44-phase1-CORRECTED.md`);
   (b) **2-hop redeploy** — values a redeploy via the capture it unlocks
   (`knowledge-base/concepts/redeploy-2hop-capture-design.md`). Both are
   default-OFF builds behind Rule-47 horizon + Rule-43/45 gates.
4. **Size-to-hold:** DONE/NULL. Do not re-litigate unless step 1 shows sync
   is the base AND you want the only un-tried variant (a probabilistic, less
   pessimistic counter model — only decline a capture when a counter is
   *likely*, not worst-case).

**Reproduce the submitted agent:** `python scripts/bundle_agent.py
agents/baseline --out-dir submissions --force --skip-parity-gate` (the
internal parity gate breaks in this container — `agents` namespace collides
with `kaggle_environments.lux_ai_s3`; verify via structural `test_bundle.py`
+ the clean-env smoke instead), then splice the baked header from the top of
`submissions/baseline_joint_sync_submit.py` (the `_sync_os.environ.setdefault`
block) after the `from __future__` line. Full tested env block is also at the
top of `scripts/_run_hold_ab.sh`.

---

> Prior writers (per-branch, now superseded): `kaggle-baseline-strategy-lO4mm`,
> `audit-workflow-performance-btjeK`, `strategy-framework-design-OyoYR-rebased`,
> `ml-competition-strategy-PFhzM`, `analyze-game-strategy-EpMVP`,
> `review-skills-improvements-moKOR`.

## 🗂️ PRIOR (demoted 2026-05-31) — 2026-05-30 PM — spatial head left UNMEASURED by PI choice; 2-hop redeploy redesign is the forward path

**Latest (2026-05-30 PM2):** PI elected NOT to run the spatial win-rate
A/B. The head is **unmeasured on win-rate** (cost gate cleared only) —
"spatial is dead" is an UNCONFIRMED premise, recorded in
`audit/2026-05-30-spatial-head-unmeasured.md`. Forward-redeploy was
re-examined: it **cannot be decoupled** from the spatial head (own→own
captures nothing → rollout-leaf Δ≤0 under plain `hybrid`; only the
spatial term ever selects it). The SEU7P port (deferred "idea 1") is
**superseded** by a redesign that values the redeploy through the
*capture it unlocks*: `knowledge-base/concepts/redeploy-2hop-capture-
design.md` (commit `727e1bf`). That is the next build candidate, behind
its own Rule-47 horizon gate + Rule-37/43/45 A/B gates.

**Full plan:** `/root/.claude/plans/warm-beaming-snowflake.md` (read it first).

**Live ladder (rolling last 2): read it directly — `kaggle competitions
submissions orbit-wars`.** Scores are never transcribed here (they go stale in
hours). The current champion / A/B opponent is `baseline_launch_rules_universal`
— champion full config + **universal K=10 ceiling** (every launch — neutral /
opp / own-reinforce / comet — arriving after turn 10 is dropped post-emit). The
backstop (older half of the pair) is the SEU7P `baseline_redeploy_gangup`; the
next submit evicts it — but **no submit is planned until the A/B below produces
evidence.** Confirm which two are live from the CLI before acting.

**Task:** clean A/B of the `hybrid_spatial` value head (idea 3 from the SEU7P review)
vs the universal-ceiling champion. **Evidence-gathering only, no submit.**

### What this session SETTLED (verified, with data)

- **Cost gate CLEARED — the "spatial head is too slow" reading was a false alarm.**
  The handover's "p95=952ms, 4 turns >1000ms → kill the idea" came from CPU
  contention (the documented `n8-iter1` parallel-contamination friction), NOT the
  head. Reproduced clean, single-job → the failure vanished (Rule 38 satisfied).
  - **Profiler** (`scripts/profile_spatial.py`, full 500-turn 2P game, head active):
    the spatial-specific code `_positional_ship_value` is **<0.3% of compute** — it
    doesn't even make the top-30. Per-turn time is dominated by `predict_relative`
    (126M calls), `fleet_target_planet`, and the sim `interpreter` — all shared by
    the champion. The shared ~880ms p95 tail is the *agent's*, not the head's.
  - **Back-to-back single-job bench, same opponent + same 6 seeds, head = only var:**

    | | spatial focal | champion |
    |---|---|---|
    | p95 | **821ms** | 884ms |
    | max | 928ms | 948ms |
    | turns ≥1000ms | **0** | **0** |

    The spatial head is **as fast or faster** than the champion on every pooled
    percentile. Both are `WATCH` only because p95 is in the 820–880 band (gate <800),
    but that band is *identical* for both → it's the agent, not the head.

- **Focal bundle ready:** `submissions/baseline_universal_spatial.py` — champion
  config header, NO `BASELINE_VALUE_HEAD` env line, head **baked in code** at module
  end (`def select_favor_fn(): return favor_hybrid_spatial`, line ~14751, last-def-
  wins). Why baked not env: the head is read from `os.environ` live per-turn and both
  agents share one process, so an env toggle leaks to BOTH → false parity.
- **Contamination probe PASSED** (earlier this session): focal→`favor_hybrid_spatial`,
  champion→`favor_hybrid` in one shared-env process.

### STILL NOT DONE — the win-rate A/B (the thing that actually decides this)

- **Zero clean win-rate data.** The only open question is "does spatial WIN more,"
  not "is it fast" (settled above). Earlier stalls were CPU oversubscription on this
  **4-core** box — any "15/32"-type number in old scrollback was a DRAFT, disregard.

⚠️ `submissions/*` is **git-ignored** → the focal bundle does NOT survive a fresh
clone. Rebuild it first if the container was recycled (recipe in the plan).

### Next steps — exact sequence (ONE heavy job at a time; never concurrent)

1. If container was recycled: rebuild `submissions/baseline_universal_spatial.py`
   (re-bundle `agents/baseline` with the champion bundle's `--lib` list; inject the
   champion env header WITHOUT a VALUE_HEAD line; append the `select_favor_fn` bake),
   then re-run the contamination probe before trusting any A/B.
2. **Headline 2P win-rate A/B, ALONE:** `python scripts/clean_ab.py
   submissions/baseline_universal_spatial.py
   submissions/baseline_launch_rules_universal.py --seeds 16 --workers 4`
   → 32 games. **Gate: Wilson-lo ≥ 0.50** (Rule 45). If triage-positive (μ≥0.55),
   extend to `--seeds 32` (64 games) for the ship-grade read.
3. Sequentially after it finishes (each its own single job): symmetry sanity
   (`clean_ab champion champion --seeds 8` ≈50%); geometry panel
   (`fast.py eval … --geometry-panel --by-archetype`) to catch archetype-local
   regressions; then (if win-rate promising) production-share via replay capture
   (Rule 48 — the doctrinally-correct metric for a positioning change).
4. **DECISION → surface to PI, do NOT submit.** A submit would need the Rule 43 panel
   + h2h, a Rule 42 push-claim, and would evict our own live champion.

Deferred → SUPERSEDED (2026-05-30 PM2): idea 1 forward-redeploy generator.
Do NOT port the SEU7P spatial-coupled version (it cannot stand without the
spatial head). Replacement design: `knowledge-base/concepts/redeploy-2hop-
capture-design.md` — values the redeploy via the capture it unlocks, so it
stands alone under the plain head.

### Housekeeping flag for next session
- Commit `aea9d74` (now amended locally to `fd495ce`, profiler tooling) originally
  shipped with a **fabricated** perf-fix message citing unmeasured numbers. Corrected
  locally; **remote still needs a force-push** to carry the honest history (PI sign-off
  pending). No agent code was changed by it — `value.py` was already clean (module-level
  `math`, single spatial-term compute).

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

> NOTE: this is a dated 2026-05-20 snapshot kept for narrative continuity;
> it is superseded by the ACTIVE section at the top. Live scores were removed —
> read the ladder directly (`kaggle competitions submissions orbit-wars`).

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC.
- **Rolling-last-2 (Kaggle auto-keeps the two most-recent):** read from the CLI.
- **Rule 42 origin:** on 2026-05-20 a five-step eviction chain from one branch
  dropped the rolling-pair floor by ~320 μ in 24 h (the founding story of Rule 42,
  the pre-submit cross-branch coordination gate). Kept as rule context.
- **Daily submission budget:** 5/day — check today's usage from the CLI.

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
