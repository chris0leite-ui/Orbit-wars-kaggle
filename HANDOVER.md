# HANDOVER.md — next-session brief

> Last written: 2026-05-28 PM by `claude/kaggle-submission-review-gZsCu`.
> Prior content archived in
> `audit/archive-2026-05-28-handover-pre-pv-eta.md` (318 lines of
> Day-N PM sections from 2026-05-17 → 2026-05-22).

## Read order (Rule 44 — mandatory)

1. **`state/MULTI_BRANCH.md`** — live Kaggle rolling pair, three-track
   registry, closed tracks, push claim board. **Refresh from
   `kaggle competitions submissions orbit-wars` at session start.**
2. **`state/PEAK_BASELINE.md`** — peak bundle truth, active vs dormant
   env-var stack, build-on-top protocol. Mandatory before any baseline
   subsystem edit.
3. **`CLAUDE.md`** — rules 1-47.
4. **This file** — next-session first action below.
5. **`knowledge-base/thoughts/2026-05-28-silent-turns-pre-existing-weakness.md`**
   — the working theory and probe plan PI directed: "begin in the first
   session with further investigation and hard thinking."
6. `audit/2026-05-28-postmortem-pv-eta-pm-pv-eta-and-silent-turns.md`
   — full session postmortem with promotion candidates pending PI input.

## Where we are (2026-05-28 PM UTC)

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. **~26 days remain.**
- **Rolling-last-2 (live, post sub 53111837):**
  - **53111837** (PV_ETA=1, **μ pending** — predicted 1100-1170, mode 1140).
  - **53099429** (peak-restore byte-identical, μ=1116.2). Note this is
    ~30μ below the peak's historical 1144-1165 band — ladder drift in
    the last 24h re-calibrates expectations downward.
- **Daily submission budget:** 5/day. 2026-05-28 UTC used: 1 (this).
  4 slots remain. The 5/27→5/28 batch (revert, peak-restore, κ=0.02,
  PV_ETA) consumed considerable PI capital.
- **Open question:** how does sub 53111837 settle? Update calibration
  ladder when ladder-μ stabilizes (~30 min - few h after submit).

## Today's session — what landed

**Chapter A — ship the `favor` leaf flight-time fix (PV_ETA).**

- Single env-var-gated change `BASELINE_PV_ETA=1`: multiply candidate
  Δ by `γ^(wait_N + eta)` in `score_candidate_v4` + `score_candidate_v4_joint`.
  No new tuning knob (γ is `BASELINE_GAMMA=0.99`). Default OFF preserves
  byte-identical peak behavior.
- Commits: `c45cf00` (feature + 5 unit tests), `a65e8b4`+`e65b50a`
  (new `scripts/ab_quick.py` parallel no-swap step-250 A/B harness),
  `0d71aa6` (bundle + wrapper), `564b70e` (push-claim row).
- Sub **53111837** submitted, **μ pending**.
- Audit: `audit/2026-05-28-postmortem-pv-eta-pm-pv-eta-and-silent-turns.md`.

**Chapter B — diagnose seed=2 panel losses.**

- 5-seed step-250 panel: 4-1 uniformly across 4 opponents (`peak_anchor`,
  `v7_0`, `v4_planner`, `v3.5.1`); pooled 16-4/20 = 80%, Wilson-lo 0.58.
- All 4 losses were seed=2. Investigated. **Verdict: PV_ETA is NOT the
  cause** — peak baseline (PV_ETA=0) ALSO loses seed=2 vs v4_planner.
  Pre-existing weakness.
- Smoking-gun: P0 chooser emits zero launches for 13-29 consecutive
  mid-game turns; opponent emits every-turn-or-near. WIN-case seed=1
  has mid-game streaks ≤ 8. Knowledge-base entry has the table.

## Falsified-or-killed this session

- **"PV_ETA causes seed=2 losses"** — falsified. Peak loses seed=2 vs
  v4_planner at step 138; PV_ETA loses at step 145. PV_ETA modestly
  reduces the worst silent streak (14 → 13).
- **"`bundle_agent.py agents/baseline --force` parity-gate is fine"** —
  falsified for 2nd consecutive day; promote to tooling-fix priority.

## Next-session first action (PI directive: "further investigation and hard thinking")

### Priority 1 — silent-turns root-cause investigation (sequential)

1. **Instrument `score_candidate_v4` on a silent turn.** Replay seed=2
   vs v4_planner. Intercept the chooser at t=22 (a known silent turn
   in the PV_ETA=1 trace). Dump every candidate's `(src, tgt, ships,
   wait_N, eta, raw_delta, post_bonus_delta, post_pv_eta_delta, status)`.
   This is the diagnostic that **answers the question "are all
   candidates truly negative, or are some positive-but-below-MIN_DELTA?"**
   Expected wallclock: 1-2 hours including writing the script. Output
   should be a CSV in `audit/2026-05-29-silent-turn-trace.csv`.

2. **Conditional on 1.** If candidates are positive-but-tiny: ablate
   `BASELINE_MIN_DELTA=-5.0` on seed=2; if P0 emits more and panel
   rate holds, we have a quick lift. Wallclock: 30 min A/B run.

3. **Conditional on 1.** If candidates are uniformly negative: build
   the opp-model mixture probe. Swap `lite_greedy_policy` in
   `opp_actions_for_snap` for a 4-way mixture of {greedy, do-nothing,
   sniper, defender} weighted by ladder priors. Hypothesis: less-
   confident rollout opp-model lets our captures look positive-EV at
   the leaf. Wallclock: half-day build (mixture sampler) + half-day A/B.

4. **Independent of 1-3.** Track down the cross-process determinism
   leak. `git grep -nE "time\.time|random\.(random|seed|sample)|id\("
   agents/baseline lib/`; audit each call site for seed-dependency.
   If we find unseeded RNG in the hot path, fix it. The leak is
   poisoning every n=5 A/B currently.

### Priority 2 — tooling fixes

- **Fix `bundle_agent.py` namespace collision** (2-day recurrence;
  promotion candidate from postmortem). Either rename the parity-gate
  subprocess's `agents.*` namespace, or bypass the parity-gate and
  trust `tests/test_bundle.py`. ~1 hour.
- **Document `scripts/ab_quick.py` in `state/TOOLS.md`** as the new
  PI-directed A/B route. 15 min.

### Priority 3 — watch sub 53111837 settle

- After ~30 min, pull settled μ via `kaggle competitions submissions
  orbit-wars | head -3` and update `state/MULTI_BRANCH.md` push-claim
  board OUTCOME field + `state/calibration-ladder.md`.
- If settles ≥1130: PV_ETA validated; build silent-turns fix on top.
- If 1080-1130: PV_ETA neutral; silent-turns fix is the next axis
  regardless.
- If <1080: PV_ETA regressed despite the panel win — investigate
  what the step-250 truncation masked. Likely candidate: PV_ETA
  over-discounts long-eta captures that mattered at full game length.

## Pointers (new today)

- `audit/2026-05-28-postmortem-pv-eta-pm-pv-eta-and-silent-turns.md` —
  session postmortem.
- `audit/archive-2026-05-28-handover-pre-pv-eta.md` — archived prior
  HANDOVER content (Day-N PM sections from 5/17-5/22).
- `knowledge-base/thoughts/2026-05-28-silent-turns-pre-existing-weakness.md`
  — silent-turns diagnostic + probe plan.
- `scripts/ab_quick.py` — new A/B route (parallel, 5-seed, step-250,
  no swap).
- `submissions/baseline_pv_eta.py` — sub 53111837's bundle (PV_ETA=1).
- `tests/test_chooser_pv_eta.py` — 5 unit tests pinning PV_ETA semantics.

## Open questions for PI (Rule 36)

1. **Ratify the 3 promotion candidates in the postmortem?** Specifically:
   (a) Rule 48 cross-run reproducibility check before trusting n≤16
   A/Bs; (b) bundler namespace fix as top-priority next-session work;
   (c) `scripts/ab_quick.py` documented in `state/TOOLS.md` as standard.
2. **Step-250 truncation:** standard for ALL future A/Bs going forward,
   or specific to today's PV_ETA evidence run?
3. **Next-session priority order:** the silent-turns Priority 1 chain
   above, OR pivot if sub 53111837 settles in a way that re-prioritizes?

## Rule reminders most relevant this session

- **Rule 26 (devil's-advocate):** fired correctly pre-submit.
- **Rule 38 (fix-verification reproduces failure):** new PV_ETA unit
  test does exactly this.
- **Rule 40 (modeling-correctness over restriction-tuning):** PV_ETA is
  modeling-correct; silent-turns fix should also be modeling-side, not
  a MIN_DELTA constant-tune band-aid.
- **Rule 42 (push-claim board):** filled out; evicted-μ 1109.9 vs
  predicted lower-band 1100 was MARGINAL → PI signoff covered.
- **Rule 43 (multi-opp panel mandatory):** 4-opp panel ran; 4/4 cleared
  pooled Wilson-lo gate.
- **Rule 45 (n≥32 minimum for lift claim):** PI overrode to n=5 step-250
  with explicit signoff. Promotion candidate addresses the underlying
  evidence-bandwidth gap.
