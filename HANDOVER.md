# HANDOVER.md — next-session brief

> **2026-05-29 ADDENDUM (claude/kaggle-submission-review-gZsCu):**
> opp-model rate-cap (top-K-by-ROI) FALSIFIED. K=2 vs live champion
> `baseline_leaf_pv_2p` lost 6/16 (Wilson [0.185, 0.614]). Below
> Rule 45 gate; no submission made. **Next-session direction: do
> NOT iterate K. Pivot to MLP-as-opp-model** — hot-swap the trained
> 3-MLP ensemble from sub 53131296 (sibling branch
> `claude/competition-objective-alignment-hqNVM`, MLP commit `4a8e4c0`)
> into `lib/opp_model.py` as a new tier. Full executable plan with
> file paths, env vars, gates, and pitfalls is in
> `audit/2026-05-29-k-cap-falsified-mlp-as-opp-model-plan.md` —
> **read that file as the next-session brief**, it is self-contained
> for compaction. K-cap code (`BASELINE_OPP_MAX_LAUNCHES`) stays
> shipped but default-off; do not delete in case a future ensemble
> wants per-tick budget as a sub-mechanism. PI thought captured at
> `knowledge-base/thoughts/2026-05-29-mlp-can-double-as-opp-model.md`.
>
> **PM3 ADDENDUM (2026-05-28 PM3):** macro layer (Item 1 below) was
> built, A/B'd, and SHOWED NO LIFT. See
> `audit/2026-05-28-postmortem-pm3-macro-layer-null-result.md`.
> The strategic-direction section below (Item 1 attack-axis bonus,
> Items 2-4 follow-ons) is **partially stale**: Item 1 is functionally
> closed (macro is opt-in at default OFF; the chooser already produces
> equivalent launches via PV_ETA + LEAF_PV_2P scoring). PI's PM3
> directive: **pivot to the opponent model.** Specifically Item 3
> below (`lite_greedy_policy` spatial restriction) — diagnostic
> instrumentation first, see
> `knowledge-base/questions/2026-05-28-opp-model-spatial-restriction-prior.md`
> for the cheap 1-2h calibration step (predicted vs actual opp ships)
> that should run BEFORE the full Item 3 build. Macro code stays
> shipped but dormant (`BASELINE_MACRO=0` default).
>
> Last written: 2026-05-28 PM2 by `claude/kaggle-submission-review-gZsCu`.
> Supersedes the PM1 (PV_ETA ship) handover written ~5h earlier today.
> Prior session content remains in
> `audit/2026-05-28-postmortem-pv-eta-pm-pv-eta-and-silent-turns.md` (PM1),
> `audit/2026-05-28-postmortem-pm2-leaf-pv-2p-compute-variance.md` (PM2),
> and `audit/2026-05-28-postmortem-pm3-macro-layer-null-result.md` (PM3).

## Read order (Rule 44 — mandatory)

1. **`state/MULTI_BRANCH.md`** — live Kaggle rolling pair, push-claim
   board. **Refresh from `kaggle competitions submissions orbit-wars` at
   session start.**
2. **`CLAUDE.md`** — rules 1-48. NEW today: Rule 48 (same-day readings
   are climb snapshots, not verdicts) and Rule 45b (confound check
   before sub-gate-strength submit).
3. **`comp-context.md::SCORES DO NOT SETTLE`** block — read before
   interpreting any sub μ taken < 4h after submit. PI corrected this
   recurring misread today; the loud block is the durable fix.
4. **This file** — next-session first action below (strategic direction).
5. `knowledge-base/flags/2026-05-28-compute-variation-ab-noise.md` —
   parked compute-variance fix; do not propose shipping until ladder
   stabilizes on leaf_pv_2p.

## Where we are (2026-05-28 PM2 UTC)

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. **~26 days remain.**
- **Live rolling pair (snapshot 14:40 UTC — NOT verdicts, see Rule 48):**
  - **53117942** (LEAF_PV_2P=1, μ=921.3 ~45min after submit, **CLIMBING**).
  - **53111837** (PV_ETA=1, μ=1163.5 — **NEW LIVE PEAK**, above historical
    1144-1165 band, stabler read).
- **Daily submission budget:** 5/day. 2026-05-28 UTC used: 2 (PV_ETA +
  leaf_pv_2p). UTC midnight resets to 5.
- **Open question:** where does sub 53117942 stabilize? See
  `knowledge-base/questions/2026-05-28-leaf-pv-2p-climb-trajectory.md`
  for the decision tree.

## Today's session — what landed (PM1 + PM2)

**PM1:** PV_ETA shipped (sub 53111837 → live peak μ=1163.5). The
`γ^(wait_N+eta)` discount supersedes the SHIP_TURN_KAPPA band-aid as the
modeling-correct present-value pull-back on candidate Δ. 5 unit tests +
55-test sibling suite green.

**PM2:** LEAF_PV_2P shipped (sub 53117942, climbing). Re-enables the 2P
leaf production-PV term disabled since 2026-05-18 to address PM1's
silent-turns thesis. Local n=10 vs anchor 7-3, 5-0 mechanism check vs
v4_planner with seed=2 flip. Compute-variance investigation: same-seed
step counts drift across A/B runs; cause is wallclock-coupling in
`affordable_validate_cap()` (`n_aff` is CPU-speed-dependent).
Confirmation A/B with pinned cores converged 4/4 outcomes. **Fix
identified but parked** — pinning `n_aff=60` changed outcomes to draws
in the diagnostic, signaling playstyle change; calibration needed.

## Falsified-or-killed this session

- **"PYTHONHASHSEED randomization is the A/B non-determinism source"** —
  rejected. Hash-fixed runs still drift in step count (218 vs 305).
- **"Compute variance alone explains 50μ peak-resubmit drift"** —
  rejected. Opp pool churn + σ shrinkage + scores-still-climbing also
  contribute. Multi-causal; cannot attribute purely to compute.

## Next-session first action — STRATEGIC DIRECTION (PI-set)

**Diagnosis (PI, end of PM2):** the agent has three observable symptoms —
fleet sizes too small, rear planets don't mobilize forward, no early
expansion. Root cause: agent is a per-move local optimizer with no
macro layer. Every leaf is conservative against opponents-from-anywhere,
so sum of locally-cautious moves = globally-defensive crouch. The fix
is to make the agent **commit to a direction**, expand early, and
mobilize the rear toward the front.

### Strategic direction: macro layer on top of the chooser

The mechanisms compose. Ordered by implementation cost (cheapest first):

**Item 1 — Attack-axis bonus (FIRST CUT, recommended next-session start)**
- New scoring term in `score_candidate_v4` / `_v4_joint`:
  `delta += ATTACK_AXIS_WEIGHT * ships * cos(capture_vector, attack_axis)`
- `attack_axis = opp_centroid − own_centroid` (recomputed each turn).
- Effects: rear-to-front launches earn a bonus → mobilizes the rear.
  Captures *away* from opp deprioritized → no wasted ship-turns behind us.
  Agent naturally commits to a flank because the bonus reinforces direction.
- Build: 1 env var (`BASELINE_ATTACK_AXIS=1`, default OFF for byte
  parity), 5 unit tests (centroid math; cos alignment; env-gating;
  byte-identical OFF; joint variant). ~½ session.
- A/B gate: Rule 45 n≥32 with Rule 45b confound check (PYTHONHASHSEED=0
  + OMP_NUM_THREADS=1 + taskset for the seed re-runs).
- Acceptance: visible directional behavior in seed traces — agent
  should *visibly* commit south-east when opp centroid is south-east.

**Item 2 — Early-expansion mandate (composes with item 1)**
- For first ~30 turns, increase production-PV weight in the 2P leaf
  (or add a separate neutral-capture bonus).
- Pairs with H40 (geometry-conditional opening book) as the lighter
  first version — no archetype classifier needed yet.
- Build: ~½ session.

**Item 3 — Opp-model spatial restriction (deeper mechanism fix)**
- `lite_greedy_policy` currently assumes opp counter-attacks can come
  from any opp planet. Restrict to planets where
  `eta(opp_planet → our_target) < safe_horizon`.
- Directly addresses PI's "expects opponents from everywhere" diagnosis
  and PM1's "lite_greedy too aggressive" finding.
- Build: 1 session.

**Item 4 — Fleet-size: commit-to-hold, not commit-to-capture**
- Replace "min-ships-to-capture under best-case opp response" with
  "min-ships-to-hold for K turns against most-likely opp recapture."
- `hold-feasibility` machinery exists; under-weighted in current chooser.
- Addresses "fleet sizes too small" symptom directly. Build: 1 session.

**H40** (map-archetype opening book) sits inside this frame as the
informed version of item 2 — once we know the attack axis matters,
choosing it from board geometry (not just turn-30 centroid) gives a
stronger seed. Defer until items 1-2 ship.

### Immediate first 5 minutes of next session

1. `kaggle competitions submissions orbit-wars --csv | head -3` — get
   lifetime μ for sub 53117942 (LEAF_PV_2P). Update
   `state/MULTI_BRANCH.md` push-claim board OUTCOME field.
2. Decision per `knowledge-base/questions/2026-05-28-leaf-pv-2p-climb-trajectory.md`:
   - μ ≥ 1100 → leaf_pv_2p stays in rolling pair; proceed to item 1 build.
   - μ 950-1100 → revert at next slot (peak-restore resubmit), still
     proceed to item 1 build on a clean baseline.
   - μ < 950 → revert immediately; item 1 build on peak baseline.
3. Then start **item 1 (attack-axis bonus)** build.

## Pointers (new today)

- `audit/2026-05-29-k-cap-falsified-mlp-as-opp-model-plan.md` —
  **2026-05-29 wrap + next-session plan, self-contained**.
  K-cap A/B results + MLP-as-opp-model implementation plan with
  file paths, commit SHAs, env vars, and gates.
- `knowledge-base/thoughts/2026-05-29-mlp-can-double-as-opp-model.md` —
  PI insight that re-routed the session.
- `submissions/baseline_opp_launches_k{0,1,2,3}.py` — built and
  tested today; K=0 control, K=1/2/3 rate-limited. K=2 failed vs
  live (6/16). Bundles retained for reference.
- `tests/test_opp_model_max_launches.py` — 5 unit tests for the K-cap.
- `audit/2026-05-28-postmortem-pm2-leaf-pv-2p-compute-variance.md` —
  PM2 postmortem (compute-variance investigation, Rule 48 + 45b
  promotions).
- `submissions/baseline_leaf_pv_2p.py` — sub 53117942's bundle
  (LEAF_PV_2P=1 layered on PV_ETA=1).
- `tests/test_leaf_pv_2p.py` — 4 unit tests pinning leaf PV semantics.
- `knowledge-base/thoughts/2026-05-28-pm2-compute-variation-and-leaf-pv-2p.md`
- `knowledge-base/flags/2026-05-28-compute-variation-ab-noise.md`
- `knowledge-base/questions/2026-05-28-leaf-pv-2p-climb-trajectory.md`

## Rule reminders most relevant for next session

- **Rule 48 (NEW):** same-day reads are climb snapshots, not verdicts.
- **Rule 45b (NEW):** confound check (PYTHONHASHSEED=0 + OMP=1 + taskset)
  before any sub-gate-strength override submit.
- **Rule 40 (modeling-correctness):** items 1-4 are all modeling-side,
  not constant-tune band-aids. Reject any bump-a-constant alternative.
- **Rule 47 (physics-primitive verification):** item 4 (commit-to-hold)
  needs eta — must go through `lib.trajectory.predict_fleet_fate`.
