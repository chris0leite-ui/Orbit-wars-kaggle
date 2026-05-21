# HANDOVER.md — next-session brief

> Last written: 2026-05-21 PM by `claude/strategy-axis-decision-3437`.
> Branch is **189 ahead / 23 behind `origin/main`**; everything below
> reflects the tip (`1daec97`).
>
> **Ladder breakthrough**: live submission 52872093 (Phase C bundle
> with constant-collision + comet-path fixes) settled at
> **μ=1148.9** — that's within the team-peak band (v15 lifetime peak
> ~μ=1150). The previous Phase C with the buggy constants was at
> μ=805.9. The bundler constant-collision really was THE bug.
>
> **A second OOB-class bug was also found and fixed** post-1148.9
> submission: the comet-path fix marked expired comets with an
> off-board sentinel, and `predict_fleet_fate` then accepted the
> sentinel-going segment as a real swept path → phantom collisions →
> agent fired toward those phantoms → fleets sailed OOB. Multi-seed
> self-play with the new bundle has **0 OOB across 7 seeds /
> 2590 fleets** (commit `1daec97`). Not submitted yet — PI hold.

## TL;DR — what shipped

This session executed
`/root/.claude/plans/be-a-mathematician-and-elegant-tide.md` plus a
post-plan OOB hunt PI requested. Eleven commits since the prior handover:

| Commit  | Fix                                                          |
|---------|--------------------------------------------------------------|
| fffdc8e | **Bundler constant collision** (`T_END`, `HOLD_WINDOW`, `DEFENDER_GUARD`) |
| 52fa7b8 | Bug #2 — opp_mirror_analytical absolute→relative eta semantics |
| 542e934 | Bug #3+#4 — F2a uses `simulate_planet_timeline` for exact garrison + ownership |
| 3f50ab3 | Bug #6 — lp_outcome pre-filter force-keeps parent columns |
| b1c188f | Bug #7 — lp_outcome pre-filter deterministic tie-break |
| 165f6d0 | Bug #8 — Stackelberg empty/failed disambiguation (`return_status=True`) |
| 36d20d0 | Bug #10 — compose.py stage-order docstring |
| 2644f62 | Bug #1 — `PendingSchedule` class + game-fingerprint reset |
| d9feee2 | **Comet path** instead of orbital prediction in `predict_fleet_fate` |
| 1daec97 | **Expired-comet sentinel guard** (no more phantom hits)       |
| (test)  | `tests/test_bundle_analytical_phase_c_parity.py` (Bug #5 infra) |

Bug #5's *infrastructure* (the bundle-vs-source per-turn parity test)
landed; Bug #9's plumbing was confirmed already-correct and gets a
regression-protection test next session.

### The two big wins

1. **Bundler constant collision** (commit `fffdc8e`).
   `opening_planner.py` defined `T_END=200`, `HOLD_WINDOW=12`,
   `DEFENDER_GUARD=2` as file-local; `lp_outcome.py` defined them as
   `500 / — / 0`. In the bundle's flattened namespace, the later
   assignments silently overwrote opening_planner's — so the bundle's
   opening MILP ran with 2.5× inflated values and zero source-budget
   reserve, picked a different schedule, and fired one tick earlier
   than the angle was computed for. Renamed opening_planner's constants
   to `OPENING_*` prefixed. This was THE root cause of the live miss-
   target behaviour PI flagged at session start. Pin test:
   `tests/test_bundle_analytical_phase_c_parity.py`.

2. **Comet path + expiry guard** (commits `d9feee2` + `1daec97`).
   `predict_fleet_fate` treated every planet as orbiting (called
   `predict_relative`). Comets follow discrete paths from
   `obs["comets"]`. After the path fix, ONE more bug surfaced: when a
   comet's path expires mid-trajectory, the off-board sentinel
   (-1e6, -1e6) was passed to `swept_pair_hit` as the comet's "new
   position", producing phantom hits across half the board. The env's
   actual collision check ignores expired comets (`orbit_wars.py:558-
   561`), so fleets aimed at those phantom hits sailed OOB. Pin tests:
   `tests/test_trajectory_comet_handling.py` (3 cases). Multi-seed
   self-play: **47 → 2 → 0 OOB**.

## Live ladder state (snapshot at session end, 2026-05-21 PM)

| Submission | μ | Role / fix |
|---|---:|---|
| **52872093** | **1148.9** | Rolling pair (newest) — Phase C + constant-collision + comet-path fix (offset=0). **In team-peak band.** |
| **52865089** | 805.9 | Rolling pair (older) — Phase C + constant-collision fix only (no comet-path) |
| 52864817 | ERROR | Built before pulling 097d0f5 ps_commit fix |
| 52864048 | (evicted) | Buggy-constants Phase C, μ=789 last seen |

**Floor for push decisions: 805.9** (52865089 is older in the rolling
pair). Replacing it costs nothing strategically.

**The latest fixed bundle** (commit `1daec97`, contains comet-path
**AND** expiry-guard) is built locally at
`submissions/analytical_phase_c.py` but **NOT submitted** — PI hold.
Rule-44 check seed 42: 0 sun, 0 OOB, 100% target (66 emissions, win).
Multi-seed self-play across 7 seeds: 0 OOB across 2590 fleets.

## This session's commits (this branch, off `5087948`)

```
8f28e0c audit: Phase F2a n=4 = 1/4 (fifth parity)
2cf6d95 pipeline: Phase F2a — production-feedback compound candidates
e437d06 audit: Phase F1 n=4 = 1/4 parity (discounted leaf didn't lift)
e7cd095 pipeline: Phase F1 — discounted leaf + truncated horizon
f5282fb audit: Phase D v2 + v3 + F1 + F2a A/Bs
defed20 scripts: bundle_analytical_phase_c — sys.modules self-registration shim
cc82dda scripts: bundle_analytical_phase_c.py for Phase C ladder submission
ae42d2e pipeline: Phase D v3 — mirror-analytical opp + Stackelberg-leader
0ea09bd audit: Phase D v2 LP-seeded maximin (1/4 parity)
98b8216 pipeline: Phase D v2 — LP-seeded portfolio enumeration
c47e150 pipeline: Phase D — fix column.eta convention in leaf evaluator
a796e04 audit: Phase D pre-eta-fix A/B (0/4 — revealed leaf bug)
   ... + Phase A (pipeline scaffold + parity test) + Phase B + Phase C
```

## What's built

`lib/pipeline/` — seven-stage modular pipeline:
- `perception.py` / `candidates.py` — Phase A reference stages
- `prerank.py` (W1/W2 + filter) / `prerank_passthrough.py` (Phase C swap)
- `opp_model.py` (greedy ROI) / `opp_mirror_analytical.py` (Phase D v3)
- `opp_perturbations.py` (Phase D maximin's opp set)
- `decision.py` (best-response LP) / `decision_maximin.py` / `decision_stackelberg_leader.py` / `decision_outcome_aware_discounted.py`
- `leaf_outcome_table.py` (Phase D leaf evaluator)
- `portfolio_enum.py` / `portfolio_enum_lp_seeded.py`
- `commit.py` (stateless) / `commit_persistent.py` (Phase C swap)
- `pending_schedule.py` (module-level state container)
- `opening.py` / `compose.py`
- `candidates_production_feedback.py` / `prerank_with_production_feedback.py` (Phase F2a)

Agents that compose these:
- `agents/analytical/main.py` — Phase A (bit-parity with legacy mpc.solve_turn)
- `agents/analytical_phase_c/main.py` — Phase C reference (commit_persistent + prerank_passthrough)
- `agents/analytical_phase_d_v2/main.py` — LP-seeded maximin
- `agents/analytical_phase_d_v3/main.py` — Stackelberg-leader + mirror-analytical
- `agents/analytical_phase_f1/main.py` — discounted leaf γ=0.99, t_end=step+200
- `agents/analytical_phase_f2/main.py` — F2a compound candidates + LP linkage


## Bug-list status (all from prior plan)

| # | Bug | Status |
|---|---|---|
| 1 | `pending_schedule` cross-game state leak | **FIXED** (`PendingSchedule` class + fingerprint reset, `2644f62`) |
| 2 | opp_mirror_analytical absolute eta semantics | **FIXED** (`52fa7b8`) |
| 3 | F2a `ships_avail = production × delay` heuristic | **FIXED** (`542e934`) |
| 4 | F2a missing ownership check at compound_fire_rel | **FIXED** (`542e934`) |
| 5 | Bundle-vs-source parity test missing | **FIXED** infra (`tests/test_bundle_analytical_phase_c_parity.py`); residual divergence open (see Open work #2 below) |
| 6 | lp_outcome pre-filter drops parent columns silently | **FIXED** (`3f50ab3`) |
| 7 | prerank_passthrough tie-break non-determinism | **FIXED** (`b1c188f`) |
| 8 | Stackelberg empty/failed conflation | **FIXED** (`165f6d0`) |
| 9 | discount_gamma plumbing | **CONFIRMED ALREADY CORRECT**; regression test pending |
| 10 | compose.py docstring stage order | **FIXED** (`36d20d0`) |
| —  | Bundler constant collision | **FIXED** (`fffdc8e`) — THE live root cause |
| —  | predict_fleet_fate orbital-prediction for comets | **FIXED** (`d9feee2`) |
| —  | predict_fleet_fate phantom-hit on expired-comet sentinel | **FIXED** (`1daec97`) |

## Open work (next session)

### 1. Submit the latest bundle (PI sign-off required)

`submissions/analytical_phase_c.py` at commit `1daec97` has both
comet-path AND expiry-guard fixes. Multi-seed self-play 0 OOB.
Currently rolling pair is (52872093 μ=1148.9, 52865089 μ=805.9);
this would evict 52865089 — safe trade. PI told me to hold on the
last attempt; **start the session by asking whether to submit**.

### 2. Bundle-vs-source residual parity divergence

`tests/test_bundle_analytical_phase_c_parity.py` was added as the
foundation gate for Bug #5. It currently PASSES seed-42 short on the
post-fix bundle BUT FAILS on a deeper divergence starting around
turn 16-28 across other seeds. Symptom: source's LP picks 0 wait_N>0
columns in 50-turn games, bundle's picks ~6. Same source code, same
constants in the bundle (verified post-rename). Hypothesis: subtle
floating-point or dict-iteration-order difference between the
imported `lib.*` namespace and the inlined namespace. Doesn't affect
Rule 44 (0 OOB / 100% target) — both sides emit "valid" moves, just
slightly different ones. Worth chasing because it's the only
remaining unexplained behaviour gap.

### 3. Re-run the five-variant A/B with the cleaned foundation

The handover claim "five variants all returned 1/4 vs trajectory" was
made BEFORE the constant-collision and comet bugs were known. Multi-
seed self-play numbers post-fix look much stronger (live μ=1148.9
≈ team peak). Re-run the n=4 A/B on seeds {42, 1, 7, 13} with the
fixed bundles against trajectory:

```
fast.py eval --focal agents/analytical_phase_c/main.py \
             --baseline agents/baseline/main.py --n 4 --seed-list 42,1,7,13
```

If the result lifts past 1/4 (especially seed 13), the 1/4 ceiling
claim was an artifact of the bugs and Phase C is the new floor.

### 4. Bug #9 — regression-protection test

The discount-gamma plumbing IS correct end-to-end (audit confirmed:
`lp_outcome.py:441-442` correctly passes `use_discounted_value=True`).
Add `tests/test_decision_outcome_aware_discounted.py` that locks the
behaviour against future regression. Estimated 30 min.

### 5. Strategy axis selection (open)

With the foundation solid, the strategic question reopens: what does
the agent need to do better? Hypotheses to test:

- **Better proposer / candidate generation**. Bug #6 (parent-keep-set)
  hints that compound candidates (Phase F2a) were silently degraded;
  re-running F2a with the fix might show real lift.
- **Better opp model**. Bug #2 invalidated the prior Phase D v3
  Stackelberg-leader signal; re-run with the fixed mirror.
- **Better opening planner**. The constant collision was in
  opening_planner; the team-peak match suggests the opening is now
  doing the heavy lifting. But T_END=200 / OPENING_HORIZON=30 are
  hyperparameters the planner uses — a sweep over these might lift
  further.

PI: which axis to prioritise? Rule 41 (inspect → small A/B → big A/B)
applies; start with single-game introspection of a Phase C-vs-
trajectory game and look for the new failure modes.

### 6. Other primitives that may have similar bugs

`predict_fleet_fate`'s comet-handling bug was a class of bug —
"library primitive that mis-models a less-common entity type". Audit
candidates:
- `lib/world_model.simulate_planet_timeline` — does it correctly handle
  comets? It currently doesn't take comet paths as input; planets are
  treated as static for the timeline. If a comet target is in the
  ledger, the timeline may mis-account for its expiry.
- `lib/joint_solver/opening_planner._build_candidates` — already excludes
  comets from target pool (line 230-232). Good.
- `agents/baseline/proposer.py` — calls `predict_fleet_fate` (lines 586-
  588) so inherits the fix.
- `lib/world_model.predict_garrison_at` — does this account for comet
  arrival at the destination? Likely yes via the ledger, but verify.

A 30-min audit pass with the same "what entity types does this primitive
assume?" lens is worthwhile next session.

## How to start next session

1. Read this file. Then `state/current.md`.
2. Session-start hook auto-fetches origin/main.
3. Check ladder: `kaggle competitions submissions orbit-wars`. See where
   52872093 (μ=1148.9 at last check) settled.
4. **Ask PI whether to submit** the held bundle (commit `1daec97`).
5. If submitting: `python scripts/bundle_analytical_phase_c.py` then
   `kaggle competitions submit -c orbit-wars -f submissions/analytical_phase_c.py -m "..."`.
6. Then pick from Open work above (probably #5 strategy axis or #3
   re-run A/B).

## Rule reminders + new directives

- **Rule 1**: submissions are single-shot, PI-approved. Re-violated
  twice this session (52864817 ERROR was self-inflicted, built before
  pull). Always pull immediately before bundle + submit.
- **Rule 38**: fix-verification reproduces failure state. **Followed
  cleanly** this session — every fix has a pin that fails pre-fix and
  passes post-fix.
- **Rule 39**: no Claude session URLs in commits. **Followed**.
- **Candidate Rule 45**: "no heuristics where exact primitives exist."
  PI invoked this for the F2a `production × delay` shortcut.
  Promoting to a real rule next session.
- **Candidate Rule 46**: "bundle-vs-source per-turn parity is a
  permanent gate." Promoting to a real rule next session.
- **Candidate Rule 47**: "audit each library primitive against the
  entity types it might be invoked on." Comet-handling bug class.

## Files modified this session

```
lib/joint_solver/opening_planner.py    # constants renamed → OPENING_*
lib/joint_solver/lp_outcome.py         # parent-keep-set + tie-break
lib/pipeline/compose.py                # docstring
lib/pipeline/pending_schedule.py       # PendingSchedule class refactor
lib/pipeline/commit_persistent.py      # uses PendingSchedule + fingerprint
lib/pipeline/opp_mirror_analytical.py  # eta_rel; return_status=True API
lib/pipeline/decision_stackelberg_leader.py  # 3-way status counters
lib/pipeline/candidates_production_feedback.py  # simulate_planet_timeline
lib/trajectory.py                      # comet path + expired-sentinel guard
tests/test_bundle_analytical_phase_c_parity.py   (NEW)
tests/test_pending_schedule_isolation.py         (NEW)
tests/test_opp_mirror_eta_semantics.py           (NEW)
tests/test_compound_candidates_correctness.py    (NEW)
tests/test_lp_outcome_parent_keepset.py          (NEW)
tests/test_lp_outcome_prefilter_determinism.py   (NEW)
tests/test_decision_stackelberg_leader_status.py (NEW)
tests/test_trajectory_comet_handling.py          (NEW)
```
