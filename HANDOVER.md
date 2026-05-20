# HANDOVER.md — next-session brief

> Last written: 2026-05-20 evening by `claude/strategy-axis-decision-3437`.
> **Production agent unchanged on the Kaggle ladder.** Default chooser
> remains `"trajectory"` (`agents/baseline/main.py:38`); the analytical
> work lives on this branch only.
>
> This session executed the plan at
> `/root/.claude/plans/spicy-marinating-token.md` — built the analytical-
> native modular pipeline (`lib/pipeline/`), ran five substrate variants
> (Phase C / D v2 / D v3 / F1 / F2a). **All five returned 1/4 vs
> trajectory on identical seeds.** Then PI flagged a sloppiness; three
> critical code-review subagents surfaced multiple correctness bugs that
> contaminate the trust in those results. **No fixes landed; this
> session ends in audit-and-document mode.**

## TL;DR

- Built `lib/pipeline/` modular framework (Phase A) + Phase B diagnostics
  + Phase C stage swaps + Phase D maximin / Stackelberg-leader + Phase F1
  discounted leaf + Phase F2a production-feedback compound candidates.
- All five substrate variants return 1/4 wins on seeds {42, 1, 7, 13}
  vs trajectory baseline; seed 13 wins, 42/1/7 lose.
- **The 1/4 ceiling claim is not currently trustworthy.** Critical-
  review subagents found multiple correctness bugs that could each
  independently produce the 1/4 result by coincidence.
- Live submission 52863860 (Phase C v2) PENDING; PI reports "ships still
  missing targets in our latest submission." Most plausible root cause:
  game-id hash collision in pending_schedule across parallel games.
- **No new submissions this session beyond 52863860.** Fix the bugs,
  re-validate, then decide on next push.

## Live ladder state (snapshot at session end)

| Submission | μ snapshot | Role |
|---|---:|---|
| **52863860** | PENDING | Rolling pair (newest) — Phase C v2 with sys.modules shim fix |
| **52863735** | ERROR | Phase C v1 — load-time dataclass crash (sys.modules unregistered) |
| **52857903** | 845.7 | Rolling pair (older) — analytical + F1/F2/F5 fixes |
| 52854094 | 836.8 | Evicted by 52863860 (analytical Phase 5 initial) |
| 52754310 | 1143.7 | Long-evicted trajectory champion |

**Floor for push decisions: 845.7.** Daily submits 2026-05-20: 2 used
(one ERROR, one PENDING). 5/day budget — 3 remaining today UTC.

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

## ⚠️  CRITICAL BUGS TO FIX NEXT SESSION

Three parallel review subagents produced detailed findings. Ranked by
**likely connection to live "ships missing targets"**:

### 1. **CRITICAL — `pending_schedule` game-id hash collision** (Review 3)
- **File**: `lib/pipeline/commit_persistent.py:35-44`
- `game_id` fallback derivation uses `hash(tuple(initial_planet_state)) % 2³¹`.
- In Kaggle's tournament harness, multiple parallel games with similar
  configs collide on this hash → pending_schedule state leaks across
  games → decanted moves at wrong steps from a different game manifest
  as "ships missing targets."
- Locally I run isolated A/Bs in separate processes → no collision →
  local tests pass while live diverges.
- **This is the most plausible single root cause of the live-miss
  behavior PI reported.**
- **Fix**: replace with UUID per env instance, require env's
  `episode_seed`, or use a counter+timestamp.

### 2. **CRITICAL — mirror-analytical opp eta semantics mismatch** (Review 2)
- **File**: `lib/pipeline/opp_mirror_analytical.py:67, 73-79`
- `_columns_to_arrivals` produces `eta_abs = step_now + wait_N + eta`
  (absolute). Passes to `merge_ledgers` and then
  `simulate_planet_timeline(planet, new_ledger.get(int(pid), []), ...)`.
- `simulate_planet_timeline` may expect relative-from-now etas, not
  absolute. **Verify the exact contract**. If relative-expected, opp's
  arrivals are scheduled at wrong (much later) times → Stackelberg-
  leader's "opp best response" is structurally wrong.
- **Fix**: verify contract of `simulate_planet_timeline.arrivals`. If
  relative, convert before passing. Add an assertion / test.

### 3. **CRITICAL — F2a `ships_avail = production × delay` heuristic** (Review 1, PI's catch)
- **File**: `lib/pipeline/candidates_production_feedback.py:96`
- Used a `production × delay` heuristic for captured-planet's post-
  capture garrison. `simulate_planet_timeline` provides the exact
  garrison at any future tick — should use that. Without exact ship
  count, compound candidates have unrealistic fleet sizes.
- **Fix**: simulate captured planet's timeline given (current ledger
  + base capture's arrival) → exact garrison at compound_fire_rel.

### 4. **CRITICAL — F2a missing ownership-transition check** (Review 1)
- **File**: `lib/pipeline/candidates_production_feedback.py:71-99`
- If opp re-captures the planet between base arrival and compound
  fire, my "fire from captured planet" is impossible. Code doesn't
  check `timeline['owner_at'][compound_fire_rel] == me`.
- **Fix**: check ownership at compound_fire_rel via the same
  `simulate_planet_timeline` call as #3.

### 5. **CRITICAL — bundle vs source never validated** (PI's broader catch)
- The 681 KB bundle (`submissions/analytical_phase_c.py`) is produced
  by `scripts/bundle_analytical_phase_c.py` with import-strip + a
  `sys.modules` self-registration shim. **No test verifies the bundle
  emits the same moves as the source agent.**
- 52863735 ERRORED on Kaggle (sys.modules unregistered → dataclass
  KW_ONLY crash). The fix shim now loads — but doesn't prove behavioral
  parity.
- **Fix**: build `tests/test_bundle_parity.py` that drives a real game
  with the bundle AND the source agent in lockstep, asserts identical
  emissions every turn.

### 6. **HIGH — F2a pre-filter drops parents silently** (Review 1)
- **File**: `lib/joint_solver/lp_outcome.py:154-157`
- If a planet has > 64 candidate columns, the pre-filter drops the
  lowest-value ones. If it drops a parent capture, the corresponding
  compound columns get force-zeroed by linkage. Silent action-space
  degradation.
- **Fix**: build a keep-set of column_ids referenced as parents; ensure
  they survive the pre-filter regardless of value rank.

### 7. **SIGNIFICANT — `prerank_passthrough` sets all values to 1.0**, but per-
planet pre-filter sorts by value → arbitrary tie-breaking on which
columns survive. Phase C / F2a behavior depends on dict iteration
order (Review 1 #6).
- **Fix**: implement secondary sort key (ships descending, or
  outcome-table-aware merit), or use `value_for_candidate` for pre-
  filter ordering only (not for amputation).

### 8. **SIGNIFICANT — Stackelberg-leader's empty == fallback ambiguity** (Review 2)
- **File**: `lib/pipeline/decision_stackelberg_leader.py:128-139`
- `predict_opp_response_to_my_portfolio` returns `[]` for both "opp
  has nothing to do" AND "LP fault." Status counter conflates them.
- **Fix**: distinguish these cases in the status string (e.g.
  `opp_mirror_empty` vs `opp_mirror_failed`).

### 9. **SIGNIFICANT — Phase F1 discount-gamma plumbing not verified
end-to-end** (Review 2)
- **File**: `lib/joint_solver/lp_outcome.py` (LP body)
- Phase F1's `decision_outcome_aware_discounted` passes
  `discount_gamma=0.99`. The LP builds `prod_stream_discounted`. But:
  does the LP's MILP cost vector actually use the discounted stream?
- **Fix**: trace through `_value_for_outcome(..., discounted=True)`
  call sites; verify cost vector uses the right field.

### 10. **MINOR — compose.py docstring stage ordering wrong** (Review 3)
- **File**: `lib/pipeline/compose.py:6-14`
- Docstring claims 1→2→3→4→5→6→7; actual code runs 1→2→4→3→5→7
  (opp before prerank). Code is correct (matches mpc.solve_turn).
  Docstring misleads.
- **Fix**: update docstring.

## What this means for the "1/4 ceiling" claim

Five substrate variants (Phase C, D v2, D v3, F1, F2a) all returned
1/4 vs trajectory. The conclusion was "decision rule / leaf / candidate-
space sophistication doesn't lift." **That conclusion is currently
unreliable.** Multiple bugs above could independently produce 1/4 by
mechanisms unrelated to the structural claim:

- If pending_schedule leaks state across games (Bug #1), local results
  are wrong in ways that don't match Kaggle's harness anyway.
- If mirror-analytical opp's etas are off (Bug #2), Stackelberg's
  results don't reflect what reactive opp would actually do.
- If F2a's ship counts are heuristic (Bug #3) and ownership-unsafe
  (Bug #4), compound candidates don't fire meaningfully → no signal.

**Rule 38 says the verification step must reproduce the failure
state.** None of the audits I did reproduced the live miss state.
Fix the bugs first, then re-run the five-variant comparison cleanly.

## How to start next session

1. **Read this file first.** Then `state/current.md`.
2. Session-start hook auto-fetches origin/main.
3. Refresh ladder: `kaggle competitions submissions orbit-wars` — note
   52863860's settled μ vs the rolling pair floor 845.7.
4. **Don't push anything to the ladder until the audit bugs are fixed
   AND a bundle-vs-source parity test passes** (Bug #5).
5. **Fix order (proposed):**
   1. Bug #5 (bundle-vs-source parity test) — turns every other fix
      into something the harness can verify.
   2. Bug #1 (game_id collision) — most likely root cause of live miss.
   3. Bug #2 (mirror-analytical eta) — closes the Stackelberg result.
   4. Bug #3, #4 (F2a ship-count + ownership) — restores F2a trust.
   5. Bug #6, #7, #8, #9, #10 — sweep.
6. Re-run the five-variant n=4 A/B with bugs closed. If still 1/4,
   the ceiling claim is real; if not, the prior results were noise.

## Submission strategy (open PI decision)

- Wait for 52863860 to settle and watch behavior on live games.
- If miss-pattern persists post-fix-#1, PI may need to inspect a
  specific replay; consider downloading via Kaggle web API (no CLI
  endpoint for replays I know of).
- Rolling pair floor 845.7. Restoring trajectory floor (~μ1120) is
  still PI-deferred from earlier sessions.

## Rule reminders + new directives from this session

- **Rule 1**: submissions are single-shot, PI-approved.
- **Rule 38**: fix-verification reproduces failure state. My local
  4-seed tests did NOT reproduce live misses — Rule 38 was violated
  in spirit when I claimed reliability.
- **Rule 45** (candidate): "no heuristics where exact primitives
  exist." PI invoked this when catching the `production × delay`
  shortcut. Promoting to a real rule next session.
- **Bundle-vs-source parity** must be a permanent test, not an
  ad-hoc smoke. Promote as a new operating rule.
