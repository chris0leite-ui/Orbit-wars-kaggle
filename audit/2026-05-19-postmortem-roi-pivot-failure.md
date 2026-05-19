# Postmortem — ROI pivot didn't ship (2026-05-19)

## TL;DR

Spent one full session implementing the ROI-prior + opp-modifier
architecture pivot. Phases 1-5 (closed-form ROI), Tier 1 (defensive
coalition), Tier 2 (fast_sim rollout posterior on top-K) all landed.
Oracle suite reached 13/14 passing. **G3 panel A/B failed
catastrophically — 0/32 vs every panel opponent including v7_0 (an
older μ≈1056 agent that the trajectory chooser handily beats).**

Three iteration rounds (modeling fixes, then reinforce scoring, then
rollout posterior) all came back 0/N. The architecture has a real
ceiling: closed-form ROI cannot match the dynamic balance that the
trajectory chooser achieves via its fast_sim rollout.

Production agent unchanged; no ladder risk. Code parked on the dev
branch for future revival.

## What we set out to do

PI direction at session start: invert the current architecture. ROI
becomes the foundation (closed-form scoring), opp model is a thin
posterior modifier. Synthetic oracles become hard requirements the
strategy satisfies by construction. Goal: replace 1270 LOC of
rollout-with-embedded-opp machinery with ~400 LOC of clean closed-form
ROI logic that better captures PI's intent (don't waste ships on
unholdable captures, attack jointly, don't send fleets past prediction
horizon).

Plan was structured (`/root/.claude/plans/okay-we-can-do-elegant-lampson.md`):
- 8 phases, ~7-9 commits, +450 net LOC.
- 4 verification gates (oracles, bench, A/B panel, 4P sub-panel) before
  any ladder push.
- Convert all 4 currently-xfail oracles to PASS as a hard gate.

## What we built

12 commits on `claude/audit-workflow-performance-btjeK`:

| Phase | Capability | Outcome |
|---|---|---|
| 1 | Env-var dispatch + stub | ✓ green |
| 2 | `solo_roi` + 2 new oracles | ✓ green |
| 3 | N-way coalition (no rollout) + 1 oracle | ✓ green |
| 4 | Source vulnerability + endgame bonus | ✓ green |
| 5 | Ship-count enumeration + xfail conversions | 3 of 4 xfails converted |
| Tier 1A | Defensive coalition (B reinforces A) | converts 4th xfail (14/14) |
| Tier 1B | Wallclock budget guard | ✓ insurance |
| Tier 2 | Fast_sim rollout posterior on top-K | 14/14 → 13/14 (broke solo_capture) |

All 12 commits passed unit tests, oracle suite, and bench wallclock.

## What broke

Two layers of failure:

### Layer 1: bench misread

`fast.py bench`'s "PASS" verdict is wallclock only (`p95<800ms AND zero
>=1000ms`). Game outcomes (`outcome=p1_win` etc.) are reported but don't
gate the verdict. I read "outcome=p1_win" across 4 seeds as "focal won"
when it actually meant "p1 won" — and the focal is p0. **Bench had
shown 0/4 from the start**; I reported it as 4/4. The actual A/B (G3)
exposed this.

Root cause: I didn't read `fast.py:706` carefully enough to confirm seat
orientation before drawing conclusions. Easy mistake to repeat — added
to the "verified findings" section in HANDOVER for next session.

### Layer 2: closed-form ROI has a structural ceiling

Three rounds of fixes all came back 0/N:

1. **Phase 1-5 + Tier 1**: ROI emits 1.4% of turns (vs ~50%+ for v7_0).
   All candidates score negative because vuln_loss (2× margin over full
   remaining game) dwarfs gross (1× margin over hold horizon).

2. **Downsize fix + vuln 2× → 1× + reinforce scoring + neutral mult=2 +
   transient vuln**: emit rate up to 6-8%. Game lengths vary (131-500).
   Still 0/8.

3. **Tier 2 rollout posterior**: validate top-K candidates by running
   8-tick fast_sim rollouts. Replace closed-form score with measured
   `delta_us_minus_them`. Still 0/4.

The pattern that emerges from across the three rounds:

- **Closed-form gross is too small** (capped at hold horizon ~10-20).
- **Closed-form vuln is too large** (PV over remaining 400+ ticks).
- **No tuning sweet spot.** Halve vuln → still loses. Cap loss window
  → loses differently. Disable vuln entirely → suicidal greedy plays.
- **Tier 2's surrogate opp doesn't generalise.** `lite_greedy_policy`
  plays differently from v7_0 (which has its own fast_sim + drop-one
  chooser), v4_planner (receding-horizon mission portfolio), v3.5.1
  (aggressive snipe). The rollout measures one opp model's reaction,
  the game plays against another.

The trajectory chooser at μ≈1120 is doing real work that closed-form
ROI can't replicate. The K=25 fast_sim rollout, even with `lite_greedy`
as opp, lets the chooser see opp's counter-attack land within the
horizon and react. Closed-form ROI tries to compute the same thing
analytically but the integral over opp's hypothetical actions over the
rest of the game doesn't simplify to something that ranks candidates
correctly.

## Lessons

1. **`fast.py bench` is a wallclock gate, not an A/B.** Use `fast.py
   eval` for outcome verdicts. Codified in HANDOVER + next-session plan.

2. **Architecture pivots need a control experiment, not just oracle
   testing.** Oracles encode PI's intent but don't predict ladder
   competitiveness. A single early small-n A/B against v7_0 (cheap;
   8-game bench EVAL takes ~5 min) would have caught the regression
   right after Phase 2, saving 4 phases of work building on a broken
   base.

3. **The opponent model used in any rollout must match the opponents
   on the ladder.** Surrogate opponents (`lite_greedy_policy`) are
   convenient but don't generalise. Any future rollout-based agent
   either needs a panel of opponent models (one per archetype) or a
   learned policy that approximates real ladder play.

4. **Closed-form opp modeling needs SYMMETRIC time windows.** Gross
   uses hold horizon (until opp counter ETA). Vuln using "rest of
   game" creates an asymmetry where every drained capture is paper-
   negative. Transient vuln (loss horizon = opp_eta) is closer to
   correct but still doesn't balance against the proven game-dynamic
   weight of capture vs defense in actual play.

5. **The proposer's filters are the most valuable shared
   infrastructure.** `_target_holdable_after_capture`,
   `_source_survives_launch`, and the trajectory-admissibility filter
   all operate BEFORE the chooser and benefit any chooser downstream.
   Improvements there compound. Next session's Phase B targets this.

6. **Oracle scenarios encode intent precisely enough that bad test
   geometries surface.** Two of the four xfail-conversions in Phase 5
   needed minor geometry adjustments because the original layouts
   created trajectory-filter collisions (collinear with sun-line or
   passing through ally planets). Real insight: tests written before
   a filter exists may stop working when the filter lands. Document
   the filter's geometry assumptions in `tests/conftest.py` if/when
   it exists, or in test docstrings.

7. **Multi-step / conditional planning is beyond closed-form ROI.**
   The `solo_capture_but_loses_source` oracle (B reinforces A in
   anticipation of opp's counter to A's attack) was structurally
   undecidable for ROI v1; Phase 5 needed asymmetric production
   (A=2, B=1) to make the math support the answer. Some game patterns
   simply require sequential reasoning that single-action evaluation
   can't capture.

## Friction tags added or confirmed this session

- `fast-py-bench-pass-is-wallclock-not-ab` — me: misread bench as A/B
  for one G2 check. Codified in HANDOVER and next-session plan.
- `bundler-modular-agent-namespace-multi-line-import-breaks-bundle` —
  already documented at main.py:71-76; tripped me when chooser_roi.py
  used a multi-line `from lib.scoring import (...)`. Single-line
  imports mandatory in any bundled module.
- `delta-us-minus-them-misses-eliminations` — fast_sim's terminal
  scoring head doesn't reflect game-end wins. Built `_terminal_value`
  wrapper at chooser_roi.py for future reuse.
- `surrogate-opp-doesnt-generalise` — using `lite_greedy_policy` as
  the rollout opp model produces measurements that don't predict play
  against v7_0/v4_planner/v3.5.1.

## What we kept

Beyond the immediate failure, the session produced:

- **14 synthetic oracle scenarios** covering: hold feasibility (3),
  drain frontier (2), defense (2), coordinated capture, solo capture,
  cleanup, sanity, horizon refusal, ROI target selection, n-way
  coalition, opp modifier, solo loses source. Useful for any chooser.
- **Source-vulnerability concept** — closed-form model of opp counter.
  Pattern is in `_cheapest_opp_counter` (factored helper, reusable).
- **Endgame elimination bonus** — explicit term for "capturing last
  opp planet ends the game; the gain is uncontested production." Worth
  considering for trajectory if its current scoring misses this case.
- **N-way coalition enumeration** — closed-form merged-arrival-ledger
  walk via `predict_garrison_at`. Trajectory's joint mechanism is
  pair-only; generalising it to N-way using ROI's enumeration
  structure could be a follow-up lift.
- **Defensive coalition (B reinforces A)** — the joint shape trajectory
  doesn't currently model. Different problem from N-way attack
  coalition.
- **Tier 2 rollout posterior infrastructure** — `_terminal_value`,
  `_rollout_baseline`, `_rollout_with_action`. Reusable if a better
  surrogate opp becomes available.

## Decision: where ROI work parks

Recommended: **opt-in via `BASELINE_CHOOSER=roi`** with the chooser file
preserved. The 14 oracles encode real game properties regardless of
which chooser is active. The defensive coalition + N-way coalition +
endgame bonus are useful concepts. Re-evaluating the direction is
cheaper from this state than rebuilding from scratch.

Alternative (if disk-cleanliness wins): delete `chooser_roi.py` and the
4 new oracles. Loses ~750 LOC + 4 test cases. Not recommended; the
overhead is minor and the option value is real.

## Recommendation for next session

`audit/2026-05-19-next-session-plan.md`. Primary work: validate the
hold-feasibility filter as a SOLO change via 64-seed A/B, then push
if it lifts. This is the smallest safe ship and was identified as
the highest-EV unvalidated mechanism in the trajectory line.

Don't revisit ROI without (a) a better surrogate opponent model, or
(b) a different architectural strategy (e.g., trajectory-with-coalition
or trajectory-with-defensive-joint, which would borrow ROI's
enumeration but use trajectory's proven rollout scoring).
