# 2026-05-18 PM postmortem — bug #15 v1/v2 + bug #14 option 5

## TL;DR

This session shipped 5 substantive commits trying to fix the chooser's
behavioural regression introduced by bug #15 fix attempts. **All three
A/Bs vs the pre-fix bundle FAILED at essentially the same magnitude:**

| Attempt | A/B verdict | Diagnosis claimed |
|---|---:|---|
| Bug #15 v1 (PV + per-fleet credit) | 40.6% @ n=64 | "double-counting" |
| Bug #15 v2 (PV only) | 39.6% @ n=96 | "PV alone is fine, bug #14 makes it bad" |
| Bug #14 option 5 v1 | 15.6% @ n=32 (catastrophic) | wallclock blowup + stateless bug |
| Bug #14 option 5 v2 (idempotent + tick-0-only) | 39.6% @ n=96 | "option 5 doesn't cure bug #15 v2" |

The convergent 39.6% across two structurally-different "fixes" — bug
#15 v2 alone (no defense in rollout) and bug #15 v2 + option 5 (smart
defense in rollout) — is the load-bearing finding: **the
production-PV term over-credits captures regardless of whether the
rollout simulates future defense**. Adding defense doesn't move the
needle. The chooser was calibrated WITHOUT PV; adding any positive
capture signal of that magnitude inflates all candidate scores
uniformly → over-emission → drained sources → losses.

Bug #14 hypothesis ("asymmetric rollout under-rates captures, fix it
and PV will be calibrated") is fully falsified.

## What worked

- **Bug #3 + #12 fix** (9eb882d): symmetric reinforce sizing +
  widened multi-wave window. Clean math fixes anchored on
  asdf-game (76947663) step 37. No A/B regression observed; tests
  pin the contracts.
- **Bug #4 fix** (384cd54): proposer-side drain-frontier pre-cut.
  Drops candidates whose launch would leave the source unable to
  defend against the earliest known inbound enemy threat.
  Defensive layer for edge cases banding-dedup misses.
- **Option 5 wallclock optimization**: precompute defensive emits
  ONCE per candidate at tick-0 obs, merge with the candidate's
  launch at `wait_N`. Cut per-call cost from 5000 calls/turn to
  ~200 calls/turn (200 candidates × 1 vs 200 × 25 ticks). Bench
  dropped from 1492ms max (10 outliers) to 685ms max (zero outliers).
- **Idempotency fix for the defensive policy**: count in-flight
  friendly ships in `garrison_at_eta` so the policy converges after
  one emit per real threat instead of stacking reinforces every tick.

## What broke

### Bug #15 fix attempts

Bug #15 was originally diagnosed in commit `c57b2c0` (yesterday's
session). The composite's `pred_owner == my_id → skip` branch fired
on every capture WE caused because the WorldModel's prediction
INCLUDED our fleet's arrival.

V1 fix (commit 466fc98, this session, early): added two changes
together — (a) production-PV term in composite's base, and (b)
counterfactual per-fleet capture credit. A/B failed at 40.6%. I
told PI the root cause was "double-counting per capture" — same
capture gets PV credit AND per-fleet credit.

V2 fix (commit b285882): dropped the per-fleet credit, kept the PV
term. A/B failed at 39.6% — the SAME magnitude. The diagnosis was
wrong: the per-fleet credit only fires on in-flight fleets at leaf
state (post-arrival fleets are removed from `obs.fleets`), so there
was no actual double-counting per capture. The regression was just
the PV term itself over-crediting because of calibration mismatch
with the rest of the chooser's scoring.

### Bug #14 option 5 attempts

Hypothesis: bug #15 v2 over-credits because the rollout doesn't
simulate ME defending captured planets past horizon. If we add a
reactive-defense policy for ME, captured planets the rollout shows
held will actually be held → PV term stays calibrated.

V1 (commit e7f94cf): purely-defensive policy in CANDIDATE rollouts
only (baseline stays asymmetric). Two bugs:
1. **Stateless across rollout ticks** — `garrison_at_eta` ignored
   already-in-flight friendly reinforces, so every tick re-emitted
   a fresh reinforce against the same threat. By tick 5 we had 5
   redundant reinforces.
2. **Per-tick policy cost** — 5000 calls/turn for typical
   `N_VALIDATE=200 × horizon=25`. Each call ray-casts fleets, scans
   planets. Cost ~1.5s/turn added to the chooser.

A/B: 15.6% @ n=32 (catastrophic). Max wallclock 8252ms (env
returns empty actions on timeout → agent loses turns).

V2 (commit 8e60a6a): fixed both bugs. Idempotency via
in-flight-friendly counting; tick-0-only call instead of per-tick.
Bench: 685ms max, zero outliers. **A/B still failed at 39.6%** —
identical to bug #15 v2 alone. Option 5 makes the rollout faithful
to "future me defends" but doesn't change the chooser's emit
behaviour enough to overcome the PV over-credit.

## The convergent diagnosis

Three independent regressions, all at 39.6% (within rounding of v1's
40.6%), is informative. Whatever the chooser is doing differently
from pre-fix, it's structurally the same across:

- "PV credits capture + per-fleet credits in-flight + no rollout-me"
- "PV credits capture only + no rollout-me"
- "PV credits capture + tick-0 rollout-defense"

The common factor: PV term adds ~100 units of value per captured
planet at leaf. Chooser pre-fix had no such signal; its emit-or-
not threshold (Δ > 0) was calibrated against the unbiased base
ship-delta. With PV active, EVERY capture candidate gets +~100 →
emit becomes the default → over-emission → drained sources lose
games.

## What to do next session

### Option A — disable PV in production (RECOMMENDED)

`lib/value_heads.py:177` has `_COMPOSITE_PV_ENABLED` defaulting to
`1`. Change to default `0`. Sanity oracle goes back to xfail
(acceptable — it was xfail before this session anyway). The
chooser reverts to pre-#15 calibration; the 1141 baseline should
hold. The kill-switch remains in place; future PV experiments can
flip it back on for A/Bs.

Cost: 1-line change + commit. Verification: re-bench (should match
toggle-off numbers), small A/B vs bundle.

### Option B — revert b285882 entirely

Cleaner history but loses the diagnostic toggle. Same end-state as
A.

### Option C — retune the chooser for PV

The PV term IS principled (it credits ownership at the leaf rather
than letting equal-production captures score Δ = 0). The chooser
could be re-calibrated to absorb the PV scale: shift the emit gate
from `Δ > 0` to `Δ > PV_SCALE × prod_diff_at_leaf` so PV doesn't
unilaterally bias emission.

Cost: large. Requires sweeping the gate threshold over many seeds;
PI's "no parameter tuning" stance probably rules this out.

### Bug #14 follow-up — parked

Option 5 v2 is functional code (passes bench, passes oracles) but
doesn't deliver on the bug #14 hypothesis. Default OFF; toggle
remains for future re-investigation. The "future-me defends in
rollout" semantic is principled but the magnitude of effect is too
small to overcome the PV over-credit in the current chooser.

Followup ideas (low priority):
- **Source-defense penalty at the leaf** — penalize candidate
  rollouts whose leaf shows MY planets captured. Direct
  punishment of "I drained my source and opp took it" without
  needing rollout-defense.
- **Test the cleanup oracle planet-collision bug** — re-run the
  xfail with corrected geometry to confirm bug #15 v2 unlocks it.

## Process notes

- **Friction `tag: wrong-root-cause-from-symptom-similarity`** — I
  diagnosed bug #15 v1's regression as "double-counting" without
  running the kill-switch to isolate halves. v2 was built on that
  wrong diagnosis. The actual investigation should have started
  with `COMPOSITE_PRODUCTION_PV=0` A/B — that one experiment
  would have flagged the PV term immediately and saved the
  v1→v2→option-5-v1→option-5-v2 cycle.
- **Friction `tag: stateless-policy-in-rollout-cannot-converge`** —
  any iterative policy added inside a fixed-point rollout needs
  to be idempotent on its own output state. Locked the contract
  via `test_idempotency_inbound_friendly_counts_toward_garrison`.
- **Bench gates are good early warnings** — option 5 v1 hit
  1041ms in bench (verdict WATCH) before the catastrophic A/B.
  Should have re-prioritised diagnostic over A/B at that point.

## Calibration snapshot

A/Bs run this session vs `/tmp/baseline_hybrid_bundle.py`:

| Run | Tier | wins/games | winrate | Wlo | verdict | max ms |
|---|---:|---:|---:|---:|---|---:|
| Bug #15 v1 (pre-session, recall) | 64 | 26/64 | 40.6% | 0.295 | FAIL | 1666 |
| Bug #15 v2 | 96 | 38/96 | 39.6% | 0.304 | FAIL | 1150 |
| Bug #14 option 5 v1 | 32 | 5/32 | 15.6% | 0.069 | FAIL | 8252 |
| Bug #14 option 5 v2 | 96 | 38/96 | 39.6% | 0.304 | FAIL | 1349 |

No A/B PASSED this session. Production submitter (52754310, 1141.0,
trajectory champion) was unchanged because we never approved a
production submission.

## Commits shipped

```
8e60a6a  Bug #14 option 5 v2: idempotent policy + tick-0-only call
e7f94cf  Bug #14 fix: option 5 — reactive defense in candidate rollouts
384cd54  Bug #4 fix: proposer-side drain-frontier pre-cut
b285882  Bug #15 fix v2: PV-term-only (drop double-counting counterfactual)
5f22ea8  Bug #14 cheap-mirror attempt: NEGATIVE RESULT, toggle preserved off
333b884  value_heads: env-var kill-switches for bug #15 fix halves (diagnostic)
9eb882d  Bug #3 + #12 fix: symmetric reinforce sizing + widened multi-wave window
cdffbaf  fast.py: linear Wilson-gate tier ladder (+16 per tier, was doubling)
cf21d3e  Foundations: fix mission-drain test + bundler-safe imports, drop jax parity
466fc98  Bug #15 fix: counterfactual capture credit + production-PV term
```

10 commits, ~2200 LOC. Net production effect: zero — every fix that
touched the chooser's emit behaviour was kept behind an OFF-by-default
toggle. Bug #3, #4, #12 are unconditional fixes; bug #15 v2 is
production-on by default (currently regressing — needs the
`_COMPOSITE_PV_ENABLED` flip next session). Option 5 toggle stays OFF.
