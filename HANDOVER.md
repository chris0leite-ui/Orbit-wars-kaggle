# HANDOVER.md — next-session brief

> Last written: 2026-05-18 PM by `claude/audit-workflow-performance-btjeK`.
> **Bug #15 v2 + bug #14 option 5 both A/B-failed at 39.6% vs the 50%
> bundle baseline. PV term is shipped as production default but
> regressing.** Next session: FLIP THE PV KILL-SWITCH.

## Live state

| Submission | μ (settled) | Status | Role |
|---|---:|---|---|
| 52766596 | **1094.1** | Active rolling pair | joint v3 (2P-only-gate) — underperformed (from prior session) |
| 52754310 | 1141.0 | Active rolling pair | trajectory champion |
| 52744856 | 1149.2 | Evicted | composite_a2 (was the floor) |

Daily submission budget: 5/18 used **0 this session**. The 1141 floor
was untouched because we never approved a production submission.

## The headline finding

**Bug #15 v2's PV term over-credits captures**, full stop. Three
independent A/Bs all converged at 39.6% vs the 50% pre-fix baseline:

| Attempt | A/B | Diagnosis claimed |
|---|---:|---|
| Bug #15 v1 (PV + per-fleet credit) | 40.6% n=64 | "double-counting" |
| Bug #15 v2 (PV only) | **39.6% n=96** | "asymmetric rollout (bug #14)" |
| Bug #14 option 5 v2 (PV + rollout-defense) | **39.6% n=96** | falsifies "bug #14" hypothesis |

The convergent failure means: the chooser was calibrated WITHOUT
the PV term, and adding any positive capture signal of that
magnitude (~100 value units per captured planet) inflates all
candidate scores uniformly → over-emission → drained sources →
losses.

**Bug #14 hypothesis (asymmetric rollout causes the over-credit)
is fully falsified.** Rollout-defense for ME doesn't change
chooser behaviour enough to overcome the PV inflation.

## What this session shipped

10 commits on `claude/audit-workflow-performance-btjeK`:

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

Net production effect: **negative**. Bug #15 v2 (`b285882`) is
default-on; the PV term it ships is the regressing signal. Bug #4
(`384cd54`), bugs #3+#12 (`9eb882d`), and the foundational
infrastructure changes are unconditionally good.

## Falsified-or-dead

- **Bug #14 option 1** (cheap-mirror with lite_greedy for ME, commit
  5f22ea8): regressed defense oracles to FAIL — lite_greedy is too
  attack-biased for a defensive baseline. Toggle OFF.
- **Bug #14 option 5** (smart reactive defense, commits e7f94cf +
  8e60a6a): A/B-failed at 39.6%. Wallclock fixes are real (685ms max
  in bench, zero outliers) but the behavioural hypothesis is falsified.
  Toggle OFF.
- **Bug #15 "double-counting" diagnosis**: wrong. The per-fleet
  credit only fires on in-flight at leaf; PV credits arrived
  captures at leaf — disjoint regimes, no double-counting per
  capture. The actual cause is chooser-calibration mismatch with
  the PV term's scale.

## Tier 1 — Critical for next session

**1. Flip the PV kill-switch.** One-line change. In
`lib/value_heads.py:177` change `_COMPOSITE_PV_ENABLED` default to
`False`, OR set `os.environ.setdefault("COMPOSITE_PRODUCTION_PV", "0")`
in `agents/baseline/main.py`. Sanity oracle reverts to xfail; chooser
returns to pre-#15 calibration. Verification:
- Bench: should match toggle-off numbers (max ~692ms).
- A/B small (n=32): Wlo should clear 0.45 (the pre-fix was 50%
  by definition; we expect ≈ 50% back).

**2. Then submit a production update** if A/B confirms 50%+:
trajectory champion (1141) is still the rolling-2 anchor — pushing
a calibrated update should hold the floor at worst, lift it at best.

## Tier 2 — only after Tier 1 lands

**3. Re-investigate PV's calibration debt** — open question filed at
`knowledge-base/questions/2026-05-18-can-chooser-be-recalibrated-for-PV.md`.
Can the chooser's emit gate be tuned to absorb the PV scale without
losing the PV signal? Two outcomes possible; investigate before
re-enabling PV.

**4. Fix the cleanup oracle test setup** —
`test_oracle_cleanup_capture_last_opp_planet` has a planet-collision
bug (P0 at i=0 in the radius-35 circle coincides with the opp
planet at (85, 50)). Offset the opp planet by a few units;
re-verify whether bug #15 v2 (PV on) unlocks the xfail. This is a
test-quality fix, not a production fix.

**5. Source-defense penalty at the leaf** — alternative to option 5.
At leaf, count MY planets captured by opp during the rollout and
subtract a penalty. Direct punishment of "I drained my source, opp
took it" without requiring rollout-defense. Lower complexity than
option 5; potentially complementary.

## Tier 3 — parked

- **Bugs #6, #7, #8** — all require option 5 or bug #14 to be
  resolved first. Currently parked.
- **Bug #13 cleanup oracle** — see Tier 2 item 4.
- **Bugs #9, #10** — won't-fix tail.

## Pointers (new this session)

- `audit/2026-05-18-bug-catalog.md` — 15-bug catalog (yesterday's
  artifact, still current).
- `audit/2026-05-18-postmortem-bug-15-v2-and-bug-14-option-5.md`
  — full postmortem of this session's three failed A/Bs.
- `knowledge-base/thoughts/2026-05-18-PV-term-recalibration-debt.md`
  — the general lesson on value-head/chooser calibration mismatch.
- `knowledge-base/flags/2026-05-18-pv-term-regression-shipped-as-default-on.md`
  — must-flip-next-session warning.
- `knowledge-base/questions/2026-05-18-can-chooser-be-recalibrated-for-PV.md`
  — investigation sketch for re-enabling PV after the flip.
- `tests/test_planner_oracles.py` — oracle suite with bug #4 +
  drain-frontier tests added.
- `tests/test_me_defensive_policy.py` — option 5 unit tests
  (idempotency contract pinned even though feature is OFF).
- `lib/opp_model.py` `me_defensive_action` — option 5 policy
  (currently dormant behind `BASELINE_ME_DEFENDS=1`).

## Rule reminders for next session

- **Rule 1**: submissions are PI-approved. Don't push a new
  submission until A/B post-flip clears 50%.
- **Rule 12**: rolling-last-2 = [52754310 (1141.0), 52766596
  (1094.1)]. Pushing 1 evicts the older. A clean 50%+ A/B before
  push.
- **Rule 32**: session-start git fetch (already done in the
  bootstrap hook — verify HEAD matches origin/main).
- **Rule 38**: fix-verification reproduces failure state. The
  oracle suite is the regression harness for the PV flip — confirm
  sanity oracle goes from PASS to xfail (expected) and ALL other
  oracles remain in their current state.
- **Rule 40**: prefer modeling correctness over restriction
  tuning. The PV flip is restriction tuning (turning off a feature)
  but it's also recognising that the FEATURE isn't yet integrated.
  Re-introducing PV later requires chooser-gate recalibration
  (Tier 2 item 3), not just leaving PV on and hoping.

## Next-session first commit suggestion

Single-line flip: `_COMPOSITE_PV_ENABLED` default → False (or
`COMPOSITE_PRODUCTION_PV=0` setdefault in `agents/baseline/main.py`).
Commit message: "value_heads: disable PV term in production (n=96
A/B regressed 39.6%); kill-switch retained." Bench + small A/B,
then push.
