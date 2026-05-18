# HANDOVER.md — next-session brief

> Last written: 2026-05-18 LATE-PM (Tier 1/2/3 implementation wrap) by
> `claude/audit-workflow-performance-btjeK`.
> **Submission 52784853 still on ladder; μ snapshot drifted UP from
> 1083.1 (briefing) to 1122.1 (late-PM check) — drift confirms
> `kaggle-mu-does-not-settle` friction.** This session implemented
> Tier 1 (bundler sha256 parity cache; 6700× warm speedup), Tier 2
> (hold-feasibility filter behind PROPOSER_HOLD_FEASIBILITY,
> default on), Tier 3 (BASELINE_OPP_TIER dispatch, default off).
> Three new oracles in test_planner_oracles.py pin Tier 2's contract.
> No new submission pushed this session.

## Live state

| Submission | μ (snapshot, drifts) | Status | Role |
|---|---:|---|---|
| **52784853** | **1122.1** | Most-recent | PV off + clean math (drifted up from 1083.1 morning snapshot) |
| 52766596 | 1109.8 | Possibly in rolling pair | Joint v3 |
| 52754310 | 1143.7 | Possibly evicted (was champion) | Trajectory v4 |
| 52744856 | 1149.2 | Older | Composite_a2 hybrid (highest snapshot) |

Daily submission budget: 5/18 used **1** (52784853). 4 unused (none
used this late-PM Tier 1/2/3 session).

**Rolling-pair ambiguity (UNRESOLVED).** State/current.md says
52766596 was evicted by 52784853 (pair = [52754310, 52784853]).
The literal "rolling LAST 2 submissions" rule says 52754310 was
evicted (pair = [52766596, 52784853]). These have opposite
implications for the protected floor (1143.7 vs 1109.8). Verify
EMPIRICALLY at next-session start by checking the Kaggle team page
(CLI doesn't expose which 2 are evaluating). Conservative floor for
push decisions: 1109.8 until confirmed.

## Headline this session (late-PM Tier 1/2/3 implementation)

**Tier 1 — Bundler sha256 parity cache.** Cold parity gate measured
8m42s. With cache, warm re-build hits 0.078s (6700× speedup). Cache
file at `audit/bundle-parity-cache.json` (gitignored).
`--ignore-parity-cache` forces re-verification.

**Tier 2 — Hold-feasibility filter (proposer.py).** Three iterations
on the design:
  - v1 (MIN_COUNTER=10, no src-distance check, 1.0× margin): too
    aggressive — 88% zero-emit vs random.
  - v2 (MIN_COUNTER=20, src-closer-than-opp check, 1.5× margin):
    better but still cut some legit captures.
  - v3 (ally-aware: filter only triggers if our NEAREST planet to
    tgt is farther than opp's nearest): final design. Sanity vs
    random matches production behavior (~75% zero-emit baseline).
Three oracles in `tests/test_planner_oracles.py` pin the contract;
oracle 1 (`neutral_near_strong_opp`) is the fix-verification anchor
(failed pre-filter, passes post-filter). Filter on by default behind
`PROPOSER_HOLD_FEASIBILITY=on/off`.

**Tier 3 — Asymmetric opp model dispatch (chooser.py).** Added
`_select_opp_policy()`: `BASELINE_OPP_TIER=1` switches rollout opp
from lite_greedy_policy to top_tier_mirror_policy (v3.5.1 aggressive
snipe). Defaults OFF — bench gate REQUIRED before enabling (the
per-call cost is 5-10× lite_greedy). Bench WAS NOT run this session
(the Tier-3-on bench got buffer-stalled and was killed; needs re-run
next session).

**Carry-over caveat — local A/B vs ladder calibration miss.**
Previous submission's 81.2% n=32 local A/B → −30μ ladder loss.
Friction `local-ab-vs-ladder-calibration-miss-30mu` is binding.
This session's A/B was running at wrap-up time and HAD NOT
completed — DO NOT push Tier 2-on bundle until the A/B verdict is
known.

## What this session shipped (late-PM, Tier 1/2/3)

3 new commits on `claude/audit-workflow-performance-btjeK`:

```
a7f9383  proposer: hold-feasibility filter to drop unholdable captures
ce14317  chooser: BASELINE_OPP_TIER dispatch for asymmetric opp model
a273f09  bundler: sha256 parity-cache to skip warm-rebuild self-play gate
```

Default behavior:
- `PROPOSER_HOLD_FEASIBILITY` on by default → Tier 2 active in any
  bundle built post-commit.
- `BASELINE_OPP_TIER` off by default → Tier 3 has no behavioral
  effect unless explicitly enabled.
- Bundler cache: no behavioral effect; dev-tooling only.

This means a re-bundled `agents/baseline` is functionally
different from 52784853's bundle (filter is active). DO NOT push
without A/B verifying the filter doesn't regress ladder μ.

Older commits this branch (prior session):
```
be7a3b8  state/current.md: submission 52784853 (PV-off + clean math fixes)
82df5b8  value_heads: disable PV term in production (n=96 A/B regressed 39.6%)
[... see git log for full history ...]
```

## Falsified-or-dead

- **Bug #15 v1** (PV + per-fleet credit): A/B-failed 40.6% n=64.
  Diagnosed as "double-counting" — wrong diagnosis.
- **Bug #15 v2** (PV only): A/B-failed 39.6% n=96. The PV term
  ITSELF over-credits (chooser calibration mismatch).
  `_COMPOSITE_PV_ENABLED=False` is now the production default;
  kill-switch retained for future PV recalibration work.
- **Bug #14 option 1** (cheap-mirror with lite_greedy for ME):
  regressed defense oracles. Toggle off.
- **Bug #14 option 5** (smart reactive defense): A/B-failed 39.6%
  n=96 — bug-#14-cures-PV hypothesis fully falsified. Toggle off.
- **Bundle-as-only-A/B-baseline**: insufficient for predicting
  ladder shift. Need panel + 4P + h2h-vs-current-floor.

## Next-session plan (audit/2026-05-18-next-session-plan-tiered.md)

Five tiers, ordered by ROI:

1. **Tier 1 — Bundling-tax cleanup.** Cheap mechanical. Make
   iteration faster. ≤ 30 min.
2. **Tier 2 — Hold-feasibility filter** (THE wasted-ships lever).
   PI observed live games where we send fleets from far to capture
   neutral planets adjacent to strong opp planets — opp counters
   cheaply from short range and recaptures. Encodable as 3
   synthetic oracles (write FIRST); fix is a proposer-side pre-cut
   sibling to `_source_survives_launch` (bug #4 drain-frontier).
   1-2 hours. **Highest expected ladder lift.**
3. **Tier 3 — Asymmetric opp model.** Replace `lite_greedy_policy`
   in the rollout with `top_tier_mirror_policy` (already exists at
   `lib/opp_model.py:92`) or a ME-targeted counter-policy. Forces
   our chooser to find ME-robust strategies. Pair with Tier 2.
4. **Tier 4 — Active-planets / coalition proposer.** PI's
   structural critique: every planet should pitch a candidate; far
   planets contribute ships to the highest-EV mission. Generalises
   the existing pair-joint path
   (`chooser_trajectory.py:score_candidate_v4_joint`) to N-way.
   Bigger redesign; commit only after Tiers 2 + 3 are ladder-
   validated.
5. **Tier 5 (parked) — PV recalibration.** Open question filed at
   `knowledge-base/questions/2026-05-18-can-chooser-be-recalibrated
   -for-PV.md`.

Full plan at
`audit/2026-05-18-next-session-plan-tiered.md` — read FIRST next
session.

## Hard-won lessons

1. **A/B vs one bundle is not enough.** The bundle's local-vs-live
   gap is unpredictable. Add 3-opp panel + 4P sub-panel + h2h
   vs current floor before any submission push.
2. **Pre-submit calibration math doesn't add up.** Multiple recent
   submissions had +20pp local A/B → current −20 to −30μ vs floor.
   Recurring pattern; the warning in state/current.md is real and
   binding.
3. **Sanity oracles detect structural bugs but not calibration
   debt.** Bug #15's sanity oracle correctly identified "captures
   score Δ ≈ 0"; the fix structurally addressed it but the chooser's
   emit gate wasn't recalibrated → over-emission → ladder loss.
4. **Convergent failure across hypotheses is informative.** Three
   independent A/Bs (bug #15 v1 / v2 / +option 5) all at 39.6% told
   us the root cause was upstream of all three "fixes."
5. **PI's intuition about wasted captures is right and encodable.**
   The "we capture a neutral that opp easily takes back" pattern is
   the SAME shape as bug #4 (drain-frontier) but applied to the
   TARGET instead of the SOURCE. Synthetic oracles + pre-cut filter
   should land cleanly.

## How to start next session

1. **Read this file first.**
2. **Empirically verify rolling-pair** by checking the Kaggle team
   page (CLI doesn't expose which 2 are evaluating). state/current.md
   and the rolling-LAST-2 rule disagree on whether 52754310 (1143.7)
   or 52766596 (1109.8) is the preserved floor.
3. **Re-run the A/B that didn't complete this session**:
   ```
   PROPOSER_HOLD_FEASIBILITY=1 python -u fast.py eval \
     /tmp/baseline_tier2_v2.py --vs submissions/baseline.py \
     --geometry-panel --by-archetype --max-seeds 32 --workers 6
   ```
   The focal bundle `/tmp/baseline_tier2_v2.py` was the ally-aware
   filter snapshot at session-end (`sha256:50ec0b79...`). Re-bundle
   from current source if `/tmp` was wiped:
   `python scripts/bundle_agent.py agents/baseline --out-dir /tmp`
   (warm cache hit if source unchanged).
4. **Run the deferred Tier 3 bench**:
   `PROPOSER_HOLD_FEASIBILITY=1 BASELINE_OPP_TIER=1 python -u fast.py
   bench agents/baseline/main.py --vs v7_0 --games 2`. Verify max <
   1000ms; if it blows budget, fall back to "top_tier only on top-N
   candidates" per plan.
5. **Run 4P sub-panel**: `python scripts/play4p.py --focal
   /tmp/baseline_tier2_v2.py --bg v7_0,v7_0,v7_0 --rotate-seats
   --workers 4 --seeds <16 random>`. First-place rate ≥ 25% gate.
6. **Submission gate** (unchanged): do NOT push until ALL FOUR
   gates clear (oracles ✓ done + bench + multi-opp panel + 4P
   sub-panel). The bundle A/B alone is insufficient.

## Pointers (new this session)

- `audit/2026-05-18-bug-catalog.md` — 15-bug catalog (yesterday's
  artifact, still current).
- `audit/2026-05-18-postmortem-bug-15-v2-and-bug-14-option-5.md`
  — full postmortem of the three failed A/Bs (PV + option 5).
- `audit/2026-05-18-next-session-plan-tiered.md` — tier-by-tier
  plan for next session (Tier 1-5).
- `knowledge-base/thoughts/2026-05-18-PV-term-recalibration-debt.md`
  — general lesson on value-head/chooser-gate calibration mismatch.
- `knowledge-base/flags/2026-05-18-pv-term-regression-shipped-as-default-on.md`
  — historical flag (PV now disabled by default).
- `knowledge-base/questions/2026-05-18-can-chooser-be-recalibrated-for-PV.md`
  — investigation sketch for re-enabling PV.
- `tests/test_planner_oracles.py` — oracle suite (8 tests +
  3 xfails; conditional xfails keyed on
  `_value_heads._COMPOSITE_PV_ENABLED`).
- `tests/test_me_defensive_policy.py` — option-5 unit tests
  (dormant feature, idempotency contract pinned).
- `agents/baseline/proposer.py:355` — `_source_survives_launch`
  (the bug-#4 drain-frontier filter; PATTERN TO MIRROR for the
  Tier 2 hold-feasibility filter).
- `lib/opp_model.py:92` — `top_tier_mirror_policy` (the
  asymmetric-opp candidate for Tier 3).

## Rule reminders for next session

- **Rule 1**: submissions are PI-approved, single-shot, no retry
  loops.
- **Rule 12** (Orbit Wars caveat): rolling-pair contents AMBIGUOUS
  this session. Verify empirically before pushing. Conservative
  floor (lower of the two recent μ): 1109.8.
- **Rule 32**: session-start git fetch is REQUIRED. Origin/main has
  diverged 21 commits since last sync — archetype-strategies +
  simpler v15 baseline track. This branch deliberately stays on
  the trajectory_chooser stack per PI directive.
- **Rule 38**: fix-verification reproduces failure state. The new
  hold-feasibility oracles in `tests/test_planner_oracles.py`
  document the failed-pre-fix / passed-post-fix contract.
- **Rule 40**: prefer modeling correctness over restriction tuning.
  Tier 2 IS a restriction (a candidate pre-cut) — it's the cheapest
  approximation of the wasted-ships pattern. The principled-correct
  fix is the rollout seeing opp counter (Tier 3 + longer horizon).
  Watch for Tier 2 over-restriction in the A/B: if it cuts too many
  legit captures, default it OFF and lean on Tier 3 instead.
