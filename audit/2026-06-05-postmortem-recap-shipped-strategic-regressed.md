# Postmortem — 2026-06-05 champion-ml-graft-majestic-storm

Session shipped `multi_tick_recap` (sub 53390700, PENDING in TrueSkill
warm-up). Three mechanism layers landed on top of `multi_opp_def`
(μ=1285): multi-tick opp projection, recapture penalty, strategic
value bonuses (denial + opening). The first two ship in the
submitted bundle. The strategic bonuses regressed in clean A/B at
default weights and were deferred for re-tuning.

## What went wrong

- **Bonus weights chosen by hand-math, not empirical calibration.**
  I set `PRODUCER_PLUS_DENIAL_WEIGHT=0.1` and
  `PRODUCER_PLUS_OPENING_WEIGHT=0.1` based on a back-of-envelope
  estimate that ~55 ship-unit bonuses would be "comparable" to the
  existing competitive_score range. They were 2-10× the typical
  magnitude. Result: 0/4 vs producer in a clean n=4 A/B. A 5-minute
  probe dumping competitive_score values from one game would have
  revealed the magnitude mismatch before the bundle was built. Bad
  decision given priors at decision-time.

- **Burned multiple A/Bs on contention-saturated runs before
  escalating to workers=1.** First run was 8-worker (0/16 at
  p50=11.5s). Correctly diagnosed contention. Then ran 2-worker A/Bs
  (p50=1.1s, still over the 1000ms cap) on three bundles before
  recognizing the harness was systematically producing
  TLE-driven outcomes. The right move was workers=1 on the second
  contention-noisy result, not the fourth. Cost: ~3h of A/B compute
  + delayed submission decision.

- **Started a 4P self-match concurrently with a running A/B.** Both
  saturate CPU; both got slower. The 4P ran 1 hour CPU without
  finishing (almost certainly hit step cap = cycle persists, but the
  data is unrecoverable). The "sequential" 4-game A/B had p50=1350ms
  instead of the clean ~80ms it should have had. Lost the 4P
  diagnostic entirely and had to re-run the sequential A/B.

- **Did not ablate strategic vs recap before code review and submit
  preparation.** A clean single-process A/B on each variant before
  the code-review pass would have surfaced the strategic regression
  in 5 minutes. Instead, we discovered it during pre-submit prep.
  Decision could have been better-sequenced: code review → tune
  → A/B, not code review → A/B → tune.

## PI overrides this session

- **Rule 12 warm-up framing**: I read μ=1159 as "regression" when the
  submission was still in TrueSkill warm-up. PI corrected — same
  pattern as earlier handover. Calibration data: don't read sub-24h
  scores as terminal.
- **"No band-aids" (carried over from handover)**: kept the strategy
  aimed at modeling-correctness (Rule 40) rather than cycle-detection
  hacks.
- **Rule 45 strict gate override** on the multi_tick_recap submit:
  accepted that 2-worker contention contaminated the n=16 measurement
  and authorized the submit despite Wilson-lo 0.28 < 0.50 gate.
  Documented in claim row with the contention reasoning.

## Frictions logged this session

(Not appended to `audit/friction.md` — postmortem invoked directly,
not via WRAPUP. Patterns captured inline in this file's "What went
wrong" section above. The two candidate-promotions were declined by
PI as not warranting framework changes at this time.)

## Promotion candidates (PI ratified: NO on both)

Two were drafted and surfaced inline for ratification:

1. **A/B harness wallclock-validity gate** — pattern that
   contention-saturated A/Bs whose focal p50 exceeds the timeout cap
   are structurally invalid. PI declined promotion: not warranting
   a framework rule at this time.
2. **Empirically calibrate new additive scorer terms** — pattern that
   default weights chosen by hand-math regress when the new term's
   magnitude exceeds the existing scorer's typical range. PI declined
   promotion: not warranting a framework rule at this time.

Both patterns are documented in this postmortem for future reference;
the next session's agent will read them as audit context. No edits to
`.claude/skills/kaggle-comp/improvements.md` this session.

## PI additions (from step 4)

None. PI: "Nothing to promote or to add."

## What landed this session

1. **Multi-tick opp projection** (commits `4e7c840` → `d714943`).
   `predict_opp_launches_via_mirror` runs K rounds per turn,
   accumulating opp launches with cumulative_bg threaded into each
   inner call. Eta-frame correctness fix (round-(k+1) opp sees etas
   decremented by k+1), parse_obs hoist, truncation policy fix
   (final pad = max(pad_to, len(records))) all came from code review.

2. **Recapture penalty** (commits `5ac1803` → `173c952`). Per-candidate
   leaf-scorer discount for thin captures opp can recapture. Subtracts
   `prod[T] × turns_lost` in ship units. `K_recap_eff = max(1, K_recap
   - K_opp)` avoids double-counting multi-tick's coverage. Code-review
   pass removed dead `target_idx` parameter, deleted no-op test,
   replaced ObsLike duck-type with real ParsedObs, extended drift-check
   to all 9 shipped shims.

3. **Strategic-value bonuses** (commit `8c0c228`). Two opt-in scorer
   credits: denial (rewards captures opp values) + opening (rewards
   early-game captures, linearly decaying to zero at the opening
   window). Both in ship units, share `_compute_captures()` helper.
   Default weight 0.1 regressed; deferred for re-tuning at ~0.02.

4. **Submission**: sub 53390700 `multi_tick_recap` (commit `68604a6`,
   bundle sha256 `049915a4ecdf...`, 272 870 B). Rolling pair after:
   #1 53390700 PENDING, #2 53384340 multi_opp_def μ=1285.

## Open questions (carry forward)

1. **What's the clean single-process A/B for multi_tick_recap?** We
   never ran it. The 2-worker n=16 was 8/16 contaminated; clean n≥4
   never run. First diagnostic the next session should add.
2. **What weight makes strategic-value bonuses non-regressive?** A
   reasonable starting bet is 0.02-0.03 (5-15% of competitive_score
   magnitude). Needs a probe to verify the per-candidate score
   distribution and pick a default empirically.
3. **Does the 4P cycle stalemate break with multi_tick_recap?** The
   1-hour-stuck 4P self-match this session strongly suggests not, but
   the test was contaminated by parallel CPU work. Need a clean 4P
   self-match against the same seed as validation game 78807326.
4. **The wallclock optimization path.** Outlined this session: skip
   regroup/greedy in opp planner calls; batch opps within a round;
   approximate rounds 1+ via eta-shifted reuse; eventual learned opp
   policy. Cumulative ~3-5× speedup expected; opens headroom for
   denser mechanisms.
5. **Force-concentration mechanism.** Discussed but not implemented.
   Restructures greedy_select to pool sources onto top targets
   instead of one-launch-per-source. Higher-altitude lever than
   scorer corrections; bigger surgery.

## Framework version at session-end

- Commit SHA: `0ef5dcea292efd8a6cf76203cd72487c8845e9f4`
- Branch: `claude/champion-ml-graft-majestic-storm` (157+ commits ahead of main)
- Active rules: 1-46 per `CLAUDE.md`. No rule promotions this session.
- Loaded skills this session: `kaggle-comp`, `code-review` (high effort, twice),
  `postmortem`. No skill edits this session.
