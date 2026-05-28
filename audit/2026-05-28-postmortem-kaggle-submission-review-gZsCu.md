# Postmortem — 2026-05-28 kaggle-submission-review-gZsCu

## What went wrong

- **Plan paper-math anchored on Δ≈300; real Δ scale was 3-50.** Step 2B
  κ-calibration sweep `{0.5, 1.0, 2.0}` was an order of magnitude too
  high relative to the candidate distribution the chooser actually
  emits. At κ=0.02 (the lowest tried in the smoke), penalty still
  suppressed 97.7% of positive-Δ candidates. Decision quality
  (given priors): poor — the trace that would have caught this took
  20 minutes and was already a known instrumentation pattern. Decision
  cost: one rolling-pair slot at μ=680.0 (sub 53099001, -445 vs anchor).

- **Predicted-μ band 1080-1170 for Step 2B submit.** Actual landed at
  μ=680, which was the 5%-tail I explicitly dismissed in the prediction
  distribution. Anchoring error: "strict addition to peak, can only
  help" framing under-weighted the reinforce/late-game interaction
  risks I had already identified in the same chat. The band should
  have been mode 1050-1100 with 30% probability of <1000.

- **Did not insist hard enough on n=32 before submit.** PI signoff was
  explicit; the rule-conflict was presented cleanly. Given sub 53083109's
  same-morning evidence (75% local → 6% panel), arguing harder for at
  least one Phase 1 n=32 run vs anchor before burning a slot would have
  been the right move. Recurrence: this is the second consecutive day
  the same shape (PI signoff "ship for early feedback" → μ regression)
  has materialised.

- **Submitted on top of the REVERT instead of restoring peak first.**
  The cleaner sequence would have been: (1) submit peak-restore early
  morning to verify the ~20μ NEUTRAL_BONUS-plumbing gap is real, then
  (2) build Step 2B on top of the verified peak baseline. Instead I
  built Step 2B on top of the REVERT (which carries the wiring),
  conflating two unknowns.

## Frictions logged this session

Today's appends to `audit/friction.md` under
`## 2026-05-27 / 2026-05-28 (claude/kaggle-submission-review-gZsCu)`:

- `paper-math-calibration-trap` — Step 2B κ tuned on estimated Δ;
  real Δ 10x smaller; 97.7% candidates suppressed; μ=680 disaster.
- `dormant-env-var-wiring-suspected-regression` — NEUTRAL_BONUS-into-v4
  plumbing (sub 53083109) suspected source of ~20μ peak→REVERT gap.
- `n8-local-ab-vs-anchor-doesnt-predict-ladder` — second-day recurrence
  of `local-AB-not-calibrated-to-live-ladder`; n=8 vs anchor has
  near-zero ladder-μ predictive value.
- `bundler-namespace-collision-on-load` — bundler parity-gate crashes
  on namespace collision with `kaggle_environments.envs.lux_ai_s3`;
  bundle file gets deleted. Manual rebundle path required.
- `scatter-symptom-survives-peak` — peak emits 2-ship 40-turn launches
  because `favor` leaf doesn't price flight time. Peak achieves
  1144-1165μ despite this. Structural fix: pass `eta` to `pv_horizon`.

## Promotion candidates (PI ratified: no)

PI declined ratification of all three candidates presented in the
postmortem step 4:

1. CLAUDE.md Rule 48 — instrumented-trace calibration before paper math.
2. CLAUDE.md Rule 49 — dormant env-var wiring requires isolation A/B.
3. WRAPUP.md step 1b — peak-anchor capture as standing practice.

Not promoted. The behaviours are documented in `state/PEAK_BASELINE.md`
(anti-patterns + build-on-top protocol) and in this friction log.
Future sessions reading either file will encounter the lessons.

## PI additions (from step 4)

"Nothing to add or to promote."

## Framework version at session-end

- Commit SHA: `6cdcf61` (foundation push) — wrap-up commit will be
  this postmortem + friction + state + knowledge-base.
- Active rules: 1..47 (CLAUDE.md `## Operating rules — concise`).
- Loaded skills this session: `code-review`, `postmortem`.

## What landed cleanly this session (not failure-coded)

For balance, the session also delivered:

1. **Step 2B implementation** (commit `3c6c6dc`) — code is correct
   per spec; 5 unit tests green; env-gated default-OFF so it can be
   tuned/disabled without code revert. The mechanism is the right
   shape; the calibration was wrong.

2. **Peak-restore submit** (sub 53099429, commit `718d0ff`) — SHA-verified
   byte-identical to the historical peak. Currently pending; if it lands
   in the 1130-1170 band we confirm the ~20μ NEUTRAL_BONUS-plumbing gap.

3. **Clean foundation for build-on-top** (commits `c1f3bef`, `6cdcf61`):
   - Git tag `peak-1165` on commit 458f663.
   - Frozen anchor `submissions/baseline_peak_1165_anchor.py` (SHA
     `9ec3af83`, verified against live submission file).
   - `state/PEAK_BASELINE.md` — single source of truth with: peak record,
     plain-English strategy summary, active vs dormant env-var tables
     (19 active + ~40 dormant), top 5 fragility risks with file:line +
     mitigation, 7-step build-on-top protocol with mandatory Rule
     43/45/46/42 gates, anti-patterns from this session.
   - Pointers from `state/MULTI_BRANCH.md` and `HANDOVER.md`.

The session's net deliverable is the foundation, not the regression.
The regression's value is calibration data for Rules 43/45 enforcement.
