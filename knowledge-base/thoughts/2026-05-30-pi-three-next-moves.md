# PI brainstorm — three next-move candidates (2026-05-30, post B.3 panel A/B)

> **Append-only.** Captured at the end of the B.3 / launch_rules_universal
> A/B session (pooled 20/32 = 62.5%, Wilson-lo 0.453 — just under the
> 0.50 submit gate). PI's words below; my take in
> `audit/2026-05-30-reframe-b3-results.md`.

## PI's three directions

1. **Remove the buggy kinematics table.** Architectural correctness fix.
   Refers to `lib/kinematic_table.py` (per-turn precomputed fleet-position
   lookup, intended bit-parity with `lib.orbit.predict_relative`). PI's
   characterization is that the table has a bug. Removing it falls back
   to the inline `predict_relative` path — slower but correct.

2. **Re-run the B.3 ML approach with the latest strong submission as
   foundation.** The current head was trained against pv_eta candidates
   + pv_eta rollouts. Retraining with `baseline_launch_rules_universal`
   (μ=1173.6 — current ladder peak) as the policy in self-play would
   produce a head calibrated against the stronger architecture.

3. **Use a B.3-style approach to build a fast, smart replacement for
   the "light greedy opponent model"** (the in-chooser policy used by
   `lib/opp_model.py` for rollouts inside `lib/v7_search.py` /
   2P/4P leaf scoring). Tier 2 is already a documented placeholder in
   `lib/opp_model.py` (intended as a "small logistic regression on
   the 37k labeled shots in `data/shot_validator/`"). PI's suggestion
   is to give that placeholder a CRN-paired training recipe similar
   to the B.3 head.

## My take

Ordering and rationale below. Asking PI for clarification on (1)
before scheduling.

### #1 — Kinematics table — **DO FIRST IF the bug is identified**

- Correctness fix, low cost. If the table returns wrong positions,
  every chooser that uses pv_eta-style leaf scoring is computing
  the wrong leaf delta — could be a non-trivial fraction of the
  62.5% loss rate vs launch_rules_universal.
- Risk: I don't currently know where the bug is. I need PI to point
  at the specific failure mode (e.g. "comet sentinel handling
  misfires for fleet ETAs past 30 turns" or similar) OR the audit
  note that documented the discovery.
- Cost estimate: 2-4 hours once the bug is identified.

### #2 — Tier 2 opp model — **HIGHEST LEVERAGE**

- The training data exists (~37k labeled shots in
  `data/shot_validator/`). The slot in `lib/opp_model.py` exists with
  the right signature. Every chooser that calls `make_opp_policy` is
  a downstream beneficiary.
- B.3-style CRN labels would be a stronger training signal than
  observational shot-validator labels: instead of "did this shot succeed
  in capturing?" (observational), the label would be "given the
  current state, would the *opponent* launch this candidate?" measured
  as a CRN-paired prediction shift. Different paradigm — the head
  needs to predict the OPP's action distribution, not its own.
- Lift mechanism: better opp predictions → tighter forward-sim leaf
  values → better chooser argmax. A weak opp model causes the chooser
  to over-attack (assumes opp won't counter) or over-defend (assumes
  opp will counter everything).
- Expected μ-gain if it lands: large (20-50 μ) because it affects
  every lookahead path, not just the per-candidate scoring.
- Cost estimate: 1 session for a first cut (logistic regression or
  small MLP on existing 37k labels). 2-3 sessions if we go full
  CRN-paired-on-fresh-self-play.

### #3 — B.3 retrained on launch_rules_universal — **DEFER until #1, #2**

- Natural iteration but the EXPENSIVE one (~24-38h overnight compute
  for stage 2 alone).
- Argument for deferring: the current B.3 head already shows 62.5%
  vs launch_rules_universal. Retraining changes the foundation but
  doesn't address the head's structural limits (one-sided predictions,
  per-candidate featurization cost). A better opp model (#2) would
  improve the B.3 corpus quality across the board, so it's better
  to do #2 FIRST and then re-train B.3 against that improved foundation.
- Caveat: launch_rules_universal includes a post-emit drop filter.
  A head trained on the post-filter prerank may learn to assign zero
  value to candidates the filter would drop — which would replicate
  the filter's logic rather than add to it. Need to think about
  whether the head adds anything to the launch_rules_universal
  architecture.

### Submission discipline note

While these tracks develop, the rolling pair stays at 1173.6 + 1017.2.
Submitting our 62.5% B.3 bundle right now would evict the 1017.2
(redeploy_gangup) — small win even if B.3 lands at μ=1100 — but
ALSO risks evicting the 1173.6 if a worse run-trace happens. Per
Rule 12 + Rule 42, hold off pushing the B.3 bundle until either:

- The opp-model work (track #2) produces a stronger bundle, or
- We're explicitly OK with the 1017.2-evict for a small
  ladder-floor improvement.

## Question for PI

For track #1 — where's the kinematics bug? Specific symptom or
audit note pointer?
