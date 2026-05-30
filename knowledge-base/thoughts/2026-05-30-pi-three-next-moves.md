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

---

## PI's ratified sequence (added 2026-05-30 ~20:50 UTC)

PI confirmed alignment with my ranking and added a submission step
between fix and opp-model work:

1. **Fix the kinematics bug**
2. **Submit the B.3 head bundle WITH the bug fix applied**
3. **Train the Tier 2 opp model** (highest-leverage architectural lift)
4. **Iterate** — stronger ML-supported agent on the better foundation

PI pointer for the fix: `claude/game-theory-winning-strategy-SEU7P`.

### Bug location + magnitude (post-investigation)

The bug is in `lib/kinematic_table.py` — a module-global singleton
position cache whose mutable state leaks across seats in any in-process
play (both agents run in the same Python process per `env.run`). Two
parallel fixes exist:

| Branch | Commit | Approach |
|---|---|---|
| `claude/game-theory-winning-strategy-SEU7P` | `d50654a` | Sets `KINEMATIC_TABLE_ENABLED` env-var default `"1" → "0"` (disables, leaves code in tree) |
| `claude/champion-strategy-rules-00JzI` | `232307c` | Deletes `lib/kinematic_table.py` + 2 test files entirely (~436 LOC removed) |

n=16 isolation A/B (per d50654a commit body) showed **+25 pp recovery
(31% → 56%)** — the kinematic-table singleton was silently regressing
pv_eta's own win rate by ~25 percentage points in any in-process A/B.

### Implication for the existing B.3 bundle

The B.3 head's CRN-paired corpus was generated with the bug active
(both seats running pv_eta in one process per self-play game) — so
every label was computed against corrupted leaf scores. The head
learned to predict the corrupted advantage, not the true one.

Three possible interactions with the fix:
1. **Net help** — the head's corrections partially compensated for the
   bug; with the bug gone, the underlying pv_eta is +25 pp stronger and
   the head's lift compounds.
2. **Net hurt slightly** — the head learned to exploit the bug's
   predictable wrongness; with the bug gone, some corrections misfire.
3. **No interaction** — bug + head touch orthogonal regions of the
   action space.

Verification path: rebuild the B.3 bundle with the fix, re-run the
n=16/32 A/B vs launch_rules_universal. If the head's 62.5% holds or
climbs → ship. If it collapses → retrain the head on a corpus
generated with the fix applied (~3 h smoke + decision point).

### Concrete next-step plan for my branch

| Step | Action | Cost |
|---|---|---|
| 1.1 | Cherry-pick `d50654a` (env-var flip — minimal) OR `232307c` (full removal — cleaner) onto `claude/competition-objective-alignment-hqNVM` | 15 min + parity tests |
| 1.2 | Rebuild `submissions/baseline_pv_eta_vh_b3smoke.py` with fix applied | 5 min |
| 1.3 | Rule 46: `pytest tests/test_bundle.py` + `fast.py play` | 5 min |
| 1.4 | n=16 A/B vs launch_rules_universal (focal=P0, fresh seeds 16-31 to avoid prior contamination) | ~17 min |
| 1.5 | If clear lift (point ≥ 70%, Wlo ≥ 0.50) → submit; else n=32 confirmation | depends |
| 2 | Submit (Rule 42 push-claim → Rule 46 gate) | 5 min |
| 3 | Tier 2 opp model first cut | 1 session |
| 4 | Re-train B.3 on improved foundation | 1-2 sessions |

Holding for PI go-ahead on whether to use d50654a (smaller diff) or
232307c (clean removal) before executing.

