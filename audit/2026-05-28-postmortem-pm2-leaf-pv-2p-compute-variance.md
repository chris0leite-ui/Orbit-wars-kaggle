# Postmortem — 2026-05-28 PM2 (kaggle-submission-review-gZsCu)

Second postmortem of the day. Companion to
`audit/2026-05-28-postmortem-pv-eta-pm-pv-eta-and-silent-turns.md`
(PM1, which covered the PV_ETA ship). This one covers the leaf_pv_2p
ship + the compute-variance investigation.

## What went wrong

- **Recurring misread: "scores don't settle."** I called sub
  53117942's μ=921 first-hour reading a "regression" / "verdict"
  and presented it to PI as such. PI corrected: new agents enter
  the ladder at μ₀=600 and climb. The reading is a snapshot on
  the climb, not a verdict. PI noted this has been corrected
  "hundreds of times" across sessions. **Decision-quality verdict:**
  this is a rule-gap failure — the fact was scattered across
  comp-context.md's σ-shrinkage line and matchmaking line but not
  stated loudly enough to bind. Fixed THIS session by adding a
  loud "SCORES DO NOT SETTLE" block to comp-context.md.
- **Submitted on n=10 Wilson-lo 0.40 when n=32 was 19 min away.**
  Same antipattern as the SHIP_TURN_KAPPA disaster (sub 53099001).
  PI override authorized at the time. POST-hoc, the investigation
  uncovered a CPU-variance confound that should have raised the
  bar to n=32 — the A/B was noisier than the Wilson interval
  suggested. Decision-quality verdict: defensible given priors at
  decision-time, but the confound was discoverable via grep in
  ~15 min and should have been priced in.
- **Submitted before investigating same-seed step-count drift.**
  The drift signal (seed=3 finishing at 250 vs 183 across two
  runs of the same A/B harness) was visible BEFORE the submit
  decision. I noted it as "worth investigating" but did not gate
  the submit on it. PI asked for the investigation post-submit;
  it turned up the CPU-variance coupling. Pre-submit it would
  have widened the implied A/B noise floor and likely changed
  the recommended sample size.

## Frictions logged this session (audit/friction.md PM2 section)

- `tag: scores-do-not-settle-recurring-misread` — promoted to
  comp-context.md::SCORES DO NOT SETTLE block.
- `tag: local-AB-cpu-variance-coupling` — flag raised at
  knowledge-base/flags/2026-05-28-compute-variation-ab-noise.md.
- `tag: n10-submit-when-n32-was-19min-away` — friction noted;
  same antipattern as ship-turn-kappa disaster.

## What went RIGHT (calibration data)

- Bundle + parity smoke (Rule 46): clean, caught zero issues.
- Cross-branch coordination (Rule 42): claim row appended pre-submit
  with eviction analysis; PI made the eviction call with full picture.
- Compute-variance hypothesis test: cheap, fast, decisive. The
  4-parallel-runs-on-pinned-cores experiment took ~12 min and
  converged 4/4 outcomes. Good experimental design.
- Did NOT panic-resubmit on the μ=921 reading. PI's "scores don't
  settle" correction landed before I could propose a panic action.

## Promotion candidates — PI RATIFIED 2026-05-28 PM2

**Outcome:** A → CLAUDE.md Rule 48 (as drafted). B → CLAUDE.md Rule 45b
(amendment to Rule 45, not standalone Rule 49 — PI's call).

### Candidate A — promote "scores don't settle" to a Rule

**Target:** `CLAUDE.md` rules block — add as Rule 48.

**Tag:** `scores-do-not-settle-recurring-misread` (this session) +
multiple prior sessions per PI's "hundreds of times" remark.

**Draft text:**

> 48. **Same-day Kaggle ladder readings are climb snapshots, not
>     verdicts.** New submissions enter at μ₀=600 and climb as they
>     play games. Older submissions have stabler reads (σ shrinks
>     with games played). NEVER call a sub "regressed" or "falsified
>     by the ladder" until it has had hours of play. Same-day verdicts
>     on freshly-pushed agents are systematically biased downward.
>     Read the comp-context.md::SCORES DO NOT SETTLE block before
>     interpreting any sub's μ reading taken < 4h after submit.
>     Origin: 2026-05-28 PM2 leaf_pv_2p ship; agent misread sub
>     53117942 μ=921 first-hour reading as regression; PI corrected
>     ("hundreds of times" across sessions). Sub-clause of Rule 8
>     (settled-once facts).

**Why:** the recurring-misread pattern has cost real cycles across
multiple sessions. PI has corrected the same agent misunderstanding
repeatedly. comp-context.md now has the loud fact but a CLAUDE.md
rule would also catch it in the Rule-scan that opens every session.

### Candidate B — promote "A/B confound check on inconclusive results"

**Target:** `CLAUDE.md` rules block — add as Rule 49, OR amend Rule 45.

**Tag:** `local-AB-cpu-variance-coupling` (this session).

**Draft text:**

> 49. **A/B confound check before any sub-gate-strength submit.**
>     When considering a submit on local A/B evidence below the
>     Rule 45 n=32 gate (Wilson-lo ≥ 0.50), first run a
>     CONFOUND CHECK: (a) re-run the smoking-gun seed(s) ≥3 times
>     with `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, and `taskset
>     -c <core>`; (b) if outcomes diverge, the local Wilson interval
>     understates true noise and the gate threshold for submit
>     should widen by ≥1 tier. Origin: 2026-05-28 PM2 leaf_pv_2p —
>     same-seed A/B step-count drift (218 / 241 / 305 / 359 same
>     seed) was visible pre-submit; post-submit investigation
>     traced it to wallclock-coupling in chooser. Confound was
>     discoverable in ~15 min.

**Why:** Rule 45 already requires n=32 with Wilson-lo ≥ 0.50, but
PI overrides land regularly. This rule makes the override more
expensive — agent must explicitly verify there's no confound
before proposing the override. Lightweight (a few minutes); high
calibration value.

## PI additions

PI answered "Nothing to add — wrap as-is."

## Framework version at session-end

- Branch: `claude/kaggle-submission-review-gZsCu`
- Commit SHA at write-time: see `git rev-parse HEAD` in the
  commit that ships this artifact
- Active CLAUDE.md rules: 1..47 (47 = physics-primitive
  verification, added 2026-05-19 PFhzM session)
- Skills loaded this session: postmortem (this); kaggle-comp
  via /improvements.md reads
- Newly added artifacts:
  - `submissions/baseline_leaf_pv_2p.py` (bundle, force-added)
  - `tests/test_leaf_pv_2p.py` (4/4 green)
  - `comp-context.md::SCORES DO NOT SETTLE` block
  - `knowledge-base/thoughts/2026-05-28-pm2-compute-variation-and-leaf-pv-2p.md`
  - `knowledge-base/flags/2026-05-28-compute-variation-ab-noise.md`
  - `knowledge-base/questions/2026-05-28-leaf-pv-2p-climb-trajectory.md`
  - This postmortem
