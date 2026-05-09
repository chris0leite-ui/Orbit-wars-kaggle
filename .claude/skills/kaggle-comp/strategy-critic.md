# Strategy-critic-loop

A self-fired evidence-base interrogation. Different from
Research-loop (which scouts EXTERNAL writeups for untried mechanisms).
Strategy-critic interrogates **our own data** to surface what we
DON'T know about our model's failure modes — before we plan more
mechanisms on top of unverified premises.

The agent runs this loop AUTOMATICALLY at the triggers below. The
PI does not invoke it. Sitting on a stale evidence base while
queueing more experiments is the failure mode this loop prevents.

## ⚠️ Auto-trigger recognition (load-bearing)

Fire the loop, no prompting required, when ANY cue lands:

1. **End-of-day audit** — the day-N-wrap is being written
   (Day-loop step 5). Strategy-critic runs as step 5.5 and its
   output drives the hypothesis re-rank in step 5(d).
2. **Gap drift** — OOF→LB gap widens by ≥2bp on consecutive
   submissions in the same family (e.g., M5 −4.4 → M5b −3.5 → M5d
   −6.0bp = 1.6bp drift over two submits → fires).
3. **Before adding a new mechanism family** — if the proposed
   experiment introduces a new model family or FE family not on the
   `mechanism_families_explored` list, run the loop FIRST. Don't
   stack untested premises.
4. **Mid-comp checkpoint** — at 50% of `time_budget_total_days`.
5. **Plateau** — same trigger as Research-loop (3 nulls / 5 sat /
   2 days no lift). Strategy-critic runs BEFORE Research-loop:
   internal interrogation precedes external scouting.

When the cue fires, write `audit/YYYY-MM-DD-strategy-critique.md`
with the five sections below. No PI confirmation required. Output
is a delta on prior critiques — only document what *changed* or
what was newly discovered.

## The 5-question template

### 1. Per-segment failure map
Compute OOF AUC of the current PRIMARY (or best single-model) on every
load-bearing segment of the data. Default segments:
- Per high-cardinality categorical (each level if ≤30 levels;
  bottom-decile + top-decile groups if more)
- Per ordinal feature, decile-bucketed
- Per time/year if a temporal feature exists
- Per class-prior tertile if priors vary across segments

Output a ranked table: **which 3 segments are dragging the mean
most**. Lift surface lives there.

### 2. Probability calibration
Brier score, ECE (expected calibration error), reliability diagram on
PRIMARY OOF. If miscalibrated:
- Pseudo-label thresholds use isotonic-calibrated probs, not raw.
- AUC-optimization may be hiding probability-mass concentration.
- Document fix path (isotonic / Platt) before downstream consumers
  use the probs.

### 3. Model-disagreement localization
For all bases in the current pool: pairwise OOF correlation matrix
(already partly tracked) PLUS the row-level disagreement count per
test row. Identify:
- Where do the "highly-correlated" bases actually disagree? (the 1%
  diversity region — that's the real signal each base contributes)
- Test rows where ≥k of N bases disagree are the residual-difficulty
  rows: candidate for targeted FE or specialist sub-model.

### 4. Unexploited structural signal scout
Re-read `comp-context.md` `structural_findings` and the pre-baseline
gate audit. For each finding, ask: **have we turned it into a
feature?** Settled-once findings (sequence-recoverability, group-key
overlaps, target-rate gradients across deciles) often go undocumented
as FE candidates because they sound like "data understanding" not
"feature ideas."

Scout with one cheap diagnostic: a single-LGBM probe on baseline +
the proposed sequence/structural features. If Strat-OOF lifts ≥5bp,
add to the EXPLORE queue at high priority.

### 5. Headroom math vs plan
- Sum H-list midpoint lifts → optimistic upper bound.
- Apply 50% additivity discount (independent levers stack
  sub-linearly).
- Compare to `headroom_to_top5pct` (CLAUDE.md current state).
- If realistic-discount lift < headroom: **single-track plan will
  NOT reach goal**. Required action: either add levers (research-loop
  fires) or revise the goal.

Document the realistic-discount projection AND the contingency move
if H-list under-delivers.

## Output spec

`audit/YYYY-MM-DD-strategy-critique.md`, ≤150 lines:

```
# Strategy critique — <event/day> (YYYY-MM-DD)
Triggered by: <which cue fired>

## 1. Per-segment failure map
| segment | OOF AUC | n_rows | Δ vs mean |
[ranked table; bottom-3 highlighted]

## 2. Calibration
Brier=X.XXXX, ECE=X.XXXX. <miscal verdict + fix path>

## 3. Disagreement localization
<rows-of-residual-difficulty count + top-3 disagreement axes>

## 4. Unexploited structural-finding scout
<cheap-probe result + new EXPLORE queue items>

## 5. Headroom math
midpoint Σ = Xbp; 50%-discount = Ybp; headroom = Zbp.
Verdict: <reach / shortfall / over-budget>. Contingency: <if shortfall>.

## Concrete plan re-rank
<rerank H-list using sections 1-5 evidence>
```

Cap at 150 lines. If results need more space, link out to a sub-audit.

## Anti-patterns

- Don't fire this loop in place of Research-loop. They're orthogonal:
  Research = external scouting; Strategy-critic = internal
  interrogation. At plateau, both fire (critic first).
- Don't write a critique that just lists known facts. Each section
  must produce a NEW table / number / surfaced gap. If a section has
  nothing new, write "no change since YYYY-MM-DD" with a one-line
  justification — and that's a signal the loop is over-firing.
- Don't skip section 5 (headroom math). The optimistic-additivity
  fallacy is the single most common process error in playground-
  series stacking work; the math has to be on the page.
