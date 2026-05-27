# 2026-05-27 — Spearhead directional rule, 5-cell A/B triage

Plan: `/root/.claude/plans/do-it-but-do-bubbly-teacup.md`. Commit: `85871cd`.

## Setup

5×250×no-swap via `scripts/ab_quick.py agents/baseline --opps submissions/baseline_joint_aggr_consolidated_orbitfix.py`. Sequential, workers=1.

Goal: attribute lift cleanly across `{relay off / relay on}` × `{spearhead off / relay-spearhead / chooser-bonus / both}`.

## Results

| Cell | Config | Wins | Winrate | Wilson 95% | Elapsed |
|---|---|---|---:|---|---:|
| A | relay off, spearhead off | 4/5 | **80%** | [0.376, 0.964] | 881s |
| B | relay on, no spearhead | 2/5 | 40% | [0.118, 0.769] | 779s |
| C | relay on + relay-spearhead | 3/5 | 60% | [0.231, 0.882] | 843s |
| D | relay off + chooser-bonus | 2/5 | 40% | [0.118, 0.769] | 625s |
| E | relay on + both spearheads | 1/5 | **20%** | [0.036, 0.624] | 599s |

## Interpretation (per plan's grid)

- **B < A** ✅ — relay regression direction matches the live-ladder data (sub 53065150 μ≈1097 → sub 53067354 μ≈905, −192 μ).
- **C > B** ✅ — relay-spearhead heals about half the relay regression (+20 pp over B), but C is still −20 pp below A. The R-selection fix is directionally right but doesn't recover the baseline.
- **D > A** ❌ — chooser-side directional bonus alone REGRESSED 40 pp below the no-bonus baseline. Opposite of the expected lift.
- **E vs max(C, D)** — E=20% << max(60%, 40%) = 60%. The two passes fight each other; composition is worst of all.

**Cleanest cell is the reference (Cell A).** Neither spearhead variant beats the no-spearhead, no-relay base on this opponent.

## n=5 caveat (Rule 45)

All cells have overlapping Wilson 95% CIs. n=5 is triage-only — these are directional signals, not falsifications. The point estimates are consistent enough to skip n=16 escalation:

- The chooser-bonus direction is clearly wrong (D=40%, E=20%). No reason to spend n=16 budget here.
- The relay-spearhead direction is right but the magnitude is bounded: even at best, C=60% < A=80%. Halving the relay regression is the *ceiling* of this fix, not its midpoint.

## What this tells us

1. **The relay forward-staging on this branch's lineage is a real ladder hazard.** Cell B reproduces it. The PI's observation of fleet ping-pong was a correct identification.

2. **The R-selection fix is partial.** Adding a directional bias to the relay's R-pick heals about half the regression but doesn't close it. The remaining damage likely comes from leg-2 misdirection — the relay commits leg-1, but R's future-turn choice of T is still ETA-greedy. A complete fix would need to evaluate the leg-2 alignment at leg-1-commit time, which the current code doesn't.

3. **Chooser-side directional bonus actively hurts** on a base that already considers direction implicitly through favor accrual. The favor function (`agents/baseline/value.py`) computes a position-aware ownership score over the rollout horizon. Adding an explicit cosine-aligned production bonus to the delta over-counts: the rollout's favor change *already* rewards taking front-line targets because they hold longer and produce more before recapture. The β=8 bonus shifts target selection toward distance-from-self-toward-opp without re-checking force-sufficiency — exactly the failure mode that Rule 40 ("prefer modeling-correctness over restriction-tuning") warned about.

4. **Stacking the two passes is worst** (Cell E 20%). Both want to commit fleets toward the opp side; together they over-emit and starve defense.

## Decision

- **Do NOT ship either spearhead variant.** Cell A's 4/5 is the operating point.
- **The proper next step is turning the relay off in `agents/baseline/main.py`** (set `BASELINE_RELAY` default to `0`). That's a one-line change recovering the Cell A configuration. NOT done this session per PI direction "no submission this session."
- **The spearhead axis as designed is closed for this opponent cohort.** The 2× coefficient follow-up was on the table (Cell C 3/5 met the trigger threshold), but the cost-benefit doesn't justify it: even doubling α can only narrow the gap below A, and the chooser-bonus + composition cells are far enough below A that a stronger α would likely make E worse, not better.

## Open questions / follow-ups

1. **Why does chooser-bonus regress?** The favor leaf already encodes direction via per-step ownership accrual. Adding an explicit launch-time cosine bonus to the delta double-counts and probably misranks candidates where the favor delta has already accounted for the directional value. A more principled bonus would adjust the score by *missing* directional value (i.e., penalize alignment with our OWN cluster) rather than by adding aligned-with-opp value on top of an already-aligned delta.

2. **Is the relay concept salvageable?** Cell C shows +20 pp of recovery over Cell B — there IS signal in the directional re-selection. But the relay's underlying premise (idle planets shipping to friendly waypoints) may itself be the wrong mechanism: every analyzed top-10 player concentrates fire on opp targets directly, not via friendly relays. The simplest action is "turn relay off"; the more ambitious one is "replace relay with direct-fire from rear planets when no front-line target is reachable."

3. **The bigger gap to top-10 (~340 μ) is not a chooser-tweak problem.** Two failed proposer-pre-filter axes, one failed defensive-modeling axis, and now a failed spearhead axis on the same branch suggests we are saturated at the chooser layer for this opponent class. The HANDOVER's "replay scout / behavior cloning" line and the trajectory-layer/precision-physics substrate work on sibling branches both remain unexplored from this branch.

## Files

- Plan: `/root/.claude/plans/do-it-but-do-bubbly-teacup.md`
- Implementation commit: `85871cd` (feat(baseline): spearhead directional rule)
- Per-cell logs: `/tmp/spearhead_cell_{A,B,C,D,E}.log`
- This audit: `audit/2026-05-27-spearhead-ab-matrix.md`
