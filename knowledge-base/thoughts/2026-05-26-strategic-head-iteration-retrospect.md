# 2026-05-26 — strategic-head iteration cycle, retrospect

PI voice-dump synthesis from today's session.

## The arc

Started with PI's question: "we have a massive improvement in 2P games
but somehow now we lose in 4P games and launching small waste fleets
of a small amount of ships. What is broken?"

The original strategic head (Phase D/E/F, my work 5/25) had 75% 2P /
10% 4P. PI noticed the 4P trickle and asked for diagnosis.

I diagnosed (correctly): asymmetric Term A + sum-weighted 4P opp aggregation
caused flat F2 / chooser-trickle. Built `fcaf414` (35% 4P / 62.5% 2P) —
that was a clean improvement.

Then I went too far: tried to "unify" the 2P/4P branch, "fix" calibrated
heuristics, ship a "modeling correct" leaf. Each step accumulated a
regression. 5 forward changes, 4 reverts to restore. Net: ~zero progress,
~−130 μ lost on the live ladder.

## What an expert would have done

After fcaf414 worked (35%/62.5% n=5 panel), PI said the new question is
"can you keep the 4P fix AND restore Phase F's 75% 2P." That's a clean
empirical objective.

The expert plays: try ONE focused change (e.g. asymmetric Term A
discount only when it pessimizes my side; symmetric only when it
pessimizes opp side — calibrated mid-ground). A/B panel. Ship or
revert. ONE axis. ONE change.

I instead made a "principled unified model" that touched 3 axes at
once. Each axis was defensible in isolation. Together they broke
the chooser. That's not bold-iteration, that's compound-axis chaos.

## The chooser's calibration is a real thing

The leaf is a number. The chooser's gate is `Δ > 0` where
Δ = leaf(action) − leaf(idle). What I missed: the leaf's ABSOLUTE
SCALE matters, not just the SIGN.

If a leaf swings ±200 across candidates and I "improve" the model
so it swings ±50, the chooser's Δ noise floor (from rollout stochasticity)
ate the signal. Δ that used to be +5 (clear positive) now might be ±5
(noise). Chooser idles or picks randomly.

This is why "modeling correctness" arguments don't always win against
calibrated heuristics. The chooser is a SECOND model that depends on the
first.

Operational lesson: when touching the leaf, hold the chooser's calibration
sacred. Pin every change with an A/B panel at the same N as the
calibration was established at (here, N≥5 per opp).

## On submitting for "learning"

The "submit to learn" framing is dangerous when:
- Local panel predicts below floor (the rolling pair lower bound)
- The agent is meaningfully different from prior submissions

In those cases the live learning is bounded by what the local panel
already told us: the agent is broken. Submitting just confirms.

Better frame: "submit to learn" when the LOCAL panel says NEUTRAL or
PROMISING but the live ladder might disagree (calibration drift,
unknown opps). NOT when local panel says BROKEN.

Today: 50% 2P + 12.5% 4P locally → submitted → μ=984. Local was right.
Live confirmed. No new information gained.

## Three rules I'd carve in stone for next time

1. **Calibrated heuristics are not approximations.** `fleet_speed(2)` in
   Phase F's threat-ETA wasn't wrong — it was tuned. The chooser
   depended on it. "Fixing" it broke calibration. Same for mean-garrison
   in Term B, asymmetric Term A in 2P, etc.

2. **One axis at a time.** Compound-axis commits are how you lose 2 hours
   on a 5-line revert. If 2+ knobs change, the test space is N×M, not N+M.

3. **Hard revert beats axis-by-axis restoration.** `git reset --hard <last-good>`
   is 1 minute. Multi-revert is hours. Default to the former when in
   "restore" mode.

## Where this leaves the comp

- Live: μ=984 on sub 53032723 (today's bad). Rolling pair {μ=1135.4,
  μ=984.1}. Floor dropped 129 μ.
- Code HEAD: `4ad192f` — fcaf414-equivalent strategic head (50% 2P / 50% 4P
  on n=2 panel; expect μ ~1100-1140 if submitted).
- Trickle-launch problem: **STILL OPEN**. Today's iteration didn't reduce
  trickle. The work to actually kill it needs different leverage (chooser
  Δ-per-ship threshold, or pivot to baseline_ev_per_ship lineage which has
  the per-ship sort already proven at live μ=1135.4).

Next session: either submit `4ad192f` to recover floor, or pivot to
`baseline_ev_per_ship` and iterate trickle-reduction from there. The
second has higher ceiling.
