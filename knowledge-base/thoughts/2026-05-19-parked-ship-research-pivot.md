# 2026-05-19 — parked-ship research, PI override, and what's underneath

Context: PI's framing was "roughly half our ship-turns are parked on
rear planets in mid-game; we want an incentive that draws them to the
frontier without too much exposure." I treated that as a starting
hypothesis to be tested empirically, ran a replay-mine across 89
episodes from three submissions, reported a 27.9 pp win-vs-loss gap
in the "parked fraction" metric, and concluded the hypothesis was
falsified. PI immediately caught the confound: the distance-based
"rear" definition (min_dist_to_nonour ≥ 35) grows automatically with
territory share, so winning → low-enemy-density → ships-look-parked
is tautological. The conclusion was unfounded. The audit doc and the
script were deleted before commit.

What's actually underneath:

1. **The parked-ship hypothesis is still UNTESTED.** Two prior fixes
   (spatial-leaf, H1 idle-drain) regressed, which is causal evidence
   that THOSE specific interventions hurt. It is not evidence that
   parking is OR is not a leak. To test the hypothesis honestly we
   need a metric that is invariant under territory share — e.g.
   launch-rate-per-surplus-ship in contested midgames (both sides
   < 55 % planets), or distance-to-engagement normalised by remaining
   enemy planet count.

2. **The Rule-22 scan and the joint-candidate scope finding stand.**
   Independent of the parking question, the strongest public notebook
   (Rahul, MCTS with 10-turn rollouts) confirms multi-step planning
   is the load-bearing idea. And `audit/2026-05-18-joint-candidates-submitted.md`
   already specified the exact `lite_greedy_policy` fix (vulnerability
   term for drained sources) that gates the 4P joint extension. That's
   a concrete next move with documented rationale and zero confound
   on the analysis.

3. **My failure mode.** I had the 87.5 %-winrate origin of the 43.8 %
   number on screen and didn't think to ask whether the metric was
   sensitive to that fact. Rule 26 says BOTEs get a devil's-advocate
   ritual; I skipped it because the data "felt clean." This is the
   same family as last week's "the framework optimises HOW to evaluate,
   not WHAT" (Rule 23) — I let process discipline (extending
   scripts/idle_trajectory_audit.py to compute win/loss split) substitute
   for thinking about what the metric was measuring.

4. **What to carry forward.** The promotion candidate in the postmortem
   (Rule 41: confound-sweep before correlational conclusion) is the
   one durable artifact from this session. Without it, the same
   pattern recurs whenever someone reads a future BOTE that compares
   group-A to group-B on a metric whose denominator moves with the
   grouping variable.

Open thread to the PI: do you want me to redo the parked-ship
analysis with a territory-share-controlled metric, or is the
"mobilize parked ships" thread itself off the table for now? The
joint-4P / lite_greedy-vulnerability path looks more compelling
either way, but I want to be sure I'm not declaring the parking
question dead just because my first measurement was confounded.
