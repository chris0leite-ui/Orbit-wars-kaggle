# 2026-06-21 — lead-gated win-equity REFUTED (both binary and smooth); we win by snowballing

> AI session note. Built the smooth/hysteretic rework of the lead-gate (the three-lens
> review's #1 lever) per PI request, tested by replay vs Producer V2. It is refuted, the
> same way the binary version was. The negative result carries a positive strategic
> insight. Lever kept default-OFF.

## What was built (the rework)

Replaced the hard ahead/behind switch with a continuous defensiveness `d in [0,1]`:
sigmoid of the production gap vs the strongest rival, EMA-smoothed across turns
(hysteresis; reset at game start). The leaf BLENDS its threat reading
`atk = (1-d)*allocate + d*max` instead of switching, and scales offense by `(1-d)`.
Threat block extracted to a shared `_blend_atk_reach` so the two leaf copies can't
drift. Knobs: LR_LEAD_GATE / LR_LEAD_STEEPNESS / LR_LEAD_EMA / LR_LEAD_OFFENSE_BOOST.
Code is clean, default-OFF, byte-identical when off, timing safe (max ~683 ms).

## Result — refuted (2P vs Producer V2, seat 0, replay)

| seed | validated default | binary lead-gate | SMOOTH lead-gate |
|------|-------------------|------------------|------------------|
| 6013 | WIN | loss | **loss** |
| 6019 | WIN | loss | **loss** |
| 1127764379 | WIN | loss | **loss** |
| 6031 | WIN | WIN | WIN |
| 6007 | loss | loss | loss |
| 6001 | loss | loss | loss |
| 6025 | loss | loss | loss |

The smooth version regresses 3 of 4 validated wins and fixes 0 losses — no better than
binary. The smoothing/hysteresis was NOT the problem; the IDEA is.

## Why (the insight)

Against V2 we WIN BY SNOWBALLING: every win-trace is monotonic production growth
(6013 26->45->...->94%, 1127764379 32->45->54->92%, 6031 9->42->57->95%). "Defend the
lead" (worst-case `max` threat when ahead -> hold a reserve, stop pressing) directly
SUPPRESSES the snowball that wins -> it converts our snowball wins into losses. The
lead-then-collapse losses are NOT "failed to defend a lead"; they are "the snowball
stalled / never got going and we got overrun." The cure is to press/expand HARDER
(which the neutral-margin lever does -> it fixed 6013/6019), not to play defense.

So the reviewers' theoretical #1 (win-equity / defend-the-lead) is the WRONG frame for
this agent-vs-V2 matchup. This contradicts the convex-win-condition argument in the
abstract, but empirically our edge over V2 is tempo/compounding, and protecting a lead
forfeits that edge. (Note: defend-the-lead might still matter in 4P placement or vs a
stronger opponent who can punish over-extension; untested. But vs V2 in 2P it loses.)

## Decision

Lever kept DEFAULT-OFF with a REFUTED status note in the `_lead_gate` docstring. Shipped
config unchanged (validated default = points 1+4 + validated neutral margin). No submit.

## Where this points next

Stop trying to make the agent cautious. The productive direction is the OPPOSITE:
sustain/extend the snowball and fix the games where it never starts (6007/6001/6025 — the
scattered-attack losses, which the hold-aware NATIVE_BUILDER addressed at the cost of
comeback aggression). The open question becomes "how to press the expansion advantage
harder without the scatter," not "how to defend a lead." Bring the live-ladder replay to
re-aim.
