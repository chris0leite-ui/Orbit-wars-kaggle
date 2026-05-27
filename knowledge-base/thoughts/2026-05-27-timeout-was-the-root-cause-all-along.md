# 2026-05-27 — timeout was the root cause all along

The lesson from today's session-EqJuT iteration (commits `ba58caf` +
`1da2652`) is one I want to remember in PI's voice, not as a rule:

**When an agent that should win a fight stalls at step 500 with the
opponent still alive, the FIRST hypothesis to falsify is "we got
disqualified by the actTimeout," not "we have a strategic blind
spot."** The strategic blind spot is a much more interesting story
and the postmortem narrative will lean toward it. But a strategic
blind spot doesn't usually produce "we WON by score with 24 planets
vs opp's 1" — that pattern is much more consistent with "the agent
stopped acting partway through and the random opponent meandered."

The accidental smoking gun in our case was seed 80504 ending with
opp=1 planet. The dogpile/strategic-coordination story said "opp
pocket no single source can match" — but a single planet isn't a
"pocket." Once I noticed that mismatch, the timeout hypothesis
walked itself in.

Sub-lesson: a try/except around `from lib.kinematic_table import
begin_turn` masked the missing optimisation completely. The agent
*claimed* it had a 50-100 ms/turn cache; in reality every turn paid
the full physics cost. Silent absence + a comment in the same file
asserting the cache exists is worse than no comment.

Sub-sub-lesson: the wave-attacks null (0/16 vs baseline, same as
before the fix on that rung) is the strongest evidence we have right
now that the rung-3 gap is positional/tempo, not firepower. The
picker has no opponent model and no goal-directed planning. No amount
of action-space expansion in the chooser will close that gap.

This is a real PI-level decision point for the next session:
keep climbing lagrange_simple by porting baseline's JOINT, or pivot
the session to the baseline lineage where the live ladder evidence
already lives. The "simplest Lagrangian agent" succeeded at its
stated goal (rungs 1+2 at 16/16) but its architectural ceiling
against a strong opponent is structural.
