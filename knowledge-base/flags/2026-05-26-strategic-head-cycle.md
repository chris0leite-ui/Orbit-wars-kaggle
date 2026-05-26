# 2026-05-26 — flags for next-session PI review

## Calibration sacrosanct after Phase F + fcaf414

Empirically established working configurations should be treated as
SACRED until evidence proves otherwise:

- **Phase F**: 75% 2P / 10% 4P (trickle in 4P; otherwise calibrated)
- **fcaf414**: 35% 4P / 62.5% 2P (4P trickle reduced via symmetric Term A + max-of-opps)

Any change to the leaf/chooser/proposer/rollout that touches these
configurations needs an A/B panel BEFORE merge, not after. Today's
loop made 4 "model-correct" changes without an empirical guard and
each broke calibration in a different way.

## Live ladder regression flag

Sub 53032723 (baseline_unified, μ=984.1) is in the live rolling pair.
Floor dropped from 1113.2 → 984.1 (−129 μ). The pair is:
- 53024913 (ev_per_ship, μ=1135.4, older half)
- 53032723 (unified, μ=984.1, newer half)

If we submit again, 53024913 drops out. To get rid of 53032723 we
need to submit TWO more agents. Decision: do we want to "burn" two
submissions to evict 53032723, or do we accept the floor at 984 for
2-3 days and submit improvements?

## Trickle-launch problem: STILL OPEN

The original PI question that triggered the strategic-head work
("we lose in 4P games and launch small waste fleets") is not solved.
Today's `4ad192f` restoration matches fcaf414's behavior, which still
has the trickle pattern (just less than Phase F).

The right next attack is probably: switch to `baseline_ev_per_ship`
lineage (per-ship-efficiency sort is proven at live μ=1135.4 with
75% 2P / 40% 4P panel), and iterate THERE on the trickle.

## Promotion candidates pending PI ratification

Three rules drafted in `audit/2026-05-26-postmortem-*.md`. Need PI
yes/no/edit on each:
- Rule 48: Don't "fix" calibrated heuristics without empirical guard
- Rule 37 addendum: HARD REVERT at N=3 consecutive falsifications
- Rule 49: When 2+ axes changed in broken commit, `git revert`, not axis-by-axis

## Rule 36 self-flag

This is the only `knowledge-base/flags/` entry today. Yesterday and
earlier days have entries from other sessions. Rule 36 was effectively
ignored during today's iteration loop until wrap-up. The WRAPUP step 4c
check is currently WARN-only; it didn't trigger an emit-during-session.
Promotion candidate (deferred): make Rule 36 mid-session check
HARD-blocking on commits, not just at wrap-up.
