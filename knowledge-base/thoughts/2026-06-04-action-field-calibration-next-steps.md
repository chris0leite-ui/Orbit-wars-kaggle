# Action-space field — where we are + next steps (2026-06-04)

## The pivot that worked

The PI reframed the "potential field" idea: the field is **not** a spatial pull
over planet positions — it lives in the **action space** (the moves we can make
from the current state). We rebuilt the probe (`agents/protoflow/main.py`) so
the field is the champion's `propose()` feasible-action set, scored by an
importance height (production x hold-horizon x location/flip x winnability x
near-bias) and selected greedily under a per-source defensive-reserve budget,
plus a same-arrival cohort supplement for defended targets.

**Result vs light-greedy (6 seeds, n=12):** the reframe fixed the spatial
version's two diseases.

| | spatial (planets) | action-space |
|---|---|---|
| win | 8/12 (67%) | **10/12 (83%)**, Wilson-lo 0.55 |
| far shots (overreach) | 18% (up to 34%) | **1%** |
| idle | 70% | **10%** |
| end planets (mean) | 10 | **20** |

In action space a far/infeasible move simply is not in the field, so the
overreach and the inertia disappeared on their own. The PI's insight is
validated on the discipline axis.

## The remaining calibration gaps (found by the synthetic harness)

`scripts/protoflow_calib.py` builds minimal hand-made states and prints the
ranked action field + what the agent emits (via `proto.get_last_field()`). One
run exposed three concrete bugs:

1. **Dribbling has no "hold" rival.** Physics: fleet speed rises with ship count
   (`1 + 5*(ln(ships)/ln1000)^1.5`), so a small fleet is a SLOW fleet. In S1 a
   3-ship launch (speed 1.32) is the only field entry and gets emitted. There is
   no representation of "hold and accumulate a faster, decisive fleet" as a rival
   action with its own value, so the slow trickle wins by default. Light-greedy
   tiny-fleet rate is still 12% (up to 21%).

2. **Friendly planets pollute the offense field.** In S2 our own planets appear
   as targets (reinforce candidates, location-weight 1.5) and rank #1-2
   (importance 637), so the agent trickles 3 ships BETWEEN its own planets
   instead of massing on the real defended target. Reinforce/defense must be a
   separate, threat-gated track, not competing in the offense importance ranking.

3. **Convergence is shadowed by wait-candidates.** In S2 the defended neutral
   (garrison 30; neither 18-ship planet can solo-take it) had a wait-then-fire
   solo candidate (33 ships after ~5 turns of accumulation) in `propose`. That
   put it in `solo_tids`, which EXCLUDES it from the cohort supplement — so the
   2-source same-arrival cohort never formed and the target fell through
   entirely. The supplement must trigger whenever no *affordable-now* solo
   exists, regardless of wait-candidates.

## Next steps (calibration, in priority order)

The PI's method: keep using **simple opponents + synthetic situations** to learn
how to define/calibrate the field. Do NOT jump to champion/Producer yet.

1. **Split defense out of the offense field.** Stop treating our own planets as
   offense targets. Defense = a separate threat-gated pass (reinforce only
   planets with real incoming threat, sized to the threat). This removes the
   friendly-reinforce pollution (bug 2).

2. **Add a "value of holding / accumulation" so dribbling is improbable (bug 1).**
   A slow small fire-now launch should only emit if its importance beats the
   value of waiting to mass a faster fleet — compare against the same target's
   best wait-then-fire candidate (per-turn value). If waiting dominates, HOLD.
   This is the calibration that makes small/slow launches naturally low-value,
   exactly as the PI predicted (slow launches "should be not probable").

3. **Make speed/mass first-class in the importance height.** Right now speed only
   enters via eta. Consider: penalize low-speed launches explicitly, and value
   margin (a fleet above the capture floor survives a counter; a barely-floor
   fleet does not).

4. **Fix convergence shadowing (bug 3)** and then decide the cohort-vs-wait
   tradeoff: when neither planet can solo-take a target, is a same-turn cohort
   (mass now) better than one planet accumulating and striking later (mass
   later)? Use a synthetic scenario to calibrate.

5. **Re-validate:** synthetic suite shows the desired property on S1/S2, then
   re-run light-greedy to confirm tiny_frac drops WITHOUT re-introducing idle /
   over-caution. Only after the field is calibrated on synthetic + light-greedy
   do we test vs champion + Producer (where dribbling is punished and convergence
   is actually needed).

## Tools

- `agents/protoflow/main.py` — the action-space field probe (NOT a submission;
  imports lib/* + agents.baseline.proposer directly; not bundled).
- `scripts/protoflow_calib.py` — synthetic-situation harness (`get_last_field()`).
- `scripts/protoflow_probe.py` — light-greedy/champion/Producer A/B with the
  waste metrics (tiny / far / idle / convergence / end-planets / winrate).

## Status

Probe only — nothing touches the live champion or the bundle. The action-space
reframe is validated on discipline (overreach + inertia gone); the open work is
calibrating mass/speed (dribble), defense-split, and convergence so the good
behavior is fully natural before any stronger-opponent test.
