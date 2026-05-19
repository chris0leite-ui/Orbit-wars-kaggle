# Is the per-class share gap causal or a side-effect?

> Logged 2026-05-19, after the
> `audit/2026-05-19-archetype-per-planet-class.md` rollup. This
> question is the load-bearing risk in
> `knowledge-base/concepts/per-class-priority-prior.md` — if the
> answer is "side-effect," the prior boosts a symptom and won't
> help winrate.

The audit shows top-10 spends +10 percentage points more of its
fleet share on `low_prod_rotating_inner` planets than we do.
Three competing explanations:

1. **Causal — those planets are undervalued by production alone.**
   Low-production-rotating-inner planets are cheap to capture
   (low garrison), close to home (fast arrival), and rotate so
   you can hit them from many angles. They compound early. Top-10
   gets there first; we don't. If true, our prior should boost
   them, and winrate moves.
2. **Spillover — those planets are by-catch of top-10's main play.**
   Maybe top-10 sends large fleets at high-production planets and
   the spare ships incidentally collect low-prod inners on the way.
   The class isn't valuable in itself; it just happens to be on
   their flight paths. If true, our prior boosts a symptom.
3. **Sample artifact — the +10 pp signal shrinks with more data.**
   16 cells averaged 2 top-10 games each (one cell had 6). With
   tighter confidence intervals the effect might be much smaller.

How to disambiguate (in priority order):

- **Ablate.** Ship the prior with `lambda_alpha = 0`; if winrate
  matches the current submission within noise, the prior code is
  neutral and we can sweep `lambda_alpha`. Confirms the prior at
  least doesn't break anything before we ask whether it helps.
- **Counterfactual sim.** Take a top-10 replay; force the focal
  agent to skip every `low_prod_rotating_inner` launch in the
  first 30 turns; re-run via the simulator with the same opponent
  policy; measure win-rate drop. If the drop is large, the class
  is causally important (explanation 1). If the drop is
  ≈0, it was by-catch (explanation 2). ~10 min per replay × 8
  classes × 30 games is too expensive for v1 but worth doing once
  the closed-form prior is on the panel.
- **Re-audit on a bigger top-10 corpus** (next pull from
  `audit/external/replays`). If the +10 pp signal stays after
  doubling the sample, explanation 3 is ruled out.

Open until the v1 prior ships and we read the panel result.
