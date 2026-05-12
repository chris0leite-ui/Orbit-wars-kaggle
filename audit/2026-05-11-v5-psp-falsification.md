# v5_psp probe — Sim<K>-filtered action selection on v3+endgame base (2026-05-11)

## Setup

agents/v5_psp/main.py — wraps v4_endgame's end-scenario routing with a
Sim<K> override layer. Candidate set per turn = {v3 incumbent, ROI
sibling}. Score each via lib.lookahead.score_action (K=50, ROI as
rollout policy for both sides). Deviate from v3 only if alternative
beats v3 score by DEVIATION_MARGIN=20 ship-units. TAU_NEAR_TIE=0
(deterministic).

## Results (8 games: 4 seeds × both sides vs v3_snipe)

| Side | seed | steps | psp | v3 | outcome |
|---|---|---|---|---|---|
| 0 | 0 | 500 | +1 | -1 | W |
| 1 | 0 | 500 | -1 | +1 | L |
| 0 | 1 | 275 | -1 | +1 | L |
| 1 | 1 | 227 | -1 | +1 | L |
| 0 | 2 | 241 | -1 | +1 | L |
| 1 | 2 | 196 | -1 | +1 | L |
| 0 | 3 | 200 | -1 | +1 | L |
| 1 | 3 | 237 | -1 | +1 | L |

Total: 1W / 0D / 7L = 12.5% W/D. **Worse than v4_endgame's 37.5%.**

Self-play (2 seeds, 200 steps): P0 wins both — asymmetric (should
draw by symmetry; the asymmetry is policy-mismatch noise compounded
by tight games).

## Root cause: policy-mismatch in Sim<K> rollout

The audit at 2026-05-11-lookahead-phase2-forward-sim.md:137-141
flagged this risk: "the rollout assumes both players use v2 [...] If
Sim<K>'s ranking of candidate this-turn intent sets is robust to
opponent identity, the agent transfers. If it's not, we need a richer
rollout policy."

Mechanism of the regression:

1. PSP rollout policy = ROI (chosen for speed; v3-as-policy was 1.5s+
   per Sim<K=50>, busting actTimeout).
2. Sim<K>(v3-incumbent, ROI rollout): our v3-action then 50 turns of
   ROI vs ROI. v3's aggressive snipe is "abandoned" by the ROI rollout
   that follows — the rest of v3's plan never executes.
3. Sim<K>(ROI-incumbent, ROI rollout): pure ROI-vs-ROI from turn 1.
   Consistent policy → cleaner rollout → often scores higher.
4. PSP picks ROI-incumbent based on the misleading delta, even though
   ROI is empirically WEAKER than v3 in real play.

The DEVIATION_MARGIN of 20 ship-units gets exceeded by this policy-
mismatch noise more often than by genuine signal. The override
mechanism is wrong-way biased.

## What would actually work

Three remediations (none implemented in this session):

1. **v3 as rollout policy with K=5**. Drops K from 50 → 5 to keep
   wallclock budget. Less horizon, more policy-match. Per call:
   2 × 5 × ~30ms = 300ms — feasible.
2. **Multi-policy rollout**: rollout with a mix of {v3, ROI, nearest}
   weighted by historical opp distribution. Approximates true
   ladder.
3. **Self-play-trained policy** as rollout. The real fix per audit
   §137-141 — but requires the RL infrastructure (Improvement F in
   the plan), which is out of 43-day budget.

## Verdict

PSP iteration as a whole: empirical falsification of the Sim<K>-
candidate-filter improvement on top of v3+endgame. v4_endgame remains
the best practical iteration in this branch (37.5% W/D vs v3).

The cannot-lose plan's improvement A (predictive opp modeling) is
not the issue — v3 implicitly handles it. The issue is improvement
B (Sim<K> filter), which suffers from policy-mismatch when the only
fast rollout policies are weaker than the incumbent strategy.

For a future session: implement (1) above (v3 rollout, K=5) as v5_psp_v2;
or skip Sim<K> entirely and pursue improvement F (self-play RL).
