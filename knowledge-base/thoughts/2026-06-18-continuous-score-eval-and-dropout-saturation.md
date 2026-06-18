# 2026-06-18 — Continuous-score evaluation + dropout saturation (confirmed at n=40)

## What I set out to do
The dropout A/B table ranked variants by binary win/loss over 28 maps (base 15,
incentive 13, winprob 12/11, deeper 6). PI's instinct: win/loss throws away most
of the signal — a map won 23786-to-456 and one won 510-to-509 both count "1", so
a 1-2 map swing is indistinguishable from noise. Build a CONTINUOUS-score
evaluation and re-run the plan's variants with the better feedback.

## What "continuous score" is here
The engine decides the winner by `argmax_i scores[i]`, where `scores[i]` = total
ships across player i's planets + fleets at game end (orbit_wars.py:703-715). The
natural continuous relaxation is the normalised ship margin

    margin = (focal_ships − best_rival_ships) / (focal_ships + best_rival_ships)  ∈ [−1, 1]

Its SIGN reproduces win/loss exactly; its MAGNITUDE measures dominance. Built:
- `scripts/_continuous_game_worker.py` — plays ONE game in a fresh subprocess
  (honours the env-leak rule: producer_plus bundles set knobs via
  os.environ.setdefault, which leak across variants in one process), emits the
  margin + win + reward + timing as JSON.
- `scripts/continuous_ab.py` — runs every variant on the SAME seed+seat as base
  (paired), reports win-rate Wilson CI (old coarse signal) AND mean margin +
  paired Δmargin-vs-base with a bootstrap CI and sign test.
- `tests/test_continuous_ab.py` — stats + the invariant `sign(margin)==engine win`
  (checked against the live log too).

## Result (n=40 paired, seeds 5000-5039, vs Producer V2)
| variant | wins | Δmargin vs base [boot CI] | maps changed |
|---|---|---|---|
| base | 21/40 | — | — |
| more_sims4 | 20/40 | −0.051 [−0.15, 0.00] | 2/40 |
| incentive | 20/40 | −0.050 [−0.15, +0.00] | 3/40 |
| winprob γ=0.5 | 19/40 | −0.100 [−0.25, +0.00] | 4/40 |
| winprob γ=1.0 | 19/40 | −0.100 (≡ γ=0.5 on all 40) | 4/40 |
| deeper_h30 | 9/40 | −0.601 [−1.00, −0.15] p=0.01 ✱ | 23/40 |

## What the better feedback actually changed
It did NOT overturn the conclusion (nothing beats base). It changed the STORY:

1. **The refinements are inert, not regressions.** Each changes the played game
   on only 2-4 of 40 maps; otherwise byte-identical to base. The binary table's
   "incentive worse / winprob worse" was over-reading a 2-map swing — paired
   margins put every refinement's Δ CI across 0. winprob γ=0.5 ≡ γ=1.0 on ALL 40
   maps: the lead-scaling never crosses a decision boundary.

2. **Only deeper-horizon is a statistically real effect**, and it's strongly
   negative (the one CI that excludes 0). Confirms "catastrophic."

3. **Base is parity with V2, not ahead.** Margin +0.051, CI [−0.26, +0.36]
   straddles 0. dropout_repl ≈ coin-flip vs V2 — and BELOW the live
   least_resistance champion (21/32 ≈ 65% vs V2). So fork (a) "ship the cheap
   replacement" would regress the live agent. That kills (a) as a ship.

4. **Mechanistic confirmation of saturation.** The perturbation is too thin a
   layer to express a measure/risk refinement — it washes out before reaching
   the one-ply chooser. Strongest evidence yet that the binding constraint is the
   value function, not the drop measure.

## Methodology notes (for reuse)
- 37/40 of these wide maps end in elimination (margin ±1) — the outcome is
  genuinely bimodal, so the continuous score's variance-reduction was MUTED on
  this distribution. Its payoff here was the clean PAIRING ("≤4 maps change") +
  deeper-horizon significance, not tighter margin CIs. On a map set with more
  step-capped (timeout) finishes the margin would do more work — worth picking
  seeds that produce close games when the continuous signal is the point.
- torch is NOT in requirements.txt; had to `pip install torch --index-url
  .../cpu` on a fresh container. Add to SETUP if the producer line continues.
- Producer V2 re-pulled via `kaggle kernels pull slawekbiel/the-producer-v2`,
  notebook `%%writefile` cell extracted to
  `audit/external/agents/slawekbiel_the-producer-v2/main.py`.

## The fork, with the new evidence
The plan's open decision was (a) ship the cheap opp-model replacement vs (b)
commit to the dropout-NATIVE rebuild. The continuous A/B resolves it:
- (a) is OUT on merit — base is parity-with-V2, worse than the live champion.
- (b) the native rebuild (ensemble of stochastic flip-hazard rollouts, value =
  mean/CVaR over a distribution with real support; `state/DROPOUT_NATIVE_DESIGN.md`)
  is the only path where measure/risk refinements could get traction — precisely
  because the bolt-on's failure is that its 2-point perturbation has no support
  to refine. Phase A is a hard kill-gate.
- Or drop the dropout line and bank the better evaluation harness, which is
  reusable for ANY agent A/B.

Surfaced to PI for the fork sign-off (working-mode step 5).
