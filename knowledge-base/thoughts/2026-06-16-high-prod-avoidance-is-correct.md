# 2026-06-16 — "Why do we avoid high-prod planets?" → because it's CORRECT

PI watched a live 2P loss (vs dungcho, seed 854925794) and asked why we avoid
the high-production planets. Investigated end-to-end; the answer closes the
economy/positional lever for good.

## Why we avoid them (verified)
Reproduced seed 854925794: prod-3 planets 8 & 9 (36 garrison, ~22 away, NOT
sun-blocked, in-shortlist) sit neutral under `seq_strength` (h18). Cause: the
18-step ROI gate. Capturing planet 8 costs ~37 ships; over 18 steps it pays back
~3×12 ≈ 36 → nets ~0 → under the 1.5 fire threshold → never fired. Producer
can't see it's worth 360+ over the 185-step game.

## Is it a bug? NO — forcing the grab LOSES
`expand` (h30) DOES grab planets 8 & 9 (5 high-prod held vs seq's 2). But:
- Head-to-head, **no timeout**, 2P, n=32, deeper horizon vs h18:
  **h24 0.53, h30 0.19, h40 0.09.** Deeper horizon loses badly on STRATEGY.
- Ladder corroborates: `expand` (h30) = 961 vs `seq_strength` ~1188.

So the avoidance is a **feature**: taking high-prod planets over-commits ships to
long-payoff captures that leave us thin/exposed; a real opponent punishes it.
The 18-step horizon is well-tuned (sweet spot ~18–24; deeper over-extends). The
dungcho loss was a one-off where the grab happened to pay.

## The economy/positional lever is now CLOSED (5 confirmations)
1. positional objective 2P A/B n=32: 0.38 (loses).
2. `pp_positional` (terminal_prod=12) ladder: ~1060 (underperforms).
3. `expand` (h30) ladder: 961.
4. deeper-horizon head-to-head no-timeout: h30 0.19, h40 0.09.
5. (4P positional was the only marginal-positive: 0.38 vs 0.25 — shipped gated
   as pp_pos4p, low-risk.)

**Lesson:** "grab the economy / value long-term production" is the wrong
intuition for this game — ship+exposure cost dominates the production gain
against a real opponent. Producer's myopia is correct. Do not re-chase this.
