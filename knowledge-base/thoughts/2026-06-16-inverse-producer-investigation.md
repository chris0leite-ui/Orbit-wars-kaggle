# 2026-06-16 — The inverse producer: faithfully built, but short games can't measure it

## What the PI asked

Iterate, overnight, on an "inverse producer": like our producer, but instead of
assuming a static opponent it assumes the opponent is *also* a producer — one
that picks the moves maximising its own ships over the next 18 steps while
assuming *we* sit still — and then we play our best response to that. Test only
in short 200-step games, ~4 seeds.

## The first finding: it already exists

The inverse producer is the "opponent-projection" mechanism we already have
(`PRODUCER_PLUS_OPP_PROJECTION`). Each turn it runs our own producer's planner
from the opponent's seat with us assumed static, reads off the launches that
opponent would fire over its 18-step horizon, and injects them so every one of
our candidate moves is scored against "the opponent does exactly those things."
That is the PI's description, mechanism-for-mechanism. It is even switched on
inside our current best ladder agent.

So "iterate on the inverse producer" became "measure and try to improve this
mechanism," not "build it."

## The core result: in 200-step games it ties the plain producer

Inverse vs the plain (static) producer, 4 seeds, both seats, 200-step cap:
**average score difference exactly zero, same as the static-vs-static control.**
- 2 of 4 seeds: perfect mirror draws.
- 2 of 4 seeds: decided entirely by which side of the board you start on — the
  same side wins whether or not the opponent model is on.
- No seed was won at BOTH seats by either agent.
Adding the multi-size option or a deeper horizon on top made it measurably
*worse* in short games; adding a denial/"race the opponent" bonus also tied.

## Why — two reasons, one shallow, one structural

1. **Shallow:** the opponent model barely changes our move. Measured on a shared
   observation stream, it is silent for the entire opening (~first 45 turns —
   the opening is uncontested, so knowing where the opponent expands doesn't
   change where we expand) and changes our chosen move only ~1 turn in 6 even
   later. 85–90% of the time the inverse producer plays identically to the plain
   producer.
2. **Structural (the real lesson):** the boards are 4-fold symmetric for
   fairness. Two equally-strong agents on a symmetric board, played at both
   seats, are *forced* to an average score difference of zero — swapping seats
   just flips the result. A 200-step head-to-head literally cannot reveal a small
   edge for one agent; it can only catch large regressions. And against anything
   weaker (our old v7 agent), both versions just blow it out 8/8. So there is no
   200-step instrument where the edge is both present and cleanly measurable.

The one positive flicker: vs the (non-mirror) v7 agent the inverse producer won
by a bigger margin than the plain producer (+1750 vs +1577 median), but both win
8/8 and the margins are blowout-noisy — a hint, not proof.

## The full-length verdict (the PI's chosen next step)

Because short games can't separate them, the PI chose to run the full-length
game (to 500 steps / elimination), where a small per-turn edge can compound and
break the symmetric tie. Result, inverse vs static producer, 16 seeds × 2 seats:

> _[VERDICT PENDING — fill from /tmp/inv/full_test.txt]_

## Practical notes / flags

- The opponent model roughly doubles per-turn cost. Mid/late-game turns hit
  1.8–2.6 s locally; a full 500-step mirror game takes minutes, and 32 of them
  ~45–50 min. Future "long enough to break symmetry but still fast" tests should
  use a ~300-step cap, not 500.
- Tooling added this session: `scripts/short_margin_ab.py` (step-capped A/B by
  competition score margin, both seats), `scripts/inv_decision_diff.py`
  (decision-level opp-model on/off diff), and a matched `bare` bundler variant
  (producer_plus with all flags off = the static producer, action-identical to
  the vendored one). Audit: `audit/2026-06-15-inverse-producer-short-game-study.md`.
- Fixed: `torch` was missing from `requirements.txt` (hand-installed every
  session); added with the pytorch CPU index.
