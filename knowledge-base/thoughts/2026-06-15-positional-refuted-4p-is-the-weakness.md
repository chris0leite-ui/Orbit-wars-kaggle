# 2026-06-15 (overnight) — Positional objective refuted at n=32; 4P is the real weakness

## Positional objective: REFUTED at scale (the single game was a fluke)

The single game (positional beats ship-flow 26-0, seed 11) and a 6-game
dbg_loop (seeds 11/29/53) were **favorable-seed cherry-picks**. At **n=32
(16 seeds × 2 seats, clean single-thread, head-to-head vs the ship-flow
champion on the exact same engine)**:

| objective (B) vs ship-flow champion (A) | winrate |
|---|---|
| flat positional (terminal_prod=12) | 0.38 (12W-20L) |
| neutral-gated (term=12) | 0.19 |
| hold_value gate (12) | 0.06 |
| hold_value gate (25) | 0.00 (0-32) |

**More economy/positional weighting → strictly worse.** Ship-flow is genuinely
the better objective. This confirms the original ladder refutation of
`terminal_prod=12`, rigorously. **Lesson (again): never claim a lift from 1 game
— n≥32 balanced, every time.**

Consequence: **`pp_positional` (sub 53722697, submitted on the single-game
evidence) will likely settle BELOW our champion.** First move tomorrow: restore
the rolling pair (re-submit a settled config) rather than leave it.

## The real weakness: 4-player (grounded, current field)

Pulled the live-ladder episodes for `pp_seq` (53708787, seq_strength champion,
current field), 30 episodes:

- **2P: 15/20 = 0.75 winrate** — strong heads-up.
- **4P: 4/10 = 0.40 winrate** — we lose 60% of free-for-alls.

4P is the clear weakness and a real μ lever (≈⅓ of episodes here). n=10 4P is
small (wide CI) — confirming with more data (pp_ws episodes). Hypotheses to
check in the 4P loss replays: gang-up / target-selection in the FFA objective
(seq_strength runs `ffa_score`+`ffa_weights=strength`), early elimination,
or under-expansion specific to 4P crowding.

## Status of overnight grind
- Objective lever: dead (refuted). Simple agents / baseline / search: all lose
  to producer (earlier). Producer ship-flow ≈ 1190 current field is our ceiling
  via known levers.
- New lead: **fix 4P play.** Mining the 4P losses next.
- Knob sweep (denial/opening/recapture/veto-upsize toggles vs champion) running
  to confirm they're dead (they're OFF in the champion for a reason).
