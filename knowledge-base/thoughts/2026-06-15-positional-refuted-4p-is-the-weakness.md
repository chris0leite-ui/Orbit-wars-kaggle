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

---

## CORRECTION (2026-06-16) — 4P is NOT a weakness (baseline misread)

I compared 4P winrate (0.40) to 0.50. Wrong baseline: in a 4-player
free-for-all the average/random winrate is **0.25** (1 of 4), not 0.50.
Combined live data (n=85): **2P 0.70 (baseline 0.50, +0.20); 4P 0.48
(baseline 0.25, +0.23).** Relative to baseline, 4P is about as strong as 2P.
**There is no 4P weakness.** (Third premature-conclusion error of the run —
single game, then 4P-weakness; discipline: state the baseline before claiming.)

## The actual lead: positional objective is 2P/4P ASYMMETRIC

4P A/B triage (candidate vs 3 champions, >0.25 beats champion), n=16:
champ 0.25 (sanity ✓), hold12 0.38, **term12 (positional) 0.44**, hold25 0.19.
So the positional objective that LOSES in 2P (term12: 0.38 there) appears to
WIN in 4P (0.44). If it holds at n=32, the move is **ship-flow in 2P +
positional in 4P** (producer has per-player-count config). Confirming now.
