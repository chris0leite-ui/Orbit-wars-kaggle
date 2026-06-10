# 2026-06-10 — 4P loss anatomy: deep mining of the 195-episode corpus

Corpus: `audit/live-episodes/53384340/` (sub 53384340, multi_opp_def,
pre-FFA-fix). Tools: `scripts/mine_4p_carving.py`,
`scripts/mine_4p_economy.py`. 116 usable 4P episodes (31 wins / 85 losses).

## Finding 1 — "drained then carved" is NOT the loss signature

Of all captures of our planets, the fraction where we had launched ≥5 ships
OUT of that planet within the prior 8 steps is **60% in losses AND 60% in
wins**. Median garrison the step before a fall: 12 (losses) vs 15 (wins).
Self-draining churn is how this game is played, not what kills us. A
threat-aware garrison-floor mechanism would target a non-separator —
deprioritized.

## Finding 2 — losses are decided in the step-20..80 brawl window

Median trajectories (loss games):

| step | our ship rank | our ships | winner ships | our prod | winner prod |
|-----:|--------------:|----------:|-------------:|---------:|------------:|
| 20   | **1** | 90  | 90  | 9  | 9  |
| 40   | 2 | 170 | 205 | 11 | 14 |
| 60   | 3 | 185 | 303 | **10 (declining)** | 18 |
| 80   | 3 | 139 | 402 | 8  | 22 |
| 120  | 3 | 52  | 675 | 2  | 31 |

We are rank 1 at step 20 **in losses too** — the opening is fine. Production
peaks ~step 40 and then declines (we are net-losing planets from ~40 on)
while the eventual winner's production doubles. Elimination at median 120 is
cleanup, not cause.

## Finding 3 — neutral expansion is NOT the separator

Our cumulative neutral captures stall at 3 in wins AND losses (winner takes
4). The separator is enemy-capture rate in the brawl window (winner 8 vs our
5 by step 80) and net planet retention. Opening-bonus-style expansion
mechanisms target a non-separator — consistent with their 2P nulls.

## Finding 4 — multi-front carving is the end state, not the entry point

2+ distinct rivals capture our planets in the final 40 steps in 54/85 losses
(vs 6/31 wins). But ≥3 of our planets falling within the 13-step scorer
horizon happens on only 9% of loss steps (2% win steps) — the 4P defensive
shortlist cap (2) binds rarely and late. Widening it is a minor lever.

## Implication

The decisive axis is mid-early brawl outcomes (steps 20–80): who wins the
contested-frontier exchanges and converts them into compounding production.
Mechanisms aimed at this window, in current priority order:

1. **FFA objective fix** (live as sub 53527125; the mutual-damage-trade bias
   pays exactly in this window) — live A/B reads ~2026-06-11 07:00 UTC.
2. **Multi-tick opponent projection, 4P-only** (`tick4p` bundle variant) —
   the planner is blind to rival launches beyond the current tick, so it
   neither anticipates incoming brawl waves nor times its own. Standalone-4P
   was never measured (the 53390700 live regression was composed with
   recapture penalty and ran K_2P=2 in the 2P games that dominated that
   eval). Measured today with clean_ffa — result appended below.
3. Force-concentration in 4P brawls — hard 2P null; only if 1–2 both null.

## Measurements (appended as they complete)

- **tick4p vs 3× vanilla producer, seeds 0–31: 10/32 (31.2%), Wilson
  [0.180, 0.486] — NO LIFT** vs baseline 13/32. Per-seed log:
  `audit/pools/2026-06-10-tick4p-vs-3xproducer-n32.log`. Suspected cause
  beyond plain null: the multi-round mirror re-plans each round from the
  same tick-0 board without debiting ships committed in earlier rounds, so
  rivals are projected attacking with the same ships up to 3×. Phantom
  aggression → the planner turtles. A budget-debited multi-tick would be a
  different mechanism; the cheap standalone variant is dead.
- **reinforce_deficit vs 3× vanilla producer, seeds 0–31: 9/32 (28.1%),
  Wilson [0.156, 0.454] — NO LIFT** vs baseline 13/32. Winning seeds
  5, 12, 13, 19, 20, 24, 27, 30, 31; tick4p's winners were 12, 13, 16,
  17, 18, 20, 21, 24, 27, 31 — six of the same seeds win under BOTH
  variants, so the map/seat draw dominates this pool and the variants
  move few games. Per-seed log:
  `audit/pools/2026-06-10-reinforce-deficit-vs-3xproducer-n32.log`.
  Baseline per-seed reconstruction (deterministic re-run) appended when
  complete.
