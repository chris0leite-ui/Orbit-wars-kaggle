# 2026-06-19 — the scorer's objective is fine; losses are decided in the step 30→50 window

## Why we looked
Deep search (more plies) was refuted at n=32, and so was scatter-reduction. Both
pointed at "the strategic value function is the constraint." The cheapest version
of that claim: the leaf scorer `_project_value` returns raw **ship** advantage
(`mine - theirs`) at a SHORT horizon (`LR_HORIZON_2P=18`, `LR_HORIZON_4P=13`) with
no new launches. Hypothesis (Rule 40): it's **myopic about territory** — it reads
"ahead on ships" in games we go on to lose because it under-credits planets /
production that pay off after the horizon.

## Test (n=32, 2-ply champion vs Producer V2, /tmp/diag_scorer.py)
Replay each game; at steps 30/50/70 and final record our ship-margin
(our_ships − opp_ships, ~what the scorer optimizes) and planet-margin
(our_planets − opp_planets, ~true standing); group by eventual win/loss.

| checkpoint | wins ship_m | wins planet_m | loss ship_m | loss planet_m | ship-ahead&planet-behind |
|---|---|---|---|---|---|
| step 30 | −0.2 | +0.18 | −0.1 | −0.13 | 0/17 win, 2/15 loss |
| step 50 | +22.0 | +1.00 | −3.5 | +0.00 | 0/17, 0/15 |
| step 70 | +113.9 | +2.18 | −17.8 | −0.47 | 3/17, 2/15 |
| final | +1533 | +19.9 | −2358 | −19.8 | 0/17, 0/15 |

## Conclusion — hypothesis REFUTED
- **Ship-margin and planet-margin move together**, not in opposition. When we
  lose we are behind on BOTH. The predicted "ship-rich, planet-poor in losses"
  divergence essentially never occurs (0–3 of 15–17 in every bucket).
- If anything the **ship margin separates EARLIER and HARDER than planets**
  (step 50: winners +22 ships but only +1 planet). So the scorer's ship-advantage
  objective is a GOOD leading indicator of who wins — not a misranking signal.
- Therefore "value planets/production more" / "lengthen the horizon" is NOT the
  lever. The scorer's *objective* is fine.

## What this leaves (triangulated across three negatives)
The constraint vs strong opponents is NOT:
- search depth (deep-search refuted, n=32 parity);
- tactical scatter (over-commit refuted, n=40 inert);
- the leaf scorer's objective (this test — ship advantage tracks outcome well).

It IS **move quality in the step 30→50 window**: at step 30 the game is ~even
(ship ±0, planet ±0.1); by step 50 it is effectively decided (winners pulled
ahead, losers fallen behind). The decisive divergence is the early midgame, and
since the scoring objective is sound, the gap is in **which moves we generate /
choose in that window**, not in how we score the resulting positions. This is
consistent with prior KB (the opening/expansion race decides the leader; we have
no opening plan — `2026-06-14-why-stuck-...` point 5, `2026-06-15-loss-mining`).

## Next-lever candidates (for PI steer — early-midgame move quality)
1. Diagnose WHAT flips between step 30 and 50 in losses — is it a specific bad
   commit (we drain/overextend), or an opponent expansion we fail to contest?
   (Replay the step-30→50 deltas on the loss seeds.)
2. Opening/early-expansion as a small plan rather than greedy nearest-target
   scoring (KB-recurring; tractable because the opening is predictable).

## Pointers
- /tmp/diag_scorer.py (ship vs planet margin by outcome).
- Scorer: `_project_value` agents/least_resistance/main.py:609-637; horizons L162-163.
