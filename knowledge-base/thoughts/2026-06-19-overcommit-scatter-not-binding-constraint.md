# 2026-06-19 — over-commit / scatter is real but NOT the binding constraint vs strong opponents

## PI observations (ladder replays, plain English)
Across four replays the PI flagged the same behaviour, sharpening each time:
1. "we launch from large planets to attack small planets and lose both"
2. "too hyperactive, throwing too many fleets — prefer fewer, keep flexible"
3. "we expose our planets by attacking, not holding defense — short-sighted"
4. "our attacks are waste, not aligned — better ONE aligned strong attack; this
   should emerge"

Seeds: 25260880 / 788834306 (4P), 1576908455 (2P).

## Diagnosis (confirmed in code + reproduced, Rule 38)
`least_resistance`'s greedy commit loop scores each capture with the producer's
ONE-PLY garrison-flow scorer, which models production + in-flight combat but NO
new opponent launches. So draining a source to fund a capture looks free (the
counterattack that flips the emptied source is invisible), and reserve ships left
at home have zero projected value (no reward for defence) -> the loop scatters
minimum-sized captures across many fronts and strips sources bare.

Quantified scatter (vs Producer V2): in the 2P LOSS our fleets averaged 18.8
(median 15, max 52) vs V2's 31.6 (median 21, max 103), and 39% of our launches
were < 10 ships; in a 4P WIN we concentrated (39.2 vs 34.3). So concentration
correlates with winning — the observation is real.

## What was tried — FIVE mechanisms, all default-OFF, all A/B-inert vs the strong panel
Paired continuous-margin A/B, n=40, fresh-process-per-game (scripts/continuous_ab.py):

| mechanism | type | knob | 4P Δmargin vs live |
|---|---|---|---|
| hold-source reserve | value/sourcing | LR_HOLD_SOURCE | +0.05 (ns); 2P −0.10 |
| dropout-perturbed score | value | LR_DROPOUT_SCORE | +0.00 (8/40 maps) |
| response-veto (real mirror reply) | value | LR_RESPONSE_VETO | +0.05 (ns); timing tail |
| concentrate (blanket size surplus) | generation | LR_CONCENTRATE | m1.0 −0.05; m2.0 −0.50 p=0.03 ✱ WORSE |
| concentrate (SELECTIVE, top-N target) | generation | LR_CONCENTRATE_FRONTS | sel_f1 −0.10; sel_f2 +0.00 |

Every one lands in the noise (best ±0.05) or is significantly worse. Per-seed
traces showed real behaviour changes (fewer lost planets, bigger fleets), and
single rendered games occasionally won — but those were seed-luck: each mechanism
changes only ~4-10 of 40 maps, and the aggregate margin shift is ≈0.

## The load-bearing conclusion
The over-commit / scatter / lose-both behaviour is REAL and visible, but it is
NOT the binding constraint on performance vs strong opponents (V2, the
V2+Roman+konbu17 panel). Fixing it five different ways — including the
*generation* lever (concentration) that historically worked (take-and-hold
14->21) — does not move aggregate margin. Strong opponents beat us through other
channels; preventing the self-inflicted blunder just changes which map we lose.
This matches the prior dropout-family plateau: the binding constraint is the
strategic value function / move quality, not these tactical blunders.

## The ONE untested angle
Every A/B here is vs STRONG opponents. The PI's replays are WEAK-opponent ladder
games (e.g. Pushkeshwar Singh @992), the one place lose-both / scatter reduction
should pay. The local panel structurally cannot see that. Only a ladder probe can.
Caveat also: a broken (ERRORED) depth-3 agent currently sits in one of our two
final-eval rolling-pair slots and must be evicted regardless.

## Small-n discipline (re-learned)
Selective concentration looked +0.10 at n=20 and regressed to ≈0 at n=40 — the
exact small-n overconfidence improvements.md warns about. Do not read Δmargin at
n<~32; single rendered games are for bug-finding, never ranking.

## Pointers
- Code: agents/least_resistance/main.py — gates `_hold_source`, `_dropout_score`,
  `_response_veto`, `_concentrate` (+ `_lr_drop_status`), all default-OFF,
  OFF-path byte-identical (entry test green).
- Harness: scripts/continuous_ab.py variant sets holdsrc/dropscore/veto/
  concentrate/selconc; scripts/_trace_overcommit.py reproduces the failure;
  /tmp/scatter.py measures fleet-size distribution.
- A/B logs: audit/continuous-ab-{holdsrc,dropscore,veto,concentrate,selconc}-4p.jsonl.
