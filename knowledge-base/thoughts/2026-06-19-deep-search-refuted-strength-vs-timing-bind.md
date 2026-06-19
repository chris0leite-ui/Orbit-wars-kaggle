# 2026-06-19 — deep search (depth-3) refuted at n=32: parity, and a strength-vs-timing bind

## Why we looked again
Depth-3 deep search was the strongest-looking lever of the prior session: the
only config to "beat the producer wall" (binary 14/28 → 17/28, margin claimed
monotone in depth). It ERRORED on the ladder (sub 53836276) on a per-turn
timeout. The plan was: make it anytime-safe and ship the strength.

## What the timing investigation found (Rule 38, reproduced)
The depth-3 per-turn cost is dominated by the **rollout opponent model**, called
O(depth × seats × candidates) per turn. With the accurate producer mirror
(`LR_DEEP_OPP=0`, ~10-50+ ms/call) a single heavy-board turn spiked to **1581 ms**
even after adding anytime deadline guards — because the unguarded `opp_now`
precompute plus the iterdeepen ply-1 setup loop already exceed the 1000 ms wall.
The cheap `lite_greedy` opponent (`LR_DEEP_OPP=1`, ~1-2 ms/call) drops max turn to
**272 ms** at a 200 ms budget — robustly wall-safe.

So: only the cheap-opponent config can actually ship under the wall.

## The strength measurement that killed the line (n=32 paired, 2P vs Producer V2)
`scripts/continuous_ab.py --set deepsearch --players 2 --seeds 32`
(`audit/deepsearch-strength-2p.jsonl`). All three arms stack on the live champion
(take-and-hold ON); only the rollout differs.

| arm | knobs | wins | mean margin [95% CI] | paired Δmargin vs live |
|---|---|---|---|---|
| `live` (2-ply) | depth 0 | 19/32 | +0.188 [−0.158, +0.533] | (reference) |
| `d3_prod` | depth 3, producer opp | 18/32 | +0.125 [−0.224, +0.474] | **−0.062 [−0.500, +0.375], 6up/7dn, p=1.00** |
| `d3_lite` | depth 3, lite_greedy opp | 8/32 | −0.500 [−0.805, −0.195] | **−0.688 [−1.125, −0.250], 2up/13dn, p=0.01 ✱** |

- **depth-3 with the ACCURATE opponent = dead parity with 2-ply.** Zero lift, CI
  straddles 0, p=1.00. The old 17/28 was small-n noise (n=28 binary, below the
  Rule 45 bar; my n=32 paired test shows no margin shift).
- **depth-3 with the only WALL-SAFE opponent = significant regression** (−0.688,
  CI excludes 0). A weak rollout opponent means deep search optimizes against a
  fantasy and plays worse than not searching at all.

## The load-bearing conclusion
Deep search is caught in a **strength-vs-timing bind** and is **not a strength
lever vs strong opponents**:
- the strength (such as it is) needs the accurate opponent, which is too slow for
  the wall;
- the opponent that fits the wall throws the strength away;
- and even the slow, accurate version is only PARITY at n=32 — adding plies to the
  one-ply garrison-flow leaf evaluator does not convert to wins.

This is the SAME conclusion as the over-commit/scatter line
(`2026-06-19-overcommit-scatter-not-binding-constraint.md`): the binding
constraint vs V2 is the **strategic value function / move quality**, not search
depth and not tactical blunders. Deeper search over a flawed leaf evaluator just
costs time. Per Rule 40, the right next lever is a better leaf/target model, not
more plies.

## What we kept (no harm, default-OFF)
The anytime guards ARE committed (`agents/least_resistance/main.py`:
intra-rollout deadline in `rollout_value()`, candidate[0] guard in the fixed-depth
loop) + `tests/test_deep_search_anytime.py`. They are correct and make the deep
path wall-safe IF ever revisited, and the default-OFF path is byte-identical. We
do NOT bake depth-3 ON. The errored depth-3 slot was already evicted last session.

## Small-n discipline (re-learned, again)
Partial eval at n≈11 showed d3_prod 7/11 "ahead"; at n=32 it was 18/32, tied. Do
not read a lift at n<32. improvements.md's small-n overconfidence warning bit twice
this week (selective concentration +0.10→0; deep search 7/11→parity).

## Pointers
- Harness: `scripts/continuous_ab.py` `"deepsearch"` variant set (live / d3_lite /
  d3_prod). Report: `--report-only audit/deepsearch-strength-2p.jsonl --ref live`.
- Timing repro: `/tmp/repro_deep_timing.py` (1581 ms producer-opp vs 272 ms lite).
- Code: `_deep_pick` / `rollout_value` / `_deep_opp` / `_iterdeepen` in main.py.
