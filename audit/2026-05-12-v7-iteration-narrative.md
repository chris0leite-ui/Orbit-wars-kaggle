# v7 minimax iteration — narrative (2026-05-12)

> Companion to `audit/2026-05-12-v7-iteration.md` (auto-generated
> results table). This file holds the plain-English story; the
> machine-written file holds the numbers.

## What v7 is

A **1-ply forward-evaluating chooser** that, on every turn, generates
a small set of candidate this-turn action bundles, scores each by
running a K=10 forward simulation under our `fast_sim` simulator with
the opponent modelled by `lib/opp_model.top_tier_mirror_policy` (the
v3.5.1 mirror — best Tier-1 ladder-opponent proxy we have without
training), and picks `argmax` of `(our ships − their ships)` at the
rollout's terminal state.

The v3.5.1 action bundle is always candidate-0. If no other candidate
scores strictly higher, or if the 700 ms watchdog trips, v7 returns
v3.5.1's action verbatim — **the parity floor is preserved by
construction.**

This is "minimax" in the loose sense:
- **Min** = we model the opponent (Tier-1 mirror) doing *its* best
  this turn — the worst-case-for-us first move under that model.
- **Max** = we pick the candidate that maximises the score after K
  turns of self-play continuation.
- Both players play `top_tier_mirror_policy` for turns 2..K.

## Why now

`fast_sim` (committed `d054f18` earlier this session) made rollouts
**183× cheaper** than the env-clone path the Phase 2 probe used. The
v3_lookahead MVP (audit/2026-05-11-v3-lookahead-mvp-parity.md) hit
50/50 vs v2 specifically because **drop-one couldn't propose anything
the incumbent hadn't already considered**. The candidate enumerator
was the bottleneck; the scorer was already oracle-quality. v7
finally pays the per-rollout cost we need to afford additive
enumeration.

## What the five variants test

| Variant | Hypothesis being falsified | Floor (if hypothesis is wrong) |
|---|---|---|
| **v7_0_drop_one** | Whether the new scorer alone (fast_sim + Tier-1 opp) buys lift, holding the enumerator constant from v3_lookahead. | Parity vs v3.5.1. |
| **v7_1_target_swap** | Whether v3.5.1's per-source greedy is ever wrong about the top target. The rollout decides between top-1 and top-2. | Parity (rollout always picks top-1). |
| **v7_2_ship_sweep** | Whether `aggressive_fraction=0.7` is universally correct, or whether some boards reward `0.5` (saturation) / `0.95` (concentrated artillery). | Parity (0.7 always wins the rollout). |
| **v7_3_archetype** | Whether top-10 archetype hot-swapping (baseline / concentrated / saturation / defensive) lifts μ. Strongest hypothesis from the fingerprint analysis. | Parity (baseline always wins) or REGRESSION (the wrong archetype on the wrong board). |
| **v7_4_hungarian** | Whether settle_plan's per-source greedy leaves global-coordination value on the table. | Parity (greedy ≈ global). |
| **v7_combined** | Stack of every passing variant — the final-form candidate if PI authorises a submission. | Parity vs the best individual variant. |

Each is run as a 2P A/B at 12 seeds × both sides (24 games) vs
`submissions/v3.5.1.py`. Gate: **Wilson 95% lower bound ≥ 55%** =
PASS. NEUTRAL if ≥ 45%; FAIL otherwise.

## Reading the numbers (decision rubric)

- **≥ 1 variant 2P PASS** → bundle that variant; run 16-seed 4P FFA
  panel; surface to PI. **No `kaggle competitions submit` runs from
  this branch** (Rule 1).
- **All variants NEUTRAL** → re-run the strongest candidate at K=20
  to test whether short rollouts are starving the lookahead. Likely
  next-session work.
- **All variants FAIL** → strong structural evidence the 1-ply form
  is wrong shape. Pivot hypotheses:
  - The `ship_totals` scoring head over-rewards offence and ignores
    defensive value (planets-not-lost). Switch to
    `Σ planet.production × is_owned_at_K` instead.
  - The opponent model is too symmetric with the baseline (both v3.5.1)
    so the rollout has no exploitable gradient. Tier-2 trained opp
    model would break the symmetry.
  - Depth-2 minimax (width-3 beam) is needed to capture 2-move
    sequences.

## Discipline checklist

- Rule 1: No `kaggle competitions submit` from this branch. ✓
- Rule 18: B.3 leaf claimed in ISSUES.md as `wip` on this branch. ✓
- Rule 2: 12 seeds × 6 variants × ~50 s/game / 4 workers ≈ 30 min.
  Under the 1 h per-probe cap. ✓
- Rule 16 Q6: scoring head = ship_totals delta at K=10 rollout
  (proxy); comp metric = win/loss. Phase 2 audit established AUC ≈
  oracle for this proxy at K=50; v7 uses K=10 per the v3_lookahead
  MVP precedent. Alignment = approximate. ✓
- Rule 21: 5 variants + combined = family falsification covered. ✓
- Rule 31: variants run sequentially in the loop. ✓

## What this session ships (regardless of sweep outcome)

- `lib/v7_search.py` — reusable enumerator + scorer.
- 5 ablation agents + 1 combined.
- `scripts/run_v7_ablation.py` — autonomous loop.
- `scripts/bench_v7.py` — p95 turn ms gate (already PASSED for all 6
  variants at p95 < 410 ms; well under the 800 ms safety budget).
- 12 sanity tests in `tests/test_v7_search.py`.

The framework itself is the load-bearing output — even if every
variant lands NEUTRAL, the next session can iterate on top of v7
(K=20, depth-2, alternative score heads) without re-building.
