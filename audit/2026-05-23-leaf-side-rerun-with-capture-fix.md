# 4P FFA A/B — favor vs projected_rank_diff_sum, post-capture-fix — TIE

Branch: `claude/strategy-framework-design-OyoYR` @ `28ce9f3`
Date: 2026-05-23
Predecessor: `audit/2026-05-19-projected-sum-4p-ab.md` (pre-fix, TIE at 32 seeds).
Fix under test: `28ce9f3` — projected_rank_diff capture classifier +
projection-transfer debit (worked example: capture frame V_diff bumps
from 0 to +196).

## Tooling

```
python -m scripts.ffa_panel \
    --focals submissions/_ab/favor_v2.py submissions/_ab/projected_sum_v2.py \
    --background submissions/v7_0_drop_one.py submissions/v4_planner.py submissions/v3.5.1.py \
    --seeds 8 --no-rotate-seats --workers 4
```

8 seeds × 1 seat × 2 focals = 16 games total. Focal always seat 0 — seat
bias cancels in the A/B because both focals are evaluated under the
identical seat assignment.

Sample size is intentionally small: PI ratified "we need a large lift,
so 8 games suffice." This panel resolves the binary "knockout lift vs
not"; it does NOT detect a marginal 3–5pp gap.

Bundles re-built from `agents/baseline/` at HEAD `28ce9f3` with
`VALUE_HEAD_CHOICE=0` (favor) and `=3` (projected_sum) respectively.
Both bundles share the post-fix `_per_seat_in_flight_credit` and
`_projected_totals` source. The fix is dead code for favor (head 0
doesn't reach `_projected_totals`); identical source on both bundles is
audit-cleanliness only.

## Result

| Focal | 1st-place | Wilson 95% | p95 turn ms |
|---|---|---|---|
| `submissions/_ab/favor_v2.py` | 5/8 (62.5%) | [30.6%, 86.3%] | 453.6 |
| `submissions/_ab/projected_sum_v2.py` | 5/8 (62.5%) | [30.6%, 86.3%] | 336.8 |

Point-estimate gap: **0.0pp**. Identical winner pattern: both focals win
on seeds {42, 7, 31, 13, 17} and lose on {1, 23, 100}. The "large lift"
gate fails.

JSON: `audit/tournaments/ffa-panel-20260523T145204Z.json`.

## Expressivity: trajectories DID diverge, winners did not

The fix is not silent at the chooser level. Per-seed step counts and
launch totals differ between favor_v2 and projected_sum_v2:

| Seed | favor_v2 steps | psum_v2 steps |
|---|---|---|
| 1 | 291 | 251 |
| 42 | 296 | 296 |
| 7 | 250 | 267 |
| 31 | 167 | 280 |
| 13 | 369 | 287 |
| 23 | 175 | 176 |
| 17 | 184 | 275 |
| 100 | 277 | 277 |

Pre-panel spot-check (`/tmp/inspect_multi_seed.py` on first 4 seeds)
confirmed:
- Seeds 42 and 13: both win, but launch counts differ by 47 and 107
  respectively.
- Seeds 1 and 7: **winner flips between favor and psum** (one in each
  direction).

The full 8-game panel did not pick up flips on the additional 4 seeds
(31, 23, 17, 100). The fix shifts chooser actions and trajectory length
substantially; the chooser still makes choices that converge to the
same final placement against this background.

## Comparison to 5/19 pre-fix result

| Run | favor | projected_sum | Gap |
|---|---|---|---|
| 5/19 panel (n=128/focal, pre-fix) | 85/128 (66.4%) | 87/128 (68.0%) | +1.6pp |
| 5/23 panel (n=8/focal, post-fix) | 5/8 (62.5%) | 5/8 (62.5%) | 0.0pp |

Both runs cluster around favor=65%, psum=65% with the gap inside the
per-game RNG noise band (5/19 documented favor self-noise of 6.3pp
between identical-config runs). The fix did not push projected_sum
above favor by a margin detectable at either sample size.

## Verdict

**Leaf-side axis stays exhausted under the baseline chooser.** The 5/19
verdict — "the leaf-side reframing axis is structurally saturated under
the current baseline-chooser architecture" — holds after the fix.

The fix is a real modeling improvement (regression-tested: capture
frames now credit attacker +196 V_diff and debit defender's projection
window correctly). But the chooser's argmax over the K=10 rollout in 4P
doesn't materially change winners against this background. The two
heads compute different scalars and the chooser produces different
trajectories, but reaches the same placement on 8/8 seeds.

## Caveat on sample size

n=8 leaves a wide miss-window. Wilson half-width is ~±28pp at p=0.5.
A 3–5pp real lift from the fix would be invisible at this n. PI
accepted that trade-off because:
1. The 5/19 panel at n=128 already showed +1.6pp pre-fix (well below
   the 6.3pp self-noise band).
2. Submitting projected_sum_v2 evicts the rolling-2 champion `52766596`
   (μ=1119.6) for an expected ±σ outcome — Rule 12 caveat against
   speculative late submits.
3. A 3–5pp marginal lift on this background is unlikely to translate
   to live-tournament μ-gain given the 50-100 Elo σ between brackets.

## Decision

**Do not submit projected_sum_v2.** Keep `28ce9f3` on the branch as a
permanent modeling fix (regression tests cover it). Pivot to
chooser-side per the 5/19 commit body's candidate list:

- **K-horizon sweep** (10 → 15 → 25) in 4P only. Tests whether
  capture-payoff visibility is the bottleneck.
- **`top_tier_mirror_policy` in rollouts** (symmetric A/B with baseline
  and action legs, CRN-safe). Tests whether a weak opp model is
  suppressing aggressive launches.
- **Proposer per-source dedup loosening** (top-K per source, not 1).
  Tests whether starvation of useful joint launches is the bottleneck.

Rule 37 status: leaf-side axis now exhausted at N=2 variants (max +
sum). Pivot is mandatory before iterating further on value-head shape.

## Artifacts

- `audit/tournaments/ffa-panel-20260523T145204Z.json` — full panel JSON.
- `submissions/_ab/favor_v2.py`, `submissions/_ab/projected_sum_v2.py`
  — post-fix bundles (sha256: `13857a1c…`, `a14ea08c…`).
- `audit/2026-05-19-projected-sum-4p-ab.md` — predecessor (pre-fix).
- `audit/2026-05-19-projected-value-head-4p-ab.md` — original (max
  aggregator), 4P.
- `audit/2026-05-19-projected-value-head-2p-ab.md` — 2P A/B (pre-fix
  TIE; not re-run because chooser-side pivot deprioritizes leaf-only
  variants regardless of player count).
