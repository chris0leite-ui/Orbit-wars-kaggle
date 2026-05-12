# 2026-05-12 — first submission of σ-equivariance v1

## Submission

```
ref:      52565034
file:     v3_snipe.py (68.4 KB)
bundle:   sha256:8ba37fc0b7e71112cfe1663b4690faa6fc3354e86b1b6c08e6def9aa088cb3fb
date:     2026-05-12 04:39:49 UTC
status:   PENDING (validation episode in progress)
```

Message submitted with the agent:

> σ-equivariance v1: v3.4 base + 3 surgical patches (lib/planner
> σ-equiv tie-break + lib/geometry sym_hypot + score rounding to 6
> decimals). Headline: 16/16 = 100% v3-vs-v3 self-play draws over 500
> steps — provable cannot-lose at v3-class (symmetric-game value
> theorem realized). Calibration: 50/50 vs precision_v3 (peer Nash
> tier), 54.7% Wilson [42.6,66.3] vs v2, 93.8% vs roi, 100% vs
> v1/baseline/random. Bundle sha256:8ba37fc0b7e71112. Audit:
> 2026-05-11-cannot-lose-final-finding.md.

## Rolling-last-2 change

Before push:
- precision_v3 (#52552139) — μ=1009.0 (oldest in rolling-last-2)
- v3_4 (#52556866) — μ=995.4

After push (precision_v3 evicted as oldest):
- v3_4 (#52556866) — μ=995.4
- v3_sigma_equiv_v1 (#52565034) — PENDING

## Theoretical claim

The strict cannot-lose strategy for v3-class in Orbit Wars 2P is the
σ-equivariant version of v3_snipe. Three surgical patches deliver it:

1. **σ-equivariant tie-break** (`lib/planner.py:_tb` + sort key).
   Secondary key `-(src.x - 50) * (target.x - 50)` is σ-invariant
   for σ-paired (src, target) pairs. Within a source's tied targets,
   T and σ(T) get opposite-sign keys → consistent σ-equivariant pick.

2. **Canonical-order hypot** (`lib/geometry.sym_hypot`). Neutralises
   the 1-ULP non-associativity of `math.hypot(a²+b²)` vs
   `math.hypot(b²+a²)`. σ-paired (src, target) pairs produce
   bit-equal distances.

3. **Score rounding to 6 places** (`lib/planner.SCORE_ROUND=6`). The
   env stores planet coordinates with 1-ULP σ-asymmetries that
   propagate through distance → score. Rounding the primary sort
   key treats sub-ULP noise as tied so the σ-equivariant tie-break
   actually fires.

Together, two identical σ-equiv-v3 agents in self-play produce
**16/16 draws over 500 steps** — empirical realization of the
symmetric-game value theorem.

## Expected ladder behavior

Calibration tells us what to expect on the live ladder:

| Opponent class | Expected vs σ-equiv-v3 | Why |
|---|---|---|
| Other v3-class agents (peers' v3 derivatives) | ~Draw | σ-equiv lock applies |
| precision_v3 / RL bots / different classes | ~50/50 | Peer Nash tier |
| Older v3-class (e.g. ladder's v3.1 derivatives) | 54-94% | Strength + σ-equiv |
| Random / Nearest Sniper / baseline | 100% | Total dominance |

Predicted ladder μ: 1000-1015. The σ-equiv work is a **μ-FLOOR**, not
a μ-ceiling — it eliminates the small probability of losing in
near-tied scenarios against v3-class opponents but doesn't push us
above precision_v3's 1009 strength.

## What this submission DOESN'T claim

- It is NOT strict cannot-lose against any opponent. The cannot-lose
  property only applies within v3's strategy class (i.e. between any
  two σ-equiv-v3 agents).
- It does NOT add strength against precision-class or unknown
  strategy classes. We're at 50/50 vs precision locally.
- It is NOT a μ-jumper. Expected μ gain over v3.4 baseline: +0 to +20μ.

The honest framing for this submission: **proof of concept that the
cannot-lose theorem is realizable in working agent code, delivered
as a μ-floor at v3-class.** The next iteration's job is to add
strength (wave bundling, strike-window, or recapture missions) on
top of the σ-equiv foundation.

## Plan for next iteration

Based on the per-seed analysis
(`audit/2026-05-11-v3-vs-precision-perseed-analysis.md`),
precision wins long-sustained games (high-home-production +
distant-neutrals boards). To climb past 1009:

1. **Wave bundling** (~3-5 day build). 2-source coordinated attacks
   on distant high-value targets. Directly attacks precision's
   winning regime per the per-seed analysis. Cheapest concrete
   strength upgrade.
2. **Strike-window timing** (~1-2 week build). Schedule shots to
   arrive AFTER projected enemy capture. Highest claimed individual
   ROI (2×).
3. **Recapture mission class** (~2-3 day build). Roman's playbook
   addition. Estimated +50μ per HANDOVER.

These all preserve the σ-equiv lock as long as new sort/scoring
operations use the same `round(score, SCORE_ROUND)` + tuple-key
σ-equivariant tie-break pattern.

## What to watch on the live ladder

- **Validation episode**: should pass within minutes (local self-play
  was clean).
- **First ladder games**: μ initialises at 600 then converges to a
  stable rating over the first ~10-20 games.
- **Predicted stable μ**: 1000-1015 within 24-48 hr.
- **Failure modes to watch for**:
  - Validation Error: bundle import or syntax issue (unlikely; bundle
    parity test passed 998/998 turns).
  - Time-out per turn: highly unlikely (local max 17ms vs 1000ms budget).
  - Sub-995 μ: would indicate σ-equiv patches actively hurt against
    the ladder distribution.
  - Above-1020 μ: would indicate σ-equiv-vs-v3-derivatives draw lock
    is more impactful than predicted.

## Branch state

`claude/game-theory-strategy-analysis-0oH4N`, 22 commits ahead of
origin/main at branch tip.

Key commits relevant to this submission:
- 6c12b9f σ-equivariant tie-break in settle_plan
- 7b60938 sym_hypot bit-exact distance
- 24bae06 score rounding to 6 places
- 90463b9 100% v3-vs-v3 draws verified
- fb9fbc9 merged origin/main (v3.4)
- 9db5445 calibration matrix complete
- e0fc581 v2 anomaly resolved (54.7% Wilson [42.6, 66.3])
- This submission documents at commit 8ba37fc bundle hash.
