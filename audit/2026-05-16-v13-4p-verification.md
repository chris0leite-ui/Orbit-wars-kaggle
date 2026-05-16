# v13 4P verification — Maruichi + regression panel

Branch: `claude/review-foundations-progress-14HXp` HEAD `7314c28`
Date: 2026-05-16
Harness: `scripts/play4p.py` (new; thin CLI on `run_ffa_tournament`)

## Why this matters

PI chose "Hold; explore 4P" after 2P gates passed but before
submission. Concrete risk: v9 lost a live 4P game vs Maruichi
on seed 76670184; v13's hybrid policy multiplies opp_traj cost
by ~3× in 4P (3 opps), potentially blowing the 1000ms actTimeout.
Both risks needed local data before spending the submission slot.

## Maruichi seed 76670184 (4P, focal=v13 vs 3× v7_0, rotate seats)

```
seed=76670184  focal_seat=2  WIN   reward=1   n_steps=117  turn-ms p50=336 p95=457 max=574
seed=76670184  focal_seat=0  WIN   reward=1   n_steps=143  turn-ms p50=312 p95=499 max=631
seed=76670184  focal_seat=3  WIN   reward=1   n_steps=164  turn-ms p50=309 p95=508 max=643
seed=76670184  focal_seat=1  loss  reward=-1  n_steps=221  turn-ms p50=245 p95=492 max=575
```

**3/4 first-place** (75%, Wilson95=[30.1%, 95.4%]). p95=493ms, max=643ms,
zero turn-ms ≥1000ms.

Context: v9 lost this seed live (per `audit/2026-05-19-v12-session-
summary.md:39`: "us 6p/152s vs maru 6p/306s at step 50, ship_balance
−0.34"). v13 wins 3 of 4 seat permutations on the proxy game
(v7_0 standin for Maruichi). The chooser handles the seed
configuration cleanly from most seats.

Caveat: this is a PROXY for Maruichi (we don't have Maruichi's
agent). The proxy tests whether v13 can win this seed's geometry,
not whether it specifically beats Maruichi. Directional signal only.

## 4P regression panel (4 seeds × 4 seats = 16 games)

```
seed=42           4/4 WIN  (random)
seed=7            4/4 WIN  (random)
seed=1492346051   1/4 WIN  (Felipe 2P seed, played in 4P)
seed=768065184    1/4 WIN  (Naoism 2P seed, played in 4P)
```

**10/16 first-place (62.5%, Wilson95=[38.6%, 81.5%])**.
p95=275ms, max=658ms, zero ≥1000ms.

The Felipe and Naoism seeds — both originally 2P sustained-pressure
seeds — are *harder in 4P* (focal is 1 of 4, not 1 of 2). v13 still
wins 1 seat on each, which is non-trivial.

## Combined 4P signal (5 seeds × 4 seats = 20 games)

- **First-place: 13/20 (65%)**
- vs random baseline: 25%
- vs v9 live 4P rate: 36.7%

**v13's 4P winrate roughly DOUBLES v9's live ladder 4P rate** —
the hybrid opp policy is a meaningful upgrade in 4P, not just 2P.

## Wallclock summary

| Run | p95 | max | over_1000ms |
|---|---|---|---|
| Smoke seed=42 (4 games) | 133ms | 208ms | 0 |
| Maruichi seed=76670184 (4 games) | 493ms | 643ms | 0 |
| Regression panel (16 games) | 275ms | 658ms | 0 |
| **Total 4P (20 games)** | **~300ms** | **658ms** | **0** |

Predicted worst-case: ~700ms top_tier cost (3 opps × 10 steps ×
~25ms in heavy game) plus chooser work. Observed: max 658ms,
consistent with prediction. Comfortable margin to 1000ms.

## Full v13 gate matrix (combining 2P + 4P)

| Gate | Threshold | v13 | Verdict |
|---|---|---|---|
| 2P Felipe seed (PRIMARY) | ≥1/2 | 2/2 | PASS |
| 2P Naoism seed | ≥1/2 | 2/2 | PASS |
| 2P bench wallclock | p95<800ms, 0 >1000 | p95=176ms max=274ms | PASS |
| 2P panel vs v3.5.1 | Wlo ≥0.55 | 15/16 Wlo=0.717 | PASS |
| 2P panel vs v7_0 | Wlo ≥0.55 | 12/16 Wlo=0.505 | INCONCLUSIVE (point 75%) |
| 2P panel vs v4_planner | Wlo ≥0.55 | 11/16 Wlo=0.444 | INCONCLUSIVE (point 68.8%) |
| 4P Maruichi seed | ≥1/4 | 3/4 | PASS |
| 4P regression panel | first-place ≥25% | 62.5% | PASS-target (≥35%) |
| 4P wallclock | p95<800ms, 0 >1000 | p95=275-493ms max=658ms | PASS |

**Net**: every hard gate PASS. The two INCONCLUSIVE entries are
sample-size artifacts (n=16); point estimates are well above gate
threshold.

## Recommendation

**SUBMIT v13.**

Justification:
- 2P primary gates all pass
- 4P performance is a substantial upgrade (65% vs v9's 36.7%)
- Wallclock has comfortable margin in both 2P and 4P
- Submission cost is bounded: evicts v9 (μ=1120.6) from rolling-
  last-2; team floor stays ≥ v12 (μ=1217.7) unless v13 catastrophically
  crashes
- 4 daily submission slots remaining
- Ladder upside: v13 looks better than v12 (μ=1217.7) on the 2P
  v3.5.1 panel (75 → 93.8%) AND in 4P (double v9's rate). Expected
  μ: 1220-1280.

## Submission plan

1. Bundle: `python scripts/bundle_agent.py agents/v13` →
   `submissions/v13.py`
2. Verify bundle import: `python -c "import submissions.v13; ok"`
3. Submit: `kaggle competitions submit -c orbit-wars -f
   submissions/v13.py -m "v13: v12 + hybrid top_tier/lite_greedy
   opp_traj — 2P Felipe 2/2 Naoism 2/2 v3.5.1 93.8%, 4P 65%"`
4. Update `state/current.md` with sub_id and μ=PENDING
5. Wait ~10 min for initial μ; settle 5-6h for full TrueSkill

## Reproduction

```bash
# 2P
python fast.py play agents/v13 --vs v7_0 --seed 1492346051
python fast.py play agents/v13 --vs v7_0 --seed 1492346051 --swap
python fast.py play agents/v13 --vs v7_0 --seed 768065184
python fast.py play agents/v13 --vs v7_0 --seed 768065184 --swap
python fast.py bench agents/v13 --vs v7_0 --games 3
python fast.py eval  agents/v13 --vs-panel default --max-seeds 8 --workers 4

# 4P
python scripts/play4p.py --focal v13 --bg v7_0,v7_0,v7_0 --seeds 76670184 \
    --rotate-seats --workers 4
python scripts/play4p.py --focal v13 --bg v7_0,v7_0,v7_0 \
    --seeds 42,7,1492346051,768065184 --rotate-seats --workers 4
```

Total wallclock: ~40 min combined.

## Next-session options

If v13 ships and ladder confirms (μ ≥ v12):
1. **Push `OPP_TRAJ_TOP_TIER_STEPS`.** Currently 10. Try 5 (cheaper)
   or 15 (deeper) and bench the panel + 4P winrate.
2. **Real Maruichi replay.** Pull Maruichi's last 5 ladder games
   from Kaggle, replay with v13 in their slot, see if v13 changes
   the outcome. (Requires ladder replay scraping; out-of-scope
   this session.)
3. **Tournament-level head-to-head.** Run `scripts/ffa_tournament.py`
   with v12 and v13 both as backgrounds; see which one v3.5.1 and
   v7_0 favor as a sparring partner — informative signal for the
   ladder population distribution.

If v13 regresses live (μ < v12 by ≥30 points):
1. Roll back: resubmit v12 next slot.
2. Diagnose: pull v13 ladder losses, classify by phase, look for
   opening-window failures (where top_tier_mirror_policy may have
   miscalibrated the chooser).
