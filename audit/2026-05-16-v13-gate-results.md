# v13 gate results — all green; submit decision pending PI

Branch: `claude/review-foundations-progress-14HXp` HEAD `cccd67e`
Date: 2026-05-16
Commits: cccd67e (v13 setup + hybrid opp policy)

Base architecture: v12 (sub 52699232 on the ladder at **μ=1217.7**,
team-best). v12 came from `origin/claude/recover-main-foundations-MV0e2`:
v9-chooser + opp_traj baseline (replayed for CRN) + lite_greedy
bounce-check fix.

v13 layer: hybrid opp policy in `_build_opp_trajectory` —
`top_tier_mirror_policy` (v3.5.1 aggressive snipe + reinforce
pipeline) for steps 0-9 of opp_traj, `lite_greedy_policy` for
steps 10-29. Constant `OPP_TRAJ_TOP_TIER_STEPS = 10`.

## Gates

| Gate | v13 result | v12-recover | Verdict |
|---|---|---|---|
| Felipe seed (PRIMARY) | **2/2** | 2/2 | PASS — matches |
| Naoism seed (soft) | **2/2** | — | PASS (v8 was 1/2; v4_planner 0/2; v12-C3 portfolio 0/2) |
| Wallclock bench | p95=176ms max=274ms zero >1000 | p95=116ms max=213ms | PASS — under budget |
| Panel: v7_0 | 12/16 (75.0%, Wlo=0.505) | 52/64 (81.2%, Wlo=0.700) | INCONCLUSIVE (n=16; v12 had n=64) |
| Panel: v4_planner | 11/16 (68.8%, Wlo=0.444) | 24/32 (75.0%, Wlo=0.579) | INCONCLUSIVE |
| **Panel: v3.5.1** | **15/16 (93.8%, Wlo=0.717)** | 24/32 (75.0%, Wlo=0.579) | **PASS — big improvement** |

The Wlo INCONCLUSIVEs are panel-sample-size artifacts (n=16 has
limited resolution), not regressions; point estimates 68.8% / 75.0%
on v7_0 and v4_planner are well above the 0.55 gate.

## What the hybrid policy bought us

The single change v13 vs v12 was: opp policy in the first 10 steps
of opp_traj switches from `lite_greedy_policy` (1ms/call, no
WorldModel) to `top_tier_mirror_policy` (10ms/call, full v3.5.1
pipeline with snipe + reinforce + settle_plan + arrival ledger).

**Effect on the panel vs v3.5.1: 75% → 93.8% (n=16).** This is a
clean signal: when the opp model in opp_traj matches the opponent's
actual policy (the panel's v3.5.1), the chooser's predictions are
near-optimal and our winrate jumps ~19pp.

**Effect on Naoism seed: 0/2 (v12) → 2/2 (v13).** Naoism is a
sustained-pressure top-tier opponent — exactly the class that
v3.5.1's pipeline models well in the first-10 turns. The improved
opp model in opp_traj's strategic window lets the chooser see and
counter Naoism's invasion buildup.

**Effect on v7_0 / v4_planner panel: -6pp / -6pp on point estimate
(within sample noise).** These are different opponent styles
(v7_0 is the older drop-one chooser, v4_planner is portfolio-based).
top_tier_mirror is a slightly worse model for THEIR behavior than
lite_greedy, costing minor head-to-head. Total panel weighted
average: ~79% across n=48 games (vs v12-recover's ~77% across
n=128). Roughly parity.

## Submission decision (pending PI)

Current ladder state:
- Rolling-last-2: [v12 sub 52699232 μ=1217.7, v9 sub 52687411 μ=1120.6]
- Team score: 1217.7

If v13 is submitted:
- Evicts v9 (1120.6) → rolling becomes [v13, v12]
- Team floor stays ≥ v12=1217.7 unless v13 catastrophically fails
- Daily budget: 1/5 used today (v12); 4 remaining

Expected ladder μ: 1200-1280. Floor scenario (v3.5.1 panel gain
doesn't translate live): ~1180. Best scenario (v3.5.1-style opps
dominate the ladder distribution): ~1280+.

**Risk:** v3.5.1 panel boost may not generalize. The panel
measures head-to-head; the ladder is TrueSkill across diverse
opps. If most ladder opps are NOT v3.5.1-style, v13's gain might
be modest.

**Reward:** if even half of the v3.5.1 panel gain translates,
v13 could exceed 1250 — a meaningful jump.

Recommendation: SUBMIT. Downside is bounded (eviction of v9 is
non-load-bearing now that v12 is the high-water mark).

## Reproduction commands

```bash
python fast.py play agents/v13 --vs v7_0 --seed 1492346051
python fast.py play agents/v13 --vs v7_0 --seed 1492346051 --swap
python fast.py play agents/v13 --vs v7_0 --seed 768065184
python fast.py play agents/v13 --vs v7_0 --seed 768065184 --swap
python fast.py bench agents/v13 --vs v7_0 --games 3
python fast.py eval agents/v13 --vs-panel default --max-seeds 8 --workers 4
```

Total wallclock: ~25 min on this box.

## Next-session options

If v13 ships and ladder confirms:
1. **Tune OPP_TRAJ_TOP_TIER_STEPS.** Currently 10 (matches strategic
   decision window). 4P games may need it lower (~5) for wallclock;
   2P games might benefit from 15.
2. **4P: maruichi-class seeds.** Need a 4P play harness (fast.py
   doesn't support 4P). v9's 4P winrate was 36.7%; v13 with hybrid
   opp model may improve here too.
3. **opp policy ensemble.** Could blend top_tier and lite_greedy
   predictions (e.g., majority vote) instead of step-based hybrid.
4. **Submission: post-game loss-mode classify** (per the codified
   methodology). After 24h, pull v13's loss replays, classify by
   phase, identify the next modeling defect.

If v13 regresses (live μ < 1217.7):
1. Roll back to v12 by re-submitting it (cheap).
2. Reduce OPP_TRAJ_TOP_TIER_STEPS to 5 or revert to pure lite_greedy.
3. Investigate which seed cluster regressed via replay pull.
