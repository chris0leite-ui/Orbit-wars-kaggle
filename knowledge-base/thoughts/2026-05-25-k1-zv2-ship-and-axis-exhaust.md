# 2026-05-25 — Session arc: K1 unlocks budget, Z v2 lifts vs phi1_only, deeper-axis chase fails vs joint_aggr

## What happened (one paragraph)

Started chasing the "BUILDUP mispriced 19-garrison neutrals" bug from
seed 1622482326. Surgical fix (commit `843cc35`) made the agent fire
the cheap captures but a) n=64 A/B was parity-band (48.4%) and b) p95
turn-time blew past the 1000 ms Kaggle budget (1101 ms). Diagnosed the
wallclock root cause via cProfile: `predict_relative` was burning 84 s
cumulative own-time in a 219-turn game; an already-built, parity-gated
caching layer (`lib/kinematic_table.py`) had been extracted to
origin/main from a sibling branch but never wired into our agent's
turn dispatch. The K1 patch (commit `c0035ff`) added a 10-line
`begin_turn(world)` priming call in two places and shipped. Re-profile:
p95 dropped 1894 → 918 ms (with cProfile overhead), production max
651 → 718 ms (well under budget). Then Z v2 (commit `603f45f`): the
effective-landing prune formula `ships - prod·eta` was missing the
`pred_ships` subtraction (defended targets got hidden ship-credit). Fix
was 5 LOC + dropping a redundant opening_planner site. n=64 A/B vs
phi1_only: **56.2%, +10.9pp** over K1-alone. Submitted as sub 53018599
(K1+Z v2 stack). Then chased a follow-up "small-recapture asymmetry"
diagnosis triggered by the user's seed-2020490432 game screenshot:
two more fixes (opp-model floor 10→5, holdability floor 20→5). The
new 5×250×no-swap A/B standard showed 80% vs phi1_only (replicates
diagnosis) but **0-20% vs joint_aggr** (the live μ=1120.1 rolling-pair
half). Three falsifications on the proposer-tightening axis in one
session = Rule 37 cap. Reverted both follow-up fixes; branch HEAD now
matches the source state of sub 53018599 plus the new `ab_quick.py`
tool.

## Load-bearing insights

1. **Profile FIRST when caching.** I tried to cache opening_plan
   (commit 9870575) based on a guess that the BUILDUP MILP was
   eating 1.7 s/call. cProfile showed each call was actually ~5 ms;
   the variance came from `predict_relative` × candidates, not from
   the MILP itself. The cache regressed behavior 34% → 9%.

2. **Wallclock has a TIER STRUCTURE.** The first per-turn budget
   bottleneck (predict_fleet_fate cumulative) was hiding the second
   (CONSOLIDATION chooser's per-candidate rollout). Fixing K1 made
   the agent USE the freed headroom for more candidate scoring,
   which materially changed picks. So K1 was not "pure wallclock"
   — it composed with chooser behaviour via the deadline-capped
   candidate loop.

3. **Per-cohort regression matters more than aggregate.** Fix A+B
   gave 47% aggregate but 20% against the strong opp. The strong
   opp is in our live rolling pair. Aggregate panel parity is fool's
   gold if you're losing to the cohort that decides ladder μ.

4. **n=5 panel is a triage signal, not a falsifier.** 0/5 vs joint_aggr
   has Wilson CI [0, 0.434]. The point estimate is alarming but the
   sample is too small to declare a regression vs a true 30% winrate.
   Next iteration: n=8-16 against the critical opp once K1+Z v2's
   live μ settles.

5. **The cheap-recapture diagnosis is real.** 80% vs phi1_only across
   two different fix configurations is reproducible, large-effect.
   But fixing it via proposer pre-filters over-restricts against
   aggressive opp. The real fix is structural — better opp model in
   the rollout (e.g., joint-aggr-priority projection or a learned
   policy), not threshold tuning.

## Open questions for next session

- Will sub 53018599 (K1+Z v2) settle above the rolling-pair floor?
  μ to watch: > 1120 = strong half stays, > 1127 = displaces sub
  53013786 as strong half.
- If sub 53018599 settles weak, was K1 alone the right move and Z v2
  the regression source? Need to un-stack and test independently.
- The opp-model `lite_greedy_policy` is a 1995s-era closed-form
  heuristic. Replacing it with something behaviour-cloned from live
  ladder replays is the natural next axis — but requires the HANDOVER
  P0 replay scout to be done first.
