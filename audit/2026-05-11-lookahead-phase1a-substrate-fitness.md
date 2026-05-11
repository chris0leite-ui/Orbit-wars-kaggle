# Lookahead Phase 1a — substrate fitness measurement

> Date: 2026-05-11
> Branch: `claude/bootstrap-agentic-systems-lqnm6`
> PI direction (chat): "check carefully how fit are we for looking in the
> future, looking steps ahead." Bundle behavior over N steps; integrate ROI
> globally; prune to stay fast.

## TL;DR

The game outcome is **decidable from step 50 with a perfect 100-step
oracle** (AUC 1.000). Our current static WorldModel substrate captures
**~14% of the available signal** at the same probe point. The bottleneck
is not horizon depth or compute — it's that the simulator ignores future
actions (ours and adversary). Per-turn budget is 1000 ms; v2 uses 1.35 ms.
We have ~700× headroom for policy-aware forward simulation.

## Compute fitness — where we stand

| Quantity | Value |
|----------|-------|
| Per-turn budget (`actTimeout`) | 1000 ms |
| v2 / v3_snipe turn median wallclock | 1.35 ms |
| Budget headroom | ~740× |
| Board diagonal | 141.4 units (100×100 square) |
| Max fleet travel time (1 ship at speed=1) | 141 turns |
| Typical fleet travel (10-50 ships, speed 2-3) | 45-72 turns |
| Game length | 500 turns |

WorldModel build wallclock vs horizon:

| Horizon | Build ms (24 planets, 0 fleets) | % of budget |
|---------|----------------------------------|-------------|
| 110 (current default) | 0.74 | 0.07% |
| 200 | 1.29 | 0.13% |
| 300 (covers all realistic fleets) | 1.86 | 0.19% |
| 500 (full remaining game) | 3.19 | 0.32% |

**Cost of horizon=500 is negligible** (< 0.4% of budget). The compute
constraint is not the limiter.

## Probe design

`scripts/lookahead_probe.py` — for a sample of (seed, midgame step)
pairs in v2-vs-roi_baseline games (asymmetric pair chosen because v2 vs
v2 yields a perfect step-500 tie, zero signal):

- **naive** predictor: current (P0 ship total - P1 ship total), planets +
  in-flight fleets, snapshot.
- **H<H>** predictor: WorldModel.from_world(world, horizon=H), then sum
  `(predicted_owner==P0 ? ships : 0) - (predicted_owner==P1 ? ships : 0)`
  at step `now + H`. Static projection: in-flight fleets only, no future
  launches.
- **O<H>** predictor (oracle): future ship delta read directly from
  `env.steps[now + H]`. Upper bound on what any scalar predictor of
  "delta at H" could achieve.

For each (probe step, predictor) we compute the AUC of `score >
threshold → P0 won` using the Mann-Whitney rank-sum equivalent.

32 seeds, v2 (P0) vs roi_baseline (P1). All 32 games decisive (no ties).
188 (step × seed) samples. Per probe step n ≈ 32.

Artifact: `audit/lookahead/20260511T061813Z.json`.

## Results

```
     step   naive    H50    O50    H100   O100   H200   O200
       25   0.466   0.502  0.864  0.493  0.986  0.493  1.000
       50   0.595   0.641  0.916  0.625  1.000  0.598  1.000
       75   0.864   0.841  0.986  0.832  1.000  0.836  1.000
      100   0.916   0.850  1.000  0.850  1.000  0.836  1.000
      150   1.000   1.000  1.000  1.000  1.000  1.000  1.000
      200   1.000   1.000  1.000  1.000  1.000  1.000  1.000
```

(AUC 0.5 = no signal; 1.000 = perfect.)

### Three load-bearing observations

1. **The future IS predictive.** The perfect oracle at H=100 from probe
   step 50 has **AUC 1.000** — every game is fully decided 50 turns into
   the future. At probe step 25 (very early), O200 = 1.000 (game is
   decidable 25→225 = step 225, 55% through the game).

2. **The static WorldModel projection captures only a sliver of that signal.**
   At probe step 50, the gap is:
   - oracle O50 advantage over naive: **+32.1pp** (0.916 vs 0.595)
   - WorldModel H50 advantage over naive: **+4.6pp** (0.641 vs 0.595)
   - **WorldModel captures ~14% of available signal at the H=50 sweet spot.**
   At H=200 the WorldModel adds zero (or slightly negative) over naive
   (0.598 vs 0.595) — the projection is no better than the snapshot.

3. **Longer static horizons monotonically REGRESS in predictive power.**
   This contradicts the "look as far ahead as possible" hypothesis IF we
   keep the static simulator. At probe step 50: H50 = 0.641, H100 = 0.625,
   H200 = 0.598, H300 = 0.589, H500 = 0.575. Each extra 50 turns of
   static projection adds noise without adding signal — because the
   projection's "no new actions" assumption diverges further from reality
   the further ahead we look.

## Diagnosis — what's missing

The static WorldModel assumes both players FREEZE — no future fleet
launches, only existing in-flight fleets and production accrual on
currently-owned planets. Real games are about the launches. So:

- At step 50, the WorldModel projecting to step 150 sees: "P0 will own
  the planets P0 owns now plus production accrual; same for P1." But P0
  will *capture more planets* in the next 100 steps, and so will P1.
  The relative balance of those captures is exactly the signal the
  oracle is reading and the static WorldModel is missing.

- The 32pp oracle lift at probe step 50 is the value of correctly
  modeling "what will both players do next." Even 14% of that = 4.5pp
  AUC improvement = real strategic value if we could capture it.

## Implication for the strategic direction

The PI hypothesis — **look far ahead and coordinate globally** — is
DATA-SUPPORTED: the signal is there. What the data correct in my earlier
read:

- The lever is NOT "extend horizon with a static simulator" (longer
  horizon hurts that simulator's AUC). Pruning to stay fast at horizon=500
  is solving the wrong problem.
- The lever IS **policy-aware forward simulation**: simulate plausible
  future launches by both players, then use the projection as a scoring
  signal. Pruning becomes useful once we're inside that loop (depth>1
  search), not before.

## Suggested Phase 2 (post-discussion, not committed)

Replace static WorldModel projection at the strategy-scoring boundary with
a **K-step self-play roll-forward** under both players using a fixed
policy (v2 or v3):

```
def project_us_minus_them(world, horizon_steps=50, policy=v2_policy):
    sim_env = clone(world)
    for _ in range(horizon_steps):
        actions = [policy(sim_env.obs(0)), policy(sim_env.obs(1))]
        sim_env.step(actions)
    return ship_delta(sim_env)
```

Cost projection: ~1.35 ms per v2 turn × 2 players × 50 simulated steps =
**~135 ms per scoring evaluation**. Budget is 1000 ms; that's room for
**~5-8 candidate this-turn action bundles** evaluated and compared per
real turn.

That's a 1-ply lookahead with bounded breadth — the simplest thing that
extracts the signal the probe says is on the table. If it lifts μ over
v3 in the panel + 4P FFA gates, we deepen toward MCTS (depth>1).

Pruning levers (low-production planets, tiny fleets) become natural
once we're inside the K-step loop and need to keep the inner-loop
per-turn cost down. Today they're premature.

## What I am NOT recommending today

- Building lib/missions/{reinforce,recapture,gang_up}.py at the surface
  level (the original Block E v3.1 plan). Per-class heuristics on a
  blind-to-the-future score function will keep us in the ~70% panel WR
  band where v2 sits.
- Extending DEFAULT_HORIZON past 200 in the current static substrate.
  Probe shows no gain.
- Pruning rules in current code. Premature; substrate is far under budget.
