# Lookahead Phase 2 — env.clone()+step() forward simulator

> Date: 2026-05-11
> Branch: `claude/bootstrap-agentic-systems-lqnm6`
> PI prompt: "there must be a simple solution for that. think and research."

## TL;DR

`kaggle_environments` ships `env.clone()` and `env.step()` out of the box.
Threaded together, they give us **the forward simulator we needed** — no
pure-Python re-implementation required. K-step v2-self-play forward
projection (Sim<K>) closes essentially the entire oracle gap from Phase
1a: at probe step 50, Sim50 AUC = 0.952 vs perfect-oracle O50 = 0.952
(identical to 3 decimals). At step 25 the gap is 0.871 vs 0.879 (0.8pp).
Cost: median ~213 ms per Sim<K> evaluation; K=30 → ~170 ms, K=50 →
~280 ms. Budget 1000 ms → 3–5 candidate forward sims per real turn.

This is the load-bearing finding of the lookahead investigation.

## The simple solution I missed

`kaggle_environments.core.Environment` exposes:
- `env.clone()` — deep-copies the env state, returning an independent
  Environment that can be advanced without touching the original.
- `env.step(actions)` — advances the env one tick given the per-player
  action list. Used internally by `env.run`.
- `env.reset(num_agents=N)` — initialises a fresh env from the configured
  seed; `env.state` is then live and steppable.

Putting them together:

```python
def forward_sim_delta(env, p0_fn, p1_fn, K: int, pov: int) -> float:
    """Clone, roll forward K turns under both players' policies, score."""
    clone = env.clone()
    for _ in range(K):
        if clone.done: break
        a0 = p0_fn(clone.state[0].observation)
        a1 = p1_fn(clone.state[1].observation)
        clone.step([a0, a1])
    return (our ship total in clone) - (their ship total in clone)
```

That's the entire forward simulator. Wallclock: ~5.6 ms per step on a
4-core box (env.step + 2 v2 turns).

## Probe extension

`scripts/lookahead_probe.py` refactored to **step the env manually**
(env.reset + per-step env.step loop) so a clone can be taken at any
probe step. Before, we ran env.run to completion and iterated env.steps;
that gave us static historical state but no live env to clone from at
mid-game. The manual loop is no slower than env.run (env.run is just a
loop calling step internally).

At each probe step we clone, run forward K turns under v2-self-play,
and record the ship-delta scoring scalar.

`--sim-ks 30,50` is the default; per-K cost adds ~K · 5.6 ms per
probe step per seed.

## Results

32 seeds, v2 (P0) vs roi_baseline (P1), 5 probe steps × 3 horizons ×
K ∈ {30, 50}. Artifact: `audit/lookahead/20260511T063556Z.json`.

### Headline — Sim<K> vs the perfect oracle

```
      step     naive       H50      Sim30     Sim50       O50
        25     0.469     0.565     0.713     0.871     0.879
        50     0.725     0.658     0.867     0.952     0.952
        75     0.879     0.846     0.950     0.979     0.979
       100     0.952     0.863     0.979     0.996     0.996
       150     0.995     0.943     1.000     1.000     1.000
```

- **Sim50 matches the oracle to 1pp or better at every probe step.**
  Step 50: Sim50 = O50 = 0.952 (identical). Step 100: 0.996 = 0.996.
- **Sim30 is ~92% as good as Sim50** at ~60% the cost. Step 50:
  Sim30 = 0.867 vs Sim50 = 0.952 vs O50 = 0.952. Still 14pp above
  naive (0.725) at the mid-game uncertainty zone.
- **The static WorldModel (H50)** captures only 23% of the available
  signal: at step 50, H50 = 0.658 vs naive 0.725 (NEGATIVE lift),
  while Sim50 = 0.952. The 4.6pp Phase 1a "lift" was within seed
  noise — the static substrate is essentially useless past naive.

### Cost

```
WorldModel.from_world build wallclock @ horizon=200: median 1.87 ms
Sim<K> forward-sim wallclock (any K in [30, 50]): median 212.8 ms, max 1105.3 ms
```

The 1105 ms max is a tail-event outlier (one game's mid-turn ran long
under both agents simultaneously). Median is solid. For agent
deployment we'd add a wallclock watchdog and use the partial-K
rollout result if a sim runs long — the env is mid-step-safe to
abort because the clone is independent.

### Per-turn budget at deployment

At K=50, **2-3 candidate forward sims per real turn** (560-840 ms of
1000 ms budget). At K=30, **5 candidates** (~850 ms). That's enough
breadth for shallow beam search over candidate this-turn intent
sets — exactly the use case from the Phase 1a strategic write-up.

## What this rules in and out

**Ruled in** — Sim<K> as the v3.1+ scoring head. Action sequence:

1. Define a candidate-intent enumerator: e.g. "v2's top choice per
   source," "v2's top-2 per source × seat-rotation," "Hungarian
   bipartite assignment over (src, tgt) scores," etc.
2. For each candidate intent set X this turn: clone the env, force
   our P0 to commit X (one turn), then let both players play v2-self
   for K-1 more turns. Read the ship-delta scalar.
3. Pick the X with the highest projected ship-delta-at-K.
4. Mechanism layer (DEFAULT_MECHANISMS) finalises the chosen X.

This is the v3.1 design that v3.0 was scaffolding for. The
`lib/planner.settle_plan` boundary already lets us swap in a Sim<K>
scorer without changing the rest of the agent.

**Ruled out** — pulling Roman's kernel for a custom forward sim, the
~1-day pure-Python re-impl, any further intermediate static-substrate
extensions (Hours / Hall / arrival-ledger variants). `env.clone()`
makes all of those redundant.

## Caveat to flag for the next session

The Sim<K> AUC of 0.952 measures *predictive power* (can we read the
winner from a v2-self-play rollout?) not *strategic strength* (does
acting on the prediction make us win more?). Two important
distinctions:

1. **Policy mismatch**: the rollout assumes both players use v2. On
   the live ladder, opponents are unknown. If Sim<K>'s ranking of
   candidate this-turn intent sets is robust to opponent identity,
   the agent transfers. If it's not, we need a richer rollout policy
   (mix of policies / learned opponent model / pessimistic minimax).
2. **Greedy bias**: at each rollout step, v2 picks the same
   per-source-ROI target it would pick "now," not a strategically
   different action a smarter agent might choose. The rollout is
   biased toward what v2 would do, which is what we're trying to
   beat.

Phase 2.5 (next step, not committed): wire `Sim<K>` as a scorer into a
v3.1 candidate-evaluator, ship it against v3 / v2 in the local 2P +
4P panels. Gate: ≥3pp lift over v3 in 2P, parity-or-better in 4P,
p95 turn < 800 ms.
