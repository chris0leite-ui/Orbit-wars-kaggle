# least_resistance

A simulation-driven expansion agent. Each turn it builds a coordinated launch
plan by **scoring candidate moves with a forward-projecting evaluator** and
keeping only the launches that improve the projected outcome — not with
hand-tuned weights.

## How it decides (plain English)

1. **List the sensible moves.** For every planet we don't own, work out the
   coordinated launch that captures it — from one planet, or several ganging
   up when one can't afford it — using the exact lead-intercept physics
   (`lib/aim`), with a cheap `path_clears_sun` pre-filter. Order the list by
   the *path of least resistance to production* (most production per turn of
   travel), tie-broken toward whatever shortens our distance to the nearest
   opponent. This ordering is the strategy's *flavour* — which moves we try
   first.

2. **Decide with a strong evaluator.** Score each candidate plan with the
   PRODUCER's (`orbit_lite`, our strongest agent) garrison-flow scorer
   `score_candidates`: it projects every planet's garrison + production +
   in-flight combat forward ~18 turns and returns the plan's competitive
   *net ship gain* (ours minus opponents'). Greedily commit a launch only if
   it improves that projected gain by at least the producer's ROI floor; stop
   when nothing helps.

The evaluator is production-aware and policy-free, so it doesn't depend on a
weak rollout policy. Reserves, gang-up-vs-solo, attack-vs-expand, "accumulate
when nothing pays," and "don't bleed ships" all fall out of the projected
ship-delta — there are no strategy weights to tune.

If `orbit_lite` / torch isn't importable, the agent falls back to a
`lib/fast_sim` rollout under `lite_greedy_policy` with a production-aware leaf
(`inflight_value`) — weaker, but keeps the agent running anywhere.

## Why the evaluator matters (the story)

- **v1 — hand-tuned weighted score:** beat `random` 100% but lost to the
  `nearest` sniper ~6% (bled ships on launches that looked good on paper).
- **v2 — fast_sim rollout under `lite_greedy`, ship-delta leaf:** beat
  `nearest` 94% but lost to v7_0 (**12%**), and a *longer* horizon made it
  *worse* — proof the ceiling was the weak rollout policy, not foresight.
- **v3 — orbit_lite `score_candidates` leaf (current):** beats v7_0
  (**62%**) with the same candidate generation. The evaluator was the
  bottleneck; swapping in the producer's projector fixed it.

## Standing (2026-06-16)

| opponent | result | notes |
|---|---|---|
| `random` | 100% | floor |
| `nearest` | ~94% | comp baseline |
| `v7_0` (former champion ~μ1115) | **10/16 = 62%** | n=16, suggestive; n=32 to confirm |
| `producer` (orbit_lite ~μ1280) | 0/10 | loses to the full producer |

Timing: fast — the orbit_lite garrison-flow score is cheaper than per-candidate
rollouts (single-game max well under the 1s budget).

It sits **between v7_0 and the producer**: it beats our former champion line
but not the full producer, which also has reactive defense and idle-ship
regroup that this agent doesn't (yet). Those are the obvious next levers to
close the producer gap.

## Parameters

Only **compute bounds** + the producer's ROI floor (not strategy weights):

| var | default | meaning |
|---|---|---|
| `LR_HORIZON_2P` / `LR_HORIZON_4P` | 18 / 13 | orbit_lite projection window |
| `LR_ROI_FLOOR` | 1.5 | min projected net-ship gain to commit a launch |
| `LR_MAX_CANDIDATES` | 28 | most candidate moves considered per turn |
| `LR_WALLCLOCK_MS` | 700 | per-turn budget (bails the greedy loop) |
| `LR_EVAL` | (auto) | force `orbit` or `fallback` evaluator |

## Run

```
python fast.py smoke agents/least_resistance     # vs random + nearest
python fast.py bench agents/least_resistance      # per-turn ms
python fast.py play  agents/least_resistance --vs v7_0 --seed 7
```

## Dependencies & bundling

This agent imports the producer's `orbit_lite` package (and `torch`). It adds
`agents/producer/` to `sys.path` at import time (same mechanism the producer
uses). torch is available on the Kaggle evaluation runtime but is **not** in
the local `requirements.txt` — install with
`pip install torch --index-url https://download.pytorch.org/whl/cpu` to run
locally.

Because it depends on the multi-file `orbit_lite` package, the single-file
`scripts/bundle_agent.py` path does **not** apply. For submission it would ship
as a `tar.gz` with `main.py` at root plus the `orbit_lite/` package and the
needed `lib/` modules — a follow-up, not yet done (we are not submitting).
