# Trajectory chooser v2 — sketch

> Filed 2026-05-17 PM after the v1 trajectory chooser lost 0/32 vs v15.
> Sister doc to `trajectory-first-architecture.md` (the architectural
> reframe) and the in-flight `proposer.PROPOSER_TRAJECTORY_FILTER`
> (Option 1 — the safe pruning-only variant).
>
> Decision criteria for picking this up: only after Option 1 lands
> A/B-validated AND we know whether the current ladder regression
> (μ=1041 on 52744856) is a strategic-depth issue (Option 2 wouldn't
> help) or a value-function issue (Option 2 might help).

## Why v1 failed

`choose_trajectory` lost 0/32 vs v15. Three structural gaps:

1. **No defense.** Only scored "captured" status; "reinforced" was
   filtered (score=0). Net: no launches to defend threatened planets.
2. **No opponent counter-launch model.** Single-turn myopic; v15's
   K-step rollout sees 25-40 ticks of opp reaction.
3. **Sparse emissions.** Hard cap of 1 launch per source per turn.
   v15 emits parallel attacks; trajectory chooser couldn't keep up.

Wallclock was excellent (p50=8ms, p95=55ms — 14× faster than composite)
but strategically the chooser played like a passive agent.

## v2 design — three additions

### Addition 1: Defense via counterfactual reinforce-credit (~20 LOC)

For each candidate where `tgt.owner == me` (reinforce candidate):

```python
ledger_no_us = base_ledger[tgt.id]
ledger_with_us = ledger_no_us + [(eta, me, ships)]
owner_no_us, _ = predict_garrison_at(tgt, eta, ledger_no_us)
owner_with_us, _ = predict_garrison_at(tgt, eta, ledger_with_us)
if owner_no_us != me and owner_with_us == me:
    # We SAVE a planet that'd otherwise fall.
    score = DEFENSE_WEIGHT * production * time_remaining
```

Plugs into the existing dispatch; only changes the "captured" /
"reinforced" branch in `score_candidate`. Net new functions: 0.

### Addition 2: 1-turn opp lookahead (~60-100 LOC)

For each opp planet, predict its likely best 1-2 launches given
current world state. Inject those projected arrivals into the ledger
before scoring our candidates.

```python
def predict_opp_responses(world, me: int, num_seats: int,
                          ) -> list[tuple[int, int, int, int]]:
    """For each opp planet, predict its best 1-2 launches.
    Returns: [(target_pid, eta, opp_owner, ships), ...].
    """
    opp_arrivals = []
    for opp_id in range(num_seats):
        if opp_id == me:
            continue
        for opp_src in world.planets_by_id.values():
            if int(opp_src.owner) != opp_id:
                continue
            # Cheap ROI scan — what's opp's best target?
            best = find_best_target_for(opp_src, world, opp_id)
            if best is not None:
                opp_arrivals.append((best.target_id, best.eta,
                                     opp_id, best.ships))
    return opp_arrivals

# Merge into ledger BEFORE scoring our candidates:
pessimistic_ledger = merge_ledgers(base_ledger, opp_projected_arrivals)
# Every predict_garrison_at call now sees the pessimistic state.
```

`find_best_target_for` re-uses the same trajectory admissibility +
scoring path (with seat swap). 

**Validity caveat:** the opp model uses OUR scoring assumptions. v15
uses `lib.opp_model.lite_greedy_policy` — a different (simpler)
heuristic. For fidelity, swap in `lite_greedy_policy` for the opp
prediction layer. Tradeoff: `lite_greedy_policy` already exists and
is fast, but it's heuristic (no real lookahead); our scoring layer
is more sophisticated but may over-estimate opp competence.

### Addition 3: Multi-launch per source via ship sub-budget (~15 LOC)

Drop the "1 launch per source per turn" dedup rule. Replace with
ship-budget tracking:

```python
src_budget = {src.id: src.ships - MIN_RESERVE for src in my_planets}
moves = []
used_tgts = set()
for score, src, tgt, ships, angle in scored_sorted:
    if src_budget[src.id] < ships:
        continue           # source out of ships
    if int(tgt.id) in used_tgts:
        continue           # don't dogpile a single target
    moves.append([int(src.id), float(angle), int(ships)])
    src_budget[src.id] -= ships
    used_tgts.add(int(tgt.id))
```

A source with 100 ships can now do `(30→A, 30→B, 30→C)` instead of
just `(90→A)`. Matches v15's emission density.

`MIN_RESERVE` keeps each planet from being completely empty
post-launch (so a fresh enemy fleet doesn't auto-capture). Probably
`MIN_FLEET_SIZE = 2` (same as proposer).

## Implementation order + effort

| step | LOC | tests | days |
|---|---:|---:|---:|
| 1. Multi-launch budget (cheapest, biggest emit-density bump) | ~15 | 1 | 0.1 |
| 2. Defense counterfactual | ~20 | 2 | 0.25 |
| 3. Opp 1-turn lookahead | ~80 | 3-4 | 0.5-1.0 |
| 4. Re-A/B vs v15 | 0 | 0 | 0.25 |
| **Total** | **~115** | **6-7** | **~1.1-1.6 days** |

## Decision criteria — when to do this

- **Don't do** Option 2 yet if Option 1 (proposer prefilter) lands at
  parity-or-better vs current baseline AND we don't have evidence the
  current composite head is the bottleneck.
- **Do Option 2** if:
  - Option 1's A/B passes (no regression from prefilter) — gives
    confidence trajectory primitives are sound.
  - AND the composite head's live μ stays well below v15's (settled
    floor not regaining), suggesting the K-step rollout's strategic
    depth isn't the load-bearing piece — and Option 2's simpler model
    might match it cheaper.
  - AND wallclock matters for a future submission (1196-1580ms max
    on heavy turns is the standing concern).

## What this design deliberately does NOT do

- **Does not model multi-turn opp dynamics.** Only 1-turn opp lookahead.
  If opp plays a sequential strategy (build up → strike), we miss it.
  Acceptable trade-off vs the K-step rollout's higher cost.
- **Does not handle wait-then-fire.** v2 still drops `wait_N>0`
  candidates. Adding wait-time-discounted scoring would be a separate
  ~30 LOC addition; defer until A/B shows wait-N matters.
- **Does not address 4P-specific dynamics.** A2 (4P weakness) is in the
  `favor` value head; trajectory chooser bypasses that head. Either
  port A2 logic into the trajectory scoring or keep `favor_hybrid` as
  4P fallback at the dispatcher level.

## Risks

- **Best case:** trajectory chooser matches or beats current baseline
  on panel AND wallclock stays ~100-300ms p95. Submission-ready
  replacement for composite.
- **Median case:** within 5-10pp of v15, faster, simpler to maintain.
  Worth shipping as the next iteration after Option 1.
- **Worst case:** 1-turn opp lookahead isn't deep enough — still loses
  to v15's K-step rollout. ~1.5 days sunk; salvage value = the
  individual primitives (defense counterfactual, multi-launch budget)
  can be ported back into the current chooser.

## Cross-references

- `knowledge-base/concepts/trajectory-first-architecture.md` — the
  architectural reframe; this v2 sketch is the implementation
  reaction to v1's empirical failure.
- `agents/baseline/chooser_trajectory.py` — v1 implementation (lost
  0/32 vs v15). Most of the score_candidate logic ports forward; the
  three additions wrap around it.
- `agents/baseline/proposer.py` — `PROPOSER_TRAJECTORY_FILTER=on`
  (Option 1, currently A/B-ing). Survives independently of v2.
- `lib/world_model.predict_garrison_at` — sparse single-tick combat
  prediction; reused in all three additions.
- `lib/trajectory.predict_fleet_fate` — engine-mirroring trajectory
  ray-cast; reused in admissibility + opp scoring.
- `lib.opp_model.lite_greedy_policy` — existing simple opp heuristic;
  candidate for the opp lookahead layer's policy.
