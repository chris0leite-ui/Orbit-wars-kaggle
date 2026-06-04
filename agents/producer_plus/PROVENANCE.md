# producer_plus — provenance

This directory is the host for our **Producer-engine migration**: Producer's
planner as the engine substrate, with our pieces ported in as
candidate-generation / scoring extensions. See `state/MIGRATION_PLAN.md`
for the full plan and rationale.

## Step 1 state (this commit)

- `producer_agent.py` — a thin entry-point shim, behaviour-identical to
  `agents/producer/producer_agent.py`. Loads `main.py` under a unique
  `sys.modules` name (`producer_plus_main`) so the vendored producer
  (`producer_main`) and producer_plus can coexist in the same process.
- `main.py` — verbatim copy of `agents/producer/main.py` as of the
  Producer vendor commit `0cc08da` (cherry-picked from main on
  2026-06-04). No edits yet. Modifications begin at Step 2 (adaptive K).
- `orbit_lite/` — NOT vendored here. We reach into the existing
  `agents/producer/orbit_lite/` via `sys.path` injection in
  `producer_agent.py`. Single source of truth for the engine modules.

## Not submittable on its own

At this step, `producer_plus` is byte-identical in output to
`agents/producer/` (which is Slawek Biel's published work). Per the
ethics note in `state/MIGRATION_PLAN.md`, neither Producer nor a thin
wrapper around it may be submitted to the Kaggle ladder. Submission is
permitted only after our pieces (adaptive K, opponent projection,
multi-source coalitions, etc.) are added and the resulting hybrid agent
carries genuine value-add of our own.

## Step 4 — multi-size enumeration per (source, target)

Producer enumerates a single candidate per (source, target) pair, sized
at `safe_drain` (the max the source can safely send within the protection
horizon). Step 4 expands this to three sizes per pair:

- `capture_floor` — the minimum ships needed to capture at that arrival
  tick (read from `capture_floor` in `planner_core.py`).
- `2 × capture_floor` — clamped to safe_drain so it never exceeds the
  budget.
- `safe_drain` — unchanged from single-size.

The three variants are packed along an extra axis so the candidate
tensor becomes `[C = S × T × 3, L = 1]`. The L axis stays at 1; future
Step 5 multi-source coalitions will use the L axis for true coalitions.
Greedy selection's target mutex naturally picks the highest-scoring
variant per target and blocks the others; the smaller variants matter
when picking a lighter launch leaves source budget for a second target
in subsequent waves.

Each variant's `eta` and `angle` are recomputed because `fleet_speed`
depends on ship count — heavier fleets travel faster, so the same
geometry yields different arrival ticks per variant.

Default OFF (env `PRODUCER_PLUS_MULTI_SIZE` unset) preserves
single-size behaviour bit-identically. Env-on shim:
`producer_plus_multi_size.py` (also sets `PRODUCER_PLUS_ADAPTIVE_K=1`
to carry Step 2).

## Verification

Bit-identical to Producer at this step. Diff at fixed seed should be
empty:

```bash
for SEED in 7 13 42; do
  python fast.py play producer      --vs v7_0 --seed $SEED > /tmp/p_$SEED.txt
  python fast.py play producer_plus --vs v7_0 --seed $SEED > /tmp/pp_$SEED.txt
  diff /tmp/p_$SEED.txt /tmp/pp_$SEED.txt && echo "seed=$SEED IDENTICAL"
done
```
