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

**Note on wallclock differences in the diff:** the bit-identical claim
covers game outcomes, step counts, and the sequence of launches issued
— NOT per-turn wallclock numbers. Single-game P50/max timing is
dominated by host CPU contention and varies run-to-run by 2-4×; the
diff dump will surface those timing differences even though the two
agents drove the simulation identically. For perf comparisons run
several games per agent on a quiescent host and aggregate, rather than
reading single-game timing lines from the bit-identical dump.
