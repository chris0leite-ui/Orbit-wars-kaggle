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

## Step 5 — multi-source coalitions

Producer's planner is structurally single-source: each candidate is
one `(source, target)` pair with `L = 1` contributor. Producer's
`LaunchSet` already supports `L > 1` end-to-end (`score_candidates`
→ `sparse_launch_flow_delta` accumulates via `scatter_add_`;
`_greedy_select` debits all L contributors via `scatter_add_` and
gates the wave all-or-nothing on coalition fundability) — what's
missing is the candidate-generation extension that actually emits
L=2 coalitions. Step 5 fills it.

For each high-value target, the planner additionally emits up to
`C(K_src, 2) = 15` (with K_src = 6 by default) two-source coalitions:
both contributors send `safe_drain[s]` and the pair is admitted only
when their independent arrival ticks differ by ≤ 1 tick (env knob
`PRODUCER_PLUS_COALITION_ETA_TOL`, default 1).

The candidate tensor packs single-source rows (padded with
`active[c, 1] = False`, `ships[c, 1] = 0`) alongside coalition rows
into a unified `[C_total, L = 2]` tensor:

- `C_total = S × T + T × C(K_src, 2)` (≈ 144 + 180 = 324 candidates
  in 2P, ≈ 72 + 180 = 252 in 4P).
- Greedy's all-or-nothing fundability check
  (`(send ≤ budget) | ~active).all(dim=-1)`) and the per-leg
  `scatter_add_` debit make padded slots no-op and coalition
  contributions correct.

Source ranking per target uses `-eta` (fastest arrivers first) under
the per-`(s, t)` validity mask, ties broken by ascending source slot
via `_stable_topk_indices` — CPU/CUDA bit-stable.

Multi-size (Step 4) is deliberately NOT carried in the coalitions
shim: composing 3 size variants × C(K_src, 2) pairs would blow the
candidate count and the wallclock budget. Step 4 vs Step 5 are
A/B'd separately; compose later as Step 5b only if both lift.

Cross-wave over-drain guard (introduced in Step 4) is also active
under coalitions — `source_budget` is capped at `drain` so a source
that fires in a coalition can't simultaneously fire as a solo wave
in a later iteration of the greedy loop.

Default OFF (env `PRODUCER_PLUS_COALITIONS` unset) preserves
single-size behaviour bit-identically. Env-on shim:
`producer_plus_coalitions.py`.

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
