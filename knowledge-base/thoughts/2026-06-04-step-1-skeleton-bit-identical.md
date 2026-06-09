# 2026-06-04 — Step 1 skeleton (`producer_plus`) is bit-identical, opp-model inserted as Step 3

Implementation kickoff of `state/MIGRATION_PLAN.md`. Step 1 is the
no-behaviour-change skeleton, gated on byte-for-byte identity to Producer.

## What landed

- `agents/producer_plus/` directory with:
  - `__init__.py` (empty package marker)
  - `main.py` — verbatim copy of `agents/producer/main.py` (368 lines)
  - `producer_agent.py` — importlib shim that injects BOTH
    `agents/producer/` (for shared `orbit_lite/`) AND `agents/producer_plus/`
    into `sys.path`, then loads our `main.py` under module name
    `producer_plus_main` (distinct from Producer's `producer_main` to
    avoid the sys.modules collision that would otherwise share state)
  - `PROVENANCE.md` — ethics note pinning the lineage to producer commit
    `0cc08da` and reminding that this is not submittable until our pieces
    are added
- `fast.py` registers the short-name `producer_plus`
- `state/MIGRATION_PLAN.md` — new Step 3 = opponent projection in scorer
  (mirror_self_policy callback + predict_opp_multi_launch pre-population),
  bumped multi-source / multiple-sizes downstream

## Why the wrap-and-modify pattern

- `orbit_lite/` is single-source-of-truth at `agents/producer/orbit_lite/`.
  No duplication; we reach into it from producer_plus via `sys.path`.
- We own our own `main.py` because Step 2 (adaptive K) will edit it. If
  we re-exported from `agents/producer/main.py` instead, the very first
  behaviour-change step would have nowhere to land.
- Producer stays a clean sparring reference — every A/B uses the
  untouched Producer as the held-fixed control.

## Why opponent modelling slots in as Step 3 (not later)

Per PI's question in plan mode. Producer's `sparse_launch_flow_delta`
runs an 18-tick combat sim with opp garrisons frozen — he has no opponent
prediction. Our champion has Tier 0 `mirror_self_policy` (~1–2 ms/call)
running every chooser-rollout tick + `predict_opp_multi_launch` (cheap
ROI-greedy 8-tick projection) inside the joint solver.

Both port cleanly INTO Producer's scorer as:
1. Per-tick opponent action callback inside the flow-diff sim.
2. Pre-population of the garrison ledger with projected opp arrivals
   before scoring runs.

Placing this BEFORE multi-source coalitions / multiple sizes raises the
quality of the scoring baseline against which every downstream step's
n=32 A/B is measured. Tier 2 (LightGBM distill) explicitly NOT ported —
falsified 2026-05-31 on chooser-budget cost, same penalty would apply to
Producer's scorer.

## Verification (Step 1 gate)

- `fast.py play producer{,_plus} --vs v7_0 --seed {7,13,42}` →
  IDENTICAL across all three seeds:
  - seed 7: P0=1 P1=-1, n_steps=129
  - seed 13: P0=1 P1=-1, n_steps=101
  - seed 42: P0=1 P1=-1, n_steps=118
- `tests/test_submissions_loadable.py` → 34 passed (unchanged)
- Secondary n=8 self-A/B `producer_plus vs producer` timed out (exit
  143 / SIGTERM at the bash timeout). Non-gating; primary identity gate
  already cleared. Will rerun with longer timeout next session if needed
  for paranoia.

## Pushed

Two commits on `claude/champion-ml-graft-majestic-storm`:
- `38b84d7` plan: insert opponent-model port as Step 3 of migration roadmap
- `c90488e` feat(producer_plus): bit-identical skeleton — wrap-and-modify host

No Kaggle submission this session. Step 1 is build-only.

## Next session

Step 2 — port adaptive K_eta schedule into `producer_plus/main.py`
(replace fixed `config.horizon = 18` with `K_eta = max(10, 20 − 10·step/30)`
into `build_target_shortlist`, keep H=18 for scoring forecast). A/B at
n=32 vs untouched `producer`, gate Wilson-lo ≥ 0.55. Needs PI sign-off
before launch.
