# Marco lineage reference — 2026-06-02

Materials for the marco-v3-3 opponent-model + adversarial-rerank work
planned in `PLAN.md`.

## What's here

- `PLAN.md` — the full plan, 6 phases, gates, risks.
- `kernels/marco-dg-v3-3.py` — the public lineage ancestor (rank 4 in
  the 78-agent round-robin, 88.3% win-rate). Source we port the opening
  planner from.
- `kernels/marco-dg-v56.py` — the private upgrade (rank 1, 95.5% wr).
  Reference for what we are NOT replicating (C-extension, true 2-ply IBR).
- `kernels/romantamrazov-lb-max-1224.ipynb` — confirmed marco-v3-3 fork.
  Used in phase-4 panel test (the marco-fork win-rate gate).
- `kernels/suntzu-launch-safety.ipynb` — separate idea (rank 2, 94.2% wr);
  reference for a continuous launch-safety multiplier that is NOT in
  the current plan but relevant to a follow-up.

## Why this is in the repo

The kernels were pulled into `/tmp/kernels/` during the 2026-06-02
session, which is ephemeral. Committing them makes the next session
self-sufficient — no kaggle CLI dependency, no re-download.

## How to use

Read `PLAN.md` start to finish. Phase 1 begins by reading
`kernels/marco-dg-v3-3.py` lines 2199-2484 (the EAM opening planner).

## Tournament context

Source: <https://www.kaggle.com/competitions/orbit-wars/discussion/703848>
— "🪐 Community Benchmark — 78-Agent Mega Tournament" (round-robin of
78 validated public agents, 6006 matches).
