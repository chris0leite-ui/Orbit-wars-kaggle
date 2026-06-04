# HANDOVER.md — next-session brief

## Mode

**Migration project**, not single-strategy iteration. We are rebuilding the
agent: Producer's engine as host, our pieces ported in as candidate-
generation / scoring extensions (no post-passes). See
`state/MIGRATION_PLAN.md` for the full plan.

## Strategy

Two strategies coexist during the migration:

- **Live**: `baseline_adaptive_k` (`state/STRATEGY.md`) — what is on the
  Kaggle ladder right now. Stays as backstop in the rolling submission
  pair until the hybrid agent beats it.
- **Build**: Producer-engine-host hybrid (`state/MIGRATION_PLAN.md`) —
  starts implementing next session.

## Live status

- **Latest submission (#1):** `champ_computeByShips_on.py`, sub **53332500**
  (2026-06-03 15:11 UTC), 697 927 B. Adaptive K + compute_by_ships baked.
  Predicted μ ≈ 1170 (parity with sibling).
- **Backstop (#2):** `champ_adaptiveK_on.py`, sub **53324164**, live
  **μ = 1185.2**. Our anchor — stays in the rolling pair until the
  hybrid agent beats it locally and we submit.
- **Producer's live μ:** ≈ 1200 (per PI). We will NOT submit Producer.

## Today's progress (2026-06-04)

1. **Frontier_circulation triplet parked — not killed.** Three
   implementations of pressure-gradient ship circulation as a post-pass
   over the chooser, all falsified (5/16, 8/16, 5/16). Root cause
   identified via code review: Producer's regroup composes with his
   scoring because both use the same scalar; our scoring is per-trade
   ROI, so pressure-routed ships go to destinations our chooser ignores.
   See `audit/2026-06-04-postmortem-champion-ml-graft-majestic-storm.md`.

2. **Cherry-picked Producer agent from main** (commit `0cc08da`).
   Available locally at `agents/producer/`, registered as the `producer`
   short-name in fast.py and in `DEFAULT_PANEL`. Sparring partner only.

3. **n=32 head-to-head A/B:** our champion vs Producer = **13/32 wins
   (40.6 %), Wilson [0.255, 0.577]**. Producer wins ~60 % of games.
   Confirmed at n=32; not noise.

4. **Read of Producer's planner** (`agents/producer/orbit_lite/
   planner_core.py` + `garrison_launch.py`). Synthesised what each agent
   does well; designed the migration plan in `state/MIGRATION_PLAN.md`.

## Next action — start the migration

Open `state/MIGRATION_PLAN.md` and execute **Step 1: skeleton
`agents/producer_plus/`** as a wrap-and-modify of the vendored Producer.
Confirm bit-identical behaviour before touching anything.

PI sign-off needed at start of next session on:
- Migration direction still active (no contradicting live observation).
- `agents/producer_plus/` directory name and wrap-and-modify pattern.
- Per-step gate threshold (Wilson-lo ≥ 0.55 unless tighter / looser).

After Step 1 lands clean, proceed through Steps 2-7 in order, gating each
on n=32 A/B lift per Rule 45.

## Critical constraint — Producer is NOT submittable

`agents/producer/` is Slawek Biel's published work. The vendored copy is
a sparring partner and engine substrate, NOT a submission target. Only
the hybrid (`producer_plus` with our extensions) is eligible for
submission. This is documented in `state/MIGRATION_PLAN.md` § Ethics
note.

## Pointers

- `state/STRATEGY.md` — current live strategy, build/smoke procedure.
- `state/MIGRATION_PLAN.md` — Producer-host migration plan (new this session).
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `CLAUDE.md` — process rules (lean).
- `audit/2026-06-04-postmortem-champion-ml-graft-majestic-storm.md` —
  full postmortem of today's circulation triplet (parked, not killed).
- `audit/2026-06-04-producer-eval-observations.md` — main's prior n=16
  observations of Producer vs our agents.
- `knowledge-base/thoughts/2026-06-04-circulation-family-parked-not-killed.md`
  — diagnosis and unblock paths (now superseded by migration plan).
- `knowledge-base/flags/2026-06-04-ship-utilization-still-open.md` —
  watch flag (now subsumed: migration is the answer).
- `knowledge-base/questions/2026-06-04-chooser-pressure-port-vs-2hop-targeting.md`
  — open question resolved: chooser pressure-port wins, via migration
  plan.
