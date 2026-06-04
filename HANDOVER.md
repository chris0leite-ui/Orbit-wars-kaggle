# HANDOVER.md — next-session brief

## Mode

**Migration project + first live submission.** We are rebuilding the
agent on Producer's engine. As of 2026-06-04 we have submitted the
first hybrid: Producer's engine + our Step 4 mechanism (multi-size
candidate enumeration). The next session's first job is reading the
live μ once TrueSkill warms up (~24 h).

## Strategy

Two strategies still coexist:

- **Live anchor**: `baseline_adaptive_k` (`state/STRATEGY.md`).
  `champ_refine_adaptivek` (sub 53336920) sits as backstop in the
  rolling pair at live μ 1148.8.
- **Build / new live**: producer_plus migration host
  (`state/MIGRATION_PLAN.md`). `producer_plus_multi_size_on` (sub
  53369848) is the first hybrid on the ladder. Reaches live μ over
  the TrueSkill warm-up (24 h).

## Live status (after 2026-06-04 17:30 UTC submit)

- **Newest (#1):** `producer_plus_multi_size_on.py`, sub **53369848**
  (2026-06-04 17:30 UTC), 211 011 B. Producer's engine + multi-size
  enumeration (Step 4). **Live μ will warm up from 600 over ~24 h.**
- **Backstop (#2):** `champ_refine_adaptivek.py`, sub **53336920**,
  live μ ≈ 1148.8.
- **Producer's live μ:** ≈ 1200 (per PI).
- **Evicted by 53369848:** `champ_computeByShips_on` (sub 53332500,
  μ = 1150.6).

## Today's progress (2026-06-04)

This session was a deep iteration on the producer_plus migration host
plus the first submission from that track.

1. **Step 3 (opp projector) rolled back** (commit `89ce8c7`). M1/M2/M3
   foresight mechanisms were also rolled back earlier in the session;
   diagnosis: the lite-greedy projector ex-ante design + Producer's
   single-size candidate set gave filter mechanisms nowhere to retreat
   to. See chat / earlier session for the architectural read.

2. **Step 4 (multi-size enumeration) implemented and shipped** (commits
   `64e2345` + two fixes `e606a3e` + `7f0a83f`). Three ship-size
   variants per (source, target): capture_floor, 2×floor, safe_drain,
   packed along the C axis (`C = S × T × 3, L = 1`). Two bugs caught
   and fixed during smoke: `clamp(min=float, max=Tensor)` invalid
   PyTorch syntax → use `torch.minimum`; greedy's source budget was
   uncapped → cap at `drain` so multi-wave from one source can't sum
   above safe_drain.

3. **Step 2 (adaptive K) stripped from the multi_size shim** (commit
   `c235358`). 16-game seat-alt A/B vs vanilla producer was 8/16
   (exactly parity). Step 2+4 composed to 5/16 (regression);
   Step 4 alone vs producer landed 10/16 (62.5%).

4. **Bundler script + first hybrid submission** (commits `bb77ca3` +
   `4005b19`). `scripts/bundle_producer_plus.py` produces a
   single-file Kaggle-submittable .py by concatenating
   `agents/producer/orbit_lite/*.py` (topologically ordered, internal
   imports stripped) + `agents/producer_plus/main.py` (orbit_lite
   imports stripped) + env-var header. Output:
   `submissions/producer_plus_multi_size_on.py` (211 KB, parses, max
   per-turn 138 ms at seed 7).

## Next action

1. **Read the live μ** after TrueSkill warm-up (~24 h post-submit).
   The leaderboard is the truth, not local estimates.
2. **If live μ ≥ ~1170:** producer_plus_multi_size is a real lift.
   Proceed to **Step 5 — multi-source coalitions** per
   `state/MIGRATION_PLAN.md`. Producer is explicitly single-source;
   adding `L > 1` same-arrival coalitions is the biggest expected
   lift in the plan.
3. **If live μ < ~1170:** producer_plus track may not transfer to
   the ladder the way the local A/B suggested. Either roll back the
   submission's slot (let it sit in the rolling pair so we can read
   it; next submission would evict it), or investigate why local
   over-predicted live. Possible causes: opponent panel difference
   (local A/B used producer only; live ladder has v7_0, v4_planner,
   v3.5.1, etc.), seat-bias artefact, or the n=16 was just lucky.

## What did NOT make it into this submission

- Step 2 (adaptive K) — gated OFF; preserved in `main.py` for future
  tuning. We may revisit K_OPEN / floor parameters.
- Step 5 (multi-source coalitions) — biggest expected lift, deferred
  to next session.
- Step 6 (wait-then-fire) — deferred.

## Critical constraints reminder

- Producer (`agents/producer/`) is Slawek Biel's published work, MIT
  licensed. Submit only the hybrid; never wrap Producer directly.
- Rule 45 was explicitly overridden for this submission (n=16
  alone). Future submissions should use seat-balanced n=32 unless
  PI overrides.

## Pointers

- `state/STRATEGY.md` — canonical strategy doc (`baseline_adaptive_k`).
- `state/MIGRATION_PLAN.md` — producer_plus migration roadmap.
- `state/MULTI_BRANCH.md` — push-claim board (Rule 42).
- `CLAUDE.md` — process rules.
- `scripts/bundle_producer_plus.py` — reproducible bundler.
- `agents/producer_plus/main.py` — host with adaptive_K and
  multi_size gated mechanisms.
