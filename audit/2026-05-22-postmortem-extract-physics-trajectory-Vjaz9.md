# 2026-05-22 — postmortem: extract-physics-trajectory-Vjaz9

**Branch:** `claude/extract-physics-trajectory-Vjaz9`
**Session shape:** physics substrate extraction → 3 conversion attempts to leaderboard lift
**Result:** parity vs ceiling at small n across every variant; substrate landed cleanly; no submission warranted

## What landed (7 commits)

1. `72fe45a` + `4980813` — kinematic-table substrate extracted from
   sibling Phase η branch (~1100 LOC). Per-turn precompute of planet
   positions; O(1) lookups after build. 39 unit tests, bit-parity-clean.
2. `923852e` — env-gated `kinematic_table.begin_turn(world)` priming
   in `agents/baseline/main.py:863-870`. `KINEMATIC_TABLE_ENABLED=1`
   wakes it; default OFF. `agents/baseline_kt/` wrapper agent
   (baseline_full stack + env var). Bench: p95 859 → 741 ms (−14%).
3. `877764b` — audit log: speed PASS, A/B vs baseline_full
   INCONCLUSIVE (n=64, Wilson [0.366, 0.604]).
4. `a3e21a4` — Phase 2: `predict_relative_smart` cached wrapper +
   swap at 9 hot-path sites in world_model / mechanism / aim /
   proposer / opp_projection. Bench p50 → 263 ms (−60% from off).
5. `5f8274a` — fix: revert two aim.py swaps after seed-0 outcomes
   flipped; caller in proposer.py:151-155 mutates tgt_list before
   aim_orbiting reads it, so the cache returns positions from the
   pre-mutation frame. Pin test added.
6. `c6a0c80` — **Phase 3a, the canonical H44 fix.** The proposer's
   trajectory admissibility filter (`PROPOSER_TRAJECTORY_FILTER=on`
   since 2026-05-17) had a stale bypass for every `wait_N > 0`
   candidate at `agents/baseline/proposer.py:998-1002`, with a
   comment claiming `predict_fleet_fate` would mis-classify. The
   function HAS a `wait_N` parameter; `lib/joint_solver/opp_projection.py:178`
   already uses it correctly for opponent projections. Removed the
   bypass; pass `wait_N=int(w)` through. Pin tests for sun-hit and
   orbital-drift cases at non-zero wait_N. **Measured impact: 707-
   1324 wait_N>0 candidates rejected per seed-0 game, 46-50% of all
   wait_N candidates considered.** Of those: ~64% would have gone
   OOB live, ~34% would have hit a wrong planet, ~1% the sun.
7. `f70a333` — Direction A: env-overridable `BASELINE_MIN_HORIZON` /
   `BASELINE_MAX_HORIZON` constants. `agents/orbitfix_kt_deep/` =
   orbitfix env stack + table + K bumped 25/40 → 40/60. Bench p95 =
   695 ms, max = 859 ms, well under the 1 s Kaggle cap. 3/3 vs v7_0.

Plus `98eeab1` — `agents/orbitfix_kt/` (orbitfix env stack + table)
created mid-session after discovering `state/MULTI_BRANCH.md` was
stale; the true Kaggle ceiling is sub 52912707 (`orbitfix`, μ=1175),
not sub 52893236 (`baseline_full`, μ=1058.6) which I had been
A/B-ing against.

## A/B verdicts

All A/Bs subprocess-isolated (`scripts/clean_ab.py`).

| Focal | Opponent | n | Score | Wilson | Verdict |
|---|---|---:|---|---|---|
| `baseline_kt` (Phase 1) | `baseline_full` (μ=1058) | 64 | 31/64 | [0.366, 0.604] | parity vs floor |
| `baseline_kt` (Phase 3a) | `baseline_full` | 74 (interrupted at 37 pairs) | 18W/19L = 48.6% | wide | parity vs floor |
| `orbitfix_kt` | `orbitfix` (μ=1175) | 4 | 2W/2L | [0.150, 0.850] | parity vs ceiling, n too small |
| `orbitfix_kt_deep` | `orbitfix` | 4 | 2W/2L | [0.150, 0.850] | parity vs ceiling, n too small |

Three independent slices, all parity-with-CI-spanning-50%. The
substrate work + Phase 3a + Direction A do not move the rating
against the current ladder ceiling at any sample size we measured.

## What we learned (load-bearing)

1. **The K-rollout structural ceiling is robust.** This codebase
   has now falsified ~10 attempts at shifting the leaderboard via
   mechanism-axis tweaks: H17 / H19 / H21 reweights, chain bonus,
   value aggregators, analytical-on-rollout, chooser_roi, asymmetric
   Tier-1, v9-v15 saturation, Phase 1 substrate, Phase 3a wait_N
   filter, Direction A K-bump. The K=25-40 rollout is doing
   sufficient work that "speed + accuracy improvements at the same
   strategic axis" don't move the rating.

2. **The wallclock-adaptive chooser is fundamentally
   non-deterministic across runs.** `agents/baseline/chooser.py:120-130`
   uses `time.perf_counter()` to size `per_cand_ms`, which drives
   `n_aff = int(budget / per_cand_ms)`. Per-run timing noise →
   different cap → different candidates evaluated → different
   chosen action. Same seed, two runs of the same code, different
   game outcome. Surfaced in 2026-05-22 diff_v3 trace: 128 of 210
   turns differ between env-OFF and env-ON despite identical
   physics; both wins. Implication: bench at n=3 is below the noise
   floor; A/B verdicts require n≥32 to overcome the chooser noise.
   **Open question:** could the chooser be made deterministic by
   replacing the time-based budget with a count-based budget? The
   table would make per-cand cost predictable enough for that.

3. **Phase 3a's H44 fix is correct, real, and doesn't lift the
   ladder.** 50% of wait_N candidates per game ARE geometrically
   doomed; pre-fix they were launched live. Post-fix they don't
   enter the prerank. The wins/losses don't shift because the K=10
   rollout's downstream physics simulation IS already catching
   most of those deaths inside the rollout — even if the leaf
   value head over-credits "fleet in flight" PV. The chooser was
   apparently picking around the doomed candidates anyway.

4. **The state-of-truth file (`state/MULTI_BRANCH.md`) was
   stale.** It claimed the rolling-pair ceiling was μ=1117.9; the
   real Kaggle ceiling was sub 52912707 at μ=1175. Mid-session
   correction: ran `kaggle competitions submissions orbit-wars`,
   discovered the gap, forked `orbitfix_kt` from the actual
   ceiling agent.

5. **Cross-branch parallel work (`claude/consolidate-codebase-refactor-dQAWA`)
   converged on the same wall.** Their `agents/coord/` Day 1-10
   work falsified 3-source bundle coordination (1.8% benefit;
   rolled back MAX_BUNDLE_SIZE 3→2). Their `agents/minimal/` is a
   1369-LOC consolidation of orbitfix — clean code, but still has
   the H44 bypass. Two independent attacks on the orbitfix ceiling
   hit "parity at small n."

## Why the substrate work is still valuable

We did NOT find the mechanism that converts kinematic-table speed
into leaderboard lift. That doesn't mean the substrate is dead
weight:

- `lib/kinematic_table.py` (436 LOC) + tests (621 LOC) are merged
  to main. Available to every future agent on this codebase.
- The Phase 3a fix is shape-correct: closes a documented bug that
  was wasting ~50% of wait_N candidates per game. Even if it
  doesn't lift the ladder, **launching ships into the void is
  intrinsically wrong.** A future strategy change that relies on
  predictably-arriving wait_N fleets (e.g., the Phase 3b leaf
  in-flight fate check, or any coordination mechanism that depends
  on multi-fleet timing) now has a foundation that doesn't lie.
- `predict_relative_smart` is the canonical cached wrapper for
  the 9 hot-path sites. Anything new built on this codebase gets
  free position-cache acceleration.
- `BASELINE_MIN_HORIZON` / `BASELINE_MAX_HORIZON` env vars make
  K-bumping a one-line config change for any future variant.

## Next-session first action

**Making the best of this session's work is the priority.** The
substrate landed; the conversions to lift failed. Two paths
forward, ordered by expected value:

1. **Cross-pollinate with the consolidate branch.** Their
   `agents/coord/` (now at MAX_BUNDLE_SIZE=2 after Day 10 Gate 3
   falsification) is a new chooser architecture that imports
   physics primitives from `agents/minimal/`. Merging our Phase 3a
   wait_N fix + kinematic-table priming into that path gives the
   coord agent both a cleaner-code base AND the correctness fix.
   If their Gate 4 multi-opponent panel passes for coord, the
   joint variant (their architecture + our substrate) is the
   natural submission candidate. Cost: low; their architecture is
   the unknown that needs validation, not ours.

2. **Investigate the chooser non-determinism.** Replace
   `time.perf_counter()` budgeting in `agents/baseline/chooser.py:120-130`
   with count-based budgeting (the kinematic table makes per-cand
   cost predictable enough). Two payoffs: (a) reproducible A/B
   results so future experiments can actually disambiguate
   parity from small lift, (b) might unlock real lift if the
   timing noise was masking signal.

3. **Direction B / C as documented in the prior plan.** Smarter
   opp model in the rollout, or opening trajectory matrix port.
   Heavier work; lower confidence; only worth it if path 1+2
   both wash.

**Decision deferred:** the user is submitting work from the
`consolidate-codebase-refactor-dQAWA` branch this cycle. Our work
on this branch is infrastructure-ready, not submission-ready.

## Frictions logged

- `stale-state-md` (2026-05-22): `state/MULTI_BRANCH.md` claimed
  ceiling was μ=1117.9 when actual ceiling was μ=1175 (sub
  52912707, +57 μ off). Root cause: state file written 2026-05-21
  without subsequent refresh after sub 52912707 landed 04:56 UTC.
  **Fix already applied to flow:** before forking baseline,
  always run `kaggle competitions submissions <comp> | head -5`
  to verify live ladder. Codified in CLAUDE.md Rule 32 (session-
  start git fetch) — extend to Rule 32b: Kaggle-CLI submissions
  fetch before any subsystem A/B target choice.

- `wallclock-adaptive-chooser-nondeterminism` (2026-05-22):
  bench at n=3 produced opposite outcomes (env-ON 0/3 wins vs
  diff_v3 single-process env-ON 1/1 win) for the same seed. Root
  cause: `time.perf_counter()` in chooser's `per_cand_ms` probe
  drives candidate cap. Per-run timing noise → different cap →
  different chosen move. **Fix:** documented; replacing the
  time-based budget with count-based is a next-session candidate.

- `bundler-refuses-cross-agent-wrapper` (2026-05-22, re-confirmed):
  Bundling `agents/baseline_kt/` via `scripts/bundle_agent.py`
  fails with "REFUSING TO LEAVE BUNDLE: bundle has no callable
  `agent`" because the bundler strips `from agents.baseline.main
  import agent` without inlining the body. This is the friction
  tag the bundler safety check was originally introduced for.
  **Mitigation:** wrappers are local-eval only; submissions are
  bundled from `agents/baseline/` directly with env vars
  post-injected (see `submissions/baseline_full.py:6-19` for the
  established pattern). Documented in `agents/baseline_kt/main.py`
  module docstring.
