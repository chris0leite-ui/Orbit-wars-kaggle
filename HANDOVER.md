# HANDOVER.md — next-session brief

> Last written: 2026-05-14 by `claude/simplify-fast-setup-azW8T` (geo
> iteration + first ladder result). Length budget ≤ 160 lines.
> Prior wraps under `audit/archive-2026-05-1*-handover-*.md`.

## Where we are

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC → **40 days remaining.**
- **Team leaderboard score:** **μ=1064.4** (= max of rolling-last-2 =
  v7_pv). **Rank 125 / 2667** (top 4.7 %). Slipped from 109/2587
  because of new entrants + lower current rolling slot.
- **Rolling-last-2:** `[geo v3.1 #52643676 (μ=984.0, σ-discounted floor),
  v7_pv #52630118 (μ=1064.4)]`. geo's μ is NOT settled — ~5-6 h since
  submit ≈ 80-130 episodes; σ still large. Could move +50-100 over
  next 24h.
- **Daily submission budget:** used 1/5 today (geo). 4 remaining but
  **every new push evicts v7_pv (our best)** unless the new score
  decisively exceeds 1064.4.

## Day-N PM simplify-fast-setup-azW8T

The session goal evolved: fast-iteration framework (`fast.py`) →
geometric strategy → game-theoretic combination → ladder submission.

### Shipped to the codebase (29 commits)

1. **`fast.py`** — single-file iteration entry point: `smoke / eval /
   play / bench / baselines` with adaptive Wilson-gated A/B. Validated
   bit-identical against audit-logged v7_1 result. Replaces 31 scripts
   for the inner loop. **(Approved Plan #1)**

2. **`lib/geo/{sense, posture, allocator}.py`** — geometric primitives:
   single-link clustering on ETA, Voronoi cells over neutrals, front
   detection, threat budget (reuses `WorldModel.ledger`), comet claims,
   4-mode posture arbiter, LP and greedy-multi allocators.

3. **`agents/geo/main.py` (v3.2 final)** — the geo agent:
   - Pipeline: `obs → sense_state + decide_posture → incumbent (snipe
     aggressive + reinforce + opening, comets filtered) settled by
     `settle_plan` → 5-7 candidate variants (opening boost, enemy
     focus, gang_up multi-source, concentrated/saturation archetypes,
     front-reinforce) → score each via `score_candidate` (K=10
     top_tier_mirror, opp_tier=1) or `score_candidate_4p` (K=8) →
     argmax`.
   - **`SIGALRM`-based hard timeout** per `score_candidate` (700 ms cap)
     — bounded max from ~2900 ms to ~1200 ms.
   - **4P branch** via `score_candidate_4p` — geo runs lookahead in 4P
     where v7_0 falls back to v3.5.1 (per `lib/v7_search.py:choose:1384`).

4. **17 unit + e2e tests** in `tests/test_geo.py` (incl. 4P e2e).

5. **`submissions/geo.py`** — bundle (240 KB, sha256:1babc39d) submitted
   as #52643676.

### Local A/B results (substrate IS good vs v7_0; live-ladder generalization is the open question)

| Matchup | n | winrate | Wlo | Notes |
|---|---|---|---|---|
| vs v3.5.1 (2P) | 128 | 57.0 % | ~0.48 | Combined v2.3/v2.6 runs |
| vs v7_0 (2P) | 192 | 57.3 % | ~0.50 | Combined v2.3/v2.6/v2.9 runs |
| vs 3× v7_0 (4P first-place) | 128 | 56.3 % | 0.48 | +31 pp over 25 % baseline |

### Live ladder result

- **geo v3.1: μ=984.0 (σ-discounted early floor, ~80-130 episodes)**.
- Local A/B predicted +7 pp 2P / +31 pp 4P. Live floor is 80 below
  v7_pv. **Could rise as σ tightens** OR could settle low (real
  regression).
- Same pattern as **v3.5.1 on 2026-05-12**: local +56.6 % Wlo vs
  v3_snipe → live μ=945.6. The local panel didn't reflect ladder
  distribution.

## Falsified or dead (this session)

| Attempt | Result | Lesson |
|---|---|---|
| **v1**: posture multipliers + LP allocator | -37 pp vs v3.5.1 | Cross-class multipliers ≥2× crush settle_plan's per-source ranking. Global score-sort multi-launch over-concentrates at strong sources. |
| **v2.4**: lite_greedy follow-up | -17 pp | Cheaper opp model → lookahead picks candidates that don't transfer to the real opponent. |
| **v2.5**: WALLCLOCK_MS 500→350 | -20 pp | Tighter gate drops valuable tilts; first-score is unbounded by gate anyway. |
| **v2.7**: K=10→K=8 | -20 pp | Too shallow for geo's candidate count. |
| **v3.0**: composite value head (`evaluate_value`) | -19 pp | Survivor_bonus (5×) dominates small-scale composite; candidate ranking gets noisy. |
| **v3.2c**: empty_out + tap_capture combined | -4 pp cumulative | Individually within noise; combined dragged candidate ranking. Reverted both; gang_up kept. |

**The v2.3 config (K=10, top_tier_mirror, WALLCLOCK=500, default
ship-delta, 4 tilts + 2 archetypes + drop-one cap 2) is a tight local
optimum.** All single-knob changes regressed. Documented in
`knowledge-base/thoughts/2026-05-14-geo-v2-iteration-results.md`.

## Deferred

- **JAX vmap scoring** — `agents/jax_v7_0/main.py` shows the
  `scalar_to_jax(...)` integration path; `score_candidate_jax_pure_jit`
  is 6 ms after JIT vs ~200-400 ms scalar (~30-70× speedup). Blocker:
  jax_v7_0 is flagged "OFFLINE-ONLY" by its author (parity risk).
  Wire-up is ~1-2 h once parity is verified; would let us score ALL
  candidates without a wallclock gate.
- **v3.2 (gang_up only)** ready in source + bundle, NOT submitted.
  v3.1 just regressed live; pushing v3.2 evicts v7_pv. Hold until
  geo's μ settles (24-48 h).
- **B and C of the user's A/B/C plan** — JAX vmap deferred, composite
  value head failed.

## Next-session first-action (ranked by EV / cost)

1. **Re-check geo's ladder Score** (5 sec). If μ has climbed to 1050+,
   the substrate is fine and we iterate. If <1000 after 24 h, it's a
   real regression and we diagnose.
2. **Loss-mode diagnostic on geo's live replays.** Pull via
   `scripts/live_episode_summary.py` + `scripts/classify_losses.py`.
   The 5/13 audit showed v7_0 was 68 % opening-determined; geo's
   losses likely cluster differently (we DO opening-grab heavily;
   weakness may be mid-game vs top archetypes we never tested locally).
3. **Broaden local A/B panel.** v3.5.1's regression (-150 μ on
   2026-05-12) and now geo's (-80 floor) both came from vs-v7_0-only
   panels. **Future local A/B must include ≥3 opponent classes**
   covering v3.5.1, v7_pv, v7_0_drop_one, ideally a top-10 bundle (e.g.,
   Roman-1224 from `external/kernels/`). Friction rule:
   **`tag: local-vs-v7_0-only-misses-ladder-distribution`**.
4. **JAX vmap proof-of-concept.** If diagnostic confirms the architecture
   is sound and only compute is the limit, JAX unlocks deeper/wider
   search. If diagnostic shows strategic gap, JAX scoring won't help.

## Pointers (added today)

- `knowledge-base/thoughts/2026-05-14-geo-v2-iteration-results.md` —
  full bisect tables + lessons.
- `knowledge-base/thoughts/2026-05-13-geo-v1-bisect-lessons.md` — v1
  parity bisect (3 layers, all regressing).
- `audit/friction.md` — entries under 2026-05-13 + 2026-05-14.
- `agents/geo/main.py` — current geo, 5+ tilt helpers retained for
  future revival even if unwired.
- `lib/geo/{sense,posture,allocator}.py` — reusable primitives, 17
  unit tests.
- `fast.py` — the iteration entry point.
- `audit/2026-05-14-postmortem-geo-session.md` — session postmortem.
