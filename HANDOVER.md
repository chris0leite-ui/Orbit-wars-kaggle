# HANDOVER.md — next-session brief

> Last written: 2026-05-14 by `claude/simplify-fast-setup-azW8T` (merge of
> `claude/research-competition-analysis-2R8I3` +
> `claude/read-handover-iLWTq` + this branch's geo iteration).
> Prior wraps: `audit/archive-2026-05-14-handover-pre-search-exhaustion.md`,
> `audit/archive-2026-05-1*-handover-*.md`.

## Where we are

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC → **40 days remaining.**
- **Team leaderboard score:** **μ=1064.4** (= max of rolling-last-2 = v7_pv).
  **Rank 125 / 2667** (top 4.7 %). Slipped from 109/2587 since 5/12 due
  to total-team growth and lower current rolling-slot.
- **Rolling-last-2:** `[geo v3.1 #52643676 (μ=984.0, σ-discounted floor),
  v7_pv #52630118 (μ=1064.4)]`. geo just submitted ~5h ago; μ still
  settling.
- **Daily submission budget:** 1/5 used today (geo). 4 remaining but
  **every new push evicts v7_pv (our best)** unless the new score
  decisively exceeds 1064.4.
- **Calibration warning** (`tag: local-overpredict-2x`): local A/B
  over-predicted ladder for 2 consecutive submissions (v3.5.1 5/12 =
  −150μ; geo v3.1 5/14 = TBD but −80μ floor). Both panels were vs
  v7_0 only. Future submissions need ≥3-opponent local panel.

## Day-N PM simplify-fast-setup-azW8T (this branch — geo iteration)

The session evolved: fast-iteration framework → geometric strategy →
game-theoretic combination → ladder submission.

**Shipped (29 commits):**

1. `fast.py` — one-file iteration entry point. Replaces 31 scripts.
   Validated bit-identical against audit-logged v7_1 reproduction.
2. `lib/geo/{sense,posture,allocator}.py` — geometric primitives:
   clustering, Voronoi, front detection, threat budget, posture
   arbiter, LP + greedy-multi allocators. 17 unit tests.
3. `agents/geo/main.py` v3.2 — K=10 lookahead + 4 sense tilts + 2
   archetypes + gang_up + 4P branch (`score_candidate_4p`)
   + **SIGALRM per-score timeout** (700 ms cap; bounded max from
   1500-2900 ms to 1100-1200 ms with no strategic cost).
4. `submissions/geo.py` — bundled (sha256:1babc39d), submitted as
   #52643676.

**Local A/B (combined runs, n large):**

| Matchup | n | winrate | Wlo |
|---|---|---|---|
| vs v7_0 (2P) | 192 | 57.3 % | ~0.50 |
| vs v3.5.1 (2P) | 128 | 57.0 % | ~0.48 |
| vs 3× v7_0 (4P 1st-place) | 128 | 56.3 % | 0.48 |

**Live ladder:** geo v3.1 μ=984.0 σ-discounted floor (~80-130 episodes).
Same local→live overpredict pattern as v3.5.1.

## Day-N PM read-handover-iLWTq (parallel branch — chooser-axis exhaustion)

Seven controlled scalar 32-game A/Bs vs `v7_0_drop_one`. **All lost.**

| Variant | Axis | Change | Winrate |
|---|---|---|---:|
| v7_1 | proposer | H11 opening grab | 35.9 % * |
| v7_2 | search | depth-2 over drop-ones | 31.3 % |
| v7_3 | opp model | min-regret over archetypes | 28.1 % |
| **v7_4** | value head | composite capture-value | **40.6 %** |
| v7_5 | action space | + ADD-one widening | 37.5 % |
| **v7_6** | action primitive | + split-source | **40.6 %** |
| v7_7 | proposer coef | enemy multiplier ×1.3 | 28.1 % |

Best ~10 pp below 50 % baseline. **Chooser-axis design space is
exhausted.** Side wins: bundler parity-gate fix (env-var override),
composite_capture_value value head, 3 new action enumerators,
opp-archetype set, JAX depth-2 parked (GPU compile too slow).

## Day-N PM research-competition-analysis-2R8I3 (parallel branch — PV win + 8 falsifications)

**v7_pv (PV target valuation, γ=0.99) shipped to ladder at μ=1064.4.**
Eight other interventions monotonically FALSIFIED:
danger3 (×3 κ), FLEET_OVERCOMMIT (×3 mults), PRE_REINFORCE (×3
windows), Renaissance trio + per-mission ablation, HAV-1 binary +
soft-floor ×2, Holding-tier alone.

**Architectural finding:** v7 + PV is a tight local optimum; pre-
discounting scoring/proposer signals the rollout already evaluates
→ monotonic regression. Productive next move is architectural.

Plumbing: 3-anchor Wilson gate, PV_GAMMA in JAX, HAV helpers,
Renaissance flags (default-off), snipe tier emission framework.

## Falsified or dead (across all three branches today)

- All 7 v7_X chooser-axis variants (chooser-axis exhausted)
- All 8 R2R3 proposer/scoring variants (PV local optimum)
- geo v1's posture multipliers, greedy-multi allocator (~-30 pp each)
- geo v2.4-2.7 wallclock "fixes" (-17 to -20 pp each)
- geo v3.0 composite value head as agent value_fn (-19 pp)
- geo v3.2 empty_out + tap_capture cumulative (-4 pp)
- JAX depth-2 game-vmap (GPU compile fundamentally too slow)

## Next-session first-action (ranked by EV / cost)

1. **Re-check geo's ladder Score** (5 sec). If μ has climbed to 1050+,
   substrate fine and we iterate. If <1000 after 24h, real regression
   → diagnose.

2. **Loss-mode diagnostic on geo's live replays** (~30 min). Pull via
   `scripts/live_episode_summary.py` + `scripts/classify_losses.py`.
   The 5/13 audit showed v7_0 was 68 % opening-determined; geo's
   losses likely cluster differently — we DO opening-grab heavily,
   weakness may be mid-game vs top archetypes we never tested locally.

3. **Broaden local A/B panel** (`tag: local-overpredict-2x`). Add a
   `--vs-panel` flag to `fast.py eval` that runs 3-opponent panel
   (v3.5.1, v7_pv, v7_0_drop_one_rebuilt) by default. Eliminates the
   v7_0-only blind spot.

4. **JAX vmap scoring on geo** (~1-2 h). `agents/jax_v7_0/main.py` shows
   the integration path; `score_candidate_jax_pure_jit` is 6 ms after
   JIT (30-70× speedup). Pre-warm at import. Lets us score ALL
   candidates without wallclock gate; could enable K=15+ lookahead.

5. **Architectural search** (per R2R3's exhaustion finding). Portfolio
   search across opp ensembles, JAX-batched depth-2 with the GPU
   compile constraint fixed, or imitation learning from top-10 replays
   (Bovard IL is referenced in audit).

## Pointers (this session)

- `audit/2026-05-14-postmortem-geo-session.md` — geo iteration
  postmortem (this branch).
- `audit/2026-05-14-postmortem-read-handover-iLWTq.md` — chooser-axis
  exhaustion postmortem.
- `audit/2026-05-14-postmortem-research-competition-analysis-2R8I3.md`
  — PV win + 8 falsifications postmortem.
- `knowledge-base/thoughts/2026-05-14-geo-v2-iteration-results.md` —
  geo bisect tables.
- `knowledge-base/thoughts/2026-05-13-geo-v1-bisect-lessons.md` — v1
  parity bisect.
- `agents/geo/main.py` + `lib/geo/*` — geo agent + reusable primitives.
- `submissions/geo.py` — live submission #52643676.
- `submissions/v7_pv.py` — live submission #52630118 (team-best).
- `fast.py` — single-file iteration entry point.
