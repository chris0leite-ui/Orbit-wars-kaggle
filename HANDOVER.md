# HANDOVER.md — next-session brief

> Last written: 2026-05-16 by `claude/fleet-strategy-optimization-fsu0t`
> (v21_compound build session). Merged in v8_scavenge/v12/v15/v20 source
> from `claude/recover-main-foundations-MV0e2` and built the new
> proposer-augmentation agent on top.

## v21_compound build (this session) — TL;DR

- **What:** `agents/v21_compound/main.py` + `lib/compound.py` +
  `lib/mission_book.py` + `lib/geo/rotation.py`. Bundled at
  `submissions/v21_compound.py` (333 KB).
- **Architecture:** v20's dogpile chooser, leaf scorer, K-rollout, opp
  model — all unchanged. ONLY the prerank stage augmented with:
  (1) sun-safe pre-filter; (2) rotation_alignment + chain_bonus +
  carryforward bonus added to cheap_marginal_value; (3) MissionBook TTL
  persistence (3 turns).
- **Rule 38 verified:** v20 sends 6 fleets/8 games into the sun + 2
  OOB; v21 sends 0/0. PI-observed sun-bug fixed.
- **A/B status:** v21 vs v20 head-to-head running at session end; check
  `/tmp/ab_v21_vs_v20.log`. Per the 5/16 postmortem, h2h vs v20 is the
  FIRST gate; if it passes Wlo≥0.55 then run 3-anchor panel vs v15 +
  v7_0.
- **Submission status:** NOT submitted (PI build-only this session).
  Next-session decision once v15 and v20 settle live.
- **Bundle parity OK (corrected 2026-05-17):** the previous session's
  MISMATCH flag was timing-noise from adaptive wallclock probing.
  With `ORBIT_WARS_PARITY_WALLCLOCK_MS=60000`, src=bundle on every
  probed seed. Full-game divergence under default 1000ms cap is
  expected (same behavior on Kaggle servers). Bundle is safe to submit
  per the bundler's per-obs parity contract.

## Next-session first-action (ranked by EV / cost)

0. **Fix bundle-source parity for v21_compound** (~1 hr). Probe: run
   `python -c "from kaggle_environments import make; e=make('orbit_wars',
   configuration={'seed':1}); e.run(['agents/v21_compound/main.py',
   'agents/v20/main.py']); print(e.steps[-1][0].reward)"` and the bundle
   equivalent — one returns 1, the other -1. v20's bundle (when it
   existed) had a `ORBIT_WARS_PARITY_WALLCLOCK_MS` env var override
   exactly for this reason — see `agents/v20/main.py` for the
   `_effective_wallclock_ms()` function. The v21 source inherits this
   knob unchanged. Suspect: bundle's import overhead at session start
   shifts the per-step probe, changing n_affordable, changing which
   candidates get validated, changing emit decisions. Fix: re-bundle
   with explicit wallclock cap, or inline _effective_wallclock_ms with
   a hardcoded value matching what the source observed in bench.

1. **Read v21 A/B result** (1 min): `cat /tmp/ab_v21_vs_v20.log | grep -E "wins|verdict"`.
   If PASS: run `python fast.py eval v21_compound --vs-panel v20,v15,v7_0
   --max-seeds 32 --gate 0.55 --workers 2`. If FAIL: see ablation menu
   below.

2. **Check live v15 + v20 μ** (1 min): `kaggle competitions submissions
   orbit-wars | head -10`. Both PENDING at session end. Floor risk: if
   both settle below v13's 1063.8, we lose the floor.

3. **If v21 PASSES h2h + panel** (~20 min): submit. `python
   scripts/bundle_agent.py agents/v21_compound` (already produced
   `submissions/v21_compound.py`). Verify timing: `python fast.py bench
   v21_compound --vs v20 --games 3`. If max < 900 ms, push:
   `kaggle competitions submit orbit-wars -f
   submissions/v21_compound.py -m "v21_compound: proposer-time
   sun-filter + compound bonus + TTL persistence"`. This evicts v15
   from rolling-last-2.

4. **If v21 FAILS h2h** (~3 hr): ablate one axis at a time per Rule 37:
   - `agents/v21a_sun_only/` — copy of v20 + sun pre-filter only.
   - `agents/v21b_chain_only/` — copy + chain_bonus only.
   - `agents/v21c_rotation_only/` — copy + rotation alignment only.
   - `agents/v21d_book_only/` — copy + MissionBook only.
   Any single axis passing is fixable next session; full failure of all
   four is rare and would indicate the proposer-augmentation axis class
   is actually dead.

5. **JAX port of the rollout** (~2 hr) — queued from prior handover.
   Independent of v21 axis; could compound with v21 if both work.

## Where v15/v20 came from (this session, FYI)

The recover-main-foundations-MV0e2 branch had the entire v8→v20 chooser
saturation line as untracked source. I merged it into this branch
(`git merge origin/claude/recover-main-foundations-MV0e2`) cleanly. v15
and v20 now live at `agents/v15/main.py` and `agents/v20/main.py`. No
conflicts.

## Files touched this session

**New:**
- `lib/geo/rotation.py` — rotation_alignment, drift_window, my_cluster_centroid
- `lib/compound.py` — fleet_path_safe, compound_bonus
- `lib/mission_book.py` — MissionBook + global BOOK
- `agents/v21_compound/main.py` — copy of v20 + 3 hooks
- `submissions/v21_compound.py` — 333 KB bundle
- `scripts/outcome_histogram.py` — fleet-outcome instrumentation (Rule 38 tool)
- `tests/test_rotation_alignment.py`, `tests/test_mission_book.py`, `tests/test_compound_filter.py`
- `audit/2026-05-16-v21-compound-plan.md` — design + Rule 38 verification

**Modified (touch only):**
- `scripts/bundle_agent.py` — added geo/rotation, mission_book, compound to DEFAULT_LIB_ORDER
- `state/current.md` — added v21_compound note
- `state/hypothesis-board.md` — added H-compound
- `state/mechanism-ledger.md` — added v21-compound-proposer-augmentation row

---

## Prior handover (Day-N PM 2026-05-14) — kept for reference

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

## Day-N PM fix-weak-game-starts-NhDQ3 (this branch — capture-and-secure probe)

Live geo v3.1 (#52643676) settled at **985.5 μ** (−77 μ vs v7_pv).
Reframed PI's "weak starts" lens via loss-mode classifier on 52 live
replays (`scripts/classify_losses.py`):

| Bucket | geo 5/14 | v7_0 5/13 |
|---|---:|---:|
| `opening_lost` | 33 % | 68 % |
| `mid_economy_lost` | **67 %** | 32 % |

Openings already work — opening_boost dropped opening-lost share 35 pp
vs v7_0. The bottleneck is **post-capture security**:
`lost_back_rate` median = **100 %** in losses vs 30 % in wins; median
turns-held before flip-back = **13**. Three lib proposers shipped but
**not wired in geo**: `propose_recapture_missions`,
`propose_drain_missions`, and any pre-emptive garrison. PI's two
observations (bounced fleets ~+11 % ships/cap gap; orbital-drift
1.10× over-rep in capture-then-lost) compound but each ~10 % effect.

**4 variants built, 3-opp panel (gate Wlo ≥ 0.50):**

| Variant | vs v7_0 | vs v4_planner | vs v3.5.1 | Verdict |
|---|---|---|---|---|
| **geo_recap** | **64.1 % PASS** | 56.2 % INCONCL | **62.5 % PASS** | **2/3 PASS, mean 60.9 %** |
| geo_garrison | 42.2 % INCONCL | 56.2 % INCONCL | 56.2 % INCONCL | flat ~52 % |
| geo_drift | 32.8 % **FAIL** | 31.2 % **FAIL** | 43.8 % INCONCL | panel FAIL |
| geo_all | 26.6 % **FAIL** | (killed by user) | (killed by user) | drift-poisoned |

**Did NOT submit** — geo_recap mean 60.9 % minus the local-overpredict-2x
~6–7 pp discount ≈ ~54 % live, not decisively above v7_pv's 1062.2 μ.
Pushing would evict v7_pv (Rule 12 caveat).

Audit: `audit/2026-05-15-secure-variants-wrap.md` (load-bearing wrap-up,
merge-to-main recommendations, and the JAX next-step queued).

**Falsified this branch:** drift-discount scoring axis
(`agents/geo_drift/main.py`); geo_all combined (drift-contaminated, not
an independent axis falsification per Rule 37). Neither merged to main.

## Falsified or dead (across all three branches today)

- All 7 v7_X chooser-axis variants (chooser-axis exhausted)
- All 8 R2R3 proposer/scoring variants (PV local optimum)
- geo v1's posture multipliers, greedy-multi allocator (~-30 pp each)
- geo v2.4-2.7 wallclock "fixes" (-17 to -20 pp each)
- geo v3.0 composite value head as agent value_fn (-19 pp)
- geo v3.2 empty_out + tap_capture cumulative (-4 pp)
- JAX depth-2 game-vmap (GPU compile fundamentally too slow)
- geo_drift (drift-discount Voronoi scoring; panel FAIL on v7_0+v4_planner)
- geo_all (drift-contaminated combined; not an axis falsification per Rule 37)

## Next-session first-action (ranked by EV / cost)

1. **Bundle + submit `geo_recap`** (~20 min). 3-opp panel pre-submit
   parity gate. Will evict v7_pv from rolling-last-2 — explicit PI
   approval required. Expected live ~54 % vs panel 60.9 % (calibrated for
   local-overpredict-2x). Builder: `scripts/bundle_agent.py
   agents/geo_recap`.

2. **JAX-port `score_candidate` inside the winner** (~1–2 h + smoke).
   `lib/game/jax/jax_score.py` has `score_candidate_jax_pure_jit`
   (30–70× speedup claim from 2026-05-13 sub-phase 1 audit). CPU-only
   on the ladder; the win is JIT-fused-Python, not GPU. Costs:
   first-turn cold compile (mitigated by import-time pre-warm),
   ~150 MB bundle (kernel-push path), float-parity drift (1e-3 tolerance
   vs scalar env). Payoff: per-turn 500 ms → 10–80 ms, frees K=15+
   search and depth-2 maximin within the same 1 s actTimeout. Template:
   `agents/jax_v7_0/main.py`. Gates: Rule 2 + Rule 30 two-tier smoke.

3. **Re-test garrison on top of recap as `geo_recap_garrison`**
   (~30 min). The 5/14 garrison probe was tested vs vanilla geo, not
   on top of recap. Combined "recap captures lost ground, garrison
   holds new ground" has plausible additivity.

4. **Re-check geo's ladder Score** (5 sec). If μ has climbed to 1050+,
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
