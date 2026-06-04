# Migration plan — Producer engine as host, our pieces ported in

**Status:** plan, not yet started. Next session starts implementation.
**Anchor data:** our champion (`champ_computeByShips_on`, live μ ≈ 1185)
vs Producer (Slawek Biel's published agent, live μ ≈ 1200) — n=32 local
A/B 2026-06-04: 13/32 wins for us (40.6%), Wilson [0.255, 0.577].
Producer wins ~60 % of games. Confirmed at n=32; not noise.

## Ethics note — Producer is a sparring partner, not a submission

We will NOT submit Producer (or any thin wrapper around it) to the Kaggle
ladder. It is Slawek Biel's publicly published work; submitting it as
ours would be plagiarism. The vendored copy at `agents/producer/`
(cherry-picked from main, commit `0cc08da`) exists only as:

1. A stronger local A/B opponent than `v7_0_drop_one`.
2. An engine substrate we port OUR mechanisms onto, building a NEW agent
   that uses Producer's scoring lens but adds the pieces our champion has
   that Producer lacks.

The deliverable is a hybrid agent: Producer's engine + our pieces. That
agent IS submittable because the value-add is genuinely ours.

## Why Producer as host (not the reverse)

The frontier_circulation failure we hit this session was a consequence of
a single root cause: Producer's regroup composes with his scoring because
both use the same scalar (distance-decayed enemy pressure / exact flow
diff). Our scoring lens is per-`(source, target)` trade ROI; pressure-
routed ships land at destinations our chooser ignores.

Fixing this by porting Producer's scoring INTO our chooser is possible
but expensive: `sparse_launch_flow_delta` is a vectorized torch
per-step combat simulation over 18 turns, deeply tied to his
`PlanetGarrisonStatus` tensor cache. Porting to numpy is multi-week work
with high risk of subtle scoring drift.

Hosting on Producer means we adopt his scoring directly and add our
mechanisms as new candidate-generation patterns INSIDE his planner, where
they will be scored consistently with everything else.

## Core principle — no post-passes

Every piece we add must become either a candidate-generation extension OR
a scoring extension. **No post-passes outside the scorer.** Post-passes
are what broke circulation; they are explicitly forbidden in this
architecture.

## Inventory: which of our pieces are worth porting

| Our piece | Port? | How it goes inside Producer | Effort | Expected lift |
|---|---|---|---|---|
| Multi-source coalitions | YES | Producer's `LaunchSet` already supports `L > 1` contributors per candidate; his scorer and greedy already handle it. Just generate same-arrival-tick bundles for high-value targets. **Drops our joint LP solver entirely — enumerate + score replaces it.** | medium | HIGH |
| Multiple sizes per (src, tgt) | YES | Add `capture_floor` and `2 × capture_floor` as additional `cand_send` alongside `safe_drain`. His scoring naturally picks the best. | small | medium |
| Wait-then-fire | YES | Add a `wait_N` axis to candidate generation. Aim/eta use projected positions `movement.x[wait_N]`. Scorer needs to know launch happens at step `wait_N`, not 0 — small extension to `sparse_launch_flow_delta`. | medium | medium |
| Adaptive K horizon | YES | Replace fixed `config.horizon = 18` with `K_eta = max(10, 20 − 10·step/30)` passed into `build_target_shortlist`. Keep H=18 for the scoring forecast. | trivial | small-medium |
| Comet-aware aim | MAYBE | Extend `intercept_angle` to handle comet trajectories (path-indexed lead, not orbital). Include comets in `attack_target_mask`. Only if A/B shows we lose comet-rich maps. | medium-high | small (map-dependent) |
| `emit_threat_reinforcements` | NO | Producer already has `friendly_flip_targets` in the same shortlist as attacks. Already covered. | — | — |
| Sniper bundles | NO | Subsumed by multi-source coalitions. | — | — |
| `compute_by_ships` per-source breadth | NO | Producer's shortlist is global; per-source breadth would need restructuring. His scoring already weights "good" targets correctly. | — | — |
| `drain_idle_stockpile_to_opp` | NO | A/B already showed it doesn't lift; under Producer's broader scoring lens, even less likely to help. | — | — |
| `chooser_trajectory`, `cheap_marginal_value`, `launch_rules`, all post-pass drains | DELETE | Replaced wholesale by Producer's planner + scoring. | — | — |

Roughly **5,000+ lines deleted, ~500-1,000 added.** Net code shrink with
better scoring.

## Order of work — cheap first, gates between

Each step gated on n=32 Wilson-lo ≥ 0.55 vs the previous step's bundle
before continuing. Per Rule 45.

**Step 1 — Build a working `producer_plus` agent skeleton.**
- Wrap `agents/producer/main.py` in our own `agents/producer_plus/`
  directory so we can modify the planner without touching the vendored
  Producer (which stays as a clean sparring reference).
- Confirm `producer_plus` is bit-identical to Producer at this point
  (no changes yet). n=8 self-A/B should be 4/8.

**Step 2 — Adaptive K_eta.**
- Replace fixed horizon-as-K with adaptive schedule.
- A/B `producer_plus_adaptive_k` vs `producer` at n=32.
- Cheap experiment; one-line scoring extension.

**Step 3 — Multiple sizes per (src, tgt).**
- Generate `[capture_floor, 2 × capture_floor, safe_drain]` per (src, tgt).
- A/B at n=32 vs Step 2.

**Step 4 — Multi-source coalitions.**
- For high-value targets, also emit L=2-4 same-arrival-tick bundles.
- Drop our LP solver from the plan entirely — enumerate + score replaces it.
- A/B at n=32 vs Step 3.
- BIGGEST expected lift; Producer is explicitly single-source.

**Step 5 — Wait-then-fire.**
- `wait_N` candidate axis (N in {1, 2, 3, 5}).
- Extend `sparse_launch_flow_delta` for launches at non-zero step.
- Add ledger persistence ONLY if wait_then_fire shows lift but is fickle
  turn-to-turn.
- A/B at n=32 vs Step 4.

**Step 6 — Comet-aware aim (optional).**
- Only if archetype split shows we lose comet-rich maps to non-Producer
  opponents (Producer himself excludes comets, so won't show in head-to-
  head).
- Real port — comet math differs from orbital.

**Step 7 — Submit.**
- Pre-submit smoke (Rule 46): bundle + parity smoke + wallclock under
  1000 ms max for seed 7.
- Rule 42 push-claim board, evict the older of the two rolling subs.
- Single-shot, explicit PI approval per Rule 1.

## Architectural mismatches we must handle

1. **Torch in final bundle.** Producer's submission proves torch works on
   Kaggle. Bundle size grows but allowed. Smoke must verify wallclock.
2. **Slot vs ID indexing.** Producer indexes by tensor slot; ours by
   planet ID. Translation only at I/O boundary — interior is all slot-
   based.
3. **Tensor vs dict garrison cache.** Producer's `PlanetGarrisonStatus`
   replaces our `WorldModel.timelines`. We delete the dict version.
4. **All `BASELINE_*` env vars go away.** Producer's planner is
   monolithic; we can't selectively disable mechanisms.

## Risks and what each risks

| Risk | What it costs | Mitigation |
|---|---|---|
| Wallclock blowup with larger candidate sets (sizes × wait × coalitions) | Producer's scorer is vectorized — cost is linear-ish. Smoke at each step. | Hard cap on candidate count via top-K pre-prune by `score_candidates` result from a coarse single-size pass. |
| Comet aim port mathematically incorrect | Bad shots → lost games on comet maps | Unit-test against engine-validated shots before integration. |
| Coalition enumeration explosion (C × L^k) | Memory and time | Cap to top-K target × top-K source pairs at each L; prune by reachability first. |
| Dev period without a stable champion | LB position erodes if Step 4-5 take weeks | Keep current `champ_computeByShips_on.py` (sub 53332500) as backstop in rolling pair; do not evict until hybrid clears live μ ≈ 1185. |
| Producer's strengths don't transfer to ladder play | Local A/B over-estimates lift | Step 7 is single submit with PI sign-off; we accept this risk knowingly. |

## What needs PI sign-off when implementation starts

1. Confirm the migration direction is still active when next session
   opens (no new live observation has invalidated it).
2. Sign off on the `agents/producer_plus/` directory name and the
   "wrap-and-modify" pattern (vs forking inside `agents/producer/`).
3. Sign off on the gate threshold per step (Wilson-lo ≥ 0.55 unless PI
   wants tighter / looser).
