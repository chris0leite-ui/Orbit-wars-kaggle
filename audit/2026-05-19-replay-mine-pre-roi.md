# replay-mine pre-ROI — what failure modes show up in live champion games?

> **Phase 1a output** of approved ROI/scenario-gate plan
> (`/root/.claude/plans/no-go-forward-test-fluttering-token.md`).
> Closes `knowledge-base/questions/2026-05-19-do-failure-modes-c-and-
> e-appear-in-live.md`.
> Date: 2026-05-19 AM.
> Branch: `claude/ml-competition-strategy-PFhzM` (post-rebase-abort;
> 21 BPJKs commits NOT on this branch — citations from there are
> off-branch references).

## Source data

`python scripts/replay_mine.py --recent 5 --pull` produced
`audit/replays/replay-mine-2026-05-19.{json,md}`: **56,842 fleets
across 501 episodes from the 5 most recent COMPLETE submissions**.

Per-submission bucket prevalence:

| sub_id   | description                          | μ      | win% | def% | waste_atk% | waste_traj% | inflight% |
|---------:|--------------------------------------|-------:|-----:|-----:|-----------:|------------:|----------:|
| 52784853 | PV-off + bug #3/#4/#12 fixes, 5/18 PM| 1124.1 | 41.0 | 43.2 | **13.7**   | 1.2         | 0.9       |
| 52766596 | Direction B joint cand v3, 5/18 AM   | 1119.0 | 43.3 | 38.3 | 16.8       | 1.0         | 0.6       |
| 52754310 | trajectory chooser v4, 5/17 PM       | 1143.7 | 44.2 | 36.3 | 17.6       | 1.2         | 0.7       |
| **52744856** | **composite+A2 hybrid (LIVE PEAK), 5/17 PM** | **1149.2** | **45.6** | 37.1 | 15.8 | 1.0 | 0.4 |
| 52721807 | v20 chooser dogpile, 5/16 PM         | 1076.7 | 40.6 | 43.5 | 14.3       | 1.0         | 0.6       |

The live peak (52744856) has the **highest capture rate (45.6%)** and
the lowest inflight% — it spends bigger and arrives more often.
52784853 (newest) has the LOWEST waste_attack% (13.7) but ALSO the
LOWEST win% — the bug-fix line traded aggression for precision and
landed below the peak.

## Per-failure-mode answer (the question's closing rule)

### (a) Recapture-loss — **YES, observed, 0.97% prevalence**

`arrived_but_lost` = 554 fleets across all 5 subs (raw count from
the replay-mine roll-up). Means: we launched, fleet arrived, target
was no longer ours-to-capture (recaptured by enemy mid-flight, or
neutral defended itself / was attacked simultaneously).

- Prevalence: 0.97% of all fleets. Equivalent to ~1 lost fleet per
  episode (501 episodes → 554 events).
- Per-sub: roughly proportional to fleet count (~110/sub).
- **Scenario implication:** R1 in V0 suite is justified by observed
  evidence; the rate is small but non-zero, and unrecovered ship
  cost is leveraged (sending ships AT a target you don't recapture
  is a 2x cost vs not sending: you lose the ships AND fail the
  capture). Worth 1 scenario.

### (b) Drift-loss / sun-blocks-raycast — **YES, observed, 1.1% prevalence**

`waste_trajectory` = 599 fleets. Breakdown from raw outcomes:
- `oob` (out-of-bounds — drifted off the map) = 453
- `vanished_in_space` = 94
- `sun` (clipped the sun) = 52

The `vanished_in_space` bucket is the old "comet collision /
misclassified planet hit" classifier residue; the 2026-05-17 swept-
pair fix at `lib.game.interpreter.swept_pair_hit` resolved most of
those. 1.1% prevalence is the irreducible geometric-bug floor; the
plan's D1 scenario is justified but lowest-priority of (a-c).

### (c) Garrison-counter — **YES, observed, 13.9% prevalence** ← biggest detectable failure mode

`bounced_enemy` = 7913 fleets across all 5 subs. Means: launched at a
target with enemy garrison, force < garrison + 1, fleet died on
arrival without capturing. (`bounced_neutral` = 433 is the neutral-
defender analogue.) Combined: 8346 / 56842 = 14.7% of all fleets.

- Prevalence: 13.9% just from the enemy-garrison case alone.
- Variance across subs: 13.7% (52784853 newest, with PV-off+bugfix)
  to 17.6% (52754310 trajectory-v4) — the 5/18 bug-fix line shaved
  ~2-4pp off this bucket; suggests #3/#4/#12 had real targeting
  precision improvements but at a cost (lower win% too).
- **NOT a pure "(c) garrison-counter" failure** — `bounced_enemy`
  also includes simple "we under-shot a static enemy with stale
  garrison estimate." Many will be true (c) (opp counter-launch
  arrived between our launch and our arrival, increasing the
  effective garrison), but the bucket cannot distinguish.
- **Scenario implication:** G1 is the highest-priority scenario
  (largest failure surface). Bundle 2-3 sub-flavours into G1: pure
  garrison-undershoot, opp-counter-arrival, simultaneous-attack
  defense-buffer.

### (d) Split-majority coordination failure — **N/A from buckets; corroborated negatively**

Bucket data cannot detect this — it's a "what we DIDN'T launch"
pattern. The plan calls for manual per-episode inspection of 2-3
mid-game turns from 52744856 (live peak — the agent closest to
top-10) to confirm/refute the canonical "100+100 vs 50, solo
exposes source" shape.

**Manual inspection deferred to next iteration:** 56,842 fleets is
a lot of data for visual; need a targeted detector. Heuristic
candidate: scan replays for `step_t` where we hold ≥2 planets with
ships >50 AND there's a neutral/enemy ≤80 ships in reach, AND we
emitted 0 launches OR 1 launch insufficient to flip. Build that
detector later (pre-Phase-2 if cheap, otherwise pre-Phase-4).

**Indirect evidence:** the BPJKs aggression-deficit data (off-branch
on origin/main) shows `mean_garrison_at_launch` d=+0.82σ vs top-10
universally — we leave bigger garrisons at source when launching.
That is consistent with (d) ("we should have split-launched from
two sources but instead picked one and undersized it"). NOT proof
of (d), but consistent.

- **Scenario implication:** SM1 in V0 suite is justified by PI's
  named pattern + BPJKs indirect evidence + plan-of-record detector
  build. Keep as priority.

### (e) Distant-planet idleness — **YES, corroborated by off-branch behavioural data**

Bucket data cannot detect this directly either. But the BPJKs audits
on origin/main (`audit/2026-05-18-archetype-action-audit-gap-vs-
even.md`, `archetype-action-audit-allcells.md`) measured the
behavioural fingerprint across 50 top-10 replays vs 100 of our
submission 52710995 (v15) replays and found two UNIVERSAL
deltas (consistent across archetypes):

- `launches_per_turn` d=+1.26σ (top-10 vs us) — top-10 launches
  ~2× more often per turn.
- `mean_garrison_at_launch` d=-0.82σ — top-10 leaves smaller
  garrisons at source after launching (i.e. send a larger fraction
  of the source planet).

Plus two CONDITIONAL deltas (in gap cells but not even cells):

- `mean_total_ships` d=-0.60σ — we hoard.
- `ships_growth_per_turn` d=-0.83σ — production accrues without
  spending.

The headline is an **aggression deficit**: we under-launch and
over-garrison. That is precisely the (e) "distant-planet idleness"
pattern at the team level. The pattern is universal (not archetype-
specific), so a single scenario DI1 capturing it is sufficient.

- **Scenario implication:** DI1 is corroborated by quantitative
  off-branch data. **Priority-1** in the V0 suite (largest single
  effect on our ceiling; universal pattern; quantified).

## Scenario priority ranking after Phase 1a

Combining bucket prevalence and behavioural-data corroboration:

1. **DI1 (e) distant-idleness** — universal aggression deficit
   (d=+1.26σ / -0.82σ). Largest single source of our ceiling.
2. **G1 (c) garrison-counter** — 13.9% of all fleets is bounced.
   Highest bucket-visible failure rate.
3. **SM1 (d) split-majority** — PI-named, BPJKs aggression-deficit
   data consistent with this. Build detector to quantify.
4. **R1 (a) recapture-loss** — 1.0% prevalence; real but low ROI.
5. **D1 (b) drift/sun-clip** — 1.1% prevalence; lowest priority.
6. **S1/S2/S3 sanity** — floor check.

This deviates from the plan's equal-priority framing for (a-e); the
revised priority is data-justified (Phase 1a output as designed).

## Newly-surfaced patterns NOT in PI's named five

None obviously. Across 56,842 fleets the bucket distribution is:
- 82.7% productive (win 42.9% + defense 39.8%)
- 15.7% waste-attack (subsumes c)
- 1.1% waste-trajectory (subsumes b)
- 0.5% other

No new bucket-visible failure mode was surfaced that the named five
don't already cover. (A 6th pattern, if it exists, would need a
behavioural-fingerprint analysis like BPJKs' — not in scope for
this branch right now.)

## Recommended Phase 1c V0 scenario list (revised)

Keep all 8 scenarios from the plan, but reorder by priority and
size DI1 + G1 first since they have the biggest evidence base:

| name | failure mode | source-evidence | size |
|---|---|---|---|
| **DI1** | (e) distant-idleness | BPJKs aggression-deficit d=+1.26/-0.82σ universal | multi-turn |
| **G1**  | (c) garrison-counter | 13.9% of all fleets `bounced_enemy` in live data | single-turn + multi-turn flavours |
| **SM1** | (d) split-majority | PI canonical 100+100 vs 50; BPJKs garrison data | multi-turn |
| **R1**  | (a) recapture-loss | 1.0% `arrived_but_lost` in live data | single-turn |
| **D1**  | (b) drift/sun | 1.1% `waste_trajectory` in live data | single-turn |
| **S1**  | sanity: obvious capture | floor check | single-turn |
| **S2**  | sanity: defense needed | floor check | single-turn |
| **S3**  | sanity: idle when no profit | floor check | single-turn |

## What stays open

- Manual per-episode walk for (d) — deferred until SM1 scenario is
  drafted; the act of authoring SM1 will surface the right
  detector predicates.
- Cherry-pick or full rebase of BPJKs's seed-panel infra
  (`lib/seed_panel.py`, `data/seed_panel_128.json`, `fast.py
  --geometry-panel`) before Phase 4 A/B. Cost: ~1-2h reconcile.
  Benefit: variance-reduced A/B with archetype-stratified breakdown
  surfaces flavour-dependent regressions before live.

## Next action

Phase 1b: `tests/scenarios/base.py` with `Scenario` ABC, single-turn
+ multi-turn flavours, reusing `tests/test_bundle_oracles.py`
helpers (`_planet`, `_obs`, `_emit`). Then Phase 1c — DI1 and G1
first (highest priority), then SM1, R1, D1, then sanity S1-S3.
