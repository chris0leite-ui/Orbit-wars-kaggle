# Block E snipe-only MVP — refactor rationale + parity proofs

> Date: 2026-05-11
> Branch: `claude/bootstrap-agentic-systems-lqnm6`
> Plan: `/root/.claude/plans/set-4-workers-as-zippy-puzzle.md`

## What landed

Mission framework primitives + a snipe-only agent that is **bit-for-bit
behaviorally identical to v2**:

- `lib/mission.py` — `@dataclass Mission` (mission_class, src_id,
  target_id, ships, score, eta, note) with `to_intent()` boundary.
- `lib/missions/snipe.py` — `propose_snipe_missions(world, model)`
  builds one Mission per (our-source, non-our-target). Same ROI score
  + same predicted-owner-at-arrival filter as v2's strategy.
- `lib/planner.py` — `settle_plan(missions, world, model)` v0 solver:
  per-source greedy (pick the top-score mission per source, NO
  no-double-commit enforcement). Refactor parity, not behavioural lift.
- `agents/v3_snipe/main.py` — composes the three primitives + the
  unchanged `DEFAULT_MECHANISMS` stack.

13 new tests covering Mission, propose_snipe_missions, settle_plan
(see tests/test_mission*.py + tests/test_planner.py). Full suite green
at 202 passed / 5 warnings.

## Why v0 settle_plan is per-source greedy (not no-double-commit)

The first draft of `settle_plan` enforced "no two sources may pick the
same target this turn." Quick smoke (v3 vs v2 at 4 seeds, both seats,
n=8) showed v3 lost 0/8 — a real regression, not noise.

Root cause: on dense boards (4P FFA in particular) **multiple sources
concentrating on the same target IS the right play**. The aggregate
fleet captures contested targets that single sources can't afford
individually. v2 already over-commits in this scenario (its WorldModel
doesn't see fleets it's about to send), and that over-commit is a
feature, not a bug, against strong defenses.

Filing this as the planner's gang_up class (v3.1+): coordinated
multi-source arrivals should be modelled as a *mission class* with its
own scoring, not as a planner-level filter. Per-source greedy at v0
preserves v2's behaviour while opening the seam for v3.1+ additions.

Logged friction promotion candidate:
`tag: planner-no-double-commit-regresses-without-gang-up-class` —
parallel to the existing `arrival-ledger-mechanism-without-planner-
regresses` friction. Pattern: filters that drop intents without a
re-allocation lever cost wins.

## Parity proofs

### 2P head-to-head (v3 vs v2, 32 games × both seats)

`audit/tournaments/20260511T055428Z.json` (seeds 42/1/7/13, 4 × 2 = 8
games, no self-play):

| Pair | P0 wins | P1 wins | Draws |
|------|---------|---------|-------|
| v3_snipe (P0) vs v2 (P1) | 0 | 0 | 4 |
| v2 (P0) vs v3_snipe (P1) | 0 | 0 | 4 |

All 8 games reach step 500 with `rewards=[1, 1]` and
`final_ship_delta_p0_minus_p1=0.0`. **v3_snipe and v2 produce the
exact same fleet stream every turn given the same seed** — proving the
Mission → settle_plan → Intent pipeline preserves v2's per-source ROI
greedy exactly.

### 2P broader panel (8 seeds, v3 vs v2 vs roi_baseline + 3 weak)

`audit/tournaments/20260511T055652Z.json` — 6-agent panel, 8 seeds,
no self-play:

| strategy      | mean panel WR | p95 ms |
|---------------|---------------|--------|
| roi_baseline  | 77.5%         | 0.4    |
| v3_snipe      | 70.0%         | 3.8    |
| v2            | 70.0%         | 3.7    |
| enemy_first   | 36.2%         | 1.6    |
| weakest       | 17.5%         | 4.6    |
| baseline      | 8.8%          | 0.1    |

v3_snipe's mean panel WR equals v2's to the decimal. Identical pairwise
records across all panel cells (each won 100% vs weakest, enemy_first,
baseline; 50% vs roi_baseline; 0% / all-ties vs each other).

### 4P FFA panel (8 seeds × 4 seats, fixed weak background)

`audit/tournaments/ffa-panel-20260511T055758Z.json` — focal-vs-fixed-
background design (`weakest, enemy_first, baseline`), 32 games per
focal:

| focal         | first-place rate | Wilson 95% | p95 ms |
|---------------|-------------------|------------|--------|
| v3_snipe      | 30/32 (93.8%)     | [79.9, 98.3] | 3.5  |
| v2            | 30/32 (93.8%)     | [79.9, 98.3] | 3.4  |
| roi_baseline  | 29/32 (90.6%)     | [75.8, 96.8] | 0.3  |

v3 matches v2 to the game in 4P FFA against weak background. The
shared-background design ensures focals are compared on the same
opponent mix — this isn't noise from differing matchups.

### E.2 hard gate

`v3_snipe` self-play, 10 episodes (seeds 0-9): **0 crashes, 0 timeouts,
all DONE.** Mechanism stack is the unchanged Block A physics
(`validate, arrival_size, lead_aim_v2, sun_avoid,
path_clears_other_planets, oob_guard`).

## What this unlocks

v3.1+ work (deferred per PI):

- `lib/missions/reinforce.py` — top up friendly planets the timeline
  says will fall (under-threat detection via WorldModel.owner_at).
- `lib/missions/recapture.py` — re-take planets predicted to fall to
  an enemy fleet.
- `lib/missions/gang_up.py` — coordinated multi-source arrivals on
  the same step. THIS is the right place for cross-source coordination
  (a mission class with a score that considers other sources), NOT in
  settle_plan as a filter.

Adding any of these is a single new file + integration line in
`agents/v3_*/main.py` + a settle_plan multi-class arbitration upgrade.
The framework boundary is in place.

## Caveat — predicted μ band overlap

The plan flagged this and it's worth reiterating: the roadmap's Block E
target is μ 1000-1150. v2 is already at μ=1025.5 trending up. v3.0 IS
v2 functionally, so **DO NOT SUBMIT v3.0 to the live ladder** —
rolling-last-2 eviction would cost v1.2/roi (μ=1001.4) for a zero-Δμ
gain. v3.1+ with a real lift over v2 should be the next submit.
