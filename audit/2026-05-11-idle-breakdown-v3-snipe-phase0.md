# Phase 0 idle-source decomposition — v3_snipe self-play

> Branch `claude/optimize-ship-strategy-tDPXx`. Phase 0 of plan at
> `/root/.claude/plans/you-are-a-mathematician-resilient-papert.md`.
> Replaces the audit's coarse `idle_source_rate` heuristic with a
> classifier that says **why** each idle source went idle.

## TL;DR

In **v3_snipe self-play**, idle sources are dominated by
`MECHANISM_DROP` (69-100% of all idle classifications). The two
sub-causes — `validate` and `arrival_size` — share one root pattern:
**a single source's garrison is smaller than the per-target capture
cost**, so the intent is dropped rather than launched undersized.

| Bucket | 2P self-play | 4P self-play | Targeted by experiment |
|--------|--------------|--------------|------------------------|
| `MECHANISM_DROP` | **100%** (19,896) | 69% (6,403) | Exp 1 (endgame), **Exp 4 (gang-up)** |
| `LEDGER_LOSS`     | 0% | **28%** (2,628) | **Exp 6 (Hungarian)** |
| `NO_PROPOSALS`    | 0% | 2.5% (236) | Exp 2 (expansion) |
| `GATE_REJECTED`   | 0% (no floor yet) | 0% | reserved for Exp 5 phase-floor |
| `RESERVE_HELD`    | 0% (no dial yet)  | 0% | reserved for Exp 1 reserve |

Caveat: this is **8 self-play games, not live-ladder replays.** Local
credentials don't have `kaggle.json`, so the v3_snipe live-archive
(submission 52544634) couldn't be fetched. The decomposition validates
the instrumentation; the actionable per-bucket priorities will likely
shift once live replays are decoded.

## Method

**Phase 0 instrumentation (this PR):**
- `lib/planner.settle_plan` now accepts an opt-in `reasons: dict[int, str]`
  out-param. Sources with no proposer output get `NO_PROPOSALS`;
  sources whose every candidate was over-committed by earlier this-turn
  picks get `LEDGER_LOSS`.
- `lib/intent.realize` accepts the same `reasons` param. Any intent
  dropped by a mechanism is recorded as `MECHANISM_DROP:<mech_name>`;
  intents that reach the final emit filter with no aim or zero ships
  get `MECHANISM_DROP:final_emit_*`.
- `scripts/episode_postmortem.py` passes a `reasons` dict through both
  surfaces, merges them per turn, and writes
  `audit/live-episodes/<sid>/postmortem/idle-trace.csv` (one row per
  idle classification: `episode_id, size, result, t, phase, src_id,
  reason_full, reason_bucket`).
- `scripts/generate_selfplay_replays.py` (new) writes self-play games
  in the `episode-<seed>-replay.json` schema the postmortem consumes.

**Data:** 4 × 2P + 4 × 4P self-play games of v3_snipe vs v3_snipe,
seeds 42-45 and 142-145. v3_snipe wins every game by construction (it
plays itself; the seed determines the spawn and orbit phasing, both
sides apply the same strategy). 2P games run to step 500; 4P games
end early (221-368 steps) because three sides eliminate one quickly.

## Findings

### 1. The "old" idle_source_rate metric undercounts

The pre-existing `idle_source_rate` is `(n_sources - settle_plan_chosen) /
n_sources` — it measures sources that **didn't pick a mission**, not
sources that **didn't launch a fleet**.

In v3_snipe 2P self-play, `idle_source_rate = 0.0%` (every owned planet
picks a mission), yet **11.8 sources per turn** still produce no
launch because `arrival_size` and `validate` drop their intents
downstream. The new bucket classifier captures all of this.

### 2. `arrival_size` and `validate` are the dominant single-source caps

```
MECHANISM_DROP sub-causes (totals across 2P + 4P, 8 games):
  arrival_size                13,627    (51%)
  validate                    12,192    (45%)
  path_clears_other_planets      385    (1.4%)
  sun_avoid                       57    (0.2%)
  oob_guard                       38    (0.1%)
```

Both top sub-causes share the same exit condition (`lib/mechanism.py:70,131`):

```python
if intent.ships > src.ships:
    continue   # drop intent — single source can't fund the capture
```

The semantics differ:
- `validate` fires on the strategy's **proposed** size (`target.ships
  + 1`).
- `arrival_size` fires on the **production-growth-adjusted** size
  (`target.ships + target.production * eta + 1` or
  `model.ships_at(target, eta) + 1`).

Together they are saying: **the mission picker is happy to choose
targets the source planet can't afford alone**, and the mechanism
layer silently drops the order.

### 3. 4P mid-game shifts LEDGER_LOSS to 40%

```
4P phase breakdown:
  early (t<150)   n=2628    MECHANISM_DROP=100%
  mid   (t<400)   n=6639    LEDGER_LOSS=40%, MECHANISM_DROP=57%, NO_PROPOSALS=4%
```

By 4P mid-game, multi-source contention is real: 28% of all 4P idle
classifications come from a source whose top target was already
covered by an earlier this-turn pick AND whose runner-up candidates
were also covered. This is exactly the case the Hungarian assignment
in Exp 6 addresses.

The early-game uniformity (100% MECHANISM_DROP) reflects the small
starting garrisons (10 ships home, 5-99 elsewhere) — every source
proposes captures it can't single-handedly fund.

### 4. NO_PROPOSALS is small but only in 4P

2.5% of 4P idle is `NO_PROPOSALS` — the snipe proposer returned no
candidates for a source-planet (likely because all remaining targets
are too distant given fleet-speed, or because the predicted owner
flipped before our arrival). In 2P self-play it's 0%. Both numbers
are likely understated for live opponents who survive longer; revisit
once live replays are available.

## Experiment priority shift (from plan §Experiment Matrix)

The Phase-0 decision rule was:

> If `NO_PROPOSALS > 30%` → Exp 2 first. If `MECHANISM_DROP > 30%` →
> Exp 4 first. Etc.

Self-play evidence overwhelmingly satisfies the `MECHANISM_DROP > 30%`
arm. Recommended re-ordering:

1. **Exp 4 (multi-source simultaneous-arrival gang-up)** — directly
   resolves the `arrival_size > src.ships` and `validate ships > src.ships`
   drops by pooling 2-4 sources. Targets the biggest bucket.
2. **Exp 1 (endgame burn-through)** — bypasses `arrival_size`
   strictness at step≥470. 2P games show `late` phase is 100%
   MECHANISM_DROP; this should yield free captures.
3. **Exp 6 (Hungarian global assignment)** — primarily for 4P mid-game
   where LEDGER_LOSS is 40%. Defer until Exp 4 + Exp 1 ship.
4. **Exp 3 (denial-value ROI)** — orthogonal to idle ships; still in
   the pipeline but doesn't move the dominant bucket.
5. **Exp 5 (phase multipliers)** — depends on Exp 1's phase scaffolding.
6. **Exp 2 (expansion mission)** — small immediate payoff (~2.5% in 4P
   self-play). Re-evaluate once live replays surface its share.

A secondary "off-plan" candidate emerges from this decomposition:
**filter the mission proposer to candidates the source can fund alone**
(or proactively redirect to a smaller-cost target) before the
mechanism layer drops the intent. This is a cheap fix to the validate
bucket and is worth a half-day spike before Exp 4.

## Critical artifacts

- `audit/live-episodes/SELFPLAY_PHASE0_2P/postmortem/{roll-up.json, idle-trace.csv}` — 2P data.
- `audit/live-episodes/SELFPLAY_PHASE0_4P/postmortem/{roll-up.json, idle-trace.csv}` — 4P data.
- `lib/planner.py::settle_plan` — `reasons` out-param.
- `lib/intent.py::realize` — `reasons` out-param.
- `scripts/episode_postmortem.py` — bucket classifier + CSV writer.
- `scripts/generate_selfplay_replays.py` — local replay generator (new).

## Next session

- Once Kaggle credentials are available, re-run on submission 52544634
  (v3_snipe live) to confirm whether `NO_PROPOSALS` and `LEDGER_LOSS`
  rise vs weaker live opponents.
- Implement Exp 4 (`lib/planner_multisrc.py`) targeting the `arrival_size`
  + `validate` bucket. Local A/B: 32-seed 2P + 16-seed 4P vs frozen
  v3.4. Secondary metric: bucket reduction via this same instrumentation.
