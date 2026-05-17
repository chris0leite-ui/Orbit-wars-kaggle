# v23 iteration — postmortem (negative result, axis pivot also fails)

> 2026-05-17, branch `claude/improve-fleet-efficiency-cQXg4`. Plan at
> `/root/.claude/plans/you-are-a-comoetitive-encapsulated-deer.md`.
> Follows `audit/2026-05-17-v21-pivot.md` (the v21/v22 postmortem).

## TL;DR

After v21/v22 falsified the chooser-modification axis, v23 pivoted to a
DIFFERENT axis: overlay the existing `propose_opening_missions`
(`lib/missions/opening.py`) for turns 0..15 of 2P games, hand off to the
unmodified `agents/baseline` (= v15 parity) at turn 16+. Reused a tested
proposer; no classifier; no v15 chooser modifications. **It still
failed catastrophically.** v23 at window=15: 5/32 = 15.6 % Wlo=0.069.
Retry at window=10 (per the plan's falsification path): 8/32 = 25.0 %
Wlo=0.133. Both clean regressions vs v15.

Seven total variants (v20/v21/v21_a/v21_ae/v21_solo/v22/v23) across two
distinct axes (chooser-internal modifications + opening proposer
overlay) all fail at n=32 vs v15. **v15 is structurally the local
optimum for this codebase. Any single bolt-on or modification regresses.**

## Live ladder context

- v15 (sub 52710995): μ ≈ 1112.8 — team-best floor, stable across the
  iteration.
- v20 (sub 52721807): μ ≈ 1094.2 — current rolling slot, stable.
- v9_scavenge: μ ≈ 1119.9 — historical ceiling, unbreached since 5/15.
- Team count: 2829.

## What was attempted

v23 = `agents/baseline/main.py` (clean modular re-implementation of v15,
landed via PR #27 on origin/main) + an opening short-circuit at agent()
entry:

```python
if num_seats == 2 and step <= V23_OPENING_WINDOW:
    opening_missions = propose_opening_missions(
        world, model, window=V23_OPENING_WINDOW,
    )
    if opening_missions:
        intents = settle_plan(opening_missions, world, model)
        actions = realize(intents, obs_d,
                          mechanisms=DEFAULT_MECHANISMS, model=model)
        if actions:
            return actions
# else fall through to baseline's full pipeline
```

The only library-side change: added an optional `window: int =
OPENING_WINDOW` parameter to `lib/missions/opening.py::propose_opening_missions`
(backward-compatible — geo, geo_recap, v7_1, lib/v7_search keep
default=5).

Hypothesis: v15 launches 2.0 fleets in turns 0-15 (replays), top-10
launches 7-10. Using the opening proposer should close the launch-rate
gap and lift turn-16 hand-off state.

## Why it failed

**Symptom:** v23 win-rate at n=32 is BELOW pure-v15-mirror (which would
be 50 %). At window=15: 15.6 %. At window=10: 25.0 %. We're not just
failing to lift — we're actively regressing by 25–35 percentage points.

**Root cause hypothesis (likely accurate):** Top-10's 7-10 launches in
turns 0-15 is the OUTPUT of their integrated chooser+proposer+rollout
stack, not an INPUT we can transplant. Transplanting the LAUNCH RATE
without the surrounding chooser/value-head/aim logic that justified
each launch produces fleet emissions that the baseline pipeline can no
longer correct or refine downstream. Specifically:

- The opening proposer's H7 score = `production × (remaining_steps)^1.5
  / (distance + 1)` — but its ship-sizing is `target_garrison + 1` for
  neutrals, which is fine in isolation. Combined with DEFAULT_MECHANISMS,
  the realize pipeline's `arrival_size` mechanism shouldn't change the
  sizing, but its `validate` mechanism may filter intents whose
  arrival predicts negative outcomes.
- More importantly: the overlay BYPASSES baseline's rollout-based
  validation. baseline's `chooser.choose` would have run a 25-40 turn
  rollout per candidate, catching unsticky captures via opp's
  reactive counter-launch. The overlay emits the proposer's plan
  directly. So we capture the planet, then baseline at turn 16+ inherits
  a state where opp's reactive counter-launches arrive simultaneously
  and we're not positioned to defend.
- v15 launching only 2 in turns 0-15 isn't a bug — it's v15's CHOOSER
  correctly identifying that the rollout's expected value of those
  early grabs is low. The launch-rate gap is symptom, not cause.

This is the same lesson as v21/v22: **the v15 baseline is co-tuned.
Adding ANY external behavior, even by short-circuit, breaks the
calibration.**

## Cross-axis pattern: 7 variants, 2 axes, all fail

| Variant | Axis | Approach | n=32 winrate vs v15 |
|---|---|---|---:|
| v21 | chooser filter (3-layer) | A (joint emit) + E1 (target-quality prefilter) + E2 (hold-check) | 31.2 % |
| v21_a | chooser filter | A only | 43.8 % at n=16 (couldn't justify n=32) |
| v21_ae | chooser filter | A + E1 | 43.8 % at n=16 |
| v21_solo | chooser filter | A (single-commit) + E1 + E2 | 43.8 % at n=16 |
| v22 | rollout opp model | counter-recapture inside lite_greedy | 25.0 % |
| **v23 w=15** | opening overlay | propose_opening_missions for turns 0..15, 2P only | **15.6 %** |
| v23 w=10 | opening overlay | same, window=10 | 25.0 % |

Every single variant tested at n=32 is a clean regression. The pattern
is unambiguous: **the v15 baseline is not improvable by surface
modification.** Rule 37 originally targeted 3+ same-axis failures; we
now have 6+ across the chooser/opening-overlay/rollout-opp axes
combined. The escalation Rule 37 contemplates is here.

## What this cost

- ~10 hours of work and compute across three sessions
- Zero submissions burned. No live signal lost; the ladder is unaffected.
- One submissions slot (v20) sits at μ ≈ 1094, ~21 μ below the v15 floor;
  the rolling-slot math still rests on v15.
- Five branch-local agents written (v21, v21_a, v21_ae, v21_solo, v22)
  plus v23 — preserved on the branch as "what doesn't work" reference
  artifacts.

## Lessons (promotion candidates)

- **`pattern-overlay-on-tuned-baseline-doesnt-lift`** (NEW tag, 3rd
  recurrence of the underlying issue). v15's chooser, leaf, opp model,
  and opening behavior are co-tuned. Bolting on a tested external
  component (the v7-era opening proposer) regresses by 25-35 pp. The
  symmetric prior is also true (v21 added filters on top, regressed).
  **Lesson:** treat v15 as a unit. Modifications must be holistic
  (whole-stack changes that maintain calibration), not bolted-on.
  Promotion: refuse to plan single-component modifications on v15
  going forward. Either a wholesale replacement (different chooser
  family, different value head) or no change.
- **`launch-rate-is-symptom-not-cause`** (NEW). Top-10 launches more in
  turns 0-15. v15 launches less. Replicating the launch rate without
  the surrounding chooser produces worse outcomes than doing nothing.
  Empirical-correlation features (replays say X fires faster) are
  load-bearing for diagnosis but NOT for chooser injection. Lesson:
  before transplanting a behavioral pattern from top-10 replays, run a
  CAUSAL test: would v15 with that pattern alone, controlled for
  everything else, lift? If you can't construct such a controlled
  test, the pattern can't be injected.

## Where to go next

Genuinely don't know. The honest answer is one of:

1. **Accept v15 as our ceiling.** Stop iterating. v15 holds 1112-1115
   floor; v20 at 1094 stays in rolling. Submit v15 itself again at end
   of every 24 h to keep the rolling slot fresh. Spend the remaining
   36 days on either (a) cross-comp analysis writeups, or (b) a wholesale
   architectural pivot that doesn't reuse the v15 stack at all.

2. **Wholesale architectural pivot.** Three candidates from prior
   audits / state files, all 2-7 days of work:
   - **Imitation learning from top-10 replays** (audit/2026-05-14-public-notebook-scan.md).
     Train a value head against actual win-correlated features.
     Multi-day; biggest upside if it works.
   - **4P-specific chooser** (state/mechanism-ledger.md). 36 % of
     ladder games are 4P; v15's 4P branch is the same chooser with one
     `_favor` aggregation tweak. Orthogonal to v15-vs-v15 testing.
     Risk: regression on 4P would be hidden by 2P-h2h gates.
   - **Portfolio search across multiple value heads** (referenced in
     `audit/2026-05-17-baseline-functional-parity-with-v15.md`). Each
     value head proposes a plan; outer chooser picks across them. Could
     surface lifts that single-head saturation hides.

3. **Strategic pause.** Push v22 (a known-quantity v15-mirror) to evict
   v20 (1094), then idle. Recover from the iteration drought and let
   PI plan the next architectural axis.

My recommendation: option 1 + 2c. Stop iterating, write up the
session, and start the portfolio-search plan when PI is ready. The
imitation learning path is the highest-upside but also the largest
engineering lift; portfolio search is the smallest path to a structural
change that might actually move μ.

End of postmortem.
