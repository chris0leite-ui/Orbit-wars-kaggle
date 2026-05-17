# 2026-05-17 — fleet-efficiency iteration, negative result (concise)

> Session: `claude/improve-fleet-efficiency-cQXg4`. Full postmortems
> and dead agent code live on that branch; this doc is the merge-to-main
> summary so the next session inherits the lessons without the
> code-archaeology overhead.

## What was attempted

PI brief: minimise wasted fleets (failed captures, lost-back, comet
shots). Empirical motivation from a replay-mining pass on 16 live
games (8 each of v15 and v20, sub-IDs 52710995 / 52721807):

- **15 % of launches target comets; 100 % MISS** (~20 ships/game wasted).
- **60–70 % of captures lost back within 50 turns**; median hold = 8 turns.
- **43–53 % of lost-backs are UNDEFENSIBLE** (outnumbered locally at
  recapture in R=30 game-unit neighborhood).

Seven variants built across two axes; all FAILED head-to-head vs v15
at n=32 (= 16 seeds × 2 seats; n=16 turned out to be too noisy and
delivered false-positive "parity" reads on multiple variants).

| Variant | Axis | Approach | n=32 winrate vs v15 |
|---|---|---|---:|
| v21 (full stack) | chooser filter | joint-emit + cheap target-quality prefilter + rollout hold-check | **31.2 %** Wlo=0.18 |
| v21_a / _ae / _solo | chooser filter | ablations of v21 — A only / A+E1 / A+E1+E2 single-commit | 43.8 % (n=16; never bumped to n=32) |
| v22 | rollout opp | wrap `lite_greedy_policy` with counter-recapture moves at every rollout step | **25.0 %** Wlo=0.13 |
| v23 (w=15) | opening overlay | `propose_opening_missions` short-circuit for turns 0..15 of 2P games | **15.6 %** Wlo=0.07 |
| v23 (w=10) | opening overlay | same as above with smaller window per falsification path | 25.0 % Wlo=0.13 |

## Why everything failed (the durable lesson)

Two axes, same root cause: **v15's chooser is co-tuned end-to-end.**
Its leaf `_favor`, its reactive-opp rollout, its multi-wait grid, and
its emit dedup were all evolved together. Adding ANY single-component
modification — explicit filter on top (v21), stronger opp inside
(v22), or specialised overlay around (v23) — breaks the calibration
in some other dimension that overcompensates.

The replay numbers above (comet 100 % miss, 60-70 % lost-back) are real
descriptions of what v15 does. They are NOT a roadmap for improvement.
Top-10 leaders launch 7-10 fleets in turns 0-15 vs v15's 2 — but
transplanting their launch rate without the surrounding stack
regresses 25-35 pp. **Behavioral patterns are the output of an
integrated chooser, not an input you can graft on.**

Five total falsifications now on this broader "v15 surface
modification" axis (counting v16-v20 from prior sessions). Rule 37
contemplates pivot at 3; we are well past escalation. **The next
iteration must be a wholesale architectural change, not a fix.**

## New friction tags (load-bearing)

- **`pattern-overlay-on-tuned-baseline-doesnt-lift`** (3rd recurrence
  of the underlying co-tuning issue across multiple sessions). Refuse
  to plan single-component modifications on the v15 stack. Either a
  whole-stack replacement (different chooser family, different value
  head, different proposer suite) or no change.
- **`launch-rate-is-symptom-not-cause`** (new). Before transplanting a
  behavioural pattern from top-10 replays into the v15 stack, construct
  a controlled test: would v15 with that pattern alone, holding
  everything else, lift? If you can't construct such a test, the
  pattern can't be safely injected.
- **`n16-falsely-shows-parity`** (recurrence of small-n-ab-noise-
  misled-panel). Wilson CI width at n=16 is ≈ 0.45 — literally cannot
  distinguish parity from a 20 pp regression. For any submission-
  gating decision, n=32 minimum. n=16 is for smoke only ("agent
  doesn't crash"), not for verdicts.

## Live ladder (no change through this session)

- v15 (sub #52710995): team-best floor, ~1112 μ.
- v20 (sub #52721807): rolling slot, ~1094 μ.
- v9_scavenge: historical ceiling at ~1120, unbreached for 5+ days.
- Team count: ~2829.

**No submissions burned this session.**

## Pointer for archaeology

Full postmortems + dead agent code + diagnostic scripts live on
`claude/improve-fleet-efficiency-cQXg4`:

- `audit/2026-05-17-v21-pivot.md` — v21/v22 detailed postmortem
- `audit/2026-05-17-v23-postmortem.md` — opening-overlay postmortem
- `agents/v21/` + variants, `agents/v22/`, `agents/v23/` — ~3000 LOC of
  falsified code; useful as "this didn't work, here's exactly what we
  tried" reference for any future session considering the same patches
- `scripts/diag_v21_vs_v15.py`, `scripts/instrument_v21.py` — per-turn
  diagnostic tooling; can be ported if a future iteration needs it

## Next-session recommendation

Don't build a v24 on top of v15. The most plausible wholesale pivots,
in order of engineering lift:

1. **Portfolio search across multiple value heads.** Each value head
   (PV-discount, hold-aware, composite-capture, etc.) proposes a plan;
   an outer chooser picks across them. Different action-space topology
   from drop-one + per-source emit. Smallest lift; touches `chooser.py`
   in the modular baseline.
2. **Imitation learning from top-10 replays.** `data/shot_validator/`
   already has 37k labeled examples (24-dim). MLP head training is
   ready; the pipeline was deferred. Biggest upside; multi-day lift.
3. **4P-specific chooser.** ~36 % of ladder games are 4P. The current
   stack inherits 2P logic with a `_favor` aggregation tweak; a
   dedicated 4P chooser is unexplored. Risk: 2P-h2h gates won't
   detect 4P regressions, so verification needs a 4P-only panel.

For options 1 and 2, the `agents/baseline/` modular foundation (PR
#27) is the right substrate — value, proposer, chooser, opp_model are
already independently swappable.
