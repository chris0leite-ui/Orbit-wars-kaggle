# 2026-06-15 — The opening-search wide step (and how we got there)

**Status at write time:** A/B pair fired to the ladder (subs **53708789**
`seq_strength_opening` = the wide step, **53708787** `seq_strength` = fresh
baseline). Warming up; read in ~24 h.

## The one thing the next session must know

**`agents/producer/` on `claude/festive-knuth-roggck` is the BARE orbit_lite
engine (0 flags). It is NOT the 1280 ladder agent.** The real agent is
`agents/producer_plus/main.py` — **70 `PRODUCER_PLUS_*` flags** (veto,
reactive-floor, FFA-leader, multi-size, opening-search…) on top of orbit_lite.
It + its matching `orbit_lite` (16 modules, has `opp_projection`) + its bundler
(`scripts/bundle_producer_plus.py`, ~20 config-variants) live on
`origin/claude/awesome-clarke-ixy57v`. I brought them onto this branch (commit
`1e2e747`); the real agent now runs here (`seq_strength` flags → 55 ms median).
**Every local A/B before that finding used the wrong, weaker base — treat those
results as void.**

## The dead-end map (don't re-walk these)

Every *compute/model* lever came up empty, confirmed two ways (local + ladder):
- Search wrapper over producer (fast_sim rollouts): **tied** at 28% (weak
  opp-model) and **25%** (producer opp-model). Search adds nothing.
- Deeper internal planning (3× horizon): **hurt** (19%). Aggression dial: tied.
- Distillation / hand-condensed fast policy: failed (strength = the expensive
  forward sim; no cheap copy preserves it).
- On the **real ladder**: imitation-learning (`oracle_rw`) 1018, RL (`rl_v7`)
  938 — both far below the producer heuristic line (1280). The leaderboard
  already proved models/compute don't beat the heuristic here.
- Defensive direction (`garval` = garrison-value + source-safety): **1230**,
  below seq_strength 1280 → the "survivor"/robust-opening idea is likely already
  refuted on the ladder. Did NOT ship it.

Conclusion: producer sits at a tuned local optimum; the ceiling is *strategy*,
not compute. **Wide** steps (different play), not **deep** ones.

## Why opening-search is the wide step

Real losses (producer_plus's 89 ladder games, 49.4% WR): a **third cluster at
step ~87–130 (early death)** — overwhelmed before economy matters. producer_plus
has a real, implemented **opening beam-search** (`_opening_search_plan`,
beam-64, reserve/hold filters) gated by `OPENING_SEARCH`, **OFF in every shipped
variant**. Turning it on is wide (a search subsystem, not a knob), spends the
~98% idle headroom exactly where we bleed, and was never ladder-tested on the
strong base. `seq_strength_opening` = seq_strength + `OPENING_SEARCH=40`.

## Methodology lesson (the real unlock)

Local self-play is **referee-blind** — it can't reward fixing a flaw the
self-opponent shares (deep & aggressive both came back inconclusive for this).
The **ladder is the A/B oracle**, and submissions are free to spend on testing
until the deadline (only the kept-2 at 2026-06-23 matters). Stop trusting local
verdicts; ladder-test wide variants.

## OPEN QUESTION (read ~2026-06-16)

`rating(53708789 opening) − rating(53708787 baseline)`:
- **opening clearly higher** → headroom-in-opening works → tune window/beam
  (try 30/60, beam 32/96), ship it, then sweep other untested flags.
- **opening lower** → opening-search hurts the strong base → revert; pick the
  next untested wide flag (force_concentration? denial_calibrated? coalition_k?).
- **parity** → need more games or a bigger opening effect (raise window).

## FLAG

Bringing the producer_plus lineage onto `festive-knuth` mixes two lineages
(this branch was the search/experiment line). If continuing producer_plus work,
consider doing it on `awesome-clarke` (its native home) to avoid drift.
