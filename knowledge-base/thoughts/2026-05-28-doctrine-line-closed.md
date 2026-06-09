# 2026-05-28 — Doctrine line closed: three nulls in a row

Branch: `claude/game-theory-winning-strategy-SEU7P`. Today wraps the
reach-frontier doctrine investigation. **Status: closed, falsified.**

## What we tried

1. **v1 chooser** (B1-B4 build sequence): closed-form ρ, Hungarian
   assignment, λ_loss=0.1, validate_physics=True post-prefilter.
   Result: 0/20 vs baseline. Diagnosed: `hold = max(0, ρ_opp − ρ_me)`
   collapses to 0 mid-game; chooser silent 87% of turns.
   `audit/2026-05-27-rf-v1-root-cause.md`.

2. **v2 chooser** with `MIN_HOLD_FLOOR=30` floor + gang-up via
   `solve_multi_turn`. Result: 0/32 vs baseline. Multi-launch
   activity 4× higher (157 vs 35 launches/game) but win-rate didn't
   move. `audit/2026-05-27-rf-v2-b5-triage.md`.

3. **4P delayed-launch cushion** as a baseline wrapper. Doctrine
   prediction (eval-metrics §5: 4P winners' first-capture median is
   step 137). Result: 4/32 vs baseline's 26/32 against the same
   nearest background — 6× worse, not just worse.
   `audit/2026-05-28-4p-cushion-falsified.md`.

## Why all three failed

The doctrine math is right (production-integral IS the score, the
n=92 share separation of 0.488 between winners and losers is real).
What the doctrine got WRONG was treating descriptive empirical
fingerprints (4P winners launch later, hold-fraction discriminates)
as prescriptive policy.

The likely truth: those fingerprints are **correlation, not
causation**. Winners launch later in 4P because their geometries
gave them time, not because waiting helped. Winners have higher
hold-fraction because their captures were positionally good, not
because they targeted high-hold candidates.

Forcing a baseline-strength agent to MATCH the fingerprint does not
put it in the geometric position where the fingerprint is
profitable. It just imposes a constraint that gives competent
opponents free expansion.

## What ships from this work despite the null

Durable artefacts that are net-positive regardless of the doctrine
verdict:

- **Rule 48 measurement substrate.** `fast.py --save-replays` +
  `scripts/measure_hold_times.py --replay-dir` + the share-of-
  integral aggregator. Reusable for any future agent eval.
- **Bundler unbreak.** `lib/joint_solver/lp.py`'s broken
  `agents.baseline.strategic_lp` import replaced with an inlined
  30-LOC greedy fallback. `DEFAULT_LIB_ORDER` extended for
  `kinematic_table`, `joint_solver/columns`, `joint_solver/lp`.
- **Doctrine + design + eval-metrics docs.** Stay in tree as
  reference. The doctrine math and the n=92 empirical study are
  durable knowledge of what's been tried and what's known.
- **Falsified-variant agents.** `agents/reach_frontier/` and
  `agents/baseline_4p_cushion/` kept in tree as the reproduce-the-
  null reference.

## Lesson — process

Before next session: when a doctrine-derived "improve baseline" idea
surfaces, the protocol should be:

1. Identify the SPECIFIC fingerprint the doctrine predicts.
2. **Run a counter-experiment first**: force baseline to match the
   fingerprint with the simplest possible wrapper. n=32 vs same-
   strength opp. If lift is negative, the fingerprint is correlation
   not causation; stop.
3. Only then build the full operationalisation.

This would have stopped the v1 chooser at the "is closed-form ρ even
correlated with winning baseline games" probe instead of the full
build. Cost saved: probably 2 sessions.

## What's next

The chooser/doctrine line is exhausted. Future improvements to the
Kaggle peak (currently μ=1125.2 from sub 53088099) need to come from
a different axis:

- baseline knob tuning (BASELINE_GAMMA, BASELINE_VALUE_HEAD).
- ML value head replacement / augmentation.
- 4P-specific behavioural changes ANCHORED to a counter-experiment
  not to the n=92 fingerprint.
- Process improvement: the Rule 48 substrate as a pre-submit gate.
