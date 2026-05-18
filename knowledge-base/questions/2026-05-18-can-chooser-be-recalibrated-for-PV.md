# Q: can the chooser's emit gate be recalibrated to absorb the PV-term scale?

Date: 2026-05-18

## Context

The chooser emits a candidate iff `Δ = leaf_value − baseline_value > 0`.
The leaf-value head was upgraded in bug #15 v2 to include a
production-PV term. The term adds ~100 value units per captured
planet at leaf. The chooser's `Δ > 0` gate was calibrated against
the pre-PV base ship-delta scale (typical Δ swings of 20-50). With
PV active, capture candidates routinely score Δ in [50, 200] vs
baseline → emit-by-default → over-emission → drained sources → 39.6%
winrate vs 50% baseline.

## The question

Is the chooser's emit-gate threshold tunable in a way that absorbs
the PV scale WITHOUT discarding PV's signal value?

Specifically: can we set the gate to `Δ > expected_PV_contribution
× num_planets_at_leaf` (so PV's uniform inflation is normalised out,
but the candidate-specific Δ remains the deciding factor)?

Or alternatively: should we normalise the PV term itself (e.g.,
divide by `EPISODE_STEPS_TOTAL` so it's in the same range as base
ship-delta)?

## Why it matters

Two outcomes are possible:

1. **YES, recalibratable**: bug #15's structural fix lives, and we
   restore the pre-fix 50%+ baseline AND keep PV's signal value.
   Net: small positive (sanity oracle passes; PV credits real
   captures; chooser still calibrated).

2. **NO, the calibration is structural** (e.g., PV interferes with
   the joint candidate scoring or the SETTLE_TURNS leaf in
   `chooser_trajectory.py`): we keep PV disabled and accept that
   the sanity oracle stays xfail. Net: zero (no regression but no
   gain).

The session-wrap recommendation is to disable PV in production now
(Option A in the postmortem) and answer this question in a future
session before re-enabling.

## Investigation sketch

1. Profile `Δ` magnitudes per candidate over 20 games with PV on
   and off. Histogram them.
2. Identify the `Δ` value that pre-fix chooser emitted with ~50%
   winrate (call it `Δ_emit_floor`).
3. With PV on, find the matching threshold `Δ_emit_floor + shift`
   where `shift` matches the PV's typical contribution.
4. A/B with `Δ > Δ_emit_floor + shift` as the new gate.

If the histogram is bimodal (PV makes "drain" candidates vs "real"
candidates separable), recalibration is easy. If it's a uniform
shift across all candidates, harder.

## Pointers

- `agents/baseline/chooser_trajectory.py:score_candidate_v4` —
  where `Δ` is computed.
- `agents/baseline/chooser_trajectory.py` emit gate (probably in
  the calling site that consumes the scored candidates).
- `lib/value_heads.py:160-200` — PV term.
- `knowledge-base/thoughts/2026-05-18-PV-term-recalibration-debt.md`
  — the broader principle.
