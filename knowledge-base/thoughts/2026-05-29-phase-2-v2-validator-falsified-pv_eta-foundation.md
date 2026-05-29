# 2026-05-29 — Phase 2 v2 LightGBM Booster validator falsified vs pv_eta; pv_eta is the new foundation

> Session: `claude/competition-objective-alignment-hqNVM`, day-N session
> resumed across multiple boots. PI present throughout.

## What happened today

Continued the Phase 2 v2 validator-filter line started in PM4/PM5
(2026-05-28). Today's work:

1. **Regenerated corpus at 1000 ms wallclock** (was 100 ms in PM5). 17
   opponent cells × 5 games × 2 seats = 85 games, 16,420 shots, balanced
   pos_rate 0.482. Goal: match the eval wallclock distribution.
2. **Re-trained LightGBM Booster** on 1000 ms corpus. val_acc 0.829,
   Brier 0.119, recall@0.30 0.900 — the model learns well.
3. **A/B sweep on threshold** (validator-on-baseline vs bare baseline):
   - Threshold 0.30 (default): 9/32 = 28.1 % (Wilson-lo 0.156). FAIL.
   - Threshold 0.05: 20/32 = 62.5 % (Wilson-lo 0.453). PASS-ish.
   - The 100 ms-corpus booster at 0.30 had also failed at 10/32 = 31.2 %.
4. **Diagnostic trace** (one game, validator-on-baseline @ 0.30): drop
   rate 85.6 % of scored emits. Median turn drops every non-self-reinforce
   shot. The per-shot filter at 0.30 was eviscerating coordinated
   multi-source attacks — each individually marginal, collectively
   decisive. This is exactly Reframe A from yesterday's brainstorm.
5. **Trace with validator-on-pv_eta @ 0.05**: drop rate **0.9 %** (1 of
   115 scored emits). pv_eta is already a highly selective emitter; the
   validator has nothing to filter. The wrapper just adds 150 ms overhead.
6. **A/B validator-on-baseline @ 0.05 vs bare pv_eta** (PI-directed
   correction — pv_eta is the live champion lineage, not bare baseline):
   - Run 1, n=32: 16/32 = 50.0 % (Wilson-lo 0.336). INCONCLUSIVE.
   - Run 2, n=32 (same seeds): 8/32 = 25.0 % (Wilson-lo 0.133). FAIL.
   - Pooled n=64: 24/64 = 37.5 %. Validator-on-baseline loses to pv_eta
     by ~12-13 pp on average.

## What the data says

The Booster has real information (val_acc 0.83), but applied as a
**per-shot post-hoc filter** it:

- **Helps a weak inner agent** (bare baseline at +34 pp from 28 → 62.5 %).
- **Does nothing for a strong inner agent** (pv_eta drop rate < 1 %).
- **Loses head-to-head against pv_eta** (37.5 % pooled).

**Diagnosis:** per-shot supervision is the wrong unit. Coordinated multi-
source attacks have per-shot P~0.10-0.30 individually but per-attempt
P~0.6-0.8. A per-shot filter cannot see the attempt. Per-shot **information
combined with the chooser's own coordination logic** might still win,
but applied as a post-filter the model competes with the chooser instead
of complementing it.

**Live-ladder context refresh (kaggle competitions submissions orbit-wars):**

| Sub | File | Date | μ (public) | In rolling pair? |
|---|---|---|---|---|
| 53131296 | baseline_validated.py (PM5 MLP filter) | 2026-05-28 23:22 | 1081.3 | YES (lower) |
| 53117942 | baseline_leaf_pv_2p.py | 2026-05-28 13:55 | 1084.5 | YES (upper) |
| 53111837 | baseline_pv_eta.py | 2026-05-28 09:42 | (evicted; historical peak ~1154) | NO |

Floor is 1081, well below pv_eta's historical peak. pv_eta is the
strongest agent we have empirically.

## PI direction at session end

> "We were going to build really on our latest champion on the latest
> successful submission pv_eta."

**Decision:** Stop iterating the per-shot-filter axis (Rule 37 — 3+
consecutive falsifications on the same axis: thresholds 0.30 / 0.10 /
0.05 across two corpora, plus pv_eta-as-inner = no-op). The "filter as
post-hoc gate" primitive is dead.

**Foundation going forward:** pv_eta. All next mechanisms wrap, augment,
or replace components of pv_eta — not bare baseline.

## Three reframes to follow up on (next-session menu)

### Reframe A — Booster P(success) as a chooser input, not a filter (PRIORITY)

Don't filter pv_eta's emits. Instead, expose the Booster's per-shot
P(success) inside the chooser's leaf-value function as one more
additive term:

```
candidate_score = ship_delta + production_term + γ * gamma_discount + λ * ML_P_success
```

Re-uses the existing booster — no retraining. Implementation surface:
modify `score_candidate_v4` in `agents/baseline/chooser_trajectory.py`
to accept an ML score. Per-emit Booster call already takes ~150 ms in
the wrapper; can amortize by predicting once and threading the result
into multiple scoring slots.

A/B: focal = pv_eta + ML-augmented chooser, opponent = bare pv_eta.
Wilson-lo 0.50 at n=32 is the submit gate (Rule 45).

**Cost:** ~1 day. Lightest path. Tests whether the model's information
is useful *as a signal*, not as a gate.

### Reframe B — Per-target value head (medium swing)

Retrain the supervision unit: not "does this shot succeed" but **"how
many future ship-deltas does owning planet T at time T+k earn the
focal seat?"** Continuous regression target, not binary classification.

This becomes a real value head — plugs into the chooser's leaf-value
slot, augmenting (or replacing) `predict_garrison_at`. There's partial
infrastructure already on disk: `data/value_head/` and
`data/value_head_distill/` from earlier PM sessions.

**Cost:** ~3-5 days (new corpus + new labels + new train + integration).
Higher ceiling than A; longer path. Do AFTER A's verdict.

### Reframe C — Opponent-emit predictor (big swing, strategic)

pv_eta's chooser does NOT model the opponent's next-turn emits.
`predict_garrison_at` assumes a static opponent. Build an ML predictor:
"given current state, what will opp send next turn?" Feed predicted opp
emits into the chooser's lookahead.

This is the "more strategic, less per-shot" angle PI raised at session
end. Could be conditional (e.g., only for the top-K target planets per
turn) so we don't blow the wallclock budget.

**Cost:** ~5-7 days. Highest ceiling, longest path. Requires retraining
+ integration + careful A/B. Reserve for after A and B verdicts.

### Excluded: full chooser replacement / RL / MCTS

PI's optional framing ("completely replace the chooser") was considered
and **not** prioritised: 25 days to deadline, zero RL/MCTS
infrastructure on this branch. The chooser is the hard-won foundation;
the leaf-value slot is the right insertion point for ML.

## Open questions for next session

1. **A: pure scoring vs hybrid.** Should the Booster score replace the
   PV-eta gamma term, or be additive? PI guess: additive first (no
   regression risk if λ=0 reproduces pv_eta byte-for-byte).
2. **A: λ tuning.** What λ value? Probably need a coarse sweep
   (0.05 / 0.10 / 0.20 / 0.50). Heuristic: choose λ such that ML term
   magnitude ≈ 10-30 % of ship_delta magnitude.
3. **B vs C ordering after A.** If A clears, do we still want B / C, or
   does the chooser-input path suffice? Likely B (target-value head) is
   the next big lift — opponent modeling (C) is more speculative.
4. **Bundle smoke + parity.** Wrapping pv_eta with a new chooser term
   needs a new bundle. test_chooser_pv_eta.py + test_bundle.py must
   stay green; new tests for the ML-augmented scoring.

## What is NOT going to be re-explored (Rule 4 → Rule 37 axis cap)

- **Per-shot filter at higher / lower thresholds** — already swept 0.05
  / 0.10 / 0.30 across two corpora. Family dead vs pv_eta.
- **Per-shot filter with re-rank instead of threshold** — same
  primitive, same supervision unit. Won't escape the "wrong question"
  problem.
- **Wrapping bare baseline (not pv_eta)** — even +34 pp on bare baseline
  only reaches pv_eta-equivalent strength. We have pv_eta already; the
  validator family on bare baseline is dominated.

## Foundation lock

pv_eta is the inner agent for everything going forward unless explicitly
overruled. All future A/Bs gate on Wilson-lo ≥ 0.50 vs bare pv_eta at
n ≥ 32 (Rule 45). The Phase 2 v2 LightGBM booster and its weights
(`data/shot_validator/validator_booster.txt`, 1000ms corpus) are
preserved as inputs for Reframes A / B but no longer treated as a
deployable artifact on their own.
