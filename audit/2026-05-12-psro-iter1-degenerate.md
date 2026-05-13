# 2026-05-12 — PSRO iter 1: degenerate Nash (pool not diverse enough)

## TL;DR

**First PSRO iteration produced a degenerate mixed Nash: {v7_minimax:
1.0, others: 0.0}** because v7 strictly dominates every other policy
in the pool. PSRO meta-agent would be functionally identical to v7
(already submitted as #52568317). NOT submitting v8_psro_meta this
iteration. Pool needs anti-v7 policies before PSRO can produce a
non-trivial Nash.

This is the #1 risk explicitly flagged in
`audit/2026-05-12-game-theory-next-iteration-research.md` (step 5
risk register). Empirical outcome validates the risk hypothesis.

## Setup

```
Pool:           {v7_minimax, v3_snipe, precision, roi}
Seeds:          3 per ordered pair, both sides
Games per pair: 6 (3 seeds × 2 seat assignments)
Total games:    36 (6 pairs × 6 games)
Tournament:     28.8 min wallclock
Output:         audit/tournaments/psro_payoff_v1.json
Solver:         nashpy support_enumeration (4 equilibria found — degenerate)
```

## Payoff matrix (antisymmetric, ∈ [-1, +1])

```
              v7    v3    prec   roi
v7         +0.00 +1.00  +0.33  +0.67
v3_snipe   -1.00 +0.00  +0.33  +1.00
precision  -0.33 -0.33  +0.00  -0.33
roi        -0.67 -1.00  +0.33  +0.00
```

Per-pair raw scores (6 games each):

| Pair | Result |
|---|---|
| v7 vs v3_snipe | **6-0** v7 dominant |
| v7 vs roi | 5-1 v7 strong |
| v7 vs precision | 4-2 v7 modest |
| v3 vs precision | 4-2 v3 modest |
| v3 vs roi | 6-0 v3 dominant |
| precision vs roi | **2-4** precision LOSES to roi |

## Nash result

```
Method:                support_enumeration
Equilibria found:      4 (degenerate; nashpy RuntimeWarning issued)
Game value (row):      +1.000
Row mixed strategy:    v7=1.0000, v3=0.0000, precision=0.0000, roi=0.0000
```

Game value = +1.0 means: when the row player plays the pure-v7
strategy, they win ALL games against the column player's worst-case
mixture (which must include the v7 column to be best-response). For
zero-sum: this is the max-min value achieved by v7 pure strategy.

## Why degenerate

v7 strictly dominates all 3 other pool members:
- v7 vs v3: +1.0 (6/6 wins)
- v7 vs precision: +0.333 (4/6)
- v7 vs roi: +0.667 (5/6)

A strictly dominant strategy in zero-sum 2P → Nash = pure that
strategy. By construction PSRO cannot improve on v7 within this pool.

## What this tells us

1. **PSRO infrastructure works**. Tournament + solver + meta-agent
   skeleton are correct. Re-usable for future pool variations.

2. **Our pool is not diverse enough**. All 4 policies share the same
   strategy class to v7 (v3-derived heuristic; precision is closest
   to "different class" but still loses to v7 4/6). PSRO can only
   add value when the pool contains genuinely anti-v7 policies.

3. **Local-tournament strength ≠ ladder strength**. v3_snipe is at
   live μ=1041.8 (our team's peak); v7 wins 6/6 vs v3_snipe locally.
   The ladder opponent distribution is different from our pool's.
   This is a CALIBRATION FACT: local probes are necessary but not
   sufficient for ladder-μ prediction.

4. **precision_v3 underperforms expectations**: 2-4 vs roi (33%);
   precision is at live μ=1009 vs roi at live μ=1006.9. Local-vs-
   ladder gap is even larger for precision than for v3.

## Why not submit anyway

A 100%-v7 meta-agent submission would:
- Take a daily submission slot
- Run on Kaggle as v7-equivalent
- Evict our current rolling-last-2 oldest entry (which is v3.5.1
  at μ=988.8)
- Likely settle at the same μ as v7_minimax (#52568317, currently
  PENDING)

Net value: zero. Slot wasted.

If v7's live μ turns out to be similar to σ-equiv (1041.8), then v7
IS our best agent and submitting a v7-clone via PSRO meta is
redundant. If v7 underperforms σ-equiv, PSRO meta would underperform
too because it just plays v7.

## What to do next

To make PSRO produce non-degenerate Nash, the pool needs an
**anti-v7 policy** — something that v7's maximin doesn't model
correctly. Three concrete paths:

### Path A — anti-v7 hand-crafted (~1-2 days)

Identify v7's exploitation pattern. v7's maximin enumerates:
- C = {v3-incumbent, drop-smallest-launch}
- O = {v3-from-opp-POV, drop-smallest-of-O0}

v7 ASSUMES the opponent plays v3-class. An opponent who plays a
DIFFERENT class (e.g., pure mirror, or strike-window-only, or
defensive-coast-only) is OUTSIDE v7's M=2 opp model. v7 would
mis-predict their actions and pick suboptimally.

Build an agent that:
- Plays defensive-coast (no offense) until step 100
- Then a burst-aggression mode
- Specifically times attacks to LAND when v7's defender prediction is wrong
- Likely will lose vs v3-class but win vs v7 (since v7 mis-models)

If this succeeds, pool = {v7, v3_snipe, anti_v7, roi} should give a
non-degenerate Nash.

### Path B — PSRO best-response loop (~3-5 days)

The full PSRO algorithm doesn't just compute mixed Nash once. It:
1. Compute mixed Nash over current pool
2. Train a best-response policy AGAINST that mixture
3. Add the best-response to the pool
4. Repeat

Implementation: needs a "best-response trainer" — either RL (which
we don't have time for) or hand-crafted minimax against a specific
opponent (essentially Path A).

### Path C — abandon PSRO for now (~0 days)

Accept that v7 is the local maximin winner and pivot to a different
game-theoretic direction (e.g., wider M in v7, or recapture missions
for absolute strength). Re-attempt PSRO later when we have RL
infrastructure to do Path B properly.

## Decision tree

```
v7 live μ (#52568317) >= σ-equiv-v1 (1041.8):
  → v7 is our best agent; iterate strength on v7 base
  → PSRO is moot until we have anti-v7 policies

v7 live μ < 1041.8:
  → v7's local 75% W/D vs v3 doesn't translate to ladder
  → calibration miss; pivot to σ-equiv-v3 + recapture missions
  → PSRO over {v7, σ-equiv-v3} would be more useful (since
     they'd be genuinely different in performance)
```

## Recommendation

Defer v8 submission. Watch v7's live μ over next 24h. Make next-iter
decision based on whether v7 confirms or violates the local 75% W/D
prediction.

If pursuing PSRO further: Path A (anti-v7 hand-crafted) is the
cheapest path. Build that as v9, re-run tournament with expanded
pool, re-solve Nash. Time: ~1-2 days.

## Calibration ledger update

| Prediction | Outcome | Note |
|---|---|---|
| PSRO Nash will be degenerate | YES (degenerate, pure v7) | Risk #1 from research note materialized as predicted |
| Expected PSRO μ: 1070-1110 | N/A (not submitting) | Would have been ≈ v7's μ since mixture = pure v7 |

Open question for v7's actual μ: if v7 settles below σ-equiv-v1's
1041.8 (regression), the entire "local 75% W/D" framework is suspect
and needs recalibration.

## Files

- `audit/tournaments/psro_payoff_v1.json` — 36-game raw + payoff matrix
- `audit/tournaments/psro_payoff_v1.nash.json` — Nash solution
- `agents/v8_psro_meta/main.py` — meta-agent (NASH_PROBS hardcoded
  to degenerate {1.0, 0.0, 0.0}; explicit comment "NOT RECOMMENDED
  to submit until pool includes anti-v7 policies")
- `scripts/psro_tournament.py` — re-usable for next iteration
- `scripts/psro_solve.py` — re-usable for next iteration
