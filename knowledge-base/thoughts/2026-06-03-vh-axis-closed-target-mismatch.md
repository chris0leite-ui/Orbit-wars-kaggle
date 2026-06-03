# 2026-06-03 — The VH target was wrong, not the wiring

## The clean experiment

Phase D shipped a value head trained from scratch on 100 self-play games
of the live state-driven-K champion (no VH, no Tier-2). Training
gates passed clean: Spearman ρ = +0.386 on game-level held-out
validation, σ(label) = 497.8, walker parity exact. The model has real
rank signal — this is not a tuning question.

Both wirings I tried (additive λ=1.0; rerank top-K=10) catastrophically
regress vs the un-VH champion mirror at n=32:

- Additive λ=1.0 → 0/32 (Wlo=0.000)
- Rerank K=10 → 2/32 (Wlo=0.017)

Reference: un-VH state-K → 62.5% (Wlo=0.453). vs random the agent
plays normally (16/16 = 100%, Wlo=0.806). So nothing about the trace
hook, corpus gen, training, or model inference is broken. The
*deployment* is sound. The *value* the model provides isn't.

## Why I think the target is wrong

The chooser's leaf-delta + PV-discount already integrates ship swings
over 10-30 turns using the state-driven horizon K. The VH target is
**K=10 ship-delta** — the same quantity the chooser already computes
on a longer, state-adaptive window with a better discount model.

In effect, we trained a regression head to predict a NOISIER
APPROXIMATION of what the chooser already had. Spearman ρ = +0.386
isn't measuring "VH adds information beyond what the chooser computes"
— it's measuring "VH agrees mildly with the chooser's existing
ordering." When we use the VH to override (additive) or reorder
(rerank), we replace a careful 30-turn PV-discounted integral with a
noisy 10-turn linear regression on 14 features. We lose information.

This explains both failure modes simultaneously:
- Additive: adds noise of σ=100 to a signal of O(10-100). Predictable.
- Rerank: replaces the chooser's careful order with a noisier order
  even within K=10 already-good candidates. Less obviously
  predictable, but same root cause — the VH doesn't carry
  marginal information.

## What this means for learned-head approaches in this codebase

A learned head should target something the chooser **cannot** compute
itself. Candidates:

1. **Terminal value at end-of-game** — what's the seat-share at step
   500 conditional on this candidate? The chooser doesn't roll out
   that far. The corpus is trivially available from existing replay
   data. Caveat: huge variance; would need careful target shaping.

2. **Opponent-response prior** — given the current state and our
   intended candidate, what's the probability mass over opponent's
   next-N moves? The chooser uses a hardcoded opp policy in
   rollouts; a learned head could replace that. (This overlaps with
   the Tier-2 effort; pivot path may be cleaner.)

3. **Capture-realisation probability** — does an emitted candidate
   actually land + own its target K turns later? This is the
   "physics-waste" question Rule 47 already flags. The chooser
   approximates it via `predict_fleet_fate`; a learned head could
   refine that estimate with opponent-aware adjustments.

4. **Opening-prior / first-N-turns advantage** — a head that scores
   opening moves only, where the chooser's PV-discount has no data
   yet (early game = mostly speculation). Narrow scope.

What NOT to do: train any head whose target is "expected ship-delta
within the chooser's existing horizon." Re-falsified today.

## What carries forward

The corpus-gen + training + inference pipeline is now intact on this
branch (was missing before — ported from hqNVM). The rerank
infrastructure (`BASELINE_VH_RERANK_K`) is in place and gated off by
default. The bundler can inline arbitrary VH models. Future VH
attempts on this branch only need to (a) generate a new corpus with a
DIFFERENT label, (b) retrain, (c) flip the env var. No re-port work.

The model file (`data/value_head/value_head_model.txt`) is gitignored,
so the broken-target Phase D3 artefact stays only on local disk; no
risk of someone bundling it into a submission accidentally.

## The session lesson

I spent ~6 hours iterating on a head where the issue wasn't
calibration, wiring, or model quality — it was that the head was
trying to predict something the chooser already computes better. Two
of the three pre-flight questions for any learned-head proposal
should be:

- Q-A: **Does the chooser already compute this target, or a strict
  super-quantity?** If yes, the head is redundant and will only add
  noise.
- Q-B: **What would the head know that the chooser doesn't?** If no
  concrete answer, the head is decorative.

Rule 6 ("heuristics before heavy compute") generalises here: the
heuristic version of the head is the chooser's existing scoring.
Adding a learned head should only happen when the heuristic version
demonstrably leaves value on the table.
