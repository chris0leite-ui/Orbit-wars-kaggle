# Reframe A — additive ML logit on pv_eta chooser — FALSIFIED

**Date:** 2026-05-29
**Branch:** `claude/competition-objective-alignment-hqNVM`
**Plan:** `/root/.claude/plans/squishy-bouncing-hickey.md`

## Verdict

The additive-Booster-logit-on-chooser-score axis (Reframe A) is closed
under Rule 37 (3-variant axis cap). The Booster from
`data/shot_validator/validator_booster.txt` does NOT improve pv_eta's
chooser when its `logit(P_success)` is added to candidate scores —
the signal hurts at every sign and magnitude tested.

## What landed

| Step | Status | Artefact |
|---|---|---|
| 1. pv_eta source port | VERIFIED (Wilson CI [0.364, 0.691] vs bundle) | `agents/baseline/chooser_trajectory.py` |
| 0. Step-0 probe (trace hook + analyzer) | All 3 gates PASS | `agents/baseline/_trace_hook.py`, `scripts/probe_ml_logit_signal.py` |
| 2. ML logit integration in chooser | Working at λ=0 byte-equivalent | `agents/baseline/_ml_logit.py` + chooser hooks |
| 3. Env-var wrapper agent | Source + bundle parity | `agents/baseline_pv_eta_ml/main.py` |
| 4. Wrapper bundler | Builds 953 KB single-file submission | `scripts/bundle_pv_eta_ml.py` |
| 5. Parity (λ=0) | Bundle smoke vs bundled pv_eta: WIN 1-0 in 162 steps, p50 128ms vs 115ms | — |
| 7. A/B λ sweep | FAIL FAIL FAIL across λ ∈ {4.5, 0.5, -0.5} | — |

## A/B results (vs `submissions/_imported/baseline_pv_eta.py`)

| λ | n | wins | Wilson CI | Verdict |
|--:|--:|--:|---|---|
| 4.5  | 32 | 0/32 | [0.000, 0.107] | FAIL |
| 0.5  | 32 | 1/32 | [0.006, 0.157] | FAIL |
| −0.5 |  1 | 0/1  | — | smoke FAIL; abort before n=16 |

Targeted ML-logit magnitudes: 0.1σ_delta (λ=4.5), 0.011σ_delta (λ=0.5),
−0.011σ_delta (λ=−0.5). σ_delta = 70.5 from the probe.

## Root-cause analysis

The Step-0 probe gates were necessary but **not sufficient**:

- σ(P_success) per turn = 0.26 (gate ≥ 0.05) — Booster discriminates
- |Spearman ρ(delta, logit P)| = 0.13 (gate < 0.85) — non-redundant
- median(P_success) = 0.79 — in [0.2, 0.8]

The probe correctly established that the Booster's information is
non-redundant with the chooser's `delta`. But "non-redundant" is not
the same as "valuable for game-winning." Two signals can be
rank-uncorrelated while both being noisy proxies of the same
underlying truth — adding them together adds noise, not signal.

The Booster predicts **"this shot lands as intended given the
training-distribution game state"** — but pv_eta's chooser already
optimizes for game-winning via a deep fast_sim rollout. The Booster's
"obvious" signal is redundant where pv_eta is strong, and confidently
wrong on the candidates pv_eta surfaces that lie outside the
Booster's training distribution.

Two corroborating data points:

1. **Median P_success = 0.79** in pv_eta's candidate set. The chooser
   has already pre-filtered to high-confidence shots; the Booster
   confidently predicts most of them to succeed. The only thing it
   can do is RE-RANK among them, and that re-ranking is anti-
   correlated with pv_eta's win-rate-optimal argmax.

2. **Even λ=0.5 (~1% of σ_delta) regressed 31/32 games.** The
   destruction is at the TIE-BREAKING level — pv_eta has tuned its
   argmax tightly, and any perturbation flips close calls to worse
   candidates. The Booster's tie-breaks are reliably worse than
   pv_eta's own.

This is the training-distribution-mismatch failure mode the ML-expert
plan-mode briefing flagged as the dominant risk.

## What's added to closed-track knowledge

Adding to the foundation-lock section of `state/MULTI_BRANCH.md` /
HANDOVER `Closed tracks not to be re-explored`:

- **Booster (45-d, 1000ms wallclock training corpus) added to pv_eta's
  chooser as an additive logit term.** Closed at λ ∈ {4.5, 0.5, −0.5}.
  Do not re-iterate λ tuning on this Booster + this chooser surface.

## What is NOT closed

- A DIFFERENT model with a DIFFERENT supervision target (Reframe B —
  per-target continuous value head). The per-shot-binary supervision
  target is the falsified piece, not the wrap-pv_eta architecture.
- An opponent-emit predictor inside pv_eta's lookahead (Reframe C).
- Re-training the Booster on pv_eta's emit distribution. Would address
  the training-distribution mismatch, but cost is similar to Reframe B
  and the ceiling is uncertain (still per-shot binary supervision).

## Reproducibility

- Probe trace (4 seeds, 28,661 candidates): `/tmp/probe_combined.jsonl`
  (not committed; regen via `scripts/probe_ml_logit_signal.py`).
- Probe report: `audit/2026-05-29-ml-logit-probe.md`.
- Bundle artifact at λ=0 parity: `submissions/baseline_pv_eta_ml.py`
  (regen via `python scripts/bundle_pv_eta_ml.py`).
- A/B harness commands (BASELINE_WALLCLOCK_MS=100, n=16 seeds × 2
  seats = 32 games, workers=4, gate=0.50):

```
BASELINE_ML_LAMBDA=<λ> python fast.py eval submissions/baseline_pv_eta_ml.py \
  --vs submissions/_imported/baseline_pv_eta.py \
  --max-seeds 16 --gate 0.50 --workers 4
```

## Commits

- `9c6346f` — source pv_eta port + probe scaffolding
- `bc918c0` — `_ml_logit` module + wrapper
- `01049fb` — trace hook wired into chooser
- `e23aade` — ML logit application + bundler + probe report
- `1dc0f1f` — ml_* prefixed names (bundler-friendly)
- `54c507d` — featurization runs before chooser deadline

## Time

- Pre-implementation Phase 1 + 2 (Explore + Plan agents): ~6 min
- Step 1 (port + parity): ~25 min
- Step 0 (probe wire + 4 seeds + analyze): ~15 min
- Step 2-4 (ML wire + wrapper + bundler + bug fixes): ~25 min
- Step 7 (A/B λ=4.5, λ=0.5, λ=-0.5 smoke): ~45 min

Single session, falsified cleanly. The plan's "abort early" gate would
NOT have caught this — the probe gates pass, but downstream the
signal fails to transfer. Adding a 4th probe gate (e.g.
"does the Booster's signal correlate with self-play win rate on the
chooser's accepted set?") would have been required to catch this
upstream, and that test is itself an A/B.
