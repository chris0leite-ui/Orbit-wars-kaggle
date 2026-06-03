# 2026-06-03 — VH-on-state-driven-K integration axis closed (6 falsifications)

## TL;DR

The Phase D retrain shipped a clean value-head model (Spearman ρ=+0.386,
σ(label)=497.8, walker parity exact). All training gates passed. vs
random the agent scores 16/16=100% — no wiring bug. Yet **both** wirings
of the VH onto the live champion (state-driven-K + joint-aggressive)
catastrophically regress vs the un-VH champion mirror:

- **Additive λ=1.0 (Phase D4):** 0/32 = 0.0% (Wlo=0.000, Whi=0.107)
- **Rerank top-K=10 (Phase H4):** 2/32 = 6.2% (Wlo=0.017, Whi=0.201)

Reference baseline (un-VH state-K vs same opponent): **62.5%**.

Combined with the Phase C falsifications on the previous (broken) shipped
model, the VH-on-state-driven-K integration axis has **6 consecutive
falsifications**. Per Rule 37, the axis is closed for this session.

## Falsification ladder (this branch)

| # | Wiring | Model | λ | BIAS | K | Wins/n | Wlo |
|---|---|---|---|---|---|---|---|
| 1 | additive, joint+solo | broken (Jun 2) | 1.0 | 0 | — | 0/32 | 0.000 |
| 2 | additive, solo-only | broken | 1.0 | 0 | — | 0/32 | 0.000 |
| 3 | additive, calibrated | broken | 0.1 | 102 | — | 4/32 | 0.050 |
| 4 | additive, real-bias | broken | 0.1 | 75.5 | — | 1/32 | 0.006 |
| 5 | additive, fresh model | **fresh ρ=+0.386** | 1.0 | 0 | — | 0/32 | 0.000 |
| 6 | **rerank, fresh model** | **fresh ρ=+0.386** | 0 | — | **10** | **2/32** | **0.017** |

Reference: un-VH state-K = 20/32 = 62.5% (Wlo=0.453) vs `baseline_launch_rules_universal_local`.

## What Phase D verified

**Corpus** (`data/value_head/corpus_runs/2026-06-03-stateK-100games/`):

- 100 self-play games of `agents/baseline_champion_stateDrivenK_local` (state-K + joint-aggr, no VH, no Tier-2).
- 47,565 accepted candidates emitted across both seats; 47,077 retained after K=10 truncation drop.
- Per-seat balanced (23,379 / 23,698); 0 missing-features rows.
- Stage 1 wall: 2,259 s (37.6 min). Stage 2 (label pairing): 3 s.

**Training** (`data/value_head/value_head_model.txt`, mtime 05:32 UTC):

- LightGBM regression_l1, best_iter=297/600, 80/20 game-level split.
- σ(label) = 497.8 ✓ in band [5, 1500].
- Spearman ρ = **+0.386** on val ✓ above 0.10 gate.
- Walker parity vs LightGBM Booster: **0.000e+00** on 500 val rows.
- RMSE=420.7, MAE=214.7, R²=+0.046.
- Sidecars (`value_head_model.meta.json`) refreshed to match — prior provenance chaos cleared.

## What Phase H tried (Option C)

`agents/baseline/chooser_trajectory.py` — between `scored.sort()` and the
emit loop, when `BASELINE_VH_RERANK_K > 0`:

1. Scan `scored` (sorted by base score desc); collect indices of first K solo entries.
2. Call `vh_predict_one(...)` on each.
3. Re-sort those K slots by head_out desc.
4. Joints untouched. Base scores preserved on each tuple. Emit gate (MIN_DELTA) unchanged.

`agents/baseline_champion_stateDrivenK_vh/main.py:49` set `BASELINE_VH_RERANK_K=10`, `BASELINE_VH_LAMBDA` unset.

Smoke vs random = 16/16 = 100% (Wlo=0.806) — wrapper plays cleanly,
turn-ms peaks at 952 ms (under the 1000 ms cap). The wiring is sound.

## Why both wirings fail (hypotheses)

The retrained model has real rank signal (ρ=+0.386 is well above noise),
but its predictions don't translate to deployment value:

1. **Target mismatch.** K=10 ship-delta is a noisy local approximation of
   what the chooser's leaf-delta + PV-discount already integrate over
   10–30 turns via the state-driven horizon. The VH adds noise where
   the chooser already has signal.

2. **Pre-filtered training distribution.** Corpus rows are accepted moves
   from the same chooser they're meant to rerank. The VH learns to rank
   among already-good candidates — minimal marginal information beyond
   what the chooser already encodes in its base score.

3. **Emit-loop coupling.** The chooser's src/tgt collision-aware greedy
   emit is sensitive to top-K ordering. Reordering even modest amounts
   commits BETTER first-choice candidates to BAD downstream collisions,
   missing free targets the original order would have picked up.

4. **Self-play distribution shift.** Corpus was generated with both seats
   using the same (no-VH) chooser. Deployment puts VH on one seat only;
   opponent responses differ; VH predictions are out of distribution.

Any single hypothesis would explain the regression; all four are
plausible. None is recoverable inside this session's compute budget.

## What ships (artefacts kept)

- `scripts/gen_b2_corpus.py`, `scripts/probe_pveta_selfplay.py` — ported
  from `claude/competition-objective-alignment-hqNVM@9d32066`.
  Standalone corpus-gen for any future VH retrain on any agent.
- `agents/baseline/_trace_hook.py` (solo-only `trace_accepted`).
  Gated by `BASELINE_ACCEPTED_TRACE` env var; no-op default.
- `agents/baseline/chooser_trajectory.py` — `accepted_trace` populated at
  solo emit points; `trace_accepted` called at end-of-turn. At env-var-
  unset default, byte-equivalent to pre-port chooser.
- `agents/baseline_champion_stateDrivenK_local/main.py` — VH-OFF,
  Tier-2-OFF, state-K-ON wrapper for any future corpus gen.
- `agents/baseline/_value_head.py:VH_RERANK_K` + `vh_rerank_k()` accessor.
  Rerank block in chooser_trajectory.py:1502. Both gated on env var
  `BASELINE_VH_RERANK_K=0` default (off).
- `scripts/_build_champion_stateDrivenK_vh_bundle.sh` — bundler that
  inlines the value-head model as base64+gzip into `_VH_MODEL_B64`.
  Unused this session; available for a future VH attempt.

The actual model file `data/value_head/value_head_model.txt` is gitignored
(`data/*`). The fresh Phase D3 model is retained on disk but not pushed.

## What's closed

**VH-on-state-driven-K integration axis** — both additive and rerank
consumption surfaces falsified at n=32 with both broken and fresh
models. Adding to `state/MULTI_BRANCH.md::Closed tracks`.

## What's NOT closed by this audit

- The VH model itself isn't broken — Spearman ρ=+0.386 means it ranks
  candidates better than chance. A different consumption surface (e.g.
  out-of-rollout threat oracle, opening-prior, multi-step value
  iteration) might still extract value.
- The general approach of grafting learned heads onto the trajectory
  chooser may still work with a different target (e.g. terminal value,
  capture-probability, opponent-response prior).

What's closed is specifically the **K=10 ship-delta head + chooser
score augmentation** pairing — not learned-heads-in-general.

## PI-locked decisions today

- 2026-06-03: PI directed Phase D retrain after Phase C broken-model
  falsifications ("go for the retraining").
- 2026-06-03: PI requested weak-opponent triage after Phase D4 catastrophic
  fail ("test against weaker components, few games only"). vs random =
  100% confirmed no wiring bug.
- 2026-06-03: PI approved Option C (chooser-level rerank) over closing
  the axis on Phase D failure ("played option c").
- 2026-06-03: Wrap-up triggered after Phase H4 also failed at 2/32.
  No new Kaggle submission this session.

## Submissions

**0 used today.** Rolling pair on Kaggle unchanged. Live champion
`baseline_state_driven_k` (sub #53280733, μ ≥ 1153.6 last read) remains
the lead submission.
