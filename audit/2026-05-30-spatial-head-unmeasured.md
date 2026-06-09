# Spatial value head (`hybrid_spatial`) — status: UNMEASURED on win-rate

> 2026-05-30, `claude/champion-strategy-rules-00JzI`. PI decision:
> skip the win-rate A/B for now; do not record a null we never ran.

## What is actually settled

- **Cost gate: CLEARED.** `_positional_ship_value` is <0.3% of compute
  (profiler, full 500-turn 2P game). Back-to-back single-job bench,
  same opponent + 6 seeds: spatial focal p95=821ms / max=928ms / 0
  turns ≥1000ms vs champion p95=884 / max=948 / 0. The spatial head is
  as-fast-or-faster than the champion; the shared ~880ms tail is the
  agent's (`predict_relative` / sim), not the head's. The earlier
  "p95=952ms, kill it" reading was CPU-contention noise (Rule 38
  reproduce-the-failure: vanished single-job).

## What is NOT settled — and is being left unsettled by choice

- **Win-rate: ZERO clean data.** The only question that decides the
  head — "does `hybrid_spatial` WIN more than plain `hybrid`?" — was
  never answered. Any "15/32"-type figure in old scrollback was a
  draft run under CPU oversubscription on this 4-core box; **disregard
  it.**
- **Therefore "the spatial head is dead" is an UNCONFIRMED premise.**
  It is neither alive nor dead on the metric that matters. It is
  *unmeasured*.

## Decision (PI, 2026-05-30)

Do not run the win-rate A/B now. Treat the **2-hop redeploy-enables-
capture redesign** (`knowledge-base/concepts/redeploy-2hop-capture-
design.md`, commit `727e1bf`) as the forward path **regardless of the
spatial head's eventual verdict**, because that design stands on its own
under the plain head and also composes with the spatial head if it ever
ships.

## Consequences for the deferred work

- The HANDOVER "deferred idea 1: port the SEU7P forward-redeploy
  generator" is **superseded**. Do NOT port the spatial-coupled
  `_enumerate_redeploy_candidates` / `cheap_marginal_redeploy` as-is:
  it captures nothing, so its rollout-leaf Δ is ≤0 under the plain head
  and it is only ever selected by the spatial term. It cannot be
  decoupled — it falls with (or stands only on) the spatial head.
- If the spatial head is revisited later, the win-rate A/B is fully
  staged and one command away:
  - Focal bundle: `submissions/baseline_universal_spatial.py` (head
    baked in code, `select_favor_fn` → `favor_hybrid_spatial`;
    contamination probe PASSED).
  - Champion: `submissions/baseline_launch_rules_universal.py`.
  - `python scripts/clean_ab.py submissions/baseline_universal_spatial.py
    submissions/baseline_launch_rules_universal.py --seeds 16` (serial —
    parallel `fast.py` contaminates n≤16 on this box). Gate: Wilson-lo
    ≥0.50, extend to `--seeds 32` if triage-positive.

## No submission

Evidence-gathering line only. No `kaggle competitions submit` was made
or planned. Live rolling pair unchanged.
