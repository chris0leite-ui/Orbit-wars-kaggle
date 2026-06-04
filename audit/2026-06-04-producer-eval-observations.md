# 2026-06-04 — Observations: local eval vs the vendored "Producer" agent

**Status: observations only.** This note records what was measured. It
deliberately does **not** draw conclusions about *why* the results came
out this way, nor recommend a fix. Open questions are listed at the end.

## What was run

- **Opponent:** "Producer" — a third-party *public* Kaggle-notebook agent
  (torch planner), vendored into `agents/producer/` and registered as the
  `producer` short-name. See `agents/producer/PROVENANCE.md`.
- **Harness:** `python fast.py eval <focal> --vs producer`, 8 seeds ×
  2 seats = 16 games, balanced seats.
- **Environment:** local CPU box, evaluations run serially. torch on CPU.
- **Sample size:** n = 16 per matchup. This is **triage-grade** under
  Rule 45 (n ≥ 32 required for any lift claim gated to submission); the
  numbers below support a *direction*, not a precise winrate.

## Measured results

| Focal (ours) | Live μ (context) | Wins vs Producer | Winrate | Wilson 95% CI |
|---|---|---|---|---|
| `champ_refine_adaptivek` (our latest push, sub 53336920) | 951.5 | 3 / 16 | 18.8% | [0.066, 0.430] |
| `champ_adaptiveK_on` (our live champion) | ≈1170 | 3 / 16 | 18.8% | [0.066, 0.430] |

- Both matchups: the **entire** 95% CI sits below 0.50.
- The two focal agents differ by ≈370 live μ, yet returned the **same**
  3/16 count against the Producer on the same 8 seeds.
- Per-archetype split (sparse at n=16): the `low_prod / mostly_static /
  big_rotating` geometry was 0/2 for both focal agents; the remainder of
  the wins came from the `<not-in-panel>` pool.

## Turn-time measurements (box-dependent — read the caveat)

Reported by `fast.py` as the *focal* agent's per-turn wall time.

| Run (focal) | p50 (ms) | p95 (ms) | max (ms) |
|---|---|---|---|
| `champ_refine_adaptivek` vs Producer | 477 | 824 | 1636 |
| `champ_adaptiveK_on` vs Producer | 493 | 952 | 1612 |
| `producer` vs random (symmetric check) | 474 | 580 | 1151 |

**Caveat.** These are wall-times on one CPU box under serial load and are
**not** directly comparable to Kaggle's turn budget. Importantly, when the
Producer was itself the focal agent (bottom row) its own p50 (~474 ms) and
max (~1151 ms, plausibly first-turn torch/cache warmup) are in the **same
range** as our agents'. An earlier verbal claim that the Producer runs
"~15 ms" was **not** measured here and is not supported by this data — no
clean speed differential between our agents and the Producer has been
established locally. Whether any agent exceeds the live per-turn budget on
Kaggle hardware is **unverified** (would need `fast.py bench` and/or a live
turn-time readout).

## Open questions (not yet answered)

1. Does the Producer beat our champion at n ≥ 32, and what is the winrate
   with a tight CI? (n=16 only fixes the direction.)
2. Are the 13 losses the *same* games for both focal agents (a shared,
   chooser-independent loss mode), or merely the same count?
3. What is the actual mechanism of a loss? (No single-game trace has been
   read yet — `low_prod/big_rotating` seed is the candidate to trace.)
4. Do our agents — or the Producer — actually exceed the live per-turn
   time budget on Kaggle hardware, or is the local ms purely a box artifact?
5. Is the local panel (which A/Bs within our own pilkwang-lineage family)
   systematically uncalibrated against architecturally-distinct opponents
   like the Producer?
