# 2026-06-18 — Dropout-native Phase A: built, kill-gate administered, refuted

## What I built
Per `state/DROPOUT_NATIVE_DESIGN.md` Phase A: a mean-field flip-hazard forward
model (`orbit_lite/native_forward.py`), wired into producer_plus behind
`PRODUCER_PLUS_NATIVE_HAZARD` (default OFF, byte-identical OFF path), scoring the
producer's existing candidate shortlist. The model:
- runs each candidate's exact engine trajectory (reusing `_run_exact_recurrence`),
- overlays a per-step flip hazard λ = sigmoid(steepness·(atk_reach − garrison)/
  (atk+def)) where atk_reach = enemy physically-routable mass by step k,
- values by the expected production-weighted ownership margin over the horizon.
Deterministic, no RNG. Unit-tested: trajectory construction reduces exactly to
the engine recurrence; holdable capture out-values a thin one; monotone in enemy
mass; deterministic.

## The kill-gate (continuous paired margin, n=40 vs V2)
native 19/40 vs base 21/40, Δmargin −0.10 (CI brackets 0). Does NOT beat base.

## The load-bearing twist (why the first run didn't count)
First run: steepness 5 ≡ 8 byte-identical on all 40 maps → the flip-hazard was
INERT. I had applied it as an INSTANTANEOUS per-step haircut (1−leak), a
second-order perturbation dominated by the large deterministic ownership term —
so Phase-A-v1 actually tested "plain ownership-margin vs the tuned
competitive_score," not the distribution. (Ironic: the same "thin layer over a
coarse value function" failure the bolt-on had.)

Fixed it to the design's spec — CUMULATIVE survival surv = Π_j(1−leak_j) applied
only while I hold the planet, so threat compounds. Verified load-bearing ON
SYNTHETIC BOARDS (value responds to steepness).

But the re-run was BYTE-IDENTICAL to the inert version (19/40, same 15 maps, same
margins; native ≡ native_s20 again). So even the cumulative, load-bearing hazard
is INERT FOR CANDIDATE RANKING in real games: it moves the value magnitude but
never the argmax.

## Root cause (the real lesson)
`atk_reach` is candidate-INDEPENDENT (current enemy ships, identical for every
candidate), so the hazard discount is a near-constant that cancels in the argmax;
where it IS candidate-dependent (touched source/target garrisons) it's dominated
by the deterministic ownership term. **A distributional layer over a
per-candidate DETERMINISTIC trajectory + a FIXED shortlist is dominated by the
deterministic core.** This GENERALIZES the bolt-on finding: both the 2-point
bolt-on and the continuous mean-field overlay fail for the same structural
reason — a thin distributional layer over a value function that already decides
the ranking.

## Verdict
Phase A kill-gate FAILED → dropout-native thesis refuted AS SCOPED. Making the
distribution load-bearing for RANKING needs candidate-dependent threat
(self-consistency, Phase D) or the v2 sampled ensemble + ensemble-driven
GENERATION (Phase C/D) — big builds the design gated behind a Phase A pass that
didn't happen. Base is only parity-with-V2 (below the live champion), so the
evidence doesn't justify funding that rebuild.

## What's durable (banked regardless)
- `scripts/continuous_ab.py` + `_continuous_game_worker.py`: continuous
  ship-margin A/B, paired by map, one game per fresh subprocess, focal/variant
  sets (dropout + champion). Reusable for ANY agent A/B — the real deliverable.
- The negative result: dropout (bolt-on AND native Phase A) does not beat the
  producer; the structural reason is a thin distributional layer over a
  dominant deterministic/static core.
- `native_forward.py` + tests: the forward model exists and is correct; if the
  PI ever funds v2, the trajectory builder + hazard are reusable.

## Open question for the PI
Bank the negative result and stop the dropout line (recommended), or fund the v2
ensemble + ensemble-driven generation (the only path where the distribution
could become load-bearing for ranking)?
