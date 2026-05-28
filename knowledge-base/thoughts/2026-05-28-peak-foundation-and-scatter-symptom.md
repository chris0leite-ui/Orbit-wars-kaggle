# 2026-05-28 — peak foundation laid; scatter symptom traced to leaf

Two structural realisations from today that shape the next iteration.

## The peak doesn't avoid scatter — it succeeds despite scatter

`favor()`'s leaf scoring (`agents/baseline/value.py:127`) returns
`(my_ships - opp_ships) + (my_prod - opp_prod) × pv_horizon(leaf_step, 0)`.
Critical: `pv_horizon` is called with `eta=0`, meaning the production
contribution from a captured planet is the same whether the fleet took
10 turns or 40 turns to land. A 2-ship fleet at a 1-prod neutral 40
turns away contributes ~99 to the leaf, passes the `> 0` gate, gets
emitted.

The peak bundle (sub 52912707, μ=1165.4) ships this behaviour and
still rates 1144-1165 on the ladder. So the "small-fleet long-path"
symptom is not a bug being hidden by some clever counter-mechanism —
it's a property of the algorithm that's net-neutral against the
opponent mix on the current ladder. Every other top agent uses
variants of the same chooser; they all scatter; scatter is roughly
zero-sum across the population.

The clean structural fix is `pv_horizon(leaf_step, eta=fleet_eta)`
— `pv_horizon`'s signature already supports `eta`, we just don't
pass it. At γ=0.99, eta=40 → production weighted by 0.99^40 ≈ 0.67;
eta=10 → 0.90. A real (not killed) discount on long flights. This is
where the next iteration should start.

## Two paths regressed; the lesson is calibration discipline

Today's two submits (sub 53099001 Step 2B → μ=680, sub 53083109 yesterday's
fix-stack → μ=921) failed for the same structural reason: **local A/B
at n=8 against one opponent has near-zero predictive value for ladder
μ**. The Wilson-lo 0.349 on 6W-2L is not "directional positive
evidence of small lift" — it's "the data is consistent with anywhere
from a 35% regression to a 95% win, we just don't know." Translating
that band to a μ-rating prediction requires either (a) a panel that
breaks the A>B>C>A symmetry, or (b) n large enough to actually pin
the rate.

The recurrence pattern — "PI signoff to ship early for feedback" →
μ regression — is the calibration data Rule 26 (devil's-advocate
ritual) is supposed to surface. Today the rule fired but I did not
argue hard enough. The PI's overrides are load-bearing context; my
job in those moments is to make the cost explicit BEFORE the submit,
not after.

## The peak foundation is the deliverable

Net session result: one regression slot (recoverable on next push),
one peak-restore slot (pending), and a clean `state/PEAK_BASELINE.md`
that future sessions read first. The PEAK_BASELINE.md captures the
plain-English strategy (one paragraph for non-coders), the 19 live
env vars and ~40 dormant ones (so nobody else "fixes" a dormant var
without isolation A/B), and the top 5 fragility risks ranked by
likelihood × severity. Git tag `peak-1165` + frozen anchor file mean
the calibration reference is always-on.

The foundation is the actual product of this session. The
regression is the calibration data that motivated the foundation.

## What the next iteration should do, concretely

Start from `git checkout peak-1165` (or apply on top of HEAD with
the SHA-verified anchor as A/B reference). Implement
`pv_horizon(leaf_step, eta=fleet_eta)` env-gated via
`BASELINE_PV_USE_ETA` (default OFF preserves byte-for-byte legacy).
Run instrumented trace BEFORE writing the plan to measure the new
Δ distribution shape. Run Phase 1 n=32 vs peak anchor, then Rule 43
panel, then Rule 42 push-coordination, then submit. The whole loop
should take half a day; the friction is in not skipping any of it.

If the peak-restore submit (sub 53099429, pending) lands in the
1130-1170 band, the NEUTRAL_BONUS-into-v4 plumbing hypothesis is
confirmed — and the dormant-env-var-wiring anti-pattern documented
in PEAK_BASELINE.md becomes ladder-validated, not just inferred.
