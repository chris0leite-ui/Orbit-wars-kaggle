# 2026-05-28 PM — distilled head fails the action-Δ rank-order test

Status: Phase A reinterpreted. Phase B-1 plan re-scoped (merge with B-3).

## What happened

Morning lens-critique pass against the Phase B roadmap added five
refinements (documented in HANDOVER.md). PI then asked for a single
sanity check from the morning Phase A debrief — `baseline_learned` vs
`baseline_favor` n=16 — predicting ≥80 % wins if the 99.8 % distillation
R² was real and `favor_hybrid >> favor` held. The result was the
opposite, which triggered two cheap follow-up diagnostics. All three
results below.

### Three diagnostics, all PM 2026-05-28

| Test | Result | Wilson 95 % CI |
|---|---:|---|
| weights-load sanity (forward batch on training corpus) | RMSE = 50.1, R² = **0.994** | — |
| `baseline_learned` vs `baseline_favor` (n=32, auto-bumped) | **9/32 = 28.1 %** | [.156, .454] |
| `baseline_hybrid` vs `baseline_favor` (n=32) | **15/32 = 46.9 %** | [.309, .636] |
| `baseline_learned` vs `baseline_pv_eta` (live μ=1154.8, n=32) | **6/32 = 18.8 %** | [.089, .353] |

Wallclock 100 ms per turn in every A/B, focal p50 158-167 ms (same
budget overshoot as Phase A).

## Interpretation

Three hypotheses entered the diagnostic round. After:

- **Hypothesis 3 (weights bug).** ❌ Falsified. Embedded weights load,
  produce R² = 0.994 against the saved labels. The trained model
  genuinely fits the scalar.
- **Hypothesis 2 (`favor_hybrid` is a weak teacher).** ✅ Confirmed at
  near-parity. `favor_hybrid` vs `favor` is 47 % wins under this
  harness (Wilson CI spans 0.5). The "team peak μ=1149" reputation of
  `favor_hybrid` came from ladder evidence against other agents, NOT
  from a self-play margin over `favor`. As a distillation TEACHER it
  is not meaningfully stronger than the head it was supposed to
  improve on.
- **Hypothesis 1 (scalar R² does not preserve action-Δ rank order).**
  ✅ Confirmed by the chain. R² = 0.994 on the scalar coexists with
  losing to `favor` (28 %) and to the live agent `baseline_pv_eta`
  (19 %). The chooser's argmax_a depends on action-Δs, not on
  absolute scalar value; scalar mimicry is the wrong target.

### Why Phase A's 44 % vs `favor_hybrid` was a misleading signal

Phase A's near-parity vs the teacher (14/32 = 44 %, CI [.28, .61])
looked like "wiring is sound, head ≈ teacher". With the chain now
visible, the right reading is: `learned ≈ favor_hybrid ≈ favor` under
this harness, and `learned < favor` under any matchup that isn't the
head's own copy. The 44 % vs `favor_hybrid` was mimicry noise — both
choosers were ranking off similarly-noisy heads, so disagreements
averaged to 50 %. As soon as one side switches to a different head
(any other policy), the rank-order divergence shows up as systematic
disadvantage. Phase A's diagnostic was self-consistent but not
discriminating.

### What was wrongly falsified in Phase A's debrief

HANDOVER.md (this morning) listed under "Falsified-or-dead":

> **"40-feature insufficiency"** — Falsified by Phase A's 99.8 % R²
> distillation result. If Phase B underperforms, blame the data /
> target, not the feature pipeline.

This claim was based on the same R² that we now know doesn't transfer
to action-Δ. The 40-feature pipeline could still be insufficient for
preserving action-Δ rank order, even with a clean scalar fit. The
claim should be RE-OPENED until a head trained directly on action-Δ
labels confirms (or refutes) feature sufficiency.

## Implications for Phase B-1

The morning's lens-critique refinement #1 said "B-1 reuses Phase A's
distribution (same opp, same games)". That's no longer right:

- The opp policy in Phase A's data-gen was `favor_hybrid`, which today
  measures as not meaningfully stronger than `favor`. A CRN advantage
  head trained against `favor_hybrid` rollouts would learn "what beats
  a parity-level policy". That's still the v1 weak-opp failure mode.
- B-1's diagnostic A/B target (`favor_hybrid`) is uninformative for
  the same reason — "did we beat a parity-level reference?" doesn't
  tell us anything useful.

Phase B-3's "strong heterogeneous opponent pool" had been scheduled
later in the roadmap. After today's diagnostics it has to MERGE INTO
B-1, because no isolated version of B-1 makes sense. The simplest
viable merged shape:

1. **Data-gen opp**: `baseline_pv_eta` (single, strong, currently on
   the rolling pair at μ=1154.8) — not the full pool yet.
2. **Label**: CRN-paired advantage A(s, a) = margin_action −
   margin_idle, with locked opp RNG on both legs.
3. **Diagnostic A/B**: vs `baseline_pv_eta` directly. Wilson-lo ≥ 0.50
   needed to declare the line worth continuing.
4. **Training-time gate**: Spearman-τ on action-Δ rank order on a
   held-out set, not just scalar R². Phase A would have failed this
   gate (R² high, action-Δ rank ~chance).

Phase B-3's full strong-opp pool is then a follow-on (B-3-prime), not
a separate sequence.

## Cost re-estimate

Morning estimate: "B-1 is hours not days because it reuses Phase A
games". That assumed favor_hybrid data was usable. It isn't.

New B-1 cost: fresh rollouts against `baseline_pv_eta`. For each
sampled state s, two paired rollouts (action vs idle) with locked
opp-RNG. A rough back-of-envelope (refinement #2 from this morning,
now mandatory):

```
N_games (focal vs pv_eta)     1500
turns/game                    ~150
samples/game (subsampled)        8
rollout cost per leg          ~10 s   (pv_eta is expensive)
legs per sample                  2
games + rollouts ≈ 1500 × (150 × ~150 ms chooser + 8 × 2 × 10 s)
                ≈ 1500 × (22 s + 160 s)
                ≈ 1500 × 180 s
                ≈ 75 hours single-core
                ≈ 10 hours on 8 workers
                ≈ Kaggle GPU corpus per Rule 13
```

So the cost is NOT "hours not days". It's a real Kaggle-GPU job, or
an overnight 8-worker local run. The data-gen-loop sketch lives at
`knowledge-base/concepts/crn-advantage-datagen-sketch.md`.

## Escape clause (Phase B option B)

If a CRN-advantage head trained against `baseline_pv_eta` STILL loses
to `pv_eta` at < 40 % on n=32, the entire "learned value head with
40 features" framing is the bottleneck, and the program should pause
for a structural rethink (option B from this session's chat):

- Direct policy distillation instead of value distillation.
- A search-based chooser (1-ply beam over candidates using the
  head's variance estimate) instead of a deeper head.
- Feature-pipeline expansion (the 40-feature claim is re-opened).
- A different head architecture entirely (e.g. GNN over the planet
  graph, since orbit-wars geometry is inherently relational).

The 19 % vs live result is a strong floor on the current setup's
quality. Anything in B that doesn't move that needle materially is
not a candidate.

## Methodological notes (worth promoting)

1. **Scalar R² is not a sufficient training-time gate for a head that
   feeds an argmax.** A head's job is to preserve rank order on the
   chooser's candidate set; scalar fit can be high while rank order
   is chance. Always include a rank-correlation diagnostic (Spearman
   or Kendall-τ) on held-out states. Already noted as a Phase B
   permanent gate.
2. **Verify the teacher is meaningfully strong before distilling.**
   "Production default" ≠ "stronger than the predecessor in
   self-play". Check h2h, not just ladder rep.
3. **"Parity vs teacher" is not a quality signal.** It's a mimicry
   signal. Quality signals require a non-teacher comparison.
