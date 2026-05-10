> **[TABULAR-ONLY] — Not applicable to Orbit Wars.** Logistic-regression
> diagnostics over a base-prediction matrix. Kept for cross-comp reference
> only. Skip on code/agent comps.

# LR-diagnostics — Day-1 toolkit for tabular comps

A 10-script battery that exposes pool redundancy, DGP interaction
structure, and meta-utility ceilings using populations of logistic
regressions. Built and validated on s6e5 (F1 pit-stops, binary AUC,
LB 0.95345 at top-5%) — see `audit/2026-05-07-lr-diagnostics-arc{A,B,C}.md`
for the falsifying validation.

## When to invoke this skill

- **Plateau confirmed (3+ days no advance, 5+ saturations same LB).**
  Diagnoses redundancy vs nonlinearity vs meta-arch ceiling.
- **About to add base #N to a stack pool of size K.** E1+E9 tell you
  whether the existing K already saturates; saves slots on doomed
  base-add probes.
- **Considering an LR/FM/NN base on the same features as the GBDT
  pool.** A2/A4 quantify the upper bound; if K=K is meta-saturated,
  representation-only base is predicted null with high confidence.
- **Day-1 of a new tabular comp.** E1/E2/E5/E6 establish a baseline
  understanding of the DGP in <2 hours of CPU.

## Three durable lessons (from s6e5 application)

1. **Pool effective rank is often << nominal pool size.** s6e5: 24
   nominal bases collapsed to entropy eff-rank 2.88; K=10 = K=24 in
   AUC. The 14 dead-weight bases are pure overfitting risk on private
   LB. Pool surgery (drop the unpicked-by-FS bases) is no-cost on OOF.
2. **The dominant interaction hub is one feature 90% of the time.**
   s6e5: 9 of top 10 cell-residual pairs include `Stint`. Adding the
   9 (Stint × *) interactions to a global LR lifts +123 bp standalone
   (E6 → A2 confirmed empirically).
3. **Representation-only diversity is meta-null on a saturated info
   space.** If GBDT residuals at top interaction cells are < 1%, no
   LR/FM/NN base on the same features will lift the meta — even with
   ρ vs PRIMARY = 0.71 (lowest possible). Lift requires new
   information OR new meta-architecture.

## The 10 scripts

Templates in `templates/scripts/lr_diag/`. Each is a standalone
Python file. Copy to your comp's `scripts/` and adapt:
- `TARGET = "<your target column>"` (e.g. `"PitNextLap"`)
- `K21_BASES = [...]` (your base list; for E1/E8/E9)
- `K10_BASES = [...]` (your forward-selected core; for A2_gate/A4)
- categorical column names (Compound/Race/Driver) → your comp's

| Script | Question | Cost | When |
|---|---|---:|---|
| **e1_svd.py** | Is your K=24 pool effectively rank-K' for K' << 24? | 1 min | Always |
| **e2_calibration.py** | Which bases are mis-calibrated? Is the meta doing implicit Platt scaling? | 30 min | After 3+ saturations |
| **e4_per_segment.py** | Where is the DGP locally linear vs nonlinear? | 45 min | Day 1 |
| **e5_bootstrap_coef.py** | Which features have stable signal vs noise? (50× bootstrap) | 10 min | Day 1 |
| **e6_residual_interactions.py** | What's the dominant 2-way interaction hub? | 5 min | Day 1 |
| **e8_grid.py** | Is your meta hyperparameter surface flat? | 5 min | After 3+ saturations |
| **e9_forward_select.py** | What's the *true* effective pool size (K=N at peak AUC)? | 15 min | After 3+ saturations |
| **a2_bagged_lr.py** | Does linear-with-engineered-interactions close the LR/GBDT gap? | 15 min | After E6 finds hub |
| **a2_gate.py** | Does the new LR base add to the K=K core via meta? | 5 min | After A2 |
| **a4_per_segment.py** | Do per-segment LR specialists add diversity the meta can use? | 5 min | After A2 |

## Recommended order

**Arc A — pool/meta diagnostics (4 cheap probes, first):**
E1 → E2 → E8 → E4. By the end you know:
- whether the pool is rank-deficient (E1)
- whether the meta is doing implicit calibration (E2)
- whether your meta hyperparameters are at ceiling (E8)
- where on the data the DGP is locally linear vs not (E4)

**Arc B — DGP archaeology (3 probes):**
E5 (signal vs noise features) → E6 (interaction hub) → E9 (true
effective pool size). At the end you have a feature-engineering
shopping list and a quantitative answer to "is my pool oversized?"

**Arc C — new-base injection tests (only if Arc A/B point to it):**
A2 (global LR + E6 interactions, vanilla vs rich gate against K=K
core) → A4 (per-segment specialists). These confirm whether
representation-only base addition can lift your meta.

## Outputs and decision rules

Each script writes a JSON to `scripts/artifacts/lr_diag_*.json`
plus a 1-paragraph console summary. After running:

| Symptom | Conclusion | Action |
|---|---|---|
| E1 entropy eff_rank << nominal K | Pool is redundant | Drop unpicked-by-E9 bases; switch PRIMARY to K=N core |
| E1 residualized eff_rank > 0.5 × nominal | Hidden structure remains | Meta-arch redesign is the lever (Path-B, BMA) |
| E2 has 1+ base with Brier-Δ > 5 bp | Mis-calibrated base | Check if it's also redundant (drop) or unique (Platt-scale) |
| E4 cells all > 100 bp gap to PRIMARY | DGP is nonlinear everywhere | Pure-LR base will be weak; need GBDT or interaction FE |
| E5 SNR=0 features | Pure noise; will hurt LR | Drop them from any LR base |
| E6 single feature dominates pairs | Found the interaction hub | Add (hub × *) interactions to LR; test in GBDT FE |
| E8 Δ within ±0.1 bp across 20 configs | Meta surface is flat | Stop tuning meta hyperparameters |
| E9 plateau at K' < K | True effective pool size = K' | Pool surgery candidate |
| A2/A4 gate Δ ≤ +0.05 bp | Saturated info space | No LR base will help; pivot to meta-arch |

## Pointers (origin material)

- `audit/2026-05-07-lr-diagnostics-arcA.md` — pool/meta archaeology
- `audit/2026-05-07-lr-diagnostics-arcB.md` — DGP archaeology
- `audit/2026-05-07-lr-diagnostics-arcC.md` — new-base injection
- `audit/2026-05-07-chris-deotte-lr-stacker-research.md` — origin
  context (s6e4 winner writeup → 7-step problem-solving)

## Anti-patterns

- **Don't use this suite as a justification to chase bp lift.**
  s6e5 demonstrated the suite tells you when lift via base-add is
  *impossible*; that's a feature, not a bug.
- **Don't forget to adapt categorical-feature names.** The
  scripts hard-code Compound/Race/Driver — these will break on
  comps with different schemas. Use the EDA output to remap.
- **Don't skip class_weight='balanced' on imbalanced binary
  AUC.** E5 showed it eliminates sign-flip noise without
  hurting AUC. Always use it for LR-as-base.
- **Don't transfer s6e4-style "logits + class_weight + multinomial"
  recipe** to binary AUC blindly. E8 showed it's metric-specific
  (multinomial balanced-accuracy); for binary AUC, all three axes
  are rank-no-op.
