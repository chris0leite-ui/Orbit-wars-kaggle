# kaggle-comp skill — cross-comp improvements log

Edits promoted here when a friction pattern appears in 2+ comps, costs > 1 LB slot,
or required a human nag. See self-improvement.md for the full distillation protocol.

---

## Cross-comp tagging (added on Orbit Wars seed)

- `[CROSS-CUTTING]` — applies to any Kaggle comp (tabular or code/agent).
- `[TABULAR-ONLY]` — applies only to tabular comps; ignore on Orbit Wars.
- `[ADAPT-FOR-CODE-COMP]` — concept transfers but the implementation differs in code/agent comps; adaptation note inline.
- `[CODE-COMP-DISCOVERED]` — new lessons that originate from a code/agent comp (Orbit Wars or successor). When promoted from this comp's friction, append entries here with the tag so future tabular comps can skip them and future code/agent comps inherit them.

---

## Pending (not yet applied to skill files)

### [ ] [TABULAR-ONLY] kickoff-runbook.md / day-1: pool eff-rank diagnostic on Day 1

`tag: pool-rank-lock-at-logit-direction-not-rank-correlation`.
Origin: s6e5 2026-05-08 PM postmortem. Run SVD eff-rank on the
base-prediction matrix as soon as ≥4 bases exist. If logit
eff-rank stalls below `log2(K) + 1`, the pool is rank-collapsed
regardless of nominal K. New bases will absorb at the LR meta if
their logit prediction lies in the existing logit subspace —
**low Spearman ρ to PRIMARY is necessary but not sufficient for
amp-eligibility**. s6e5 evidence: ρ=0.41 (EXP-4 Head A) still
absorbed at K=10+1 within ±0.05 bp; ρ=0.47 (EXP-3 inter-stint)
also absorbed. The K=27 pool sat at logit eff-rank 3.23 — adding
17 bases bought 1.7 bp on LB. If the diagnostic had fired Day 1,
half a session of dead-axis exploration would have been saved.
Code: `scripts/lr_diag_e1_svd.py` is the reusable template.

### [ ] [CROSS-CUTTING] do-and-dont.md — ISO date convention in prose; never invent Day-N

`tag: day-counter-drift`. Origin: s6e5 2026-05-08 PM (regression
of prior fix `ba4d531`). Prose uses ISO dates ("2026-05-08") or
comp-day-N anchored to comp start. **Never invent a "Day N"
counter that is not calendar-anchored.** The `dN` short-codes in
script names and historical audit prose are FROZEN sequencing
identifiers and MUST NOT be reused as date references in new
prose. Add a session-start sanity check: grep `state/*.md
HANDOVER.md ASSUMPTIONS.md` for "Day N" patterns where N >
days-since-comp-start, and surface as friction. PI noticed the
regression instantly; one PI override + one session of cleanup.

### [ ] [CROSS-CUTTING] do-and-dont.md — concurrent-CPU-heavy-job cap

`tag: concurrent-CPU-job-violation`. Origin: s6e5 2026-05-08 PM.
When dispatching 3+ Python jobs in parallel: if each uses
`n_jobs=-1`, three-way CPU contention can saturate the machine
to a near-halt (no fold completed in 25 min wall on 8-core, ≈75
CPU-min wasted). Rule 31 already caps concurrent CPU-heavy jobs
at 2; this addendum codifies the explicit n_jobs guidance. **If
3 jobs are absolutely required, set `n_jobs=floor(N_CORES/3)` per
process explicitly. Don't trust the OS scheduler to share fairly
under LightGBM/CatBoost OpenMP.**

### [ ] [ADAPT-FOR-CODE-COMP] kickoff-runbook.md / probe-template — persist OOF arrays for every gate probe

`tag: gate-probe-oof-not-persisted`. Origin: s6e5 2026-05-08 PM.
Gate scripts (`probe_min_meta`, `probe_field_state`, etc.) MUST
`np.save` their OOF and test arrays even when the verdict is
NULL. Add `oof_<slug>_strat.npy` and `test_<slug>_strat.npy` in
the script's success path unconditionally. Cost evidence: 2026-
05-08 EXP-1 re-tested d16 GRU at K=10+1 in 5 min because its OOF
was persisted; field-state, H9 transductive, combined-frame
lead/lag would each have required ~30 min to rerun their producer
scripts (90 min total deferred). With persistence, the
triangulation would have been free.

**Code-comp adaptation:** persist ALL evaluation artefacts on every
probe — replays (game-state JSON), per-opponent winrate panels,
action-distribution histograms, episode-return curves — not just
the headline metric. Same triangulation argument: a NULL probe
whose evaluation residuals you saved is a free re-test six days
later.

### [ ] [TABULAR-ONLY] kickoff-runbook.md / day-1: simple-LR baseline as Day-1 ceiling probe

`tag: lr-recipe-portable`. Day-1 of any new tabular comp, run the
30-second LR baseline (`KBins(20, quantile, onehot)` on every numeric +
`OneHot` on every cat → `LogisticRegression(C=1, solver='liblinear')`).
On s6e5: AUC 0.92038 in 22 s, closing 88% of the GBDT-vs-`lr_raw` gap.
Then run the mega LR (~8 min CPU, all FE families concatenated) — its
gap to single-GBDT tells you if stacking is necessary (>100 bp gap →
yes). Recipe + per-fold mechanics + mechanism map (LR vs GBDT vs NN
FE preferences) at `examples/fe-recipe-simple-lr.md`. **Origin:** s6e5
LR-bank experiment; `lr_kbins20_ohe` 0.92038 / `lr_mega` 0.92776 /
GBDT pool 0.95385. Anti-patterns codified: tree-engineered FE *hurts*
LR (Rozen FE: 0.857 vs raw+OHE 0.854 baseline); class_weight/L1/L2/
C-sweep are AUC rank-no-ops (skip the variants).

### [ ] [CROSS-CUTTING] kickoff-runbook.md Q5b — data + task description (≤10 sentences)

`tag: settled-once`. After Q5 EDA. (1) Each feature in domain terms,
(2) prediction task in real-world terms, (3) class balance →
metric/threshold implication, (4) top-3 features by F-score and why
they make domain sense. Write to `audit/<date>-day-1-kickoff.md`.

### [ ] [TABULAR-ONLY] guardrails.md G13 — single-model-first / kitchen-sink FE before stacking

`tag: recipe-over-judgment`. Before adding 2nd base or LR-meta in
first 3 days, build kitchen-sink FE (≥30 engineered features + CV TE
on every high-card combo) and the BEST single model. That OOF is the
floor; stacking adds on top, does NOT replace it. **Origin:** s6e5
ran K=22 + Path B for 13 days; a single LGBM with FE matched it on
Day-16 (after FS_A leak fix, OOF 0.946 — still −5bp under stack).

### [ ] [CROSS-CUTTING] guardrails.md G14 — family falsification requires ≥3 variants

`tag: family-falsification-too-quick`. A mechanism family (TE, FM,
lag, target-reform, pseudo, calibration) is only "dead" after ≥3
distinct configs of its key hyperparameter. Single-variant nulls
update the prior on that variant, not on the family. **Origin:** s6e5
TE family closed Day-3 on one 2-way × one smoothing variant; the 3-way
(Driver, Race, Year) was the load-bearing trick.

### [ ] [CROSS-CUTTING] guardrails.md G15 — framework is scaffolding, not authorship

`tag: recipe-over-judgment`. Reserve ≥1 slot per 3-day cycle for FE
creativity uncoupled from existing pool. Triggered when 3+ days
without a probe whose source idea is NOT a 1-step variant of an
existing experiment.

### [ ] [TABULAR-ONLY] guardrails.md G16 — fold-safe label-conditional aggregates

`tag: target-construction-layer-leakage`. Any feature derived from
labels via groupby aggregation (target encoding, mean-of-positives-
per-group, target-conditional ratios) MUST be re-fit per CV fold
using ti rows only. For test prediction either refit on full train
+ apply to test, or 5-fold-average models each with their own
ti-fitted aggregate. **Origin:** s6e5 Day-17 — `compound_avg_life`,
`race_avg_pit_lap`, `dc_avg_stint_life` fit on full train inflated
OOF +490 bp (0.95128 vs holdout 0.94637); v1 single LB 0.94107
(−863 bp gap); K=2 LR-meta LB −63 bp. **Diagnostic:** strict 80/20
holdout test (independent seed, FE state on 80% only, eval on 20%)
detects this in <10 min CPU without burning a slot.

### [ ] [TABULAR-ONLY] guardrails.md G18 — strict-fold-safe variant before treating group-aggregate OOF as honest

`tag: cross-row-aggregates-survive-strict-fold-safe-audit`. Even when
a group-aggregate FE uses a FEATURE column (not the label) as its
input, run a strict per-fold variant before treating standalone OOF
lift as honest. Pattern: for each CV fold, compute the aggregate from
tr_fold rows only (or tr_fold + test for the combined-frame variant);
merge into both tr and val rows. Compare strict-OOF to full-train OOF.
Day-17 leakage (label-derived) collapsed 88-100% under the
counterpart strict audit. Feature-derived cross-row aggregates may
collapse less or not at all — but the audit is the only honest way
to find out. **Origin:** s6e5 2026-05-07 PM — probe 4 field-state
aggregates over (Race, Year, LapNumber) full-train OOF +15.58 bp
standalone; strict per-fold OOF +13.73 bp; collapse 12% — fold-safe
real, distinct from Day-17 leakage family. **Diagnostic cost:**
~5 min CPU re-run after the full-train probe. PI sealed prediction
flagged the audit as the necessary defensive step.

### [ ] [TABULAR-ONLY] guardrails.md G17 — transductive features need AV check

`tag: transductive-features-need-AV-check`. Any FE that fits on
combined train+test (frequency encoding, quantile binning, factorize
maps, PCA/AE) requires adversarial-validation: train-vs-test
classifier AUC. If AV-AUC ≈ 0.5, combined is safe. If AV-AUC > ~0.55,
fit on train only. Even feature VALUES (not labels) can encode
distributional structure differing between train/test (or
public/private LB). **Origin:** PI s6e5 Day-17 lesson; companion to
G16. (s6e5 AV-AUC = 0.502 so combined-FE was safe here.)

### [ ] [TABULAR-ONLY] pre-baseline-gate.md items 8-11

`tag: eda-thin` + `public-notebook-scan-missing`.

```markdown
8. Public-notebook scan. `kaggle kernels list -s "<comp>" --sort-by voteCount`;
   pull top 5; list OOF AUCs, FE tricks, model classes. Re-scan at every plateau.
9. High-card TE inventory. List every cat × cat (and cat³) combo with
   unique-key count in (50, n_train/4). Flag the 3-way combo with largest
   unique count as load-bearing.
10. Domain-physics feature list. 5-10 features a domain expert would compute,
    each with one-line physics rationale. Implement ALL.
11. Single-model OOF target. Predict what kitchen-sink single LGBM should
    hit, calibrated against top public-notebook OOFs (step 8).
```

### [ ] [ADAPT-FOR-CODE-COMP] day-loop.md — public-notebook re-scan + 80/20 holdout diagnostic

```markdown
### Auto-trigger: public-notebook re-scan
On 3 nulls / 5 saturations / 50% checkpoint / "redecompose": pull top
5 notebooks (≥10 votes); ask which features are NOT in our pool.

### 80/20 holdout (mandatory before any new-FE-family LB submit)
StratifiedKFold with INDEPENDENT seed; fold 0 as 20% holdout; fit FE
+ inner-CV TE on 80% only; train + eval on 20%. If holdout ≪ OOF by
> 10 bp, leak present — debug before submit.
```

**Code-comp adaptation:** notebook re-scan applies as-is (read top
agents and ask which mechanisms are missing). The 80/20 holdout
becomes a **hold-out-opponent eval**: evaluate the new agent against
a roster of opponents not seen during training; if hold-out winrate
≪ self-play winrate by > 5 percentage points, overfitting to the
training opponents — debug before submit.

### [ ] [CROSS-CUTTING] kickoff-runbook.md / day-loop.md — keep top public notebooks as repo reference

`tag: recipe-over-judgment`. Keep top 3-5 public Kaggle notebooks
under `external/kernels/` as reference examples (not copy-pasted
code). Use them to (1) reverse-engineer FE at every plateau,
(2) sanity-check our feature factory vs published recipes, (3) build
a cross-comp recipe library. End-of-comp: review and promote durable
patterns to skill `examples/` or `recipes/`.

**Recipe library (seed entries):**
- `s6e5/romanrozen/f1-pit-driver-race-year-encoding-0-95354.ipynb`
  CV TE on 6 high-card combos (incl. 3-way), ~50 engineered FE,
  Rozen-LGBM hparams (lr=0.025, leaves=255, max_depth=10, ff=0.65).
  CAVEAT: Rozen's reported OOF 0.95241 likely inflated by FS_A leak
  per s6e5 Day-17 audit; honest single-LGBM ceiling ~0.946.
- `s6e5/yekenot/ps-s6-e5-realmlp-pytabkit.ipynb` —
  6 load-bearing FE items (arithmetic ratios + floor-cat + count-
  encoding + KBins + 2-way combo cats + CV TE inside fold loop) +
  per-fold orig-aug stratified 4/5. Verified 5-fold OOF 0.95257
  standalone on s6e5 (matches yekenot pub 0.95273 within 1.6 bp).
  Full audit at `.claude/skills/kaggle-comp/examples/fe-recipe-
  yekenot-realmlp-kitchen-sink.md`.

### [ ] [TABULAR-ONLY] examples/ — yekenot FE transfers to GBDT (CatBoost) too

`tag: recipe-over-judgment`. Research-branch audit caveats yekenot
items 2 (floor-cat), 3 (count-encoding), 4 (KBinsDiscretizer) as
"NN-specific (RealMLP can't derive these; CatBoost CAN via CTR +
split-finding; expected lift smaller for CB)." **Empirically false
on s6e5 Day-17 PM**: applying items 2/3/4 + item 7 (orig-aug) to a
research-recipe CatBoost-GPU (Bernoulli + min_data_in_leaf=20 +
Year/Stint cat + default CTR; "v3" → "v4") lifted standalone 5-fold
OOF by **+20.7 bp** (0.94993 → 0.95200) and DOUBLED the K=21+1
LR-meta contribution (+12.06 bp → +24.21 bp). Stacked K=23 + h1d
landed **LB 0.95354** (s6e5 PRIMARY). Mechanism hypothesis: explicit
floor/count/KBins as direct numeric/cat inputs interact with
CatBoost's CTR + split-finding in ways pure native CTR doesn't
capture — the GBDT split-finder benefits from pre-discretized
columns at the same fineness yekenot tuned for the NN.

**Apply yekenot's full FE recipe to both NN AND GBDT bases.**

Origin: s6e5 Day-17 PM, commit 7d179d6 on
`claude/optimize-model-performance-rruC2`. Friction tag candidate:
`yekenot-floor-count-kbins-fires-on-gbdt-too`. Promote to
`examples/cb-yekenot-transfer.md` when seen in a 2nd comp.

### [ ] [TABULAR-ONLY] kickoff-runbook.md / day-loop.md — original-data row-augmentation default

`tag: recipe-over-judgment`. For synthetic-tabular Playground comps
with AV-classifier AUC < 0.55 (train/test ≈ i.i.d. with original):
default to per-fold concat of the original (real-DGP) data,
stratified 4/5 split, weight 1.0 (or downweighted if synthesizer
has heavy label distribution shift). On s6e5 Day-17 PM, the v3 → v4
single-CB lift was driven by the combination of yekenot FE items +
this orig-aug item; never documented as a default kickoff move.
Cross-comp: irrigation-water used the same trick at weight 0.5.

**Pre-condition:** AV-classifier AUC < 0.55. Skip if AV > 0.55
(distribution shift risk; orig rows pull predictions off synth-test
marginal).

### [ ] [TABULAR-ONLY] recipes/ — Path-B pre-run base-routing audit gate

`tag: pathb-amp-dead-when-pool-already-routes-segmentation-variable`.
Before any new Path-B (per-segment hierarchical-meta LR-stacker)
candidate, run a 1-minute audit: enumerate the K-pool's bases; for
each, check whether it natively routes by the candidate segmentation
axis (e.g. `Year` is routed by `cb_year-cat`; `Position` is routed
by any continuous base that uses Position; `Stint` is NOT routed by
any current K=24 base). If 1+ bases natively route by the axis,
predict NULL ≥ 90% and require + 0.5 bp gate before allocating
τ-sweep compute. Origin: s6e5 Day-18 — d18 (Compound × Year) +
d18b (3 alt axes) ALL NULL/sub-gate after 18 min CPU each, all
predictable from base-routing audit. Saves estimated 30-60 min
CPU per future Path-B candidate. New file:
`.claude/skills/kaggle-comp/recipes/path-b-prerun-base-routing-audit.md`.

### [ ] [CROSS-CUTTING] guardrails.md — Monitor-discipline rule

`tag: stale-monitor-noise-fills-chat-after-process-ends`. When arming
a Monitor for a known-end-time process, use a natural-end command
(e.g. `until [ -f result.json ]; do sleep 5; done` or a `tail -f log
| grep -E "FINAL|→.*results" ; sleep 2; exit 0`) instead of an
unbounded `tail -f`. Cancel stale monitors via TaskStop the moment
the watched artifact appears. Origin: s6e5 Day-18 — ~12 stale
events fired on d18 watcher after the run ended; chat-token cost
visible.

### [~] [CROSS-CUTTING] PI-protocol — Sealed-prediction is non-optional even on "do it now" (SUPERSEDED 2026-05-07 PM)

~~`tag: sealed-prediction-skipped-on-do-it-now-commands`. PI's
authorisation to start a probe is NOT implicit ratification of the
agent's BOTE (back-of-envelope expected-value calculation). When PI
says "do it" / "do it now" / "execute this" on a probe with a written
spec, the agent MUST still run Rule 26a (sealed-prediction): ask PI
for LB Δ prediction in 1 line *before* allocating compute. Failing
this poisons the calibration loop for the resulting decision. Origin:
s6e5 Day-18 — d18 + d18b 18 min CPU each, no PI prediction logged in
`audit/decisions.jsonl`; calibration data lost.~~

**SUPERSEDED 2026-05-07 PM (Day-19 wrap-up postmortem):** PI directive
"remove asking for the sealed prediction" retired Rule 26a entirely.
Sealed-prediction protocol no longer applies. See entry below for the
removal record.

### [x] [CROSS-CUTTING] PI-protocol — Sealed-prediction protocol REMOVED (Day-19)

`tag: rule-26a-removed-by-pi-directive`. CLAUDE.md Rule 26a
(sealed-prediction order) and Rule 26b reference to it removed
2026-05-07 PM during Day-19 wrap-up postmortem on PI's verbatim
directive: "remove asking for the sealed prediction". Calibration
loop continues with agent-only predictions (`pi_predicted_lb_bp`
optional). Rule 26b reduced from three required questions to two
(Q6 metric-alignment + precedent citation). Rule 19f calibration
loop description updated accordingly. **Removal rationale (PI):**
sealed-prediction protocol added cognitive overhead for non-coding
PI without proportional calibration gain; agent BOTE alone is the
load-bearing prediction; PI intervenes via direct correction
(override-rate per Rule 26e) rather than per-probe sealed numbers.
**Origin:** s6e5 Day-19 postmortem. **Cross-ref:** the previous
`sealed-prediction-skipped-on-do-it-now-commands` candidate above
(now superseded) was a same-comp Day-18 friction; PI chose retirement
over enforcement.

### [x] [TABULAR-ONLY] comp-context — meta-arch redesign 9-variant-tally (Day-19)

`tag: meta-arch-redesign-family-empirically-exhausted-on-k27-pool`.
On s6e5 K=27 pool, meta-arch redesign family is empirically closed
after 9 variants tested across Days 14-19:
- Path-B alt-axes (Y×S, R×C, Driver_clustered×Stint, etc.) — 4 NULL
- Twin-meta blend ρ=0.967 — −1.79 bp (`twin-pool-2-meta-collapses-rank-info`)
- Conformal isotonic (4 schemes) — −2.5 to −9.6 bp
- Multi-level 4-tier (5 configs) — NULL
- K=10 forward-selected Path-B (9 configs) — sub-bp NULL
- C1 V3 Yao/Vehtari covariance-Σ (3 τ) — −0.47 to −0.59 bp REGRESS
Compound × Stint with plain shrinkage τ=100k IS the local optimum
on this pool. Future probes targeting meta-arch redesign on similar
pools require either (a) a fundamentally different segmentation axis
that Compound × Stint cannot capture (e.g. sequence-conditional), or
(b) a meta-objective change (e.g. row-AUC-aligned listwise loss;
tested via LambdaRank Day-12, −86 bp REGRESSED; Q6 origin). 9-variant
tally is a pool-saturation diagnostic; consult before proposing
variant 10. **Origin:** s6e5 Day-19 C1 V3 Yao/Vehtari falsification.
**Promoted:** PI directive Day-19 wrap-up.

### [x] [CROSS-CUTTING] operational-tip — bash watcher pgrep self-match (Day-19)

`tag: bash-watcher-pgrep-self-match-zombie-loops`. When polling for
a Python process completion in a bash watcher, `pgrep -f
"<script_name>"` matches the bash wrapper itself because Claude Code
bash wrappers `eval` the command string. Symptoms: until-loop never
exits, etime grows past expected wall, you end up with multiple
zombie watchers. **Fixes (preferred order):** (1) `until [ -f
<artifact_sentinel> ]; do sleep N; done` — file-existence is
unambiguous; (2) `pgrep -f "^python.*<script>"` — anchor against
bash; (3) use the Monitor tool (`tail -F` with grep on a sentinel
line in the script's output). **Origin:** s6e5 Day-19 overnight: 4
zombie watchers, ~10 min debugging + manual kills. Operational tier,
not framework rule. **Promoted:** PI directive Day-19 wrap-up.

### [ ] [CROSS-CUTTING] PI-protocol — No-unexplained-abbreviations rule

`tag: pi-comm-no-unexplained-abbreviations`. Every abbreviation MUST
be expanded on first use in a session. Methods/slang MUST include a
one-line plain-English restatement when first introduced. Postmortem
artifacts MUST include a glossary block at the top covering all
jargon used inside. Reference list (s6e5): LR (Logistic Regression),
GBDT (Gradient-Boosted Decision Trees), FM (Factorization Machines),
CB (CatBoost), NN (Neural Network), DAE (Denoising Auto-Encoder),
FE (Feature Engineering), TE (Target Encoding), OOF (Out-Of-Fold
predictions), LB (Leaderboard), AUC (Area Under ROC), CV
(Cross-Validation), K=N (pool of N base models), Path-B (per-segment
hierarchical-meta LR-stacker), τ (tau, shrinkage strength), ρ (rho,
Pearson correlation), bp (basis points = 0.0001 AUC), eff_rank
(effective SVD rank), BOTE (back-of-envelope), PI (Principal
Investigator = the human), HEDGE (backup submission). PI verbatim
2026-05-07: "I often struggle to understand what we are doing with
so many abbreviations and specific methods and slang. i need you to
explain it to me in simple terms and for abbreviations always tell
me what it is." This is a load-bearing communication rule, not a
one-off ask.

---

## Applied
<!-- log completed edits here: date · file · one-line description -->
