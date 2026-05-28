# 2026-05-28 PM2 — feature-sufficiency probe + research synthesis

Status: cheap probe (≈10 min total compute, no fresh data-gen) PLUS public-notebook scan + literature search. Phase B-1 plan amended.

Companion to `knowledge-base/thoughts/2026-05-28-pm-distillation-action-rank-collapse.md`.

## The question we were trying to answer

Phase A distilled head failed: R²=0.994 globally but lost 9/32 vs `baseline_favor` and 6/32 vs `baseline_pv_eta`. Two competing hypotheses on what to fix:

1. **Loss function.** Scalar MSE optimises absolute fit, not rank order. A rank-aware loss on the SAME features might recover argmax behaviour.
2. **Features.** The 40-dim pooled feature set cannot encode the per-candidate distinctions the chooser argmax needs.

Resolving this before sinking 9–28 h into the proposed CRN data-gen.

## What was run (three stages, all on existing Phase A corpus, no new rollouts)

### Stage 1 — quantify the rank-collapse vs Δy

For 200k random pairs (i, j) from the 10k-example corpus, bucket P(rank-agree) = P(sign(ŷᵢ − ŷⱼ) == sign(yᵢ − yⱼ)) by |yᵢ − yⱼ|.

| \|Δy\| bucket | n | P(agree) |
|---|---:|---:|
| [0.5, 1) | 481 | **0.528** ← chance |
| [1, 2) | 1097 | 0.521 |
| [2, 5) | 3199 | 0.542 |
| [5, 10) | 6028 | 0.570 |
| [10, 20) | 11307 | 0.623 |
| [20, 50) | 30337 | 0.739 |
| [50, 100) | 44057 | 0.870 |
| [100, 200) | 53102 | 0.959 |
| [200, ∞) | 48898 | 0.994+ |

**Finding.** R²=0.994 globally coexists with chance-level rank below RMSE=50. The chooser argmax operates exactly in the |Δy| < 10 regime (sibling candidate emits produce small Δscalar). The head is structurally uninformed there — formal mechanism for the observed 28 % h2h vs `baseline_favor`.

### Stage 2 — train RankNet on the same 40 features, see if rank recovers

Same architecture (40→128→128→1 ReLU). RankNet (pairwise logistic) loss. 60 epochs Adam. 80/20 train/val split.

| \|Δy\| bucket on val | n | RankNet P(agree) | Fresh MSE MLP P(agree) | Embedded MSE P(agree) |
|---|---:|---:|---:|---:|
| [1, 2) | 1019 | 0.516 | 0.482 | 0.521 |
| [2, 5) | 3341 | 0.564 | 0.541 | 0.542 |
| [5, 10) | 6261 | 0.585 | 0.535 | 0.570 |
| [10, 20) | 11260 | 0.651 | 0.605 | 0.623 |
| [20, 50) | 29498 | 0.744 | 0.716 | 0.739 |
| [50, 100) | 43416 | 0.867 | 0.848 | 0.870 |

**Finding.** RankNet gives +2–5 pp in the close-pair buckets vs MSE. Real but tiny — still essentially chance below |Δy|=10. The loss function is NOT the dominant bottleneck.

### Stage 3 — permutation importance on the embedded head

Shuffle one feature column at a time, recompute global R² and close-pair P(agree). Rank by ΔR².

| Rank | Feature | ΔR² | Δclose | univariate Spearman vs y |
|---|---|---:|---:|---:|
| 1 | `opp_ship_total` | +1.14 | +0.10 | −0.61 |
| 2 | `me_ship_total` | +0.56 | +0.08 | −0.33 |
| 3 | `me_planet_ships` | +0.52 | +0.09 | −0.16 |
| 4 | `opp_planet_ships` | +0.21 | +0.08 | −0.36 |
| 5 | `opp_inflight_ships` | +0.03 | +0.08 | −0.80 |
| — | **TOP 5 CUMULATIVE** | **+2.47** | | |
| 6–40 | All others | each < +0.003 | each < +0.04 | varied |

**Finding.** The embedded MLP uses **5 of 40 features substantively** — all are global ship counts. The remaining 35 features (per-planet means/maxes, distances, centroids, production breakdowns, step, threat, etc.) contribute ΔR² < 0.003 each. The MLP correctly learned that `favor_hybrid` is essentially a ship-count delta and ignored the rest.

## Mechanistic synthesis — why the head's "richness" was illusory

`favor_hybrid` is dominated by `delta_us_minus_them` (us-ship-total − them-ship-total) with a small inflight-credit correction. The distillation target is therefore mostly a function of 4 of the 40 features. The other 36 carry no signal for THIS target.

So the head learned what was asked of it. The problem is that the chooser argmax needs to distinguish candidate emits whose ship totals are nearly identical (one emit shifts 10 ships from planet A to planet C; another shifts 10 from A to D — same totals, very different futures). The features and the target together cannot represent that distinction.

**Two axes are simultaneously degenerate:**
1. **Target degeneracy.** A scalar that's near-translation-invariant under emit choice (because ship totals are conserved across legal emits) cannot label rank order on the chooser's candidate set.
2. **Feature degeneracy.** A pooled 40-feature representation cannot encode per-candidate target identity (which planet the emit goes to, ETA, target's defender count).

Fixing one without the other still fails:
- CRN-paired advantage labels on the same 40 pooled features: labels now reward target choice, but the features don't differ between candidates that target different planets with the same ship counts. The MLP gets variance with no controllable signal.
- Per-candidate features on a scalar value target: features differ, but the target still doesn't reward target choice, so the rank signal is weak.

## Research synthesis (Kaggle public + literature)

### Kaggle precedent (public notebooks ≥ 70 votes)

| Author | Approach | Features | Result |
|---|---|---|---|
| AidenSong123 (72 ▲) | Search + GBC value head, 1-ply minimax | 16 features over STATE (ship_lead, prod_lead, %s, centrality, in-flight frac, phase, 2P/4P) | AUC 0.976, LB ≥ 1000 |
| konbu17 (72 ▲) | **MLP shot-validator filter on rule-base proposals** | **24 features PER SHOT** (src/tgt ships+prod+radius, owner one-hot, shot ships/fraction/dist/ETA/fleet-speed, in-flight counts, turn) | **+19 pp vs rule-base alone; +43 pp vs tier4** |
| kashiwaba (222 ▲) | PPO, per-planet decision | Three groups: self / per-candidate / global, separate encoders | Educational, mid-pack LB |
| istinetz (31 ▲ disc.) | Closed-form target value `pv = prod × (γ^arrival − γ^horizon)/(1−γ)`, γ=0.99 | Per-planet pv | "LB 1000 with just this + trajectory calc" |

The two value-head archetypes that have demonstrably moved the LB on this competition are:
- **GBC over global features used INSIDE a search** (AidenSong) — the search does the action discrimination.
- **MLP over PER-SHOT features used as a filter** (konbu17) — features differ across candidates by construction.

Both architectures sidestep the degeneracy our setup hit. Neither does "global features → direct argmax", which is what Phase A built.

### Literature (academic + AlphaZero-family)

- **Leiden CoG 2019 ("Policy or Value?")** — value-MSE is a poor proxy for action rank in AlphaZero-style heads; policy loss dominates strength. Confirmed empirically by Stage 1+2.
- **Planet Wars 2010 top bots** (Jay Scott / oddshrimp, zvold) — all used per-planet scoring with safe-departure / ship-deadline / growth-field / fleet-ETA features. Scored *Δ vs do-nothing*, not absolute value. Direct precedent for what we now plan.
- **CRN paired-advantage + pairwise ranking loss** (Bradley-Terry, RankNet, Cao et al.) — dominate pointwise MSE for rank. No published Galcon/Halite/Lux precedent for CRN-paired training — possible edge, but feature design is the prerequisite.
- **GNN over planet graph** — natural fit; no published precedent in this game family.

## What this changes about Phase B-1

The 2026-05-28 AM lens-critique and PM re-refinement (handover §"Post-diagnostic re-refinement") landed on:
> Phase B-1 merges with B-3: CRN-paired advantage labels generated against `baseline_pv_eta`, Spearman-τ training-time gate, 500-game overnight or Kaggle GPU corpus.

This is still a sound RECIPE for the target. But applied to the existing 40 pooled features, the data-gen budget is wasted. The features cannot represent what the labels are now trying to teach.

**Amendment to Phase B-1:** feature redesign is a prerequisite to the corpus build. Two viable directions, both with public-LB precedent in this competition:

### Direction 1 — Per-emit MLP filter (konbu17 architecture, +19 pp evidence)

Don't try to replace the chooser's argmax. Build a per-emit binary classifier that vetoes bad-looking shots from the rule-based proposer. ~24 features per candidate shot, computed at proposal time. MLP is tiny (~3500 params). Embeds as base64 inside the bundle.

**Pros.** Proven mechanism on this exact competition. Bounded compute (no rollouts needed — just labelled emits from self-play replays). Composes additively with the live μ=1149 production stack rather than replacing it. Fits the rolling-pair coordination posture (lower risk of eviction by surprise regression).

**Cons.** Filter-only doesn't add new actions, only removes; ceiling capped by proposer's recall.

### Direction 2 — Per-candidate score head (kashiwaba architecture, generalised)

Reshape to a chooser-aware MLP that scores each candidate independently using per-candidate features (src+tgt planet stats, ETA, opp-eta, growth-field) PLUS global state features broadcast. Chooser argmaxes the head's per-candidate scores directly. Trained with CRN-paired pairwise hinge / RankNet against `baseline_pv_eta` rollouts.

**Pros.** Single end-to-end head; can replace existing chooser at the argmax step. Carries the structural information (per-candidate distinction) the current head can't encode. Direct fit to the chooser's argmax pattern.

**Cons.** Bigger compute (still needs the CRN rollout corpus). Higher integration risk vs the production stack. Untested on this competition.

### Recommendation

Run **Direction 1 first** (cheaper, evidence-backed, +19 pp precedent). It earns or kills feature-redesign as a category in <1 session of compute. If it lifts, the next slot is **Direction 2** as the longer-horizon investment, with the feature work already done.

The CRN-paired advantage label idea is preserved; it now applies to Direction 2's per-candidate head, not the existing 40-pool head.

## What to falsify next

| Hypothesis | How to test |
|---|---|
| 40-pool features are sufficient if loss is rank-aware | Stage 2 said no (RankNet gave +5 pp; still chance in close pairs) |
| 40-pool features are sufficient if MLP capacity is larger | Bigger model on same target & features won't help (target is degenerate; permutation importance shows 35/40 features dead). Skip. |
| Per-emit features + binary label achieves ≥ +5 pp vs production bundle | **Direction 1** — n=32 A/B after training; konbu17's labelling protocol as the recipe |
| Per-candidate features + CRN paired-advantage label beats `baseline_pv_eta` | **Direction 2** — only run if Direction 1 cleared |
| 4 of the existing 40 features are doing all the work | Stage 3 confirmed (top 5 = 2.47 ΔR², rest each < 0.003). Already done. |

## Methodological notes worth promoting to CLAUDE.md or improvements.md

1. **Distillation target dictates which features the head will use.** A teacher that depends on only a few input features renders the rest dead, even if the input set is nominally rich. Always check permutation importance before declaring "features are sufficient".
2. **Cheap stage-3 probes (permutation importance, close-pair rank agreement) on existing artifacts can falsify a data-gen plan before any compute is spent.** Each of these stages took seconds; together they redirected ~10–28 GPU h.
3. **"Pooled features → direct argmax" is structurally degenerate when sibling candidates conserve the pooled quantities.** Either expand features per-candidate, or use the pooled head inside a search step (à la AidenSong's 1-ply minimax) so the comparison is in scalar space across DIFFERENT terminal states.
