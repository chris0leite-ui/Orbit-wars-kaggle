# 2026-05-28 — open questions for Phase B

**For PI before Phase B kicks off:**

1. **Phase B-1 baseline isolation: CRN advantage head alone first, or
   advantage + multi-horizon together?**
   - Pro CRN-alone: cleanest signal for whether the variance-reduction
     argument actually holds. If it does, multi-horizon is upside; if
     not, we know exactly where to decompose.
   - Pro combined: single training run, single A/B slot, faster end-
     to-end. Risk: if it fails, we don't know whether CRN or multi-
     horizon (or both) is the broken piece.
   - Default if no PI input: CRN alone first.

2. **Strong opponent pool composition — five agents I have in mind:**
   `composite_a2_hybrid` (μ=1149), `trajectory_v4_wait_N` (μ=1143),
   `hold_feasibility_solo` (μ=1135), `favor_hybrid`, `favor`. Spread
   ~200 μ. Should we also include the analytical track agents
   (μ=806 / 829) as "below us" opponents to teach the head about
   weaknesses to exploit, or does that just add noise?

3. **Live-ladder calibration cadence — should Phase B candidates run
   the rolling-pair A/B (Rule 43 / 45) at every B-step (B-1, B-2,
   B-3) or only on the final submission candidate?** Every-step burns
   compute but catches early divergence; final-only is cheaper but
   risks a Phase-B-3 candidate that beats favor_hybrid but regresses
   vs the live pair.

4. **Kaggle GPU kernel template — same as the existing
   `kernels/value_head_train/` setup, or do we need a new one for
   the advantage-head + CRN data pipeline?** The Phase A kernel is
   single-target; CRN doubles the per-state forward passes (action
   + idle). Two-tier smoke (Rule 2 GPU clause) is mandatory before
   any production T4 push.

**Not blocking Phase B kickoff** — defaults exist for each. But each
answer changes the budget shape.
