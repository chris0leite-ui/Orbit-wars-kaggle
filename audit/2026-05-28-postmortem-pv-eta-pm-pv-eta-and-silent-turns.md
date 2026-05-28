# Postmortem — 2026-05-28 PM pv-eta + silent-turns

Session on `claude/kaggle-submission-review-gZsCu`. Two distinct chapters:
(A) plan + ship `BASELINE_PV_ETA=1` (the modeling-correct PV pull-back
on candidate Δ); (B) post-submission investigation of "why do we lose
the seed=2 panel game across all 4 opponents," which redirected to a
pre-existing peak weakness ("silent turns").

## What went right

- **PV_ETA designed and shipped cleanly.** Single env-var-gated change,
  default-off byte-identical to peak, no new tuning knob (γ is the
  existing chooser discount). 5/5 new unit tests including the Rule 38
  fix-verification (reproduces failure state). 55/55 broader chooser/
  value/scoring/bundle sweep green. Wrapper smoke vs v7_0 seed=7 won
  in 164 steps at max-turn 774ms.
- **PI-directed A/B protocol adopted.** New `scripts/ab_quick.py`:
  parallel, 5 seeds, no seat swap, episodeSteps=250 truncation. Used
  successfully for single-opp + multi-opp panel.
- **Multi-opp panel cleared Rule 43.** 4-1 uniformly across {peak_anchor,
  v7_0, v4_planner, v3.5.1}; pooled Wilson-lo 0.58 ≥ 0.55. The 4-fold
  identical rate is hard to reconcile with a noise spike under any
  fair-coin null (joint probability 0.12%).
- **Devil's-advocate ritual fired correctly (Rule 26).** Before submit,
  surfaced the n=5 ≤ Rule 45 floor + step-250 metric tilt + recent
  two-disaster pattern; PI chose "wait for full panel" first, then
  "submit" after panel landed. Submission decision had complete
  evidence in front of it. Sub 53111837 (μ pending).
- **Pre-submit research caught a structural weakness ungilded by the
  earlier wins.** The seed=2 deep-dive correctly diagnosed silent
  mid-game turns as the failure mode, AND ruled out PV_ETA as the
  cause (peak also loses seed=2 vs v4_planner).

## What went wrong / risk-loaded

- **n=5 was below Rule 45's n=32 floor for any submit decision.** PI
  override given, but the recent 2 disasters (53083109, 53099001) came
  from similar small-n overrides. The 4-fold panel consistency partly
  but not fully compensates.
- **Step-250 truncation isn't the live metric.** 14 of 20 panel games
  hit the cap; we're measuring "leading at midgame," not "won at
  endgame." PV_ETA literally up-weights early payoffs — same bias as
  the truncated metric. Some risk of self-confirmation.
- **Cross-process determinism leak discovered, not diagnosed.** Same
  seed (0, 2) flipped outcomes between single-opp run and panel run
  with PV_ETA=1, focal-as-P0 both times. Within-process deterministic;
  across-ProcessPoolExecutor not. Source not located. Means: every
  past n=5 A/B carries unknown variance from this leak. Friction:
  `cross-process-determinism-leak`.
- **Bundler namespace collision (recurring 2nd day).** `bundle_agent.py
  agents/baseline --force` crashes in the parity-gate auto-import on
  `kaggle_environments.envs.lux_ai_s3` shadowing `agents.*`. Bundle
  file produced anyway; manual rebundle path used. This fired twice
  in 24 h — escalation due. Friction: `bundler-namespace-collision-on-load`.

## Frictions logged this session

Today's appends to `audit/friction.md` under `## 2026-05-28 PM`:

- `silent-turns-mid-game-pre-existing` — chooser emits zero for
  13-29 consecutive mid-game turns on contested-expansion seeds;
  weakness exists at peak; not introduced by PV_ETA.
- `cross-process-determinism-leak` — same seed flips outcomes across
  ProcessPool invocations; deterministic within a single process.
- `bundler-namespace-collision-on-load` — recurring 2nd day; manual
  rebundle workaround.

## Promotion candidates (for PI ratification)

1. **CLAUDE.md Rule 48 — "Cross-run reproducibility check before
   trusting any n≤16 A/B"** — given the determinism leak, n=5
   protocol now needs a same-seed double-check (run twice in different
   process invocations; if outcomes differ, the noise floor exceeds
   the lift signal and the result is meaningless). Mirrors the
   "small-n-ab-noise-misled-panel" friction tag pattern that drove
   Rule 45. Worth a rule until the underlying RNG leak is plugged.

2. **Bundler patch as TOP-PRIORITY tooling work next session** —
   `bundler-namespace-collision-on-load` has now fired 2 days running.
   Promote from friction note to action item.

3. **`scripts/ab_quick.py` documented as the new A/B route in
   `state/TOOLS.md`** — it's where new agents should be A/B'd given
   PI's "parallel, 5 seeds, step-250, no swap" directive. Currently
   sits as orphan tooling.

## PI input requested

This postmortem is being written as part of "wrap up". PI input on:

- Ratify promotion candidates 1-3?
- Is the n=5 step-250 panel protocol now standard for all future
  A/Bs, or specific to the PV_ETA evidence run?
- Next session priority: silent-turns investigation (modeling fix
  on rollout opp-model) vs other open candidates (bundler tooling,
  determinism leak, await sub 53111837 settlement)?

## What landed cleanly (deliverables)

- Commit `c45cf00` — feat: PV_ETA env-gated γ^(wait_N+eta) discount on
  candidate Δ, default off.
- Commit `a65e8b4` — tooling: `scripts/ab_quick.py` parallel no-swap
  step-truncated A/B harness.
- Commit `e65b50a` — fix: ab_quick agent-spec resolution.
- Commit `0d71aa6` — bundles: regenerate `submissions/baseline.py` +
  add `submissions/baseline_pv_eta.py` wrapper.
- Commit `564b70e` — submit: push-claim board row + sub 53111837 id.
- **Sub 53111837** at Kaggle (PV_ETA=1; μ pending; predicted band
  1100-1170, mode 1140).
- Knowledge-base: `2026-05-28-peak-foundation-and-scatter-symptom.md`
  (earlier in session) + `2026-05-28-silent-turns-pre-existing-weakness.md`
  (this PM).
- HANDOVER.md rewrite (this commit) — next-session investigation focus
  on silent-turns mechanism.

## Framework version at session-end

- Latest SHA: `564b70e` (push-claim board update).
- Active rules: 1..47.
- Loaded skills this session: `code-review`, `postmortem`.
