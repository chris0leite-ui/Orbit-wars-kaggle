# Postmortem — 2026-06-19 dropout-plan-review-rb5817

Session arc: dropout-NATIVE forward model — caught a fatal bug, reformulated the
value to ship-margin (0/40 → 13/40 vs Producer V2), then four closing-levers all
failed to reach base's 21/40. Decision-quality: one breakthrough, one serious
process failure.

## What went wrong

- **Rule 38 bypass (the costly one).** The native scorer threw a shape error on
  100% of turns (`garrison_status.arrivals_by_owner` is `[P,H+1,A]`; the code
  derived `H` as if `[P,H,A]`), silently swallowed by `except Exception: pass`.
  For the entire first half of the session I ran kill-gate A/Bs and recorded
  "kill-gate FAILED / hazard inert / refuted as scoped" — all measuring the
  STATIC fallback, not native. Green unit tests + green bundle smoke were trusted
  as sufficient; Rule 38 ("synthetic unit tests necessary but NOT sufficient;
  reproduce the real failing state") existed, applied, and was not applied — I
  never verified the native path executed in a real game. Caught only by the
  PI-requested code review. Cost: ~5 A/B runs (hundreds of games) + a false
  recorded verdict (later marked VOID).

- **Silent `except Exception: pass`** converted the shape bug into an invisible
  no-op. A gated scorer that REPLACES the production value must fail loud (strict
  raise) or assert an executed-count — otherwise the A/B silently measures the
  fallback. Fixed mid-session (warn-once + `PRODUCER_PLUS_NATIVE_STRICT`).

- **Threading nondeterminism in ad-hoc renders.** Multi-threaded `env.render`
  games diverged from the single-threaded A/B (a "loss/idle at step 25" render
  was actually a canonical active WIN). Misled a diagnosis round and the PI.
  Flagged but left unfixed (out-of-scope).

- **Lever-guessing before diagnosis (mild).** After the ship-margin win I tried
  wide-shortlist + λ-sweep + force-concentration before the PI said "trace losses
  first." Each was cheap and PI-approved, but the observation-driven loop says
  diagnose before prescribing. The trace (once done) found the real mechanism
  (mid-game frontier collapse) in one pass.

## PI-overrides (calibration data points)

- "passivity, not over-expansion" (×2) — I mislabeled the failure mode from
  traces/renders (the threading divergence compounded it).
- "why isn't dropout solving that?" — led directly to the ship-margin
  reformulation (the session's breakthrough); I had not questioned the ownership
  objective myself.
- "trace losses first" — redirected from knob-guessing to diagnosis.

## What went right (decision quality)

- Reformulating production-weighted OWNERSHIP → expected SHIP-MARGIN
  (engine-aligned) was the correct response to the churn diagnosis: optimize the
  engine's actual win condition (total ships), not a proxy. 0/40 → 13/40, churn
  eliminated. Durable modeling insight.
- Four negative levers cleanly bounded (each A/B-gated, paired-margin):
  shortlist (no change), λ (≤ optimal, higher hurts), force-concentration (worse),
  threat-growth (α=0.25 exact parity Δ=0.000, higher over-suppresses). The
  remaining gap to base is structural (multi-ply / coalition machinery), not a
  single knob.

## Frictions logged this session

- None written to `audit/friction.md` this session (postmortem invoked directly,
  not via WRAPUP step 4). The two patterns below are the friction-equivalents.

## Promotion candidates (PI ratified: NO — recorded here only, per "note on main")

### [ ] [CODE-COMP-DISCOVERED] Verify a gated scorer actually EXECUTES before trusting its A/B
**Tag:** `gated-scorer-silent-fallback-measures-baseline`
**Where:** kaggle-comp skill, code-comp eval section.
**What:** A scorer that replaces the production value behind `except: pass` makes
the A/B silently measure the FALLBACK, not the new code. Require a strict-raise
env flag (e.g. `*_STRICT=1`) or an executed-count assert, and one real-game smoke
with it ON, before recording any verdict. Synthetic unit tests passing is NOT
sufficient (Rule 38).
**Why:** This session — ~half the compute + a false "refuted" verdict on code
that threw every turn.

### [ ] [CODE-COMP-DISCOVERED] Pin torch threads in ad-hoc render/analysis to match eval workers
**Tag:** `render-threading-nondeterminism-diverges-from-ab`
**Where:** kaggle-comp skill, code-comp tooling notes.
**What:** Ad-hoc `env.render` / analysis scripts must set `torch.set_num_threads(1)`
+ `OMP_NUM_THREADS=1` to match the single-threaded eval workers, or float
reductions diverge and the rendered game contradicts the canonical A/B result.
**Why:** This session — a multi-threaded render showed a loss/idle game that was a
canonical WIN, misleading the PI's diagnosis.

## Framework version at session-end

- Commit SHA: see `git rev-parse HEAD` at wrap (branch
  claude/dropout-plan-review-rb5817).
- Active rules: CLAUDE.md Rules 0, 1, 12, 32, 35, 36, 38, 39, 40, 42, 45, 46.
- Loaded skills this session: code-review, postmortem.
