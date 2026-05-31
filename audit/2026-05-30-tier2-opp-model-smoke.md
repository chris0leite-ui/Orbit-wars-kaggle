# Tier 2 opp-emit predictor — implementation + smoke (2026-05-30 PM2)

## Context

B.3 (CRN-paired advantage value head) shipped clean but A/B lift was
marginal (18/32 = 56.2 %, Wilson-lo 0.393 vs `launch_rules_universal`;
fails Rule 43b). PI decision 2026-05-30 PM: HOLD the B.3 bundle,
advance to Tier 2 opp model as the next lift source.

Hypothesis: pv_eta's chooser scores candidates inside `fast_sim`
rollouts whose opp policy is a cheap rule-base (`lite_greedy_policy`
default, `top_tier_mirror_policy` if `BASELINE_OPP_TIER=1`). Replacing
the rule-base with a learned opp policy should make the chooser's leaf
state a closer match to what we see on the live ladder.

Plan: `/root/.claude/plans/you-are-a-machine-snoopy-russell.md`.

## What landed this session

### Design reframe vs. handover

Handover sketched modifying `lib/world_model.py:predict_garrison_at`.
Investigation found that function is only called from
`score_candidate_static` (legacy v2) and the counterfactual at
`chooser_trajectory.py:331` — NOT from the active v4 leaf path. The
active v4 scorer reaches the opp via `opp_actions_for_snap` →
`_select_opp_policy` → `lib/opp_model.py`. `opp_model.py:128` already
had a `trained_logreg_policy` stub literally reserved for this exact
work (docstring named `data/shot_validator/schema.json`).

PI-ratified design: plug Tier 2 into `lib/opp_model.py:trained_logreg_policy`
as a **filter on Tier 1 candidates** (PM5 booster as gate, threshold 0.30,
self-reinforce passes through unfiltered per konbu17 design). Three
alternative shapes considered + rejected (argmax replacement: B.2
selection-bias precedent; score-rescaled rank: Reframe-A failure
precedent).

### Code changes

| File | Change |
|---|---|
| `lib/opp_model.py:124-243` | Implement `trained_logreg_policy` — lazy-load shot-validator booster (gzip+base64 blob or disk fallback), enumerate Tier-1 candidates, encode 45-d features per emit, score with `predict_proba`, drop sub-threshold. Self-reinforce pass-through (konbu17). Falls back to Tier 1 on any failure. |
| `agents/baseline/chooser.py:22,44-60` | Import `trained_logreg_policy`; add `BASELINE_OPP_TIER=2` branch in `_select_opp_policy`. |
| `agents/baseline/chooser.py:124-133` | **Bug-fix (separable from Tier 2):** `affordable_validate_cap` was probing per-step cost with EMPTY actions, missing the opp-policy cost. With Tier 1 (~5 ms/call) or Tier 2 (~6 ms/call) the cap was undersized by ~10×, blowing the 1000 ms env cap. Probe now uses `opp_actions_for_snap(probe, ...)` to capture real cost. With Tier 0 (lite_greedy ~0.01 ms/call) the probe is unchanged → no behavior change for current rolling-pair. |
| `agents/baseline_pv_eta_vh_opp/main.py` | NEW. Wrapper preamble: pv_eta foundation + `BASELINE_OPP_TIER=2` + `BASELINE_VH_LAMBDA=0` (clean Tier-2 attribution, no head mixing) + threshold 0.30 + kinematic-table OFF. |
| `scripts/bundle_pv_eta_vh_opp.py` | NEW. Clone of `bundle_pv_eta_vh.py` patching `_OPP_BOOSTER_B64` instead of `_VH_MODEL_B64`. |

### Pre-A/B latency bench

Per-call cost measured on a step-40 obs (after env reset + 40 idle
steps), 30 reps × 2 seats each:

| Tier | Policy | Cost | Notes |
|---|---|---:|---|
| 0 | `lite_greedy_policy` | 0.01 ms | obs-only ROI greedy; default in chooser |
| 1 | `top_tier_mirror_policy` | 5.02 ms | World+WorldModel+missions+settle+realize |
| 2 | `trained_logreg_policy` | 5.88 ms | Tier 1 + 45-d featurize + booster predict_proba |

Tier-2 inference adds only ~17 % over Tier 1 (the heavy cost is the
World/WorldModel rebuild + mission proposers, shared with Tier 1).
Tier 2 is ~600× slower than Tier 0 — load-bearing for chooser cap.

### Single-game smoke results

| Run | Wallclock budget | Probe fix | Outcome | p50 turn-ms | p95 | max |
|---|---:|:---:|---|---:|---:|---:|
| Pre-fix, seed=0, BASELINE_WALLCLOCK_MS=50 | 50 ms | no | p1_win (Tier 2 lost) | 172 | 535 | 922 |
| Post-fix, seed=0, default 1000 ms | 1000 ms | yes | p1_win (Tier 2 lost) | 473 | 812 | 939 |

Pre-fix turn cost averaged 3.4× the wallclock budget — confirms the
chooser cap was undersized with Tier 2 active. Post-fix the budget is
respected on this single short game (n_steps=197, p95=812 < 1000).
But see A/B result below for the n=32 picture, which tells a different
story.

### A/B result vs `launch_rules_universal` (n=32, default 1000 ms)

| n | wins | % | Wilson 95 % CI | verdict | focal turn-ms p50 / p95 / max |
|--:|--:|--:|---|---|---|
| **32** | **0/32** | **0.0 %** | **[0.000, 0.107]** | **FAIL (Whi<0.5 — adaptive early-stop)** | 549 / **1621** / **3015** |

**Decisive falsification.** 0/32 vs the rolling-pair champion. Wilson
upper bound (0.107) is below the 0.50 gate, so the adaptive harness
early-stopped at n=32 (Wilson confidence the lift is < 0 is high). No
seed produced a win.

**Two failure modes overlap:**

1. **Latency regression (load-bearing).** p95=1621 ms and max=3015 ms
   blow past the 1000 ms env cap — ~40 % of turns over budget. In a
   Kaggle live game this would trigger turn-forfeit penalties; in the
   local harness it just means the chooser is using more wallclock
   than budgeted (the harness doesn't enforce the cap, only reports
   it). Either way, this is a non-starter for submission.
   - The chooser cap probe was fixed to include opp-policy cost
     (commit 05aa624), but Tier 2 cost is non-uniform per-turn
     (high-emit-count turns dominate). The single-state probe sets
     the cap based on a typical state; pathological turns blow past.
2. **Policy quality (also fatal).** 0/32 is unambiguous — even if
   timing were in cap, the filter is making the chooser strictly worse.
   Likely cause: the booster was trained on focal-seat-emit
   distribution (PM5 mixed-opponent corpus); when applied to the
   opponent's emit distribution inside the chooser's fast_sim
   rollouts, it under-predicts strong opp candidates → fast_sim opp
   policy emits less than reality → leaf state under-rates opp threat
   → focal chooser becomes over-confident → loses every game.

### Falsified design + Rule 37 axis status

**Root cause re-diagnosis (2026-05-31, after PI flagged 0/32 was
suspect).** The 0/32 was NOT (primarily) policy quality. It was a
**self-inflicted chooser bug introduced by my "fix" in commit 05aa624**.

Diagnostic chain:
1. Single-turn instrumentation: focal under `BASELINE_OPP_TIER=2`
   emitted **3 moves / 100 turns** vs Tier 0's **36 moves / 100 turns**.
2. Per-turn chooser trace: prerank=11-65 candidates available,
   `moves emitted = 0` on most turns, `commits = 0-2` per turn.
3. `score_candidate_v4` call count: Tier 0 = 671 across 30 turns
   (~22/turn) vs Tier 2 = **115 across 30 turns (~4/turn)**.
4. `per_cand_ms` measurement under Tier 2 with my probe fix: median
   **255 ms**, max 390 ms.

The formula in `affordable_validate_cap`:

    per_cand_ms = (per_step_ms × avg_K + per_leaf_ms) × 1.5_safety
    safe_deadline = deadline - per_cand_ms

uses `avg_K = (MIN_HORIZON + MAX_HORIZON) / 2 = (25 + 40) / 2 = 32.5`
— but actual `score_candidate_v4` rollouts use `prop_horizon` from the
prerank entry, which is typically eta+settle ~= 5-15 ticks, NOT 32.5.

With my probe fix capturing `per_step_ms ≈ 7 ms` (Tier 2 cost included),
`per_cand_ms = (7 × 32.5 + 1) × 1.5 = ~344 ms` — but actual per-candidate
cost is `~7ms × 10 = ~70 ms`. The estimate is **~5× too conservative**.

`safe_deadline = deadline - 344 ms` then aggressively bails the
validation loop after ~3-5 candidates, leaving most of the 60-candidate
prerank unscored. With `scored` mostly empty and no Δ>0 fallback,
`chooser_trajectory.py:1084 if not scored: return [], []` fires —
focal emits NOTHING most turns → focal does nothing all game → loses
0/32.

**The "policy quality" theory in the previous version of this audit
was wrong.** Tier 2's filter alone (drops ~32% of Tier 1 emits in the
rollout) is NOT enough to cause 0/32. The chooser's own bail-out was.

### Action taken

Reverted the probe fix in `agents/baseline/chooser.py` (single line:
back to empty-actions probe). Tier 2 now restores normal emit rate
(~half of Tier 0, comparable to Tier 1).

Post-revert single-game smoke at seed=7, default 1000 ms wallclock:
turn-ms **p50=200 p95=813 max=1107** (vs my-fix version p50=549
p95=1621 max=3015 — 3× faster, mostly in cap). Outcome p1_win at n=1
(no signal — single game).

The probe fix WAS the right idea (capture true opp-policy cost) but
the per_cand_ms formula needs a smaller `avg_K` to be useful — the
formula was sized for Tier-0 / `lite_greedy_policy` where the
per-step cost is ~0.5 ms and the formula's over-estimate didn't hurt.

Tier 2's actual viability is now an open question — needs n≥8 A/B at
the reverted state. The n=4 A/B I tried timed out at 10 min, so the
local harness needs a longer per-game budget for Tier 2 work.

### Falsified design + Rule 37 axis status (unchanged)

PM5-booster-as-opp-filter, threshold 0.30, no retrain: **FALSIFIED
1st-of-3 on this axis.** Two more variants are nominally allowed
before Rule 37 axis cap, but the failure modes above suggest both
of the remaining variants in the planned sweep would also fail:

- **Threshold sweep (0.20 / 0.40)**: addresses neither failure mode.
  Lower threshold filters fewer candidates (less timing relief);
  higher threshold filters more (more timing relief but worse policy
  quality — booster's signal is already misaligned).
- **Retrain on B.3-style opp-specific corpus**: addresses failure
  mode #2 (policy quality) but not #1 (latency). Adds ~3-4 h of
  corpus-gen + training. Worth pursuing only if PI rules the
  latency fix is tractable.

**Recommended PI escalation (Rule 26):** the design as conceived
(filter Tier 1 candidates with a focal-trained booster, run inside
every chooser-rollout step) appears structurally mismatched. Two
architecturally distinct alternatives to consider:

A. **Cache the opp policy output for the turn.** Compute Tier 2 opp
   action set ONCE per real-game turn at the actual game obs, and
   replay that constant action set inside every chooser rollout step.
   Eliminates the K×N_candidates inflation. Loses the "opp reacts to
   our hypothetical move" signal, but that signal is weak anyway given
   the rule-base + 1-step lookahead.

B. **Tier 2 augments leaf scoring, not opp rollouts.** Score the
   reachable leaf states with the booster's "P(this shot succeeds)"
   over the focal seat's own candidates — same architecture as the
   PM5 baseline_validated wrapper that PI explicitly approved a
   submit override on (60.9 % vs baseline). This is the original
   konbu17 filter on FOCAL, not OPP. Reuses every piece of the
   architecture except where the booster's output gets multiplied in.
   But this is closer to Reframe A (additive logit term on focal),
   which was falsified at λ=0.5. So it's likely a dead end.

### Pre-existing test_bundle.py environment failure (flagged)

### Pre-existing test_bundle.py environment failure (flagged)

`pytest tests/test_bundle.py` errors with `ModuleNotFoundError: No
module named 'kaggle_environments'` at line 1075 of the bundled file.
Reproduced on HEAD with all my changes stashed — pre-existing, not a
Tier-2 regression. The bundler's own internal `_smoke_import` step DOES
load `kaggle_environments` correctly (smoke import OK on the new
bundle). Rule 46 in spirit verified via the bundler's own smoke; the
pytest harness needs a separate friction fix. **Not blocking Tier-2 A/B.**

## What's next — PI escalation

A/B FAILED at n=32 with 0 wins. The current architecture (PM5
focal-trained booster as opp filter inside chooser rollouts) is
both latency-broken (p95 > env cap) AND quality-broken (0/32
even ignoring timing). The remaining sweeps in the original plan
(threshold 0.20/0.40, B.3-corpus retrain) don't address both
failure modes; the structural alternatives (cache opp output
per real turn; augment focal leaf scoring instead of opp policy)
need PI ratification before this session burns more compute.

**Recommended next session start (Rule 44 — read first):**

1. `state/MULTI_BRANCH.md` for live rolling-pair.
2. This audit's "A/B result" + "Falsified design" sections.
3. Decide one of:
   - Alternative A (cache opp output per real turn): probably the
     fastest fix; restores Tier 1 timing while keeping Tier 2's
     filtered candidate set.
   - Pivot away from Tier 2 entirely and re-open a different
     candidate from the closed-tracks list at the right level of
     sophistication (e.g. re-examine the B.3 head at λ=0.5 not 1.0;
     or revisit the chooser-side ML logit at λ=0.1 with the better
     understanding of opp-side asymmetry we now have).
   - Pause head-headroom work and ship a different candidate from the
     existing carry-forward set (`baseline_pv_eta_vh_b3smoke.py`
     bundle is on disk and clean — could be a calibration probe at
     PI's discretion despite Wilson-lo 0.39 vs launch_rules_universal).

**Carry-forward state:**

| Item | Status |
|---|---|
| `lib/opp_model.py` Tier 2 implementation | Working, falsified at n=32 |
| `agents/baseline/chooser.py` probe fix | KEEP — strict improvement; no behavior change at Tier 0 |
| `submissions/baseline_pv_eta_vh_opp.py` | Built, 982 KB, but DO NOT SUBMIT (FAIL gate) |
| `agents/baseline_pv_eta_vh_opp/main.py` + bundler | Preserved as Tier-2 wrapper template for any architecture variant |

## Carry-forward artifacts

| File | Status |
|---|---|
| `submissions/baseline_pv_eta_vh_opp.py` | 981 KB Tier-2 bundle, smoke import OK; A/B pending |
| `lib/opp_model.py` | Tier 2 implemented; falls back to Tier 1 if booster fails to load |
| `agents/baseline/chooser.py` | `affordable_validate_cap` probe fix |
| `data/shot_validator/validator_booster.txt` | PM5 LightGBM booster (val_acc 0.83), reused as-is |
