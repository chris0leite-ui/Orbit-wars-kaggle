# HANDOVER.md — next-session brief

> Last written: 2026-06-01 07:35 UTC (sub 53243763 slotres still climbing
> from μ=600 prior — last seen μ=792-815, oscillating, NOT settled;
> integration plan with 00JzI ready to execute) by
> `claude/competition-objective-alignment-hqNVM`.

## 2026-06-01 session summary — slot reservation diagnosed + shipped + integration plan

### What landed (5 commits this session)

| Commit | Change |
|---|---|
| `ec43cac` | postmortem(composite): sub 53239342 settled μ=460 (later 545 then 537) — opposite pathology diagnosed |
| `6549eeb` | probe(candidate-distribution): dump per-candidate leaf_delta by target class |
| `097b474` | feat(chooser): per-class slot reservation (env-var gated, default OFF) — fix for the diagnosed bottleneck |
| `cbb862d` | fix(slotres): drop wallclock bump (was over 1000 ms env cap) |
| `3045e18` | submit(slotres): sub 53243763 — slot reservation + composite, PI submit |

### Load-bearing findings

1. **The chooser literally never scored attacks or expansions** in the
   composite (sub 53239342). Direct evidence from ep 78367540 step 100:
   - `time_to_enemy_threat` flagged 11/11 of our planets as threatened
   - Proposer surfaced 63 candidates (32 defense / 24 expansion / 7 attack)
   - Cheap_delta ranked defenses +12.5 median, attacks −1.0 median
   - Chooser's wallclock allowed only top ~5 by cheap_delta → all defenses
   - Action-distribution check on full game: defense 87%, attack 6%
2. **Slot reservation fixes it.** Same state, with `BASELINE_SLOT_RESERVATION=3/2/2`:
   - Attack candidate src=21→tgt=29 leaf_delta=**+40.3** — now scored
   - Expansion candidates +29.4 and +20.7 — now scored
   - Action distribution: defense 47%, attack 26%, expansion 27%
   - n=1 vs v7_0 WIN 106 steps
3. **Live calibration still in flux.** Sub 53243763 started at the
   Kaggle Elo prior μ=600, climbed up through the 700s, last seen
   μ=792-815 oscillating. **NOT settled — needs more games (typically
   12-24 hours after submit for the rating to stabilize).** The
   trajectory is upward overall, but the final value is unknown.
   Re-check `kaggle competitions submissions orbit-wars | head -5` at
   session start before any planning decisions depend on this number.
4. **00JzI has compounding work.** Their `BASELINE_JOINT_SYNC` shipped
   live as μ=1147; their `BASELINE_SIZE_BALANCE` passed n=16 75% local
   today (not yet submitted). Combined with our slot reservation, the
   three fixes are orthogonal and should compound. **Integration plan
   at `audit/2026-06-01-integration-plan/README.md`.**

### Live ladder state (2026-06-01 07:35 UTC)

| Sub ID | Agent | μ | Role |
|---|---|---:|---|
| **53243763** | baseline_pv_eta_vh_dist_slotres | **792-815 (climbing)** | Rolling pair (newest) |
| **53239342** | baseline_pv_eta_vh_dist_composite | **537.7** | Rolling pair (oldest) |
| 53227546 | baseline_pv_eta_vh_dist (EVICTED) | 801.1 | historical |
| 53223160 | baseline_joint_sync_submit (EVICTED) | ~1147 | 00JzI's best |
| 53212044 | baseline_pv_eta_vh_b3smoke (EVICTED) | ~1142 | hqNVM's best |
| 53182323 | baseline_launch_rules_universal (EVICTED) | 1183.7 | live champion |

- **Daily submission budget**: 3/5 used today. 2 slots remain.
- **Floor-at-risk**: TRUE — rolling pair sits 330+ μ below the
  evicted-pair peak.

### Next-session priority — execute the integration plan

**Read first:** `audit/2026-06-01-integration-plan/README.md`. Self-contained
step-by-step plan for combining slot-reservation (ours) with joint_sync +
size_balance (00JzI's) into a single bundle. ~3-4 hours of focused work.

Concrete first action: `cat audit/2026-06-01-integration-plan/README.md`,
then execute Step 1 (branch creation) and Step 2 (cherry-pick).

### What's CLOSED — do not re-explore (this session's addition)

- **Bumping `BASELINE_WALLCLOCK_MS` from 600 → 800 ms.** Single-game
  smoke went over the 1000 ms env cap (max=1195 ms / p95=1025 ms).
  Reverted in `cbb862d`. The composite already uses ~885 ms p95; there
  is no safe headroom for a wallclock bump in this chooser.
- **Slot reservation as a standalone agent at 3/2/2 against the existing
  composite stack** — sub 53243763 climbed from μ=600 prior into the
  792-815 range and is still settling. **Trajectory is upward but
  final μ unknown.** Mechanism works directionally; whether it alone
  is enough vs the historical joint_sync μ=1147 floor depends on where
  the rating settles. Integration with joint_sync + size_balance is
  the next axis regardless.

## 2026-05-30 PM session — B.3 built, A/B'd, DECISION = HOLD + advance

### What happened

1. **B.3 head trained + bundled.** Wrapper agent
   `agents/baseline_pv_eta_vh/` loads a small regressor that adds a
   λ-weighted residual to the pv_eta chooser's leaf score, trained
   on CRN-paired advantage labels (counterfactual A(s,a) =
   focal_margin(after a, K=10) − focal_margin(after idle, K=10)).
   Bundle: `submissions/baseline_pv_eta_vh_b3smoke.py` (773 KB).
   All Rule 46 gates clear: `pytest tests/test_bundle.py` 10/10
   GREEN; `python fast.py play <bundle>` runs a full game without
   crash.
2. **Kinematic-table substrate bug discovered + fixed.** Local A/Bs
   were running in a shared process where a sibling bundle's
   `os.environ.setdefault('KINEMATIC_TABLE_ENABLED', '1')` activated
   a buggy trajectory cache for BOTH agents. Fix: prepend
   `os.environ['KINEMATIC_TABLE_ENABLED'] = '0'` (hard set, not
   setdefault) to every bundled wrapper. Verified by reproducing
   the propagation: opponent bundle loads first → sets var to "1";
   B.3 bundle loads second → forces to "0";
   `_kinematic_table_enabled()` returns False. Commit `5bc88f8`.
3. **Clean A/B vs `launch_rules_universal` (Tier 1 rolling-pair
   top), n=32, P0-only, kinematic table OFF for both:**

   | Batch | Wins | % | Wilson 95% CI |
   |---|---|---|---|
   | Seeds 0-15 | 11/16 | 68.8% | [0.444, 0.858] |
   | Seeds 16-31 | 7/16 | 43.8% | [0.231, 0.668] |
   | **Combined n=32** | **18/32** | **56.2%** | **[0.393, 0.718]** |

   First batch was an upward sampling fluctuation; pooled rate
   sits at ~56% with Wilson-lo 0.393 — **FAILS Rule 43b**
   (lo ≥ 0.50) and is too noisy to claim lift under Rule 45.
   Timing improved (max 934 ms < 1000 ms cap; was 1154 ms with
   the buggy cache active).

### PI decision (2026-05-30 PM)

**HOLD the B.3 bundle. Do NOT submit. Advance to Tier 2 opp model
as the next lift source.**

Rationale:
- Local 56% point estimate over a μ=1173 opponent doesn't justify
  a slot spend — the
  `local-AB-not-calibrated-to-live-ladder` friction (0/16 local →
  μ=711 live) shows small-sample local lift can lie.
- Rolling pair is healthy (μ=1173 + μ=1017); the μ=1017 floor
  isn't desperate, so we can afford to invest one more session
  in the upstream fix instead of probing the ladder with a
  marginal head.
- The B.3 head's chooser is being fed pv_eta's STATIC opp model
  (assumes opponent does nothing in lookahead). Replacing that
  with a learned/adaptive opp model (Tier 2) is the larger
  lever; the current B.3 lift is plausibly capped by that.

### Next-session priority — Tier 2 opp model

Build an opp-emit predictor that pv_eta's chooser consumes inside
its lookahead, replacing the static "opponent does nothing"
assumption in `predict_garrison_at`. This is the Reframe-C
direction from the older PM3 menu, surfaced earlier because B.3
(Reframe B) shows a ceiling consistent with chooser-blindness to
opp action.

Concrete first moves (sketch — flesh out at session start):

1. **Define the prediction unit.** Per (state, top-K target
   planets) → predicted opp emit (ships sent / target / eta).
   Top-K to keep wallclock safe (start K=5).
2. **Corpus.** Reuse B.3's CRN self-play traces (already on disk)
   with a second pass: extract the opponent's actual emits from
   each state, write `(state_features, target_id, opp_emit)`
   tuples.
3. **Model.** Small classifier per target — does opp send? If
   yes, regress ships. Same featurizer family as the B.3 head,
   no GPU needed at first cut.
4. **Insertion.** `agents/baseline/chooser_trajectory.py:
   predict_garrison_at` — replace the static "opp does nothing"
   prior with the predictor's expected emit. Env-gated
   (`BASELINE_OPP_MODEL=1`) for clean A/B against the B.3 head
   as ground.
5. **A/B gate.** vs `launch_rules_universal`, n=32 minimum
   (Rule 45), Wilson-lo ≥ 0.50 (Rule 43b) to clear for submit.

### Carry-forward artifacts

| File | Status |
|---|---|
| `submissions/baseline_pv_eta_vh_b3smoke.py` | B.3 bundle, 773 KB, smoke + parity clean, HOLD — do NOT push |
| `agents/baseline_pv_eta_vh/main.py` | B.3 wrapper (kinematic-table override at top) — pattern for Tier 2 wrapper |
| `scripts/bundle_pv_eta_vh.py` | Bundler — adapt for Tier 2 (different blob payload) |
| `scripts/train_value_head.py` | Accepts B.3 14-d corpus schema — adapt featurizer for opp-emit labels |
| `audit/2026-05-30-reframe-b3-smoke.md` | B.3 head smoke + early A/B + panel results |

### What's CLOSED — do not re-explore (this session's addition)

- **B.3 head as a deployment candidate at current λ on current
  opp-model.** 18/32 = 56.2% (Wilson-lo 0.393) vs
  `launch_rules_universal`. NOT closing the wrap-pv_eta + head
  family — Tier 2 opp model may re-open this at a higher lift.
  Closing only "ship the current B.3 bundle as-is."

## Start here — first 30 min of work

1. **Refresh state-of-truth** (Rule 44).
   - `cat state/MULTI_BRANCH.md` for cross-branch live status.
   - `kaggle competitions submissions orbit-wars | head -5` for the
     rolling pair (snapshot 2026-05-30 09:58 UTC: sub 53131296
     `baseline_validated` μ=1097 + sub 53117942 `baseline_leaf_pv_2p`
     μ=1092; both rolling-pair slots preserved — B.2 pushed nothing).
2. **Read the B.2 closure + B.3 plan.**
   - `audit/2026-05-29-reframe-b2-value-head.md` — full B.2 result
     including the 0/32 verdict + selection-bias diagnosis.
   - `audit/2026-05-30-reframe-b3-crn-advantage-plan.md` — the B.3
     handover plan. Concrete next moves, file-by-file change list,
     verification gates, cost estimate.
3. **Run Step 0 fast_sim verification (~1 h).** Three small benches
   that decide whether B.3 stage 2 cost is 7.5 h CPU (feasible) or
   150 h CPU (untenable). Details in the B.3 plan §V0.

## B.2 verdict (2026-05-30 AM session)

| λ | Wins | n | Win rate | Wilson 95 % CI | Verdict |
|---:|---:|---:|---:|---|---|
| 1.0 (default) | 0 | 32 | 0.0 % | [0.000, 0.107] | **FAIL** |
| 0.1 (sweep) | 0 | 32 | 0.0 % | [0.000, 0.107] | **FAIL** |

Training-time gates all passed (Spearman ρ=+0.359 on val, walker
parity 0.000e+00, latency p95=691 ms inside cap), but the head's
predictions on **out-of-distribution candidates** (those pv_eta would
have rejected) drive the chooser to systematically losing actions.
The `trace_accepted` training trace only sees accepted candidates;
that's the **selection bias** that closes this axis.

**Closed under Rule 37**: observational-label additive-term head on
pv_eta's chooser. Do NOT re-iterate with adjusted λ, different K, or
different feature subsets — the bottleneck is the **label semantics**.

## Reframe B.3 — CRN-paired advantage (next session priority)

Replace observational labels with **counterfactual** ones:

```
A(s, a) = focal_margin(s after action a, K=10)
        − focal_margin(s after idle, K=10)
```

Both rollouts use pv_eta as the focal-seat policy from step 2 onward
AND as the opp policy throughout. The label is "the advantage this
candidate buys you vs doing nothing." Coverage is **all top-N
prerank candidates per state**, not just accepted ones — eliminates
B.2's selection bias.

**All B.2 infrastructure reused unchanged** — `lib/value_head_features.py`,
`agents/baseline/_value_head.py`, `agents/baseline_pv_eta_vh/main.py`,
`scripts/bundle_pv_eta_vh.py`, `scripts/train_value_head.py`,
`scripts/inspect_value_head_corpus.py`. Only the **labels** are wrong.

**New components**:
- `trace_prerank()` in `_trace_hook.py` (env var `BASELINE_PRERANK_TRACE`)
- `scripts/compute_crn_advantage.py` (advantage labelling driver)
- `scripts/gen_b3_corpus.py` (3-stage pipeline runner)
- `lib/fast_sim_pv_eta.py` (pv_eta-as-policy wrappers for fast_sim)

**Critical path**: `lib/fast_sim.py` lines 464-494 — `rollout(snap, K,
policies)` accepts agent callbacks. Step 0 verification (~1 h) tests
whether pv_eta works as a fast_sim policy without state leakage, at
<100 ms per K=10 rollout, with parity vs env.clone+step. If yes,
stage 2 cost is ~7.5 h; if no, fallback is ~150 h.

**Cost estimate**: ~13 h total in the best case (1 h V0 + 4 h stage 1
self-play + 7.5 h stage 2 advantage labelling + 5 min train + 30 min
bundle/A/B). Stage 2 is the dominant cost; reducible via top-N=5
instead of 10 (halves) or K=5 instead of 10 (halves again).

**Open design questions for PI to resolve at session start** (full
list in §"Open design questions" of the B.3 plan):

1. Top-N candidates per state: 10 (richer, 2× cost) or 5 (cheaper)?
2. Rollout horizon K: 10 or 5?
3. Stage 2 compute: local CPU or Kaggle parallel notebooks?
4. Combine with focal-wins-only filter as bias correction?

## What's CLOSED — do not re-explore

- **Per-shot binary Booster (Reframe A) as additive term in pv_eta
  chooser** — λ ∈ {4.5, 0.5, −0.5} catastrophic FAIL. Rule 37 axis
  closed 2026-05-29.
- **Per-shot Booster as hard FILTER on pv_eta** — closed PM5.
- **Wrapping bare baseline as deployment target** — closed PM5.
- **Observational K=10 ship-delta value head additive term on
  pv_eta** — λ ∈ {1.0, 0.1} both 0/32. Rule 37 axis closed
  **2026-05-30**. Bottleneck is label semantics, not λ.


2. **Read the B.1 probe report.**
   - `audit/2026-05-29-pveta-leaf-residual-probe.md` — full per-K and
     per-stratification tables.
3. **Move into Reframe B.2 design** — concrete next steps below.

## B.1 verdict (refreshed 2026-05-29 PM4)

16 games of pv_eta self-play; 3002 seat-0 accepted candidates analysed.

| K | n | σ(actual) | σ(pred) | σ(residual)/σ(actual) | R² | Spearman ρ |
|--:|--:|--:|--:|--:|--:|--:|
| 5 | 2995 | 327.2 | 224.9 | **1.242** | 0.003 | 0.007 |
| 10 | 2976 | 501.9 | 225.5 | **1.126** | 0.006 | −0.036 |
| 20 | 2895 | 699.5 | 228.1 | **1.070** | 0.004 | −0.023 |

**The pv_eta chooser's leaf-Δ explains essentially zero variance of
future seat-0 ship-delta** (R² ≈ 0.005 across all K). σ(residual) >
σ(actual) at every horizon — the leaf is worse than the unconditional
mean as a predictor of game-winning ship-delta on the chooser's own
accepted set.

ANOVA F-stats on the residual:

| K | ship_quintile | eta_bucket | owner_at_launch | top-5 tgt |
|--:|--:|--:|--:|--:|
| 5 | 6.4 | 10.9 | **43.2** | 1.6 |
| 10 | 7.1 | 11.5 | **35.9** | 3.3 |
| 20 | 8.0 | 14.5 | **43.7** | 4.1 |

The dominant axis is **target ownership at launch** (F = 35-43 across
K). Residual means (K=5):

- launching at MY OWN planets: residual −83 (overpredicted by ~83 ships)
- launching at NEUTRAL planets: residual +5 (near-perfect calibration)
- launching at ENEMY planets: residual −201 (overpredicted by ~201 ships)
- eta-bucket [0] (already-at-target): residual +16, std 53 (well-calibrated;
  longer-eta buckets show 6-9× larger spreads)

Mechanistic read: the chooser's leaf overestimates the ship payoff of
captures, especially enemy captures (which the opponent contests), and
the [0]-eta near-term snapshot is the ONLY regime where it's calibrated.
**The chooser is value-blind beyond 1 turn.** This is structured
headroom a per-target value head with `owner_at_launch` × `eta` features
can plausibly exploit.

**Decision: GREENLIT Reframe B.2.**

## Reframe B.2 — next-session concrete moves

Design implications from B.1:

1. **Supervision target:** per-(state, target, K=10) seat-0 ship-delta
   over the next 10 turns. K=10 has the cleanest signal/cost trade-off
   in B.1 (similar F-stats to K=5 and K=20 but larger σ(actual) so the
   labels carry more signal than K=5; less truncation than K=20).
2. **Features that MUST be in the head's input (B.1-evidenced):**
   - `owner_at_launch` one-hot (me / enemy / neutral) — F=43.
   - `eta` (raw integer) — F=11-15.
   - `ships` sent (raw integer) — F=6-8.
   - Plus the existing pv_eta chooser leaf-Δ as a feature (so the head
     learns the *residual* against the strong baseline, not from
     scratch).
3. **Don't waste capacity on per-`target_id` features.** F-stat 1.6-4.1
   is borderline. The signal is in the SEMANTIC features (ownership /
   eta / ships), not in target identity.
4. **Insertion point:** chooser leaf-Δ + λ · head_output as the chooser's
   ranking score. λ is a sweep (start with λ = σ(leaf-Δ)/σ(head-out)
   such that contributions are comparable).
5. **Reframe-A lesson applied:** the additive form was right; what
   killed Reframe A was that the per-shot Booster's `P_success` doesn't
   correlate with game-winning. B.1 confirms that ship-delta-over-K *is*
   what we want to predict.

**Cost estimate:** ~2-3 sessions. Stage 1 = corpus regen using the
B.1 runner pattern (`scripts/probe_pveta_selfplay.py` already writes the
right format). Stage 2 = train a small regressor (GBT or MLP) on the
per-candidate features → K=10 ship-delta. Stage 3 = embed in a wrapper
agent + bundle + A/B vs pv_eta.

## Reuse from B.1 infrastructure

| File | What it does | Status |
|---|---|---|
| `agents/baseline/_trace_hook.py:trace_accepted` | Per-turn accepted-candidate JSONL writer | Reuse as-is for B.2 corpus gen |
| `agents/baseline/chooser_trajectory.py` (emit loop) | Emits delta_pred + eta per accepted candidate | Reuse as-is |
| `agents/baseline_pv_eta_probe/main.py` | Probe wrapper (BASELINE_PV_ETA=1, λ=0) | Reuse for corpus gen; B.2 will need a sibling wrapper that loads the head + adds λ·head_out |
| `scripts/probe_pveta_selfplay.py` | ProcessPool runner with `maxtasksperchild=1` | Reuse as the B.2 corpus generator |
| `scripts/probe_pveta_leaf_residual.py` | Residual analyser | Adapt for held-out val-set evaluation of the trained head |

## Live ladder state (refreshed 2026-05-29 17:53 UTC)

| Sub ID   | Agent                             | μ        | Role               |
|----------|-----------------------------------|---------:|--------------------|
| 53131296 | `baseline_validated.py` (PM5 MLP) | **1096.9** | rolling pair (top) |
| 53117942 | `baseline_leaf_pv_2p.py`          | **1091.9** | rolling pair (bot) |
| 53111837 | `baseline_pv_eta.py` (EVICTED)    | 1154.8   | historical peak    |

- Daily submission slots: **5/5 free** (PM4 did not push).
- Deadline: 2026-06-23 (25 days).
- Team count: 3451.
- Floor-at-risk: **MODERATE** (rolling pair ~58 μ below evicted
  pv_eta peak).

## Sample size note

Plan called for 25 games. Container resumes between sessions kept
killing the long-running sweep; PM4 settled on 16 games run in the
foreground (~8 min) because the verdict was unambiguous (F = 35-43
already at this n; further games would tighten CIs but not change
the call). If B.2 corpus gen needs more, increase `--games` and break
it into multiple foreground runs.

---

## PM3 resume (2026-05-29 evening) — Reframe A → B handoff

The PM3-resume content below documents the Reframe-A falsification +
the B.1 vs B.2 fork before B.1 ran. Now superseded by the verdict
above; kept for traceability.

### Start here — first 30 min of work (HISTORICAL)

1. **Refresh state-of-truth** (Rule 44).
   - `cat state/MULTI_BRANCH.md` for cross-branch live status.
   - `kaggle competitions submissions orbit-wars | head -5` for the
     rolling pair. (Confirmed 17:53 UTC today — see numbers below.)
2. **Read the Reframe A postmortem.**
   - `audit/2026-05-29-reframe-a-falsified.md` — what's closed, what
     carries forward.
3. **Pick one of two Reframe B opening moves** (decision below).

## Reframe B — concrete opening move (PICK ONE)

The Plan-agent's Reframe B spec is "per-target continuous value
head: how many future ship-deltas does owning planet T at T+k earn
the focal seat?" That's a multi-stage program. Two viable openers
fit in a single session:

### B.1 — Diagnostic probe FIRST (recommended)

Before training any model, characterize where pv_eta's chooser
leaves value on the table. Concrete:

1. Reuse the trace hook (`agents/baseline/_trace_hook.py`) to log
   **(state hash, candidate emit, chooser's leaf value at horizon,
   ACTUAL ship-delta on the focal seat over the next K turns)** for
   every accepted candidate, in pv_eta self-play. K ∈ {5, 10, 20}.
2. Analyze the residual: `actual_delta − chooser_leaf`. If
   residual variance is high AND the residual correlates with
   identifiable features (target_id, ship-count, eta), there's
   headroom for a per-target value head.
3. If residual variance is LOW, pv_eta's chooser is already
   value-optimal at its accepted set → Reframe B has no ceiling →
   pivot to **Reframe C** (opponent-emit predictor) instead.

Cost: ~2 h. Output: `audit/<date>-pveta-leaf-residual-probe.md`.
Decision-grade evidence before committing to multi-day training.
This is the Reframe-A lesson applied: probe the GAME-WINNING
correlation, not just the model's signal quality.

### B.2 — Reuse existing distilled head, swap the target

`data/value_head_distill/` has a Phase-A distilled head fit to
`favor_hybrid` (scalar, val_acc R²≈0.998). The Phase-A finding was
that the head's high R² didn't preserve action-Δ rank order (knowledge-base/thoughts/2026-05-28-pm-distillation-action-rank-collapse.md).
If the distillation infrastructure still works, regenerate with a
per-target ship-delta target instead. Cost: ~4 h training + bundle
+ A/B. Higher risk — re-uses the failed Phase-A architecture.

### Recommendation: do B.1 first.

B.1's verdict either greenlights B.2 (or a richer Reframe B
training run) with calibrated expectations, OR redirects to
Reframe C BEFORE we burn a multi-day training cycle.

## Infrastructure already on disk (carry-forward from PM2)

| File | What it does | Status |
|---|---|---|
| `agents/baseline/chooser_trajectory.py` `PV_ETA_ENABLED` | Source-side pv_eta γ-discount | Verified parity vs bundled live champion (Wilson CI [0.364, 0.691]) |
| `agents/baseline/_ml_logit.py` | Lazy-load LightGBM, batched featurize + predict, centered-logit term | Working at λ=0; **do not use** at λ≠0 (Reframe A falsified) |
| `agents/baseline/_trace_hook.py` + `scripts/probe_ml_logit_signal.py` | Opt-in candidate trace + per-turn σ/ρ/histogram | **Reuse** for B.1: change the keying and the analysis script |
| `agents/baseline_pv_eta_ml/main.py` | Env-var wrapper template | Template for `agents/baseline_pv_eta_vh/main.py` |
| `scripts/bundle_pv_eta_ml.py` | Wrapper bundler with `_BOOSTER_B64` patch | Pattern for `scripts/bundle_pv_eta_vh.py` |
| `submissions/baseline_pv_eta_ml.py` | 953-KB single-file bundle, λ=0 = byte-equivalent to bundled pv_eta | Available as a clean wrap-pv_eta starting point |

## What's CLOSED — do not re-explore

- **Per-shot binary Booster (45-d, 1000ms training corpus) as ANY
  additive term on pv_eta's chooser.** λ ∈ {4.5, 0.5, −0.5}
  catastrophic FAIL. Rule 37 axis closed.
- Per-shot Booster as a hard FILTER on pv_eta (closed PM5).
- Wrapping bare baseline as a deployment target (PM5).

---

## Reframe A — falsified mechanism summary (PM2 session)

What was tested: a centered-logit additive term inside
`score_candidate_v4` and `score_candidate_v4_joint`:
`score = score + λ * (logit(P_success) - logit(0.5))`, where
P_success is the existing 45-d LightGBM Booster's prediction. λ swept
in {4.5, 0.5, −0.5} (0.1σ_delta, 0.011σ_delta, and the negative
mirror). All three regressed catastrophically vs bare pv_eta. Rule 37
axis closed.

Step-0 probe gates PASSed (σ(P)=0.26, |Spearman ρ|=0.13, median P=0.79)
but the verdict didn't translate. The Booster's training distribution
(baseline-proposer emits) doesn't match pv_eta's surfaced candidate
distribution. The Booster confidently re-ranks pv_eta's argmax in a
direction that's anti-correlated with game-winning. Even λ=0.5
(~1% of σ_delta) regressed 31/32 — the destruction happens at
tie-breaking.

Lessons logged to `audit/2026-05-29-reframe-a-falsified.md`:

- **Probe gates are necessary but not sufficient.** Non-redundant
  information can still be noise. A 4th gate ("does the ML term
  correlate with self-play win rate on the chooser's accepted set?")
  would be needed to catch this upstream, but that gate is itself an
  A/B.
- **Wrap-pv_eta architecture is fine** — the pv_eta source port +
  env-var wrapper + bundler all ship working at λ=0. Reuse for B.
- **Per-shot binary supervision is closed for this Booster on
  pv_eta's chooser.** Re-training on pv_eta's emit distribution
  remains theoretically open, but ceiling is uncertain and cost is
  similar to Reframe B.

## Infrastructure delivered (carries to Reframe B)

| File | What it does | Reuse for B |
|---|---|---|
| `agents/baseline/chooser_trajectory.py` PV_ETA_ENABLED | Source-side pv_eta γ-discount, env-gated (verbatim port from bundled live champion) | Yes — Reframe B wraps pv_eta the same way |
| `agents/baseline/_ml_logit.py` | Lazy-load Booster, batched featurize + predict, centered-logit term, candidate keying | Adapt the scoring helper shape; swap the model + featurizer |
| `agents/baseline/_trace_hook.py` + `scripts/probe_ml_logit_signal.py` | Opt-in candidate trace via env var + per-turn σ/ρ/histogram report | Yes — use the trace hook to characterise B's per-target signal before wiring it |
| `agents/baseline_pv_eta_ml/main.py` | Env-var-only wrapper (peak orbitfix preamble + BASELINE_PV_ETA=1 + BASELINE_ML_LAMBDA) | Template for Reframe B's wrapper |
| `scripts/bundle_pv_eta_ml.py` | Wrapper bundler — inlines inner baseline, prepends env-var preamble, patches `_BOOSTER_B64`, single-file Kaggle submit | Reuse pattern for B's bundler (swap embedded blob source) |
| `submissions/baseline_pv_eta_ml.py` | Working 953-KB bundle at λ=0 = bundled pv_eta byte-equivalent | Sanity smoke for any wrap-pv_eta variant |

## Live ladder state (unchanged this session)

- 53131296 — `baseline_validated.py` (PM5 25-d MLP filter) — μ = **1081.3**
- 53117942 — `baseline_leaf_pv_2p.py` — μ = **1084.5**
- Historical peak (EVICTED): μ ≈ 1154.8 (sub 53111837, `baseline_pv_eta.py`)
- Daily submission slots remaining: **5/5** (0 used today — Reframe A produced nothing submittable)
- Floor-at-risk flag: **TRUE** (rolling pair sits ~70 μ below historical pv_eta peak)

## Next session — read order (Rule 44)

1. `state/MULTI_BRANCH.md` — live Kaggle rolling pair, closed tracks,
   push claim board.
2. **`audit/2026-05-29-reframe-a-falsified.md`** — full Reframe A
   postmortem and what's closed vs open.
3. This file's "Reframe B" section below.
4. `state/TOOLS.md` for A/B harness conventions.

## Reframe B — next session priority

Per-target continuous value head. Different supervision target from
the falsified per-shot binary classifier.

**Supervision unit:** "how many future ship-deltas does owning planet
T at time T+k earn the focal seat, conditional on a candidate
mission decision?" Continuous regression label per (state, target,
horizon) tuple.

**Insertion point:** chooser's leaf-value slot, augmenting (not
replacing) `predict_garrison_at`. The chooser's hand-coded leaf
already encodes a strong scalar; B adds a learned per-target
correction.

**Why this might succeed where A failed:**

- Per-shot binary supervision conflates "lands as intended" with
  "wins game" — the failure mode of A. Per-target continuous
  supervision IS game-winning ship-delta.
- B re-uses pv_eta's chooser unchanged structurally; A was an
  additive perturbation that fought pv_eta's tuned argmax.
- The existing scaffolding (`data/value_head/`,
  `data/value_head_distill/`) has Phase A artefacts that may be
  adaptable.

**Infrastructure to reuse from Reframe A:**

- `agents/baseline_pv_eta_ml/main.py` → clone as
  `agents/baseline_pv_eta_vh/main.py`. Same env-var wrapper shape.
- `scripts/bundle_pv_eta_ml.py` → clone as
  `scripts/bundle_pv_eta_vh.py`. Same B64 embed pattern with a
  different blob target.
- `agents/baseline/_trace_hook.py` → adapt for per-target tracing
  (key by `(target_id, horizon)` instead of per-shot keys).
- `scripts/probe_ml_logit_signal.py` → adapt for B's gate
  characterisation. **Add a 4th gate** beyond σ/ρ/median: a
  self-play correlation check — does the model's predicted value
  correlate with the chooser's eventual ship-delta on accepted
  shots?

**Decision rule (Rule 37 axis):** B is its own axis (per-target
continuous, not per-shot binary). New 3-variant budget. Failure
modes to watch: (1) chooser-already-optimal — the hand-coded
leaf may already capture most of the per-target signal; B adds
nothing; (2) value-head-distillation-collapse — the model
learns to mimic the leaf instead of correcting it.

## Reframe C — deferred (after B's verdict)

Opponent-emit predictor inside pv_eta's lookahead. Cost ~5-7 days
even with the reframe-A scaffolding reuse. Defer to after B.

---

## Read order (Rule 44 — mandatory)

1. **`state/MULTI_BRANCH.md`** — live Kaggle rolling pair, closed
   tracks, push claim board.
2. **`state/TOOLS.md`** — A/B harnesses, single-game diagnostics,
   validation suite.
3. **`CLAUDE.md`** — rules 1-47.
4. **This file (top section first)** — today's session result + next-
   session menu of A / B / C.
5. **`knowledge-base/thoughts/2026-05-29-phase-2-v2-validator-falsified-pv_eta-foundation.md`**
   — full reasoning for the falsification + foundation lock.
6. `audit/friction.md` if you're about to touch a fragile path.

## Where we are (2026-05-29 14:30 UTC)

- **Comp:** Orbit Wars. Deadline 2026-06-23 23:59 UTC. **25 days remain.**
- **Live rolling pair (Kaggle auto-keeps these two):**
  - 53131296 — `baseline_validated.py` (PM5 25-d MLP filter) — μ = **1081.3**
  - 53117942 — `baseline_leaf_pv_2p.py` — μ = **1084.5**
- **Historical peak (EVICTED from rolling pair):** μ ≈ 1154.8
  (sub 53111837, `baseline_pv_eta.py`, 2026-05-28 09:42).
- **Daily submission budget:** 5/day. 5/29 used: 0. 5 slots remain.
- **Floor-at-risk flag:** **TRUE** — both rolling-pair submits sit
  ~70 μ below the historical pv_eta peak.

## Day-N session 2026-05-29 — Phase 2 v2 Booster falsified; pv_eta locked

### What landed (4 commits)

| Commit | Change |
|---|---|
| `04f44d1` | Phase 2 v2 Stages 3-5 — corpus regen, booster train, embed (100 ms wallclock, leftover from PM5) |
| `8e30f7b` | Re-embed Booster trained on 1000 ms-wallclock corpus (matches eval) |
| `eafed05` | Per-turn validator trace + drop-analysis diagnostic |
| `a3d8301` | Fix: planets schema is tuple not dict in trace block |
| `963dada` | `BASELINE_VALIDATOR_THRESHOLD` env override (no re-embed needed for sweep) |

### Load-bearing findings

1. **Per-shot LightGBM Booster filter is the wrong primitive vs pv_eta.**
   On bare baseline at threshold 0.05 it adds +34 pp (28 → 62.5 % vs
   bare baseline). On pv_eta inner at threshold 0.05 it drops 1 of 115
   scored emits — a no-op + 150 ms overhead drag. Head-to-head
   validator-on-baseline vs bare pv_eta, pooled n=64: 24/64 = 37.5 %
   (Wilson-lo 0.267). **Loses to pv_eta by ~12-13 pp.**
2. **The Booster has real information** (val_acc 0.83, Brier 0.119,
   recall@0.30 0.90). The model learned. We applied it wrong — as a
   post-hoc gate instead of a chooser input.
3. **pv_eta is the strongest agent we have empirically** (peak μ=1154.8
   historical). Per PI direction, all subsequent mechanisms wrap /
   augment / replace components of pv_eta.
4. **Rule 37 axis cap reached on per-shot filter family.** Thresholds
   0.30 / 0.10 / 0.05 swept across two corpora (100 ms, 1000 ms);
   wrapping bare baseline vs wrapping pv_eta — all variants fail to
   improve on pv_eta. Family closed.

### NOT yet known (open against the next session)

- Whether the Booster's information transfers when surfaced inside the
  chooser's leaf-value (Reframe A below). The val_acc-0.83 signal
  must have *some* use; the right insertion point is unproven.
- Whether per-target supervision (Reframe B) makes the model strategically
  competent rather than per-shot-aware.
- Whether opponent-emit prediction (Reframe C) is the missing piece
  pv_eta's chooser is blind to.

## Next-session menu — three reframes, A first

> Full reasoning + open questions in
> `knowledge-base/thoughts/2026-05-29-phase-2-v2-validator-falsified-pv_eta-foundation.md`.

### Reframe A — Booster P(success) as a chooser input, not a filter (PRIORITY)

Don't filter pv_eta's emits. Expose the existing Booster's per-shot
P(success) inside `score_candidate_v4` /
`score_candidate_v4_joint` as an additive term:

```
candidate_score = ship_delta + production_term + γ * gamma_discount + λ * ML_P_success
```

Re-uses the booster already on disk
(`data/shot_validator/validator_booster.txt`, 45-d 1000ms corpus, 111
trees). No retraining. Implementation surface:

- `agents/baseline/chooser_trajectory.py` — `score_candidate_v4`,
  `score_candidate_v4_joint`. Accept `ml_score` arg, add `λ * ml_score`
  to the final Δ.
- New thin wrapper agent `agents/baseline_pv_eta_ml/main.py` (or env-
  var-gated extension of `baseline_validated`) that wraps pv_eta and
  threads ML scores through.
- λ sweep: 0.05 / 0.10 / 0.20 / 0.50. Heuristic: ML term magnitude
  ≈ 10-30 % of ship_delta magnitude.

A/B target: focal = ML-augmented pv_eta, opponent = bare pv_eta.
Gate: Wilson-lo ≥ 0.50 at n ≥ 32 (Rule 45).
Cost: ~1 day. Lightest path; tests the "signal not gate" hypothesis.

**Decision rule:** if A clears, ship it. If A fails, move to B with
prior "per-shot information doesn't transfer through chooser surface."

### Reframe B — Per-target value head (medium swing, AFTER A)

Change the supervision unit from per-shot binary to per-target
continuous: "how many future ship-deltas does owning planet T at
T+k earn the focal seat?" Continuous regression label. Plugs into the
chooser's leaf-value slot, augmenting `predict_garrison_at`.

Partial infrastructure already on disk: `data/value_head/`,
`data/value_head_distill/` from PM Phase A sessions.

Cost: ~3-5 days. Higher ceiling than A; longer path. Do AFTER A's verdict.

### Reframe C — Opponent-emit predictor (big swing, strategic)

pv_eta's chooser assumes a static opponent in `predict_garrison_at`.
Train an ML predictor: "given state, what will opp send next turn?"
Feed predicted opp emits into the chooser's lookahead. Restrict
prediction to top-K target planets per turn for wallclock safety.

Cost: ~5-7 days. Reserve for after A and B verdicts. This is the
"more strategic, less per-shot" angle PI raised at session end.

### Excluded — full chooser replacement / RL / MCTS

Considered and rejected for the remaining 25 days. Zero existing
RL/MCTS infrastructure on this branch; building from scratch is too
speculative at this point. The chooser is the hard-won foundation;
the leaf-value slot is the right insertion point for ML.

## Next-session first action (concrete)

1. `cat state/MULTI_BRANCH.md` to refresh live ladder state (Rule 44).
2. Read this file's "Day-N session 2026-05-29" section + the linked
   2026-05-29 thought.
3. Implement Reframe A — modify
   `agents/baseline/chooser_trajectory.py:score_candidate_v4` to take
   `ml_score: float = 0.0` and add `λ * ml_score` to the final Δ.
4. Build `agents/baseline_pv_eta_ml/main.py` (or env-var-gated extension
   of the existing wrapper) that wraps pv_eta and threads booster
   P(success) into the chooser.
5. Bundle smoke + parity (Rule 46).
6. A/B vs bare pv_eta at n=32 (Rule 45 gate). λ sweep 0.05 / 0.10 / 0.20
   / 0.50 if first λ fails to clear.

## Foundation lock — pv_eta

> PI direction 2026-05-29: "We were going to build really on our latest
> champion on the latest successful submission pv_eta."

**Inner agent for all future wrappers**: pv_eta logic (env var
`BASELINE_PV_ETA=1` activates; bundled in
`submissions/_imported/baseline_pv_eta.py`).
**A/B opponent for all future submit gates**: bare pv_eta (n ≥ 32,
Wilson-lo ≥ 0.50, Rule 45).
**Closed tracks not to be re-explored**:

- Per-shot filter at any threshold / re-rank / corpus — axis cap
  (Rule 37).
- Wrapping bare baseline as a deployment target — even +34 pp only
  reaches pv_eta-equivalent strength.

---

## Day-N PM competition-objective-alignment-hqNVM (2026-05-28)

**Session shape:** Phase A of the learned-value-head cycle. Goal was a
binary diagnostic — does the chooser-with-learned-head wiring work at
all, or was the previous failure (v1) doomed by architecture?
**Method:** distill a known-strong scalar value function (`favor_hybrid`,
the head behind the EVICTED team peak at μ=1149) into the 21889-param
MLP from 40 hand-crafted features, then A/B vs `favor_hybrid` itself.

### What landed (5 commits this branch line)

| Commit | Change |
|---|---|
| `9008010` | MVP learned value head infra + GPU training kernel |
| `0157bf0` | A/B variant maker + end-to-end cycle script |
| `132fa2b` | embed first trained value-head weights + canonical bundle |
| `26138b8` | Phase A distillation infra (favor_hybrid label mode) |
| `fb74d22` | Phase A distillation cycle — wiring verified |

Phase A artifacts:
- `data/value_head_distill/training.npz` — distillation corpus.
- `data/value_head_distill/value_head_weights.npz` — trained weights.
- `data/value_head_distill/training_history.json` — `val_rmse 48.4`
  vs `y_std ≈ 1029` ⇒ ~99.8 % variance explained.
- `submissions/baseline.py` — re-bundled with distilled head embedded.

### Load-bearing findings

1. **WIRING IS SOUND.** `baseline_learned` (chooser + distilled head)
   vs `baseline_hybrid` (chooser + `favor_hybrid`), `n=32` (harness
   auto-bumped from 16), `BASELINE_WALLCLOCK_MS=100`:
   **14/32 = 43.8 % wins, Wilson 95 % CI [0.282, 0.607]** — near-parity,
   formally INCONCLUSIVE. v1 was 2/32 = 6.2 % vs plain `favor` on the
   same harness.
2. **40-dim feature set is mostly sufficient.** Distillation R² is
   high (~99.8 %); no major feature-insufficiency diagnostic fired.
   The ~6 pp gap to 50 % parity is consistent with normal distillation
   loss — RMSE 48 is small absolute but ~10× the typical close-call
   action-Δ, so a small fraction of close decisions flip.
3. **Latency budget holds under the chooser hot path.** p50 = 164 ms,
   p95 = 240 ms, max = 459 ms per turn (env actTimeout 1000 ms).
4. **v1's failure was target + data, not wiring.** Margin-on-
   lite_greedy-self-play (the v1 setup) produced 43 % val variance
   explained; favor_hybrid distillation produced 99.8 %. The signal
   is just better when the teacher is competent.

### Falsified or weakened this session

- **"v1 architecture is broken."** Falsified — same architecture
  with a competent teacher near-parities the teacher. The proposer +
  chooser + 40-feature MLP substrate is fine **at the consume-the-head
  level**. PM 2026-05-28 diagnostics narrowed this: the chooser
  correctly CONSUMES whatever the head emits, but the head's signal
  quality vs the live ladder is poor (19 % vs `baseline_pv_eta`).
- **"Distillation will fall to 70-80 % R² because the features can't
  recover favor_hybrid."** Falsified — 99.8 % R² on scalar.
  **CAVEAT (PM 2026-05-28):** the 99.8 % R² is on a scalar target and
  does NOT preserve action-Δ rank order, which is what the chooser
  actually uses. R² is the wrong gate. See
  `knowledge-base/thoughts/2026-05-28-pm-distillation-action-rank-collapse.md`.
- **"40-feature insufficiency."** Previously listed as falsified.
  **RE-OPENED PM 2026-05-28.** The falsification rested on the same
  scalar R² that doesn't transfer to action-Δ. Feature sufficiency for
  rank-order preservation has not been tested. Verdict: unknown until
  a CRN-advantage head trained on direct action-Δ labels either
  succeeds or fails.

### NOT yet known (open against Phase B)

- **Live ladder calibration.** A/B was vs the EVICTED μ=1149 agent,
  not vs the current rolling pair (μ=806 / μ=829). We have not
  measured whether `baseline_learned` would beat the current floor.
  Rule 43 (multi-opponent panel) + Rule 45 (n≥32 vs rolling champion)
  must clear before any submission.
- **Does the distilled head add anything new?** A learned head that
  faithfully mimics a hand-coded head is an inference-cost regression
  with no upside. Phase A is a substrate test, not a candidate. The
  upside has to come from Phase B's richer training signal.

## Roadmap — Phase B and beyond

The learned-value-head program. Sequenced so each phase is its own
diagnostic; a phase only ships if its predecessor cleared.

### Phase B — richer training signal (next session, greenlit)

The Phase A result frees us to invest in the parts that actually
matter for headroom. Five changes, expected order of impact:

1. **Advantage head with Common Random Numbers (CRN).**
   - Train `A(s, a) = margin_action − margin_idle` against the SAME
     opp-model RNG seed for both legs (so the opp noise cancels in
     the difference).
   - Expected: 50–95 % variance reduction on the Δ signal that the
     chooser actually uses. The chooser doesn't care about V(s)
     accuracy — it cares about argmax_a (V(s,a) − V(s,a')).
   - Direct fix for the "RMSE 48 ≫ action-Δ" failure mode that caps
     Phase A at parity.
2. **Multi-horizon target (KataGo-style auxiliary heads).**
   - Outputs: final-margin, K-turn margin (K = 10), win-probability.
   - Weighted loss; auxiliaries regularise. Won't change rank order
     much on its own but stabilises training.
3. **Strong heterogeneous opponent pool.**
   - Pool: `composite_a2_hybrid` (μ=1149), `trajectory_v4_wait_N`
     (μ=1143), `hold_feasibility_solo` (μ=1135), `favor_hybrid`,
     `favor`. ~200 μ spread.
   - Why: v1's single-opponent (`lite_greedy`) self-play gave the
     head no signal for what beats a competent opponent. The Phase A
     fix was a better TARGET (favor_hybrid scalar); the Phase B fix
     is better DATA (decisions that matter against strong opp).
4. **Geometry-archetype-stratified self-play generation.**
   - Use the 32-archetype taxonomy already defined in
     `data/seed_panel_128.json` (audit/2026-05-18-seed-panel.md) —
     32 archetypes × 4 seeds = 128 reference geometries.
   - Generate the training corpus stratified by archetype: M games
     per archetype × 32 archetypes, rather than M total games
     sampled from whatever distribution the default seed generator
     happens to land in.
   - Optionally inverse-frequency-weight the loss so rare archetypes
     (3-planet sparse, comet-heavy, tight-orbit clusters) get
     proportional gradient — otherwise the head fits the modal
     archetype and miscalibrates at the edges.
   - Compositional with step 3: each (opponent × archetype) cell
     gets ≥ ceil(M/32) games. With 5 opponents × 32 archetypes that
     is 160 cells; a 25 600-game corpus is 160 games/cell.
   - Why: the same logic that justified the 128-seed eval panel
     applies to training data. The chooser is asked to handle wildly
     different geometries; a head trained on a non-stratified
     distribution will be miscalibrated on archetype edges that
     happen to come up on the live ladder.
   - Diagnostic to add: per-archetype val loss breakdown. Catches
     "all loss is from one archetype" failure modes before they
     reach the A/B.
5. **Kaggle GPU training.**
   - Local 5-fold > 1 h on this corpus size ⇒ GPU per Rule 13.
   - Use existing kernel template (`machine_shape: GpuT4x2`, Rule 30).
   - Two-tier smoke before production push (Rule 2 GPU clause):
     (i) local CPU single-state with JIT compile + memory recorded,
     (ii) small-scale GPU ≤4 games × ≤50 turns inside 10 min.

### Plan refinements from 2026-05-28 PM lens-critique pass

Three-lens review (mathematician / senior-ML-engineer / sim-game)
against the 5-step roadmap above. The ladder shape is unchanged;
five concrete modifications:

1. **B-1 explicitly reuses Phase A's data distribution.** "CRN
   advantage only" means: same games / opp as Phase A
   (`favor_hybrid` self-play, single opp), only the LABEL changes
   to a CRN-paired advantage `A(s, a) = margin_action − margin_idle`
   with the same opp-RNG seed on both legs. Each Phase A game already
   yields many (s, idle, a) triples; cost is hours not days. Earns
   or kills the line cheaply before any strong-opp / archetype
   corpus is built. Train/val seed-disjointness mandatory to avoid
   val leakage.
2. **Pre-B-3 compute-budget gate (back-of-envelope BEFORE the loop).**
   Before generating the strong-opp + archetype corpus, compute
   `games × turns × per-turn-ms × N_opps × N_archetypes`. If projected
   wallclock exceeds the planned compute window (Rule 2 1h CPU cap →
   Kaggle GPU per Rule 13), revise corpus shape (subsample turns,
   cache rollouts to harvest many (s, a) labels per game) BEFORE
   writing the data-gen loop, not after.
3. **Player-count branching is a roadmap decision, not an
   afterthought.** Team peak μ=1149.2 (sub 52744856) was
   `composite_a2_hybrid` — a 2P/4P branched architecture. The one-head
   Phase B plan must explicitly pick: (a) train two heads (proven by
   team peak, doubles param count), or (b) one head with
   `player_count` as a 41st feature AND ensured 4P coverage in the
   corpus. Decide before B-3 data-gen.
4. **Latency engineering lands with B-1, not deferred.** Phase A's
   p50 = 164 ms already exceeds the 100 ms chooser wallclock budget.
   Per-candidate MLP calls don't scale; the right shape is a single
   batched MLP forward over the chooser's full candidate set per
   turn. Deliver alongside B-1.
5. **Falsification clause needs a chooser-ceiling escape.** Current
   doc: "if Phase B underperforms, blame data/target, not features."
   Add a third candidate: *or chooser is the ceiling*. Concrete probe
   at the end of any failing phase — swap `favor_hybrid` back in as
   the value function while holding proposer + chooser + bundle
   constant; if that doesn't beat the learned head either, the head
   isn't the bottleneck and the head-headroom line should be paused.

**Comparison-baseline decision (2026-05-28 PI):** keep chooser
unchanged (no PV_ETA layered) during the B-1 / B-2 diagnostic phases,
so Δ-attribution stays clean. PV_ETA adoption re-opens only at the
submission gate, after the head itself has been A/B-validated in
isolation against `favor_hybrid`.

### Live-ladder state correction (2026-05-28 PM)

The "Where we are" section above is from 2026-05-20 and is **stale**.
Refreshed snapshot for context (do not edit history — use this for
Phase B baseline-choice only):

| Sub ID | Date (UTC) | Agent | μ | Role |
|---|---|---|---:|---|
| 53111837 | 2026-05-28 09:42 | `baseline_pv_eta` (sibling) | **1154.8** | rolling pair (top) |
| 53099429 | 2026-05-28 00:13 | peak-restore (orbital safety) | 1114.5 | rolling pair (bottom) |

Floor-at-risk flag is **FALSE** (was TRUE in the 5/20 snapshot — the
five-day intervening work on sibling branches recovered the floor).
Phase B's submission bar is now **~μ=1155**, not the evicted ~μ=1149
referenced in the older Phase A debrief. The diagnostic A/B target
(`favor_hybrid`) is still the right comparison for B-1 / B-2; the
submission gate must be re-checked against the live rolling top.

**Phase B decision rule.** Each addition is gated by an A/B vs
favor_hybrid at `n ≥ 32` with `BASELINE_WALLCLOCK_MS=100`:
- B-1 (CRN advantage only): need Wilson-lo ≥ 0.50 (parity-or-better).
  If we don't beat parity here, decompose CRN failure before piling on.
- B-2 (+ multi-horizon): need ≥ B-1 with delta within noise (Wilson
  CIs overlap) OR clearer lift.
- B-3 (+ strong opp pool): need ≥ B-2 with Wilson-lo ≥ 0.50.
- B-4 (+ archetype-stratified gen): the candidate move. Need
  Wilson-lo ≥ 0.55 vs favor_hybrid AND Wilson-lo ≥ 0.50 vs the
  current rolling-pair champion (Rule 43 + Rule 45). Plus the
  per-archetype A/B from the seed panel
  (`--vs-panel --by-archetype`) showing no archetype regresses
  > 10 pp vs the B-3 baseline — catches "we lifted the average by
  tanking one archetype" failure modes.

### Post-diagnostic re-refinement (2026-05-28 PM)

Three PM diagnostics overturned key premises of the morning roadmap.
Full write-up at
`knowledge-base/thoughts/2026-05-28-pm-distillation-action-rank-collapse.md`.
Headline:

| Test | Result |
|---|---:|
| weights-load sanity (forward batch on training corpus) | R² = 0.994 — load is clean |
| `baseline_learned` vs `baseline_favor` n=32 | **9/32 = 28 %** [.16, .45] FAIL |
| `baseline_hybrid` vs `baseline_favor` n=32 | 15/32 = 47 % [.31, .64] near-parity |
| `baseline_learned` vs `baseline_pv_eta` (live μ=1154.8) n=32 | **6/32 = 19 %** [.09, .35] FAIL |

Two surviving hypotheses (no weights bug):
1. Scalar R² does not preserve action-Δ rank order. (The chooser uses
   `argmax_a`, which depends on rank order, not scalar fit.)
2. `favor_hybrid` is not meaningfully stronger than `favor` in 2P
   self-play — i.e. Phase A's "teacher" was at parity with the older
   head it was supposed to improve on, so the distillation target
   was the wrong policy to mimic in the first place.

**Plan changes to the morning's roadmap** (refinement #1 from the
lens-critique pass is invalidated):

- **B-1 merges with B-3.** No version of B-1 that uses `favor_hybrid`
  as the rollout opp policy makes sense after diagnostic 3. Simplest
  merged shape: CRN-paired advantage labels generated against
  `baseline_pv_eta` (single strong opp at live μ=1154.8). Full
  Phase B-3 strong-opp pool defers to a follow-on (B-3-prime).
- **Diagnostic A/B target changes from `favor_hybrid` to
  `baseline_pv_eta`.** Wilson-lo ≥ 0.50 required to declare lift.
- **Training-time gate added.** Spearman-τ on action-Δ rank order on
  a held-out set must be meaningfully > 0 before bundling. Phase A
  would have failed this gate.
- **Cost re-estimate.** "Hours not days" is wrong. Realistic shape is
  500-game overnight local (8 workers) → 1500-game Kaggle GPU follow-on.
  Back-of-envelope in
  `knowledge-base/concepts/crn-advantage-datagen-sketch.md`.
- **Option B escape clause added.** If a CRN-advantage head trained
  against `baseline_pv_eta` still loses to `pv_eta` at < 40 % on n=32,
  the program pauses and re-scopes (structural rethink: direct policy
  distillation / search-based chooser / feature expansion / different
  architecture).

**Data-gen rewrite sketch landed at
`knowledge-base/concepts/crn-advantage-datagen-sketch.md`** — full
pseudocode, locked-opp-RNG mechanics, compute budget, "known
unknowns to verify before code" list, sequenced rollout plan.

### Post-diagnostic re-refinement #2 (2026-05-28 PM2 — feature-sufficiency probe)

A second probe ran on the existing Phase A corpus (no fresh rollouts; ≈10 min
compute) plus a public-notebook scan and literature search. Full write-up at
`knowledge-base/thoughts/2026-05-28-pm2-feature-sufficiency-probe.md`.

Three stages, all on the existing 10k-example artifact:

| Stage | Diagnostic | Result |
|---|---|---|
| 1 | Random-pair P(rank-agree) vs \|Δy\| | Chance (~0.52–0.57) below \|Δy\|=10; 0.87 only above \|Δy\|=50 |
| 2 | Train RankNet on SAME 40 features | +2–5 pp in close-pair buckets vs MSE — loss is NOT the bottleneck |
| 3 | Permutation importance on embedded head | **5 of 40 features** carry +2.47 ΔR²; remaining 35 each ΔR² < 0.003 |

**Mechanistic finding.** `favor_hybrid` is essentially `delta_us_minus_them` + small inflight correction. The distilled head correctly learned ship-delta prediction and ignored the rest. The chooser argmax then can't distinguish candidates whose ship totals are conserved (i.e. all legal sibling emits) — formal mechanism behind 28 % h2h vs `baseline_favor`.

**Both axes of Phase B-1 must change, not just the target:**

- *Target:* scalar favor_hybrid → CRN-paired advantage (already planned).
- *Features:* pooled state-features → per-candidate features (NEW; was assumed sufficient on the basis of Phase A's R²=0.994, which Stage 3 now reveals was illusory).

**Kaggle public precedent (≥70 votes) supports two value-head archetypes that have moved THIS competition's LB:**

| Archetype | Author | Features | Result |
|---|---|---|---|
| GBC value head used INSIDE 1-ply search | AidenSong123 | 16 global STATE features | AUC 0.976, LB ≥ 1000 |
| **MLP shot-validator FILTER on rule-base proposals** | konbu17 | **24 features PER SHOT** | **+19 pp vs rule-base alone, +43 pp vs tier4** |
| PPO with per-candidate encoder | kashiwaba | self / candidate / global groups | educational |

Neither uses "global features → direct argmax" (which is what Phase A built). The architectures that work either use search to compare scalars across different terminal states, or use per-shot features so candidates differ by construction.

### Phase B-1 amended sequencing (supersedes the morning roadmap on this point)

**Direction 1 — Per-emit MLP filter (konbu17 architecture, evidence-backed, cheapest):**

1. Inventory existing infrastructure for "labelled emits" (self-play replays with per-emit outcome).
2. Define per-emit 24-feature extractor (src/tgt planet stats, owner one-hot, ETA, fleet-speed, in-flight counts, turn, totals).
3. Train binary "good emit" MLP filter, embed in bundle, A/B vs production stack n=32.
4. Decision gate: Wilson-lo ≥ 0.50 vs production stack at n=32 → continue to Direction 2; else iterate filter design.

**Direction 2 — Per-candidate score head + CRN-paired advantage (only if Direction 1 cleared):**

5. CRN-paired advantage rollouts (already designed in `crn-advantage-datagen-sketch.md`) BUT now feeding per-candidate features, not pooled-state features.
6. Per-candidate score head replaces argmax over pooled-state-score.
7. Spearman-τ training-time gate.
8. A/B vs `baseline_pv_eta` n=32.

### Falsified or weakened by PM2

- **"Feature-sufficient" claim is settled-DEAD for pooled 40-feature setup.** Stage 3 permutation importance shows 35/40 are unused. Do not iterate on pooled features.
- **"Phase A's 99.8% R² implies features carry rank info."** Falsified by Stage 1+2; the R² is a property of a degenerate target.
- **"Rank-aware loss alone (RankNet/CRN) will save the existing feature set."** Falsified by Stage 2; rank-aware loss on same features = +5 pp at most.

### Post-diagnostic re-refinement #3 (2026-05-28 PM3 — H14 recipe locked from konbu17 + cross-comp synthesis)

Full write-up at `knowledge-base/thoughts/2026-05-28-pm3-h14-recipe-locked-from-konbu17.md`.

PM2 pointed at konbu17-style filter as Direction 1. PM3 did the depth research on (a) konbu17's actual notebook code and (b) cross-comp simulation winners (Halite III/IV ttvand/teccles/0Zeta/mlomb, Lux S1 Toad Brigade, Planet Wars 2010 oddshrimp/zvold, Hungry Geese GeeseZero). Three findings re-shape the next session:

1. **The 24-d feature schema in `data/shot_validator/schema.json` matches konbu17 verbatim.** This branch already has H14 scaffolded (label pipeline + schema + README); only training + embed + A/B was deferred. ~70% of the build is in-place.

2. **Cross-comp unanimous architecture pattern.** Every Halite/Planet Wars winner that had a "pick from N candidates" pattern used per-candidate features (closed-form or learned classifier), never pooled-state scalar. The two Orbit Wars LB ≥ 1000 learned-component agents either:
   - Embed per-candidate features directly (konbu17 — 24-d per-shot, BCE filter, +19 pp)
   - Use per-state features inside a 1-ply minimax over forward-simulated terminal states (AidenSong — GBC over 16-d state, sim handles the sibling-distinction problem)
   Direct-argmax over pooled-state-scalar (what Phase A built) never appears as a winning architecture.

3. **konbu17 recipe is now mechanically pinned, not stylistic.**
   - Arch: 3-MLP ensemble (seeds [42, 100, 7]) of `Linear(24,64)→ReLU→Linear(64,32)→ReLU→Linear(32,1)`, BCEWithLogitsLoss, Adam lr=1e-3, 40 epochs, batch 512.
   - **Pos_rate calibration is THE load-bearing decision.** Exclude self-reinforcement shots ENTIRELY (target already owned by side) — drops pos_rate from ~0.96 to ~0.71. Then mix opponent strengths to keep filtered pos_rate in 0.50-0.75. Without this, BCE `pos_weight = (1-p)/p` collapses, validator outputs uniform low, threshold rejects everything, topk1 sends nothing.
   - Game-level 80/20 split (NOT row-level — shots within a game correlate, leaks 15-20 pp val acc).
   - Threshold 0.30 with ensemble (0.40 single-model). topk1 (keep largest-ship survivor per turn).
   - Strict-improvement FILTER, not re-ranker. Cannot regress vs proposer alone.

### Sequencing for next session

1. **Smoke the label pipeline.** 4-8 wallclock-capped self-play games → `audit/external/replays/`, then `python -m scripts.label_shot_outcomes`. Sanity-check filtered pos_rate ≈ 0.5-0.8. (Previous smoke run timed out at 180s with full-wallclock baseline self-play — needs `BASELINE_WALLCLOCK_MS=100` and worker parallelism.)
2. **Generate corpus.** ~140 games with adapted konbu17 mix:
   - vs `baseline_favor` (weak), vs `baseline_full` (moderate), vs `baseline_joint_aggr_consolidated_orbitfix` (strong), vs `baseline_pv_eta` (live champ at μ=1154.8), self-vs-self.
   - Tune mix to keep filtered pos_rate ≈ 0.6.
   - Compute: ~30 min on 8 workers with WALLCLOCK_MS=100.
3. **Train 3-model ensemble.** Seconds on CPU (3.5k params, 20k examples, 40 epochs).
4. **Embed via `scripts/bundle_agent.py`** (base64 npz pattern, no numpy/torch at submit time — konbu17 uses pure-Python forward with numpy `@` + manual sigmoid).
5. **Bundle parity + Rule 46 smoke** (`pytest tests/test_bundle.py` + `python fast.py play <bundle>`).
6. **A/B vs production** (`baseline` no validator vs `baseline` w/ validator) at n=32. Wilson-lo ≥ 0.50.
7. **A/B vs `baseline_pv_eta`** at n=32 if step 6 cleared. Wilson-lo ≥ 0.50 to gate submission.
8. **Push (Rules 1, 42, 43, 46, 47 checklist)** — coordinate with sibling-branch push board.

### Cost re-estimate

Original H14 estimate in `top-performer-strategies.md`: "high EV, ML, ~15 days". With (a) existing scaffold (label pipeline + schema + README in `data/shot_validator/`), (b) mechanically-pinned konbu17 recipe, and (c) per-candidate architecture validated by 6 cross-comp winners — **realistic cost is ~1 session** (overnight corpus-gen + morning train/embed/A/B). The "15 days" predates both the scaffold and the recipe being pinned.

### Decision rule for next session

Same gates as PM2 Direction 1, sharpened:
- Step 6 clears (Wilson-lo ≥ 0.50 vs production) → proceed to step 7.
- Step 7 clears (Wilson-lo ≥ 0.50 vs `baseline_pv_eta`) → Rule 42-43 pre-submit checklist → push.
- Either fails → diagnose pos_rate calibration first (most common konbu17 failure mode), then opponent mix, then threshold.

### Falsified or weakened by PM3

- **"Direction 2 (per-candidate score head + CRN-paired advantage labels) is the right longer-horizon investment."** Now ambiguous. Cross-comp evidence (TheDuck314's per-action collision classifier in Halite III with "basically no impact on mu", plus 0/5 successful pure-learned RL/IL in Orbit Wars from konbu17's postmortem) suggests per-candidate score head is not a guaranteed step up from per-candidate filter. Re-evaluate only after Direction 1 ships.
- **The "15-day" sizing estimate on H14.** With scaffold + locked recipe, ~1 session.

### PM4 — H14 MVP executed end-to-end (2026-05-28 evening)

Full write-up: `knowledge-base/thoughts/2026-05-28-pm4-validator-mvp-results-and-roadmap.md`. Commits this session: `9d20d19` (infrastructure), `e72ea9e` (embedded weights).

**What landed:**

| File | Purpose |
|---|---|
| `lib/shot_features.py` | 24-d per-shot feature encoder (matches `data/shot_validator/schema.json` verbatim) |
| `scripts/gen_validator_corpus.py` | End-to-end mixed-opp corpus generator; labels both seats inline; self-reinforce-filtered |
| `scripts/train_validator.py` | 3-MLP ensemble training with BCE + pos_weight calibration; aborts if pos_rate out-of-band |
| `agents/baseline_validated/main.py` | Filter wrapper; weights embedded as base64; no topk1 (preserves multi-source coord) |
| `scripts/embed_validator_weights.py` | Patches base64-npz into the wrapper |
| `tests/test_validator_smoke.py` | 6 smoke tests, all green |
| `scripts/bundle_agent.py` | Added `shot_features` to DEFAULT_LIB_ORDER |

**MVP results:**

- Corpus: 30 games × 3 opp pairings = **5,366 labeled shots**, pos_rate **0.529** (in healthy [0.40, 0.85])
- Per-pair pos_rate: baseline-self 0.502 / vs baseline_full 0.459 / vs v3_snipe 0.680
- Training: 3-MLP ensemble (seeds 42, 100, 7), val_acc **0.777**, Brier 0.147; precision 0.72 / recall 0.89 at threshold 0.30
- A/B vs `agents/baseline` at n=64: **39/64 = 60.9 %**, Wilson 95 % CI [**0.487**, **0.719**]
- Tier breakdown: t1 n=32 was 50.0 %, t2 n=32 was 71.9 % — the lift is real but variance is high at this n
- Verdict: **INCONCLUSIVE by Wilson-lo ≥ 0.50 gate** (lo = 0.487, 0.013 short) but **+10.9 pp directional lift**
- Latency: focal p50 = 185 ms, p95 = 282 ms, max = 692 ms (env actTimeout 1000 ms; comfortable)

**Pulled from sibling branch this session** (`origin/claude/kaggle-submission-review-gZsCu`, `submissions/_imported/`, gitignored under `submissions/*` — move to `submissions/imported/` and update `.gitignore` to track in Phase 2):

- `baseline_pv_eta.py` — **μ=1154.8 LIVE CHAMPION** (top of rolling pair)
- `baseline_leaf_pv_2p.py` — μ=1105.4 (just submitted; bottom of rolling pair)
- `baseline_peak_1165_anchor.py` — peak μ=1149 reference

### PM5 — Phase 2 Stage 2 (F2 only) executed; PI redirected next session to kitchen-sink + GBT + 17-opp mix (2026-05-28 night)

Commits this session: `5c5cfc9` (F2 + bump FEATURE_DIM 24→25), `5199c78` (re-embed 25-d weights + tighten no-weights smoke test).

**What landed in PM5:**

| Change | File |
|---|---|
| F2 `combat_margin_at_arrival` (production-walk approx; ignores in-flight defenders) | `lib/shot_features.py` |
| `FEATURE_DIM` 24 → 25 | `lib/shot_features.py`, `scripts/train_validator.py` |
| Schema bumped to v2 + name appended | `data/shot_validator/schema.json` |
| Embedder reads `FEATURE_DIM` dynamically | `scripts/embed_validator_weights.py` |
| Smoke test fixed: now actually blanks `_WEIGHTS_B64` to exercise no-weights path (was previously only resetting in-memory cache, which the embedded blob immediately re-populated) | `tests/test_validator_smoke.py` |
| Corpus regen on same 3-opp mix as MVP (5370 shots, pos_rate 0.512) | `data/shot_validator/labels.jsonl` |
| Retrained 3-MLP ensemble | `data/shot_validator/validator_ensemble_weights.npz` |

**Stage 2 training results (single-feature add):**

| Metric | MVP (24-d) | Stage 2 (25-d + F2) |
|---|---|---|
| val_acc @ 0.5 | 0.777 | **0.816** (+3.9 pp) |
| Brier | 0.147 | **0.121** (better calibration) |
| Precision @ thr=0.30 | 0.72 | 0.671 |
| Recall @ thr=0.30 | 0.89 | **0.947** (+5.7 pp) |

**A/Bs (`BASELINE_WALLCLOCK_MS=100`):**

| Opponent | n | Win rate | Wilson lo | Wilson hi |
|---|---|---|---|---|
| `agents/baseline` | 10 | 40 % | 0.168 | 0.687 |
| `v7_0` | 10 | 60 % | 0.313 | 0.832 |
| `v4_planner` | 10 | 80 % | 0.490 | 0.943 |
| `v3.5.1` | 10 | 90 % | 0.596 | 0.982 |
| **`baseline_pv_eta`** (live μ=1154.8) | **32** | **53.1 %** | **0.364** | 0.691 |

- Validator clears 3 of 4 panel opps but **loses vs unwrapped `baseline` at n=10** — small-n noise vs real regression unclear at this n (Rule 45: n=10 is sub-triage).
- vs live champion at n=32: 53 % is directionally above parity but **Wilson-lo 0.364 << 0.50** (Rule 43 submission gate). Not submittable as-is.
- Latency: focal p50 ≈ 190-230 ms, p95 ≈ 280-370 ms vs the 1000 ms env cap → **~70 % of per-turn budget unused**. This is the headroom PI flagged.

### Next session — Phase 2 v2 (PI-directed in PM5 conversation)

Three changes from PM4's original Phase 2 spec (which is still below for substrate reference but superseded as a plan):

1. **Switch model from MLP ensemble to gradient boosted trees (GBT).** A 7.5k-param MLP on 5k labeled shots is under-capacity for the planned ~40-d input. konbu17's reference notebook itself used LightGBM; we'd been on MLPs only because base64-npz embedding was operationally trivial. Decision: switch to GBT (LightGBM or XGBoost — choose during scoping based on inference-side packaging cost). Tabular features at 5-30k rows are GBT's home territory; the existing val_acc ceiling at 0.816 likely reflects model capacity, not feature insufficiency alone.
2. **Kitchen-sink the features (Rule 20).** Add all Tier 1 (F6, F3, F10, F8, F4 — 5 features, ~+9 dims net) AND all Tier 2 (F11, F7, F13, F9 — 4 features, ~+5 dims) from PM4's deep-dive. Target ~39 d input. Defer Tier 3 (F5/F12/F14/comet/mission-tag) only because the last two need chooser-side instrumentation. All Tier 1+2 substrate lives in `lib/world_model.py` / `lib/trajectory.py` / `lib/scoring.py` and is parity-tested; encoder calls are 3-5 lines each.
3. **17-opp training mix across 4 tiers** (Axis B from PM4 — spec'd then but PM5 skipped to F2-only). 5 games per cell × 17 cells = 85 games, ~16-20k labeled shots (~3-4× PM5 corpus). Cells:
   - Weak (3): `agents/simple`, `agents/geo`, `agents/v1_orbitfix`
   - Moderate (3): `agents/analytical`, `agents/v3.5.1`, `agents/v3_snipe`
   - Strong (7): `agents/baseline` (self-play), `agents/baseline_full`, `agents/baseline_joint_aggr_consolidated_orbitfix`, `submissions/baseline_hybrid.py`, `submissions/baseline_favor.py`, `submissions/baseline_learned.py`, `submissions/v7_minimax.py`
   - Live (3): `submissions/_imported/baseline_pv_eta.py`, `submissions/_imported/baseline_leaf_pv_2p.py`, `submissions/_imported/baseline_peak_1165_anchor.py`

**Sequencing for next session (~5-7 h projected):**

| Stage | Time | Output | Gate |
|---|---|---|---|
| 1. Implement Tier 1+2 features (9 features, +14d) | ~2 h | `lib/shot_features.py` v3 (~39 d), schema v3, smoke updated | All smoke tests green; one single-game trace through `lib.trajectory.predict_fleet_fate` confirms sun / OOB / comet-expiry waste <2% (Rule 47) |
| 2. GBT inference-side scaffolding | ~1 h | tree predictor in `agents/baseline_validated/main.py` (LightGBM `model_to_string` + pure-python eval OR XGBoost dump + numpy walker) | Inference parity with the trained sklearn-side model (decision-equivalence on ≥100 random rows) |
| 3. Expanded corpus gen (17 cells × 5 games) | ~20-30 min | regen `data/shot_validator/labels.jsonl` | Per-cell pos_rate in [0.40, 0.85] — rebalance if weak opps inflate it |
| 4. Train GBT | ~5-10 min | model artifact | val_acc ≥ 0.85 (vs current MLP 0.816) |
| 5. Embed + smoke + bundle parity | ~30 min | `agents/baseline_validated/main.py` with new model embedded | `pytest tests/test_validator_smoke.py` GREEN + `fast.py play submissions/baseline_validated.py` clean (Rule 46) |
| 6. A/B vs `agents/baseline` n=32 | ~25 min | adaptive eval | Wilson-lo ≥ 0.50 |
| 7. A/B vs `baseline_pv_eta` n=32 | ~25 min | only if step 6 cleared | Wilson-lo ≥ 0.50 (Rule 43) |
| 8. Pre-submit gates (Rules 42/43/45/46) | ~30 min | submittable bundle | PI sign-off (Rule 1) |

**Risks:**

- **GBT inference-side packaging.** LightGBM `model_to_string` + pure-python predictor is ~300 LOC; XGBoost `dump_model` is similar. The Kaggle single-file constraint matters — decide early between (a) raw text-format model embedded + custom tree walker, (b) base64-pickle + `lightgbm.Booster.predict` (needs lightgbm in submission env — check Kaggle env), (c) `m2cgen` or similar transpilation. Smoke-test packaging on PM5's existing 25-d corpus before re-corpus-genning.
- **17-opp pos_rate rebalancing.** Weak opps (simple/geo) likely inflate pos_rate above 0.85; trainer aborts (`gen_validator_corpus.py` checks). If so, drop weak count or oversample mid/strong.
- **Tier 2 feature substrate verification.** F11/F7/F13/F9 substrate exists in `lib/world_model.py:333-480` per PM4 audit but hasn't been exercised from `shot_features` yet. Rule 47: do a single-game trace before committing to full training run.
- **Bundle wrapper-style packaging (carried from PM4).** Still unsolved. `scripts/bundle_agent.py`'s `_INTRA_IMPORT_RE` strips `from agents.baseline.main import agent as _inner_agent` and topo-sort doesn't cross agent packages. Either write `scripts/bundle_validator.py` (one-off wrapper bundler, ~50 LOC outlined in `/root/.claude/plans/go-with-phase-2-snappy-frost.md`), or generalise `bundle_agent.py` with a `--wrap` flag. Required before Stage 8 submission.

### Phase 2 — feature expansion + opponent diversity (NEXT SESSION)

**Axis A — feature expansion (24 d → ~30 d).** Deep-dive research (PM4 evening) found that every high-EV per-shot feature from Halite IV / Planet Wars 2010 / Lux winners has a substrate function ALREADY IMPLEMENTED in our `lib/`. The encoder just doesn't call them. Top 6 to add (priority-ranked):

| # | Feature | Formula → existing primitive | Why |
|---|---|---|---|
| **F2** | `combat_margin_at_arrival` (1 d) | `(ships_sent − predicted_defenders) / max(1, predicted_defenders)`, clipped [-1, +1], via `lib/world_model.predict_garrison_at(tgt, eta, ledger[tgt.id]).ships` | The single number "did we send enough to beat predicted defenders" — literally the binary label, made explicit. Highest expected lift. |
| **F6** | `path_fate_one_hot` (4 d) | `lib/trajectory.predict_fleet_fate(src, tgt, angle, ships, world, max_steps=eta+5)`; one-hot over {target, planet-collide, sun, oob} | Encodes H44 finding (65 % fleet-destroyed-in-flight). audit/2026-05-21-h44 directly maps to this feature. |
| **F3** | `owner_at_arrival_one_hot` (3 d) | `predict_garrison_at(...).owner`, replace current launch-time owner one-hot | Roman/Pilkwang/oddshrimp all gate on arrival-time owner, not launch-time. Free swap — same slot count, more predictive. |
| **F10** | `same_target_friendly_inflight_{count, ships}` (2 d) | `[(eta, ships) for (eta, owner, ships) in ledger[tgt.id] if owner == focal_seat]` | Closes redundant-swarm failure mode. The pooled in-flight totals in v1 can't see "I already have 200 ships landing here." |
| **F8** | `src_safe_departure_ratio` + `shot_drains_safely` (2 d) | `safe_dep = src.ships + prod·enemy_eta − inbound − 1` using `WorldModel.incoming_enemy_eta(src.id, focal_seat)` + `ledger[src.id]` + `WAVE_LOOKAHEAD=12` from `lib/world_model.py:53` | Source-emptying discipline is the top-10 differentiator (mean garrison-at-launch 11 vs midpack 22). |
| **F4** | `pv_capture` (1 d) | `pv_horizon(step, eta, gamma=0.99, t_total=step + eta + expected_hold(tgt.id, eta, world)) × tgt.production` via `lib/scoring.py:89-140` | Late-game scoring asymmetry: γ=0.99 over `expected_hold`-truncated horizon penalises captures we'll lose quickly. Live μ=1064 anchor for un-truncated; truncated form is what HAV-1 was designed for. |

**Total: 24 d → ~30 d** (F3 replaces 3 d, the rest are net adds).

**Critical implementation note:** all 6 features are 3-5 line additions to `lib/shot_features.py`. The hard work (predict_garrison_at, predict_fleet_fate, time_to_enemy_threat, expected_hold) is already in lib/ and parity-tested. The feature encoder just needs to call these and normalise.

Tier 2 (slots 7-10 if budget permits): F11 `joint_arrival_count_at_eta` (±1-step same-owner stack count), F7 `intercept_enemy_eta` (earliest enemy arrival at tgt before our eta), F13 `target_growth_field_diff` (zvold's inverse-square electrostatic field), F9 `src_time_to_nearest_enemy_threat` + `src_is_frontier` (Roman ROTATION_LIMIT analogue).

Tier 3 (defer, exploratory): F5 `uncertainty_at_arrival` (0Zeta's drift × eta), F12 `target_indirect_wealth` (oddshrimp's neighbour-growth bonus), F14 `target_dominance_3nn` (3-NN ownership signal), `comet_remaining_lifetime`.

**Sources** (cited in PM4 doc):
- oddshrimp / melisgl 2010 Planet Wars winners — `safe_departure`, `pv_capture`, `indirect_wealth`
- zvold — `growth_field`
- 0Zeta Halite IV 4th — `uncertainty_at_arrival`
- Pilkwang / Roman / AidenSong Orbit Wars — `arrival-time owner/ships` via `simulate_planet_timeline`
- konbu17 — original 24-d state-at-launch schema

**Axis B — expanded opponent pool (3 → 18 opp cells, 90 games).** Full table in PM4 doc; high-level mix:
- **Weak** (3): `simple`, `geo`, `v1_orbitfix`
- **Moderate** (5): `analytical` (ANALYTICAL track), `v3.5.1`, `v3_lookahead`, `submissions/v4_planner.py`, `submissions/v7_0_drop_one.py`
- **Strong** (6): `baseline_full`, `baseline_joint_aggr_consolidated_orbitfix`, `submissions/{baseline_hybrid,baseline_favor,baseline_learned,v7_minimax}.py`
- **Live** (3): `baseline_pv_eta`, `baseline_leaf_pv_2p`, `baseline_peak_1165_anchor`
- **Self** (1): baseline self-play

5 games per cell × 18 cells = 90 games. Compute: ~15-20 min on 8 workers with `BASELINE_WALLCLOCK_MS=100`. Both seats labeled ≈ **16-20k labeled shots** (≈3-4× MVP).

**Axis C (defer):** 4P training data, synthetic emit augmentation, per-candidate score head.

### Phase 2 session sequencing

| Stage | Time | Output | Gate |
|---|---|---|---|
| 1. Feature expansion code | ~2-3 h | `lib/shot_features.py` v2, schema bump, wrapper update, tests | All tests green |
| 2. Expanded corpus gen | ~15-20 min | `data/shot_validator/labels_v2.jsonl` | pos_rate in [0.40, 0.85] |
| 3. Train 3-MLP ensemble | ~1 min | `validator_ensemble_weights_v2.npz` | val_acc ≥ 0.80 |
| 4. Threshold + topk sweep | ~30 min | held-out per-game eval × 5 cells | Pick best |
| 5. A/B vs `agents/baseline` | ~30-60 min | n=32 → n=64 adaptive | Wilson-lo ≥ 0.50 |
| 6. A/B vs `baseline_pv_eta` (live) | ~30-60 min | only if step 5 cleared | Wilson-lo ≥ 0.50 (Rule 43) |
| 7. Bundle + parity (Rule 46) | ~10 min | submittable bundle | clean imports + crash-free game |
| 8. Submission gate | depends | PI sign-off | Rule 1 + Rule 42 |

Total: ~5-7 hours. Plan for a single focused session; if Stage 1 (feature code) overruns, ship without Tier 2 features and revisit next session.

### Phase 2 decision branches

- **A/B vs baseline passes Wilson-lo ≥ 0.50:** proceed to live-champ A/B and submission preparation.
- **A/B vs baseline still inconclusive (Wilson-lo ∈ [0.40, 0.50]):** debug via `--debug-validator` instrumentation flag (log every veto + post-game outcome), eyeball categories, identify what the validator catches that baseline misses. Targeted feature fix.
- **A/B vs baseline regresses (Wilson-lo < 0.40):** something is wrong with the new features. Roll back to MVP and try a different Tier 1 subset.

### Phase 2 risks

- Feature expansion increases per-turn latency. Bench-verify with `scripts/bench_value_head_inference.py` pattern.
- Pos_rate spikes outside [0.40, 0.85] from the weak opps. Trainer aborts; adjust opp ratio.
- Bundle workflow remains unsolved for wrapper-style agents. MVP A/B used non-bundled path (`fast.py eval` loads main.py directly). For Kaggle submission, write a small wrapper bundler that base-bundles `agents/baseline` then appends validator wrapper code with weights inline (deferred; design TBD).

### Falsified / re-opened by PM4

- **"50 % at n=32 ⇒ filter does nothing."** Falsified by tier 2's 71.9 %. The MVP gives a +10.9 pp directional lift; the filter does meaningful work.
- **"konbu17's +19 pp will transfer to our stack."** Tempered. konbu17 worked vs a weaker rule-base; ours filters most bad shots already. The headroom for our filter is smaller; expect +5-15 pp post-Phase-2, not +19.
- **"PM2 / PM3 Direction 2 (per-candidate score head with CRN-advantage)" remains deferred.** Direction 1 isn't exhausted yet; Phase 2 carries it further before pivoting.

### Pre-submit checklist when Phase B clears

Apply in this exact order to avoid Rule 42 / 43 / 46 violations:

1. `kaggle competitions submissions orbit-wars | head -5` — read
   rolling-last-2 state.
2. `python scripts/bundle_agent.py agents/baseline` — bundle.
3. `pytest tests/test_bundle.py` + `python fast.py play <bundle>` —
   parity + crash-free game (Rule 46).
4. `fast.py eval <bundle> --vs-panel` — Wilson-lo ≥ 0.55 per opponent.
5. `fast.py eval <bundle> --vs <rolling_champion>` at n ≥ 32 —
   Wilson-lo ≥ 0.50 (Rule 45).
6. Append claim row to `state/MULTI_BRANCH.md` push board (Rule 42);
   verify evicted-μ < predicted candidate μ.
7. PI sign-off (Rule 1).

### Phase C — only if Phase B clears (speculative)

- **Population-based self-play.** Train multiple heads against a
  shifting opponent league (each Phase B agent enters the pool).
  Risk: many-day compute investment for a marginal lift; not
  scheduled until Phase B is on the ladder.
- **Search over the chooser's candidate set.** Replace the scalar
  ranking with a 1-ply beam search using the advantage head's
  variance estimate to prune. Touches the chooser, not just the
  head — higher integration risk.

### Falsified-or-dead so this isn't re-explored

- **Margin-on-lite_greedy-self-play as the value-head target.** v1
  result: 2/32 = 6 %. Target was too noisy AND the opp was too weak.
  Do NOT revisit this combination.
- **40-feature insufficiency.** Falsified by Phase A's 99.8 % R²
  distillation result. If Phase B underperforms, blame the data /
  target, not the feature pipeline. Expanding feature count is NOT
  the move.

---

## Day-N PM extract-physics-trajectory-Vjaz9 (2026-05-22)

**Session shape:** surgical, additive extraction of physics substrate
from the sibling Phase η branch (`claude/strategy-axis-decision-3437`).
No strategy/agent code copied; no experiments; no submissions.

**What landed (sole commit `72fe45a`):**

- `lib/kinematic_table.py` (NEW, 436 lines) — per-turn precompute of
  planet positions (static / orbital / comet). Bit-identical to
  `predict_relative` by construction. Singleton + fingerprint rebuild.
- `lib/orbit.py` (+37) — `predict_relative_cached(planet, ω, lead, *,
  table=None)` lookup wrapper; falls through on any miss.
- `lib/trajectory.py` (+47) — gated behind `KINEMATIC_TABLE_ENABLED=1`
  env var. When primed AND the table covers the needed window, one
  `table.window()` replaces the per-step inline build. Default OFF;
  existing call sites unchanged.
- `tests/test_kinematic_table_parity.py` (NEW, 621 lines) — `==`
  parity pins (no tolerance) for every cache type.

**Deliberately skipped:** `lib/joint_solver/trajectory_matrix.py`
(Phase η.1 opening matrix — couples to `agents.baseline.proposer`,
not pure physics) and the full-game parity test (imports specific
agents). Strategy / chooser / pipeline / missions / value heads from
the sibling branch all left where they are.

**Verification:** 39/39 unit tests green; 80/80 in the wider
geometry+orbital-safety+proposer+snipe sweep; end-to-end parity smoke
on a 2-planet world identical cold vs primed.

**Next-session first action:** build a fresh agent on top of this
substrate. Opt-in protocol + usage example in
`audit/2026-05-22-extract-physics-trajectory.md`. Default-OFF means
no existing agent regressed by this commit.

---

## Day-N PM review-skills-improvements-moKOR (2026-05-20 evening)

**Session shape:** n=8-capped A/B iteration loop attempting to beat sub
52827111 ("comet-aim + reactor-aware", μ=1122). PI directive: no
submission until a candidate shows significant lift at n=8 (gate
≥14/16 = Wilson-lo 0.524). Result: no candidate found. Pivot
direction surfaced at end of session.

### What landed (code + docs)

- **Setup (3 commits + bundler fix):** targeted `git checkout` of sub
  52827111's mechanism source from `claude/audit-workflow-performance-btjeK`
  onto this branch (`d642593`). Imported files:
  `agents/baseline/{proposer,chooser,value,main,chooser_trajectory,chooser_roi}.py`,
  `lib/{world_model,trajectory,aim,opp_model,value_heads}.py`,
  matching tests, and `scripts/bundle_agent.py` (btjeK upgrade with
  parity-gate cache).
  - Bundler indent-preservation fix (`9a45fea`): bundler was breaking
    function-local intra-package imports by hoisting alias rebinds to
    column 0 inside function bodies → IndentationError. Fixed at
    `scripts/bundle_agent.py:268-275`.
  - `.gitignore` for `audit/bundle-parity-cache.json` (`3f123c3`).
- **Pinned baseline:** `submissions/iter_baseline.py` = clean re-build
  of the deployed sub-52827111 bundle (parity-gate green).
- **Iter 1 audit:** `audit/2026-05-21-n8-iter1-reactor-ablation.md`
  (filename off by one day vs UTC; content correct). Documents the
  parallel-vs-serial discrepancy that invalidated the original Iter 1
  diagnostic.

### Load-bearing findings

1. **CPU-contention contaminates n=8 A/Bs.** Three parallel `fast.py
   eval` instances (24 worker processes) produced focal p95=1248ms (over
   the 1000ms env actTimeout). Variant 1b reported 12/16 (75%) under
   contention; same bundle re-tested serially gave **6/16 (37.5%)**.
   Variant 1a similarly fell from 11/16 to 7/16 serial. **Mandatory
   convention going forward: all n=8 A/Bs run serially, no parallel
   fast.py invocations.**

2. **No env-var ablation produces ≥14/16 lift over the deployed
   baseline.** Four serial n=8 runs (all clean wallclock):

   | Variant | Δ vs deployed | Wins | Wlo |
   |---|---|---:|---:|
   | A1 — comet-aim solo (reactor-aware OFF) | 7/16 (43.8%) | 0.231 |
   | A2 — Part B (reactor candidates) OFF | 7/16 (43.8%) | 0.231 |
   | A3 — BASELINE_COMET_AIM=off | 9/16 (56.2%) | 0.332 |
   | A4 | killed before completion (PI directive — see #3) |

   Three runs all landed at 7/16, A3 at 9/16. All INCONCLUSIVE; no
   candidate cleared the gate.

3. **PI verdict mid-loop ("your tests are meaningless, we need a big
   lift"):** env-var ablations tap out at ±5pp which is invisible at
   n=8 (Wilson CI ~±20pp). To produce a ≥14/16 lift over a near-optimal
   bundle requires a STRUCTURAL change, not a knob flip. Loop halted
   at A3 result.

4. **Structural-change candidates that are NOT yet new code on this
   branch:**
   - **`used_tgts` lock removal in `chooser_trajectory.py:898`.**
     Currently blocks multi-source-same-target SOLO emits even when
     JOINT is on; JOINT only fires for pre-paired candidates (capped
     JOINT_TOP_K_PER_TARGET=3, JOINT_MAX_PAIRS=20).
   - **JOINT expansion** — raise the per-target / global pair caps by
     5-10×; remove the lock-checks at `chooser_trajectory.py:885-888`.
   - **Composite value head + A2 restoration** (the μ=1149 team-peak
     architecture). `value.py` has `BASELINE_VALUE_HEAD=composite` opt-in;
     A2 4P-weakness logic also imported.
   - **New chooser** (MCTS / beam search over candidate set) — 1+
     day build.
   - **Increase N_VALIDATE / WALLCLOCK budget** — squeezes the existing
     chooser only marginally; unlikely to be a "big lift."

5. **Confirmed already-implemented (not new work):** `BASELINE_LEDGER=on`
   (wait-N inter-turn commitment memory, the original Iter 4 idea —
   already in chooser_trajectory.py lines 904-915, gated by env var
   defaulting to "off"). `BASELINE_JOINT=1` multi-source coalitions
   (already ON by default, just capped low).

### Verified gaps in the current chooser

- **`agents/baseline/proposer.py:926-928`**: wait_N>0 candidates bypass
  the trajectory filter (`predict_fleet_fate` returns wrong results
  because it doesn't pre-rotate src/tgt to launch time). This is real
  H44 surface: filter has zero coverage for the multi-wait grid.
  Iter 3 (planned, not yet implemented) would extend
  `predict_fleet_fate` with a `launch_step` arg.
- **`predict_fleet_fate` does NOT check enemy-fleet intercepts.** This
  is correct behavior — game rules confirm fleet-vs-fleet collision
  doesn't exist. Original Iter 3 framing ("add enemy fleet ray-cast")
  was based on a misread of the game spec.

### Falsified or weakened this session

- **"Part A (cost-parity filter) is the regressor."** Iter 1's
  parallel-run 12/16 was CPU-contention noise; clean serial gives
  parity-or-loss (6/16). Cannot blame Part A based on this data.
- **"Comet-aim is the key lift in sub 52827111."** A3 turned comet-aim
  OFF and got 9/16 (better than 7/16 from other ablations).
  Directional signal that comet-aim itself may be neutral-or-mildly-
  harmful, not the value-add of the push.
- **Floor-recovery via rebundle of `iter_baseline.py` (== sub 52827111).**
  PI rejected: "we can learn nothing from that." Path is OFF.

## Next-session first action (this session's pivot)

**Priority 1 — Pick one structural change from the list above and ship
it (~few hundred LOC, single axis).** Recommend `used_tgts` lock
removal + JOINT cap expansion in `chooser_trajectory.py` as the
cheapest structural-shape change: combat rule 1 (same-owner same-step
arrivals stack) is well-understood; the existing lock literally
forbids the most powerful combat pattern. Risk: Plan agent flagged
this as needing n=32 minimum (prior asymmetric chooser attempts
0/32). Run n=8 serial first; if directional, escalate to n=32.

**Priority 2 — if Priority 1 doesn't clear:** Composite value head +
A2 restoration (μ=1149 architecture). Code already imported; needs
the right env-var combo + bundle bake. Significant ladder evidence
(sub 52744856 live μ=1149).

**Priority 3 — out-of-session-scope:** Konbu17 shot-validator MLP
(~1 week build, but the only ML attack with empirical precedent
+19pp panel lift).

**Reading order for the next agent:** this section first, then
`audit/2026-05-21-n8-iter1-reactor-ablation.md`, then
`/root/.claude/plans/go-effervescent-mochi.md` for the full
iteration ladder context.

## What just landed (2026-05-20, this session)

This session was a **doc-only consolidation pass** across 8 active
branches. No code changed. New / edited docs:

| File | Change |
|---|---|
| `state/MULTI_BRANCH.md` | **NEW.** Single source of truth across branches. |
| `state/TOOLS.md` | **NEW.** Tools registry (A/B + diag + validation). |
| `CLAUDE.md` | Rules 41-47 appended. Pointers section adds MULTI_BRANCH + TOOLS. |
| `.claude/skills/kaggle-comp/SKILL.md` | Step 0 "load MULTI_BRANCH + TOOLS first" preamble. |
| `.claude/skills/kaggle-comp/day-loop.md` | Step 1 amendment for code-comp branch coordination. |
| `.claude/skills/kaggle-comp/improvements.md` | Rotated: 7 items promoted to rules; 2 superseded. |
| `.claude/skills/kaggle-comp/improvements-archive-2026-05-20.md` | **NEW.** Rotation archive. |
| `state/current.md` | Deprecated to pointer-only banner. |
| `state/mechanism-ledger.md` | Appended 2026-05-18 → 5-20 entries. |
| `HANDOVER.md` | Rewritten (this file). |

**Rules 41-47 summary (read CLAUDE.md for full text):**

- **41.** Confound-sweep before correlational conclusion (btjeK origin).
- **42.** Pre-submit cross-branch coordination gate (the ~320 μ loss origin).
- **43.** Multi-opponent panel mandatory pre-submit (supersedes `--vs-panel` pending item).
- **44.** State-of-truth read before subsystem edits (supersedes "read state docs" pending item).
- **45.** n ≥ 32 minimum for A/B lift claims.
- **46.** Bundle + parity smoke before any submission.
- **47.** Physics-primitive verification before agent design (PFhzM origin).

## Three parallel tracks — current state

| Track | Lead branch | Best result | Status | Next action |
|---|---|---|---|---|
| **A — Analytical chooser** | `strategy-framework-design-OyoYR-rebased` | μ 829.1 (sub 52854094) — both live pushes regressed | knowledge-base 5/20: "axis closed (10 slices, 0 lift)"; architectural bind: analytical needs multi-turn glue OR must replace rollout entirely | Decide: park, or pivot to analytical-leaf-inside-rollout |
| **B — Hybrid-sim production** | `audit-workflow-performance-btjeK` (production) + `analyze-game-strategy-EpMVP` (phases) | μ 1149.2 (EVICTED) | Live champion lineage. H44 finding 5/20: 65% fleet-destroyed-in-flight — new physics-driven mechanism candidate | (i) hold-feasibility solo validation (btjeK Phase B); (ii) H44 defensive mechanism design; (iii) EpMVP Phase 4/6 commissioning |
| **C — Verify-first + Goal-directed** | `ml-competition-strategy-PFhzM` (+ `precision-physics-engine-ymJkA` substrate) | Phase A Test 3 PASS; wrap-baseline 12/32 = 37.5% (only positive signal vs production) | greedy_expand (60 LOC) tied goal_planner (500 LOC); chooser axis confirmed neutral | Decide: is wrap-baseline-as-veto a viable design? Or is Track-C work substrate-only? |

## Next-session first actions (ranked by EV / cost)

### Priority 1 — code consolidation pass (start small, parity-tested)

Following the 6-step consolidation-merge gate in `state/TOOLS.md`:

1. **Substrate primitives** (no chooser changes piggy-backed):
   - Merge `lib/trajectory_layer.py` + `tests/test_trajectory_layer_positions.py` from PFhzM.
   - Merge `agents/precision/sim.py` + `agents/precision/intercept.py` + `agents/precision/tests/` from precision-physics branch.
   - Run consolidation gate; expected: GREEN.
2. **Bundler upgrade** from EpMVP — "inline agent submodules + explicit-name imports."
3. **Diagnostic scripts** (zero-risk leaf merges):
   - `scripts/h44_landing_capture_diagnostic.py` (btjeK)
   - `scripts/probe_emits_via_fate.py` (PFhzM)
   - `scripts/inspect_goal_planner_game.py` (PFhzM)
4. **Oracle tests** (test-only, zero-risk):
   - `tests/test_baseline_replay_regression.py` (EpMVP)
   - `tests/test_migration_solver.py` (EpMVP)

### Priority 2 — recovery submission planning

The rolling-last-2 is 320 μ below team peak. Three sub-IDs have evidence
of being strong:

- **52744856** (μ 1149.2, composite_a2_hybrid 2P + A2 4P)
- **52754310** (μ 1143.7, trajectory v4 + wait_N + wallclock)
- **52811320** (μ 1135.1, hold-feasibility solo)

**Open PI question:** which lineage to rebundle and push? The push will
itself need to clear Rule 42 (claim board) and Rule 43 (panel + h2h)
before submit. Do NOT push without explicit PI sign-off.

### Priority 3 — Track-B physics mechanism design

H44 finding: 65% of landing-capture failures are fleet-destroyed-in-flight.
This is a substrate-level signal that the trajectory chooser's
fleet-safety filter is not catching. Design a defensive mechanism
(NOT a restriction-tuning constant bump — Rule 40) that emerges from
the underlying physics.

## Pointers

- `state/MULTI_BRANCH.md` — cross-branch state-of-truth.
- `state/TOOLS.md` — tools registry + consolidation-merge gate.
- `state/mechanism-ledger.md` — every agent family tried.
- `state/hypothesis-board.md` — open ideas, killed list.
- `CLAUDE.md` — rules 1-47 + R-defaults.
- `audit/friction.md` — current friction summary.
- `.claude/skills/kaggle-comp/` — skill (now multi-branch-aware).
- `audit/2026-05-21-h44-phase1-CORRECTED.md` (btjeK) — physics-failure analysis.
- `audit/2026-05-20-postmortem-strategy-framework-design-OyoYR-rebased.md` — analytical axis closure.
- `audit/2026-05-19-postmortem-PFhzM-physics-gate-and-mvp-confirmation.md` — Track-C verdict.
- `audit/2026-05-21-n8-iter1-reactor-ablation.md` (this branch, filename off by one UTC day) — Iter 1 ablation results + the parallel/serial contention finding.
- `audit/2026-05-22-extract-physics-trajectory.md` (Vjaz9) — physics substrate extraction.
- `audit/2026-05-28-postmortem-competition-objective-alignment-hqNVM.md` — Phase A wrap.
- `knowledge-base/thoughts/2026-05-28-value-head-phase-a-distillation-passes.md` — Phase A debrief + Phase B framing.
- `/root/.claude/plans/go-effervescent-mochi.md` — full iteration-loop plan including the structural-change pivot list.
- `/root/.claude/plans/let-s-do-it-um-cozy-peach.md` — original value-head Phase A/B plan (this branch).

## Rule reminders (most relevant this session)

- **Rule 1:** submissions PI-approved, single-shot, no retry loops.
- **Rule 12:** rolling-last-2; weak late submits unrecoverable for ~24h.
- **Rule 32:** session-start git fetch; verify rolling pair via Kaggle CLI.
- **Rule 35-36:** PI thoughts append-only; session-end second-brain update.
- **Rule 37:** 3-variant axis cap. v9-v15 chooser hit it; chain-bonus hit it; analytical-slice hit it (10/3+).
- **Rule 40:** prefer modeling-correctness over restriction-tuning.
- **Rules 41-47 (new today):** see CLAUDE.md.

## Open questions for PI

1. Track A (analytical) — park or pivot to analytical-leaf-inside-rollout?
2. Track C — wrap-baseline-as-veto, or substrate-only contribution?
3. Recovery submission — which lineage to rebundle for the next push?
4. Should the SessionStart hook implementation (improvements.md TOP
   PRIORITY) get priority over the code consolidation pass?
