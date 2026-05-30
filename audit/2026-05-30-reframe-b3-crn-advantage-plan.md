# Reframe B.3 — CRN-paired advantage labels (handover plan)

> Written 2026-05-30 AM by `claude/competition-objective-alignment-hqNVM`
> at the close of the B.2 first-cut session. PI direction: option (2)
> from the B.2 post-failure menu — redesign with causal labels.

## Why B.2 failed (the load-bearing finding)

The B.2 first cut (commit `316e206`, audit
`audit/2026-05-29-reframe-b2-value-head.md`) trained a per-candidate
LightGBM regressor on **observational K=10 ship-delta** from pv_eta
self-play. Training metrics looked great:

- **Spearman ρ = +0.359 on held-out val** (3× the 0.10 mandatory gate)
- R² = +0.032, RMSE = 377, walker parity 0.000e+00
- Feature importance dominated by per-planet covariates (as the
  within-owner probe predicted)

H2H vs bare pv_eta at n=32 was **0/32 (Wilson 95% CI [0.000, 0.107])
at λ=1.0 AND at λ=0.1**. Latency fine (p95 = 691ms vs 1000ms cap).

### Diagnosis: selection bias

The training labels were **state-conditional**, not **action-causal**.
Concretely:

- `trace_accepted` only writes features for candidates the chooser
  **accepted**. The training distribution is `{candidates pv_eta picks}`.
- Labels are "the focal-seat ship-delta that happened over the next 10
  turns" — purely observational.
- At inference, the chooser uses `head_out` to re-rank **all** prerank
  candidates, including ones pv_eta would have rejected. The head's
  predictions on those candidates are unconstrained LightGBM
  extrapolation.
- Even at λ=0.1, the head's bad predictions on rejected candidates flip
  the argmax in counter-causal directions.

This is the classic off-policy/observational-data failure mode that
PM3 warned about (`knowledge-base/thoughts/2026-05-28-pm-distillation-
action-rank-collapse.md`). The Spearman gate detects rank-order
preservation **within the training distribution**, not generalization
to the deployment distribution.

### What's closed (Rule 37 axis)

The "observational-label additive-term head on pv_eta's chooser" axis is
**falsified**. Do not re-run with adjusted λ, different K, or different
feature subset — the bottleneck is the LABEL, not the model.

## B.3 objective

Train a per-candidate regressor with **counterfactual** labels:

```
A(s, a) = focal_margin(s after action a, K=10) − focal_margin(s after idle, K=10)
```

where both rollouts use **pv_eta** as the focal-seat policy from step 2
onward AND as the opp policy throughout. The label is "the advantage
this candidate buys vs doing nothing, conditional on opp plays pv_eta."

Why this fixes B.2:
- Label is **action-causal**, not state-conditional.
- Coverage is **all top-N prerank candidates** per state, not just
  accepted ones — no selection bias at training time.
- Common Random Numbers (the opp's pv_eta is deterministic given the
  obs) → state-noise cancels in `(action - idle)` and label variance
  drops sharply.

## Architecture changes vs B.2

| Aspect | B.2 (failed) | B.3 (new) |
|---|---|---|
| Label | Observational K=10 ship-delta | Advantage = K=10 margin(action) − margin(idle) |
| Candidate coverage | Only accepted candidates (~15-20 per state) | Top-N prerank candidates (N=10 recommended) |
| Distribution at inference | OOD on rejected candidates | In-distribution on all top-N candidates |
| Insertion | Additive `λ · head_out` | Same (no chooser changes) |
| Features | 14-d + leaf_delta | Same (no encoder changes) |
| Bundler / wrapper / walker | Reused | Reused unchanged |

**Reusable from B.2 (no re-work):**
- `lib/value_head_features.py` — feature encoder, 7 unit tests GREEN
- `agents/baseline/_value_head.py` — lazy-load + featurize + predict
- `agents/baseline_pv_eta_vh/main.py` — wrapper agent
- `scripts/bundle_pv_eta_vh.py` — single-file Kaggle bundler
- `scripts/inspect_value_head_corpus.py` — pre-train sanity inspector
- `scripts/train_value_head.py` — LightGBM trainer (minor swap to advantage labels)
- The whole inference pipeline is bundle-tested. Walker parity 0.000e+00.

Only the **labels** are wrong.

## Critical-path verification (Step 0, MANDATORY before corpus gen)

Before building any infrastructure, verify three things — each is a
single Python script that fits in <30 LOC and runs in seconds.

### V0.1 — Does `fast_sim.rollout(snap, K, [pv_eta, pv_eta])` work?

```python
# Load a fresh game state, snapshot it, run a 10-step rollout with
# pv_eta as both policies, confirm no crashes and snapshot updates.
```

Risk: pv_eta maintains module-level state (`_PENDING_LAUNCHES`,
ledger, etc.) that may leak across rollouts within one process. If
fast_sim's rollout doesn't reset that state cleanly, advantages will
be polluted by stale commits. **Fix**: reset module-level state
between rollouts (or use clean process per rollout — slower).

### V0.2 — Speed: K=10 rollout time < 100ms

`lib/fast_sim.py` claims ~20× speedup vs env.clone+step. For K=10
rollouts of pv_eta vs pv_eta (real per-turn ~40-80ms in production),
expected fast_sim K=10 rollout: 20-50ms. **Target**: <100ms per K=10
rollout. If exceeded, stage 2 cost blows up.

### V0.3 — Parity: fast_sim vs env.clone+step

`tests/test_game_parity.py` already pins fast_sim's `step()` parity at
byte-exactness for **simple actions**. We need to confirm parity for:
- pv_eta's joint-emit actions (multi-source, multi-target)
- Ledger-driven wait_N commits (BASELINE_LEDGER=on path)
- Sun-avoidance / OOB / comet-aim mechanism layer outputs

If parity holds at K=10 ship-totals to integer precision on 10 random
seeds, the rollout is trustworthy.

**If V0.1, V0.2, or V0.3 fails: fall back to env.clone+step.** Stage 2
cost goes from ~7.5 h to ~150 h on CPU (untenable for local; would
need Kaggle parallel notebooks). Strongly motivates fixing fast_sim.

## Corpus generation pipeline

### Stage 1 — self-play with top-N prerank trace (~4 h)

Identical to B.2's self-play except a NEW trace hook captures the
top-N prerank candidates per state, not just accepted ones.

| New env var | Effect |
|---|---|
| `BASELINE_PRERANK_TRACE=<path>` | Enables prerank trace emission |
| `BASELINE_PRERANK_TOP_N=10` | How many top candidates to record per state (cheap ranking by `cheap_delta`) |

Trace hook surface (in `agents/baseline/_trace_hook.py`):

```python
def trace_prerank(world, world_model, me, prerank, top_n=10):
    """For each prerank candidate among the top-N by cheap_delta,
    write {step, me, src_id, tgt_id, ships, angle, wait_N, eta_hint,
    cheap_delta, features} to BASELINE_PRERANK_TRACE."""
```

Call site: `chooser_trajectory.py` after prerank list is finalized
(around line 881) and BEFORE the scoring loop. Cheap (~10ms per state
at top-10).

Output schema (per game):
- `accepted.jsonl` (reused from B.2 — for delta_pred reference)
- `replay.jsonl` (reused from B.2 — for label computation)
- **NEW** `prerank.jsonl` — top-N candidates per state with features
  pre-encoded

### Stage 2 — CRN advantage labelling (~7.5 h target / 150 h fallback)

For each `(state, candidate)` in `prerank.jsonl`:

```python
snap = fast_sim.from_obs(replay[step], configuration, episode_seed)
# Idle rollout: focal=no-op at step t, both pv_eta thereafter
snap_idle = fast_sim.rollout(
    snap, K=10, policies=[idle_at_step_0(pv_eta), pv_eta]
)
margin_idle = fast_sim.delta_us_minus_them(snap_idle, focal_seat)

# Action rollout: focal=candidate at step t, both pv_eta thereafter
snap_action = fast_sim.rollout(
    snap, K=10, policies=[candidate_at_step_0(pv_eta), pv_eta]
)
margin_action = fast_sim.delta_us_minus_them(snap_action, focal_seat)

label = margin_action - margin_idle
```

The `idle_at_step_0` and `candidate_at_step_0` are simple wrappers that
override the focal seat's action at step 0 with a specific action,
then delegate to pv_eta thereafter.

Output: `corpus.jsonl` with rows:
```
{game_id, seat, step, src_id, tgt_id, ships, eta, angle, wait_N,
 delta_pred, features, label=advantage}
```

### Stage 3 — train (~5 min)

Reuse `scripts/train_value_head.py` with a single change: relax the
σ(label) sanity band from [200, 1500] to **[20, 500]**. CRN-paired
advantage is bounded by what one action buys you over 10 turns —
typically much smaller than the absolute ship-delta. σ ≈ 50-150
ships expected.

Training-time gates unchanged:
- Game-level 80/20 split
- Spearman ρ ≥ 0.10 on held-out val (MANDATORY)
- Walker parity check before save

### Stage 4 — bundle + A/B (~30 min)

Reuse `scripts/bundle_pv_eta_vh.py`. The new model.txt overwrites the
B.2 artifact at `data/value_head/value_head_model.txt`. A/B vs bare
pv_eta at n=32, gate Wilson-lo ≥ 0.50 (Rule 43, 45).

## Cost estimate

| Stage | Best case | Fallback (env.clone+step) |
|---|---:|---:|
| V0 verification | 1 h | 1 h |
| Stage 1 (self-play, 100 games) | 4 h | 4 h |
| Stage 2 (advantage labels) | 7.5 h | ~150 h |
| Stage 3 (train) | 5 min | 5 min |
| Stage 4 (bundle + A/B) | 30 min | 30 min |
| **Total** | **~13 h** | **~155 h** |

Reducible knobs:
- **Top-N = 5 instead of 10**: halves stage 2 (3.75 h or 75 h)
- **K = 5 instead of 10**: halves rollout cost; labels shorter-horizon
- **50 games instead of 100**: halves stage 1 + stage 2
- **Skip turns 0-50 and 400+**: focus labelling on middle-game decisions

A reasonable first cut: 100 games × top-5 × K=10 ≈ 6 h stage 2 in the
best case. Smoke first with 4 games × top-5 to confirm pipeline.

## Open design questions for PI (resolve in fresh session before corpus gen)

1. **Top-N candidates per state**: 10 (richer coverage, 2× cost) or
   5 (cheaper, may miss alternative-action signal)?
2. **Rollout horizon K**: 10 (matches B.2's horizon) or 5 (cheaper,
   shorter causal window)?
3. **Stage 2 compute strategy**: local CPU (estimated ~6-7 h if
   fast_sim works) or Kaggle parallel CPU notebooks (4 notebooks ×
   25 games each = 4× speedup with setup overhead)?
4. **Add focal-wins-only filter**: keep only labels from games where
   focal-seat won? Simple causal-bias correction; halves corpus but
   improves label quality. Combine with CRN or stand alone?
5. **Trace `cheap_delta` AND `leaf_delta`**: B.2 used the finalized
   `leaf_delta` (after pv_eta discount). For B.3, do we use `cheap_delta`
   (pre-scoring, cheaper) or `leaf_delta` (richer)? Affects featurizer.

## File changes vs B.2 infrastructure

| File | Change for B.3 |
|---|---|
| `agents/baseline/_trace_hook.py` | NEW `trace_prerank()` gated by `BASELINE_PRERANK_TRACE` |
| `agents/baseline/chooser_trajectory.py` | NEW one-line call to `trace_prerank` after prerank is built |
| `lib/fast_sim_pv_eta.py` (NEW) | Wrappers `pv_eta_policy(obs)` and `override_at_step_0(action, base_policy)` so fast_sim can drive pv_eta |
| `scripts/compute_crn_advantage.py` (NEW) | The advantage-labelling driver — reads prerank.jsonl + replay.jsonl, writes corpus.jsonl |
| `scripts/gen_b3_corpus.py` (NEW) | Three-stage pipeline runner (selfplay → label → assemble) |
| `scripts/train_value_head.py` | Edit: σ(label) sanity band [20, 500] |
| `scripts/inspect_value_head_corpus.py` | Reusable; bands match the new range |
| `agents/baseline/_value_head.py` | UNCHANGED |
| `lib/value_head_features.py` | UNCHANGED |
| `agents/baseline_pv_eta_vh/main.py` | UNCHANGED |
| `scripts/bundle_pv_eta_vh.py` | UNCHANGED |

## Verification gates (in order)

| Gate | Pass criteria | If fails |
|---|---|---|
| V0.1 | fast_sim K=10 rollout with pv_eta as policy completes, no state leakage | Investigate state leakage; consider per-rollout module reload |
| V0.2 | K=10 rollout time < 100ms | Use env.clone+step; rescope corpus size |
| V0.3 | fast_sim K=10 ship-totals match env.clone+step to integer precision on 10 random seeds | Investigate fast_sim parity holes for pv_eta-specific actions |
| Smoke | 4-game × top-5 corpus produces 200-500 rows with σ(label) > 5 | Inspect by-owner breakdown; check label sign distribution |
| Train | Spearman ρ ≥ 0.10 on held-out val AND σ(pred) < σ(label) AND walker parity < 1e-4 | Lower the bar to ρ ≥ 0.05 with PI signoff; or pivot to filter use |
| A/B | Wilson-lo ≥ 0.50 vs bare pv_eta at n=32 | Try filter-form (option 1 from B.2 post-failure menu) before declaring axis closed |

## What's still open after B.3 (regardless of outcome)

If B.3 passes: ship it via the existing bundler/wrapper. Submission
goes through Rules 42 (claim board), 43 (panel), 45 (n≥32), 46 (smoke).

If B.3 fails: the **two-axis closure** (observational label fails AND
CRN-paired advantage fails) means the head-as-additive-term axis is
falsified at the architectural level. Next pivot would be:
- **Reframe B-filter** (option 1 from B.2's post-failure menu — use
  the head as a top-K filter instead of additive term; in-distribution
  on accepted candidates only)
- **Reframe C** (opponent-emit predictor — the static-opp assumption
  is the chooser's blind spot per the within-owner probe; this is the
  biggest swing but ~5-7 day investment)

## Reading order for fresh session

1. **`audit/2026-05-29-reframe-b2-value-head.md`** — full B.2 result
   record (training metrics + A/B 0/32 verdict; will be updated with
   this session's failure analysis)
2. **`audit/2026-05-30-reframe-b3-crn-advantage-plan.md`** — this file
3. `state/MULTI_BRANCH.md` — live Kaggle rolling pair, push board
4. `CLAUDE.md` rules (especially 13 GPU budget, 32 git fetch, 37
   axis cap, 38 fix verification, 47 physics-primitive verification)
5. `lib/fast_sim.py` lines 464-494 (rollout API) and
   `tests/test_game_parity.py` (parity test to extend)
6. `agents/baseline/_value_head.py` + `lib/value_head_features.py`
   (reused as-is)
7. `knowledge-base/concepts/crn-advantage-datagen-sketch.md` — PM3's
   original CRN-advantage cost sketch (pre-B.2; some numbers may have
   shifted but the design is right)

## Live ladder state (snapshot 2026-05-30 09:58 UTC)

Unchanged from B.2 session start:

| Sub ID   | Agent                             | μ        | Role               |
|----------|-----------------------------------|---------:|--------------------|
| 53131296 | `baseline_validated.py` (PM5 MLP) | **1096.9** | rolling pair (top) |
| 53117942 | `baseline_leaf_pv_2p.py`          | **1091.9** | rolling pair (bot) |
| 53111837 | `baseline_pv_eta.py` (EVICTED)    | 1154.8   | historical peak    |

- Daily submission slots: 5/5 free today.
- Deadline: 2026-06-23 — **24 days remaining**.
- Team count: 3467.
- Floor-at-risk: MODERATE (rolling pair ~58 μ below evicted peak).
- B.2 session pushed nothing (0/32 verdict — no candidate worth
  evicting the rolling pair for).

## Decision rule

B.3 is a **new axis** under Rule 37 (different supervision unit,
different distribution — CRN-paired advantage, not observational
ship-delta). New 3-variant budget. If 3 variants of CRN-advantage
heads fail to clear Wilson-lo ≥ 0.50, the entire head-as-additive
axis closes (combined with B.2's failure) and we pivot to filter or
Reframe C.

The variants axis for B.3 hyperparam sweep should be: K ∈ {5, 10, 20}
or top-N ∈ {5, 10, 20}, NOT λ (B.2 proved λ is the wrong knob when
labels are bad).
