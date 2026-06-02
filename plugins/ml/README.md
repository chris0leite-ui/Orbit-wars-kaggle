# `plugins/ml/` — portable ML plug-ins for non-ML target branches

Two reusable ML enhancements lifted from this branch's
`baseline_pv_eta_vh_dist_composite` agent (the strongest stack
on hqNVM as of 2026-06-02). Designed to drop onto a target branch
whose chooser has no ML wiring — primary target is
`claude/champion-strategy-rules-00JzI` (the joint-sync coalition
champion), but the recipe applies to any sibling that builds on
`agents/baseline/`.

## What's in here

| Path | Destination on target branch | Role |
|---|---|---|
| `lib/_validator_tree_walker.py`       | `lib/_validator_tree_walker.py`       | Pure-Python LightGBM tree walker. Submit-time inference for both plug-ins; **no `lightgbm` dep at submit time**. Shared dependency. |
| `lib/opp_features_lite.py`            | `lib/opp_features_lite.py`            | Vectorized 30-d "lite" encoder for the opp model. Pure NumPy; no `WorldModel` rebuild. ~1 ms median per call. |
| `lib/value_head_features.py`          | `lib/value_head_features.py`          | 14-d per-candidate encoder for the B.3 value head (plus `leaf_delta` injected inside the chooser). |
| `agents/baseline/_value_head.py`      | `agents/baseline/_value_head.py`      | Value-head module: `vh_is_enabled()`, `vh_featurize_prerank()`, `vh_predict_one()`. |
| `data/opp_distill/distill_booster.txt`     | `data/opp_distill/distill_booster.txt`     | Trained Tier-2 opp model (18 KB). LightGBM dump from `scripts/train_opp_distill.py --lite`. |
| `data/value_head/value_head_model.txt`     | `data/value_head/value_head_model.txt`     | Trained B.3 value head (654 KB). LightGBM regression dump from `scripts/train_value_head.py`. |
| `wiring/opp_model_trained_logreg.py.snippet` | INSERT into `lib/opp_model.py` (replaces Tier-2 stub) | Source for `_load_opp_booster`, `trained_logreg_policy`, and the lite-encoder enumerate-and-score logic. |
| `wiring/chooser_trajectory_vh_wiring.py.snippet` | INSERT into `agents/baseline/chooser_trajectory.py` | Two code blocks (prerank-init + per-candidate scoring) that add the VH additive term. |

Total folder size: ~700 KB.

---

## Plug-in 1 — Distilled-ladder opponent model (Tier 2 v2)

**What it does.** Replaces the rollout-time opponent policy. Rollouts
inside `score_candidate_v4` currently call
`lib.opp_model.lite_greedy_policy` (Tier 0) or `top_tier_mirror_policy`
(Tier 1). Tier 2 here is a LightGBM binary classifier trained on
~50k positive (src, tgt, ships, angle) labels from 775 top-10
Kaggle ladder 2P replays. At inference: enumerates `MIN_SRC_SHIPS=5,
TOP_K=8` candidates, encodes with the 30-d lite features (no
`WorldModel.from_world` build), scores, emits anything above
`BASELINE_OPP_FILTER_THRESHOLD` (default 0.30) — capped one emit per
source per turn.

**Speed.** ~1.6 ms median per call. Tier 0 is ~1 ms, Tier 1 is ~10 ms.

**Activation.** Set env `BASELINE_OPP_TIER=2` on the wrapper. At
`BASELINE_OPP_TIER=0` (default) the plug-in is dormant — the booster
is never loaded — so dropping these files in is risk-free for
existing agents.

**Ladder evidence.** `baseline_pv_eta_vh_dist` (Tier 2 alone) lives in
the rolling pair at μ ≈ 1149 on hqNVM. Tier 1 baseline µ ≈ 1135.
**~14 μ lift attributable to the Tier-2 swap alone.**

**Files to copy.**
```
plugins/ml/lib/_validator_tree_walker.py   ->  lib/_validator_tree_walker.py
plugins/ml/lib/opp_features_lite.py        ->  lib/opp_features_lite.py
plugins/ml/data/opp_distill/distill_booster.txt  ->  data/opp_distill/distill_booster.txt
```

**Patch to apply (target's `lib/opp_model.py`).** Replace the existing
Tier-2 stub (which currently calls `top_tier_mirror_policy` as a
fallback) with the contents of
`plugins/ml/wiring/opp_model_trained_logreg.py.snippet`. The snippet
is a verbatim section from hqNVM's `lib/opp_model.py` lines 124-365:

1. **Find** the `# Tier 2 — placeholder for the trained launch-decision
   classifier` block + `def trained_logreg_policy(obs)` stub
   (on 00JzI: around lines 124-148).
2. **Replace** that range with the entire snippet contents.
3. **Confirm** the `_TIER_REGISTRY` dict at the end of the file
   still has `2: trained_logreg_policy` (it already does on 00JzI).

The snippet adds: `_OPP_BOOSTER_B64` blob, `_OPP_PARSED` /
`_OPP_LOAD_FAILED` / `_OPP_THRESHOLD` state, `_load_opp_booster()`,
the `_DIST_*` constants, and the full `trained_logreg_policy()` with
lite-encoder fast path + slow-path fallback (slow path is dead in
prod; safe to keep for parity testing).

**Failure mode.** Any load/encode/score exception in
`trained_logreg_policy` returns `lite_greedy_policy(obs)` for that
step. Never emits garbage launches.

---

## Plug-in 2 — B.3 value-head advantage term

**What it does.** Per-candidate LightGBM regressor that predicts the
seat-0 ship-delta over the next K=10 turns, conditional on the
candidate decision. The chooser adds `λ_vh · head_output` to each
candidate's scalar score. Featurization runs once over the prerank
(14 base features per candidate); per-candidate prediction injects the
15th feature (`leaf_delta`) at scoring time.

**Speed.** ~50 µs per `vh_predict_one` call (pure-Python tree walker
on a small regression dump). At ~30 candidates per turn: ~1.5 ms
per turn.

**Activation.** Set env `BASELINE_VH_LAMBDA=1.0` on the wrapper. The
regressor is in ship units, so λ=1.0 is the head's natural scale. At
λ=0.0 (default), the plug-in is byte-equivalent to no-port — same
parity invariant as ML logit.

**Ladder evidence.** `baseline_pv_eta_vh_dist_composite` (Tier 2 opp
model + B.3 head together) is the strongest hqNVM composite. The
value head's marginal contribution on top of Tier 2 alone is
**directional (+5-10 μ)** in local A/B at n=32 with Wilson-lo > 0.50
but not yet rolling-pair-confirmed. **Caveat:** the head was trained
on `baseline_pv_eta` self-play, not 00JzI joint-sync self-play.
Without retraining (see below), expect smaller lift on 00JzI than on
hqNVM.

**Files to copy.**
```
plugins/ml/lib/_validator_tree_walker.py        ->  lib/_validator_tree_walker.py
plugins/ml/lib/value_head_features.py           ->  lib/value_head_features.py
plugins/ml/agents/baseline/_value_head.py       ->  agents/baseline/_value_head.py
plugins/ml/data/value_head/value_head_model.txt ->  data/value_head/value_head_model.txt
```

**Patch to apply (target's `agents/baseline/chooser_trajectory.py`).**
Insert the TWO code blocks from
`plugins/ml/wiring/chooser_trajectory_vh_wiring.py.snippet`:

- **Block A** (prerank-init): insert immediately AFTER the existing
  `ml_score_candidates` init block, BEFORE the deadline-loop starts.
  On hqNVM this lives at lines 923-935; on 00JzI the corresponding
  landmark is the end of the prerank-rank section right after
  `score_candidate_v4`'s arg-pack is staged.
- **Block B** (per-candidate scoring): insert INSIDE the chooser's
  candidate-scoring loop, AFTER the leaf `score` is computed and
  AFTER any ML-logit correction, BEFORE any class-specific
  multipliers (`_surplus_boost`, `_aim_boost`).

Both blocks are gated on `vh_feats` (the dict from Block A) being
non-empty, so dropping the snippets in without setting
`BASELINE_VH_LAMBDA` is a no-op.

**Variable contract.** Block B expects these names in scope: `vh_feats`,
`src.id`, `tgt.id`, `ships`, `angle`, `wait_N`, `score`. On 00JzI's
chooser these names match (it shares the v4 score function signature
with hqNVM up to the diverged dead-code paths). If 00JzI renames any
of these, fix the snippet to match.

---

## Apply order for the 00JzI target branch

```bash
# From the 00JzI worktree:
PLUGIN_SRC=/path/to/hqNVM/plugins/ml      # or: git show hqNVM:plugins/ml/...

# 1. Drop-in modules (no surgery)
cp $PLUGIN_SRC/lib/_validator_tree_walker.py  lib/
cp $PLUGIN_SRC/lib/opp_features_lite.py       lib/
cp $PLUGIN_SRC/lib/value_head_features.py     lib/
cp $PLUGIN_SRC/agents/baseline/_value_head.py agents/baseline/

# 2. Trained model files
mkdir -p data/opp_distill data/value_head
cp $PLUGIN_SRC/data/opp_distill/distill_booster.txt    data/opp_distill/
cp $PLUGIN_SRC/data/value_head/value_head_model.txt    data/value_head/

# 3. opp_model.py surgery (find Tier-2 stub, replace with snippet)
$EDITOR lib/opp_model.py
# ... follow Plug-in 1 instructions ...

# 4. chooser_trajectory.py surgery (insert Blocks A and B)
$EDITOR agents/baseline/chooser_trajectory.py
# ... follow Plug-in 2 instructions ...

# 5. Smoke test (no env vars set → both plug-ins dormant, parity check)
python -c "from agents.baseline.main import agent; print(agent({'planets':[],'fleets':[],'step':0,'player':0,'episodeSteps':500}))"

# 6. Activate both, run head-to-head vs pre-port champion
BASELINE_OPP_TIER=2 BASELINE_VH_LAMBDA=1.0 \
  python fast.py eval agents/baseline_<wrapper> --vs <pre_port_baseline> --geometry-panel
```

---

## Bundler updates (submit-time inlining)

The submit-time bundle inlines selected `lib/` modules and patches
the trained-model blobs as gzip+base64 strings. **Two patch sites**
on a composite submission:

| Module | B64 var name | Source file |
|---|---|---|
| `lib/opp_model.py` (inlined) | `_OPP_BOOSTER_B64` | `data/opp_distill/distill_booster.txt` |
| `agents/baseline/_value_head.py` (inlined) | `_VH_MODEL_B64` | `data/value_head/value_head_model.txt` |

The composite bundler on hqNVM is
`scripts/bundle_pv_eta_vh_dist_composite.py`. Copy it to the target
branch as a starting point and edit the `DEFAULT_OPP_BOOSTER` /
`DEFAULT_VH_MODEL` paths if needed. Also add these to the target's
`DEFAULT_LIB_ORDER` in `scripts/bundle_agent.py`:

```python
DEFAULT_LIB_ORDER = [
    ...,
    "_validator_tree_walker",   # pure-Python LightGBM tree walker
    "opp_features_lite",        # 30-d encoder for opp model
    "value_head_features",      # 14-d encoder for value head
    ...,
]
```

(These three names go AFTER `shot_features` and BEFORE `opp_model` in
the order list — `opp_model.py` and `_value_head.py` both import them.)

**Bundle invariant (Rule 46).** Every submission MUST clear:
(a) `python scripts/bundle_agent.py <agent>` succeeds;
(b) `pytest tests/test_bundle.py` GREEN;
(c) `python fast.py play <bundled_submission>` runs one full game
without crash.

---

## Retraining for the target agent ("trained to fit the agent")

The trained models in this folder were fit to **hqNVM's
`baseline_pv_eta` self-play distribution**. The B.3 head's CRN-paired
advantage labels in particular are agent-specific — the head learned
"ship-delta given the *baseline_pv_eta* chooser's downstream play."
Porting it to 00JzI without retraining is the *partial* lift.

**When NOT to retrain (port as-is).**
- Opp model (Plug-in 1): retraining is optional. The opp model
  predicts what *ladder opponents* do, which is independent of which
  agent uses it. Drop-in transfer should be ~lossless.
- Smoke / triage runs where you want a quick directional signal.

**When TO retrain (recommended for ladder submission).**
- B.3 value head: yes, retrain on 00JzI self-play. Without
  retraining, expect the head to underestimate joint-sync coalition
  candidates (the ones 00JzI's chooser shines on) because those
  decisions never appeared in the training distribution.

**Retraining recipe (B.3 head on the target agent).**

1. **Collect CRN-paired self-play replays** (~2k games):
   ```bash
   # On the target branch:
   python scripts/train_value_head.py \
     --collect-only \
     --agent agents/baseline_joint_aggr_consolidated \   # 00JzI's strongest base
     --opponent agents/baseline_joint_aggr_consolidated \
     --games 2000 \
     --seeds-per-pair 4 \
     --out data/value_head/joint_aggr_corpus.jsonl
   ```
   CPU cost: ~2-3 h on 8 workers. Disk: ~6 MB per 1k games.

2. **Train the regressor**:
   ```bash
   python scripts/train_value_head.py \
     --corpus data/value_head/joint_aggr_corpus.jsonl \
     --K 10 \
     --num-leaves 31 \
     --num-iterations 200 \
     --out data/value_head/value_head_model.txt
   ```
   CPU cost: ~5-15 min. The 14-d encoder is identical to
   hqNVM — no encoder retraining needed.

3. **Re-bundle and smoke-test parity at λ=0** (must be byte-equivalent
   to pre-VH chooser):
   ```bash
   BASELINE_VH_LAMBDA=0.0 python fast.py play submissions/...
   ```

4. **A/B at λ=1.0** with the seeded geometry panel (Rule 43, n≥32):
   ```bash
   BASELINE_VH_LAMBDA=1.0 python fast.py eval <vh_agent> --vs <base_agent> --geometry-panel
   ```

**Retraining for the opp model (optional).** Same script family
(`scripts/train_opp_distill.py --lite`), but the input is replay
data from the *target ladder population* (not the target agent's
self-play). Re-collecting top-10 ladder replays only matters if the
ladder distribution has shifted significantly from the 2026-05-25
snapshot used here. As of 2026-06-02, no evidence it has.

---

## Compatibility prerequisites

The target chooser must have these landmarks for the VH wiring to
land cleanly:

| Required landmark | What it must contain | Where (on 00JzI) |
|---|---|---|
| `prerank` list | rows of `(cheap_delta, src, tgt, ships, angle, eta_hint, prop_horizon, wait_N)` where `src`/`tgt` are Planet objects with `.id`, `.owner`, `.x`, `.y`, `.ships`, `.production` | Built by `score_candidate_v4`'s pre-loop. |
| Candidate-scoring loop with `score` accumulator | A scalar `score` value updated per (src, tgt, ships, angle, wait_N) candidate before final emit | Inside the main `for candidate in prerank:` loop in `chooser_trajectory.py`. |
| `_TIER_REGISTRY` in `lib/opp_model.py` | A dict registering Tier 2 as `trained_logreg_policy` | Already present on 00JzI (line 151). |

If any landmark differs (e.g., 00JzI's chooser_trajectory.py renames
`score` to `total` or removes the prerank's wait_N column), edit the
snippets to match before pasting. Both snippets are <30 lines each;
the variable names are the entire surface.

---

## Verification smoke (target branch, post-port)

```bash
# A. Parity at all λ=0 / Tier 0 (default env)
python -c "
import os
os.environ['BASELINE_VH_LAMBDA'] = '0.0'
os.environ['BASELINE_OPP_TIER'] = '0'
from agents.baseline.main import agent
print('parity import OK')
"

# B. Tier 2 active + VH active
python -c "
import os
os.environ['BASELINE_VH_LAMBDA'] = '1.0'
os.environ['BASELINE_OPP_TIER'] = '2'
from agents.baseline._value_head import vh_is_enabled, vh_get_lambda
from lib.opp_model import trained_logreg_policy, _load_opp_booster
assert vh_is_enabled() and vh_get_lambda() == 1.0
assert _load_opp_booster() is not None, 'booster load failed'
print('plug-ins active OK')
"

# C. One full game wall-clock (sanity check; expect < 5 s for 500 turns)
BASELINE_VH_LAMBDA=1.0 BASELINE_OPP_TIER=2 \
  python fast.py play agents/baseline_<target_wrapper>

# D. Bundle round-trip (Rule 46)
python scripts/bundle_agent.py agents/baseline_<target_wrapper>
pytest tests/test_bundle.py -q
python fast.py play submissions/baseline_<target_wrapper>.py
```

---

## Provenance

Source-of-truth files on hqNVM (for cherry-picking individual updates):

- `lib/_validator_tree_walker.py` — origin commit when this folder was staged.
- `lib/opp_features_lite.py` — added in the Phase 6c distillation work
  (2026-05-31). Vectorized 30-d encoder, derived from the 45-d
  `lib/shot_features.py` by dropping WorldModel-dependent features.
- `lib/value_head_features.py` — added in the B.2 reframe (2026-05-29).
  14-d base encoder; 15th feature (`leaf_delta`) is injected in the
  chooser scoring loop.
- `agents/baseline/_value_head.py` — added in the B.2 reframe.
- `data/opp_distill/distill_booster.txt` — distilled on 50k positive
  labels from 775 top-10 Kaggle ladder 2P replays (2026-05-31).
- `data/value_head/value_head_model.txt` — CRN-paired self-play labels
  from `baseline_pv_eta`-vs-`baseline_pv_eta` (2026-05-30). **This is
  the agent-specific artifact; retrain for non-pv_eta targets.**

For the full evidence trail, see:
- `audit/2026-05-31-postmortem-tier2-falsification.md` (Tier 2 v1 → v2)
- `audit/2026-05-29-pveta-probe-data/` (B.2 reframe genesis)
- `state/MULTI_BRANCH.md` (rolling-pair μ history)
