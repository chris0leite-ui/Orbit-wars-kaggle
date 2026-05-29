# Nearest opp-model preview A/B — directional null at n=5

Date: 2026-05-29 PM3 (this session).
Hypothesis: a nearest-target opp policy in the chooser's rollouts may
be more predictive of live-ladder opp behavior than the current
ROI-greedy `lite_greedy_policy`, addressing the "expects opponents
from everywhere" diagnosis from PM3.

## Setup

Three policies routed through `agents/baseline/chooser._select_opp_policy`:

- **lite_greedy_policy** (status quo) — ROI-greedy: target =
  argmax(prod / (d + 1)).
- **nearest_opp_policy** (candidate, new) — distance-only: target =
  argmin(d). Identical scaffolding to lite_greedy (ships ≥ 10 source
  gate, capture-size estimate, affordability skip, neutral defender
  non-accrual). Diff is one line of scoring.

Env-var routing: `BASELINE_OPP_MODEL=nearest` selects the new policy
when no other agent-process shares `os.environ` (subprocess workers).

## What was run

### 1. Anchor vs v7_0_drop_one — both at the ceiling

| Variant | Seeds | Result | Wilson 95% |
|---|---|---|---|
| `BASELINE_OPP_MODEL=lite_greedy` | 5 | 5/5 | [0.566, 1.000] |
| `BASELINE_OPP_MODEL=nearest`     | 5 | 5/5 | [0.566, 1.000] |

v7_0 is too easy at n=5; both variants saturated. No signal.

### 2. Asymmetric anchor vs anchor (the actual test)

Direct n=5 anchor-internal-lite_greedy vs anchor-internal-nearest
using `submissions/baseline_pv_eta_nearest_opp.py` (gitignored; baked
copy of the anchor bundle with `_select_opp_policy` returning
`nearest_opp_policy` unconditionally). The bake is required because
both agents in `env.run([p0, p1])` share `os.environ` so a parent-shell
env var cannot route them asymmetrically.

| Seed (archetype-stratified) | Winner |
|---|---|
| 2083 | P1 (nearest) |
| 1649 | P1 (nearest) |
| 5199 | P0 (lite_greedy) |
| 3233 | P0 (lite_greedy) |
| 3493 | P0 (lite_greedy) |

**Final:** P0 (lite_greedy) wins 3/5, Wilson 95% [0.231, 0.882].

### 3. Asymmetric anchor (lite_greedy) vs anchor (top_tier_mirror) — clean null

Same paired-seed protocol, baked bundle path:
`submissions/baseline_pv_eta_mirror_opp.py` (gitignored;
`_select_opp_policy` returning `top_tier_mirror_policy`).

| Seed | Winner |
|---|---|
| 2083 | P0 (lite_greedy) |
| 3493 | P0 (lite_greedy) |
| 5199 | P0 (lite_greedy) |
| 3233 | P0 (lite_greedy) |
| 1649 | P0 (lite_greedy) |

**Final:** P0 (lite_greedy) wins 5/5, Wilson 95% [0.566, 1.000].
Lower bound clears 50% — cleanly directional even at n=5. Matches the
prior-session code-comment intuition that top_tier_mirror is "too
attack-biased" as a rollout opp model.

## Verdict

**Directional null at n=5.** The point estimate is wrong-direction
for promoting nearest (lite_greedy ahead 0.600), and the Wilson lower
bound 0.231 is below the [0.45, 0.55] preview-null band, so this
isn't a clean falsification either. The honest read: n=5 is too
thin to call, and the cheap variant of the opp-model axis doesn't
look like a winner.

Decision: **DO NOT** escalate "nearest" to n=32. Reasons:

1. Point estimate wrong-direction (60% lite_greedy).
2. n=32 anchor self-play ≈ 90-120 min wallclock to firm up a probably-null direction.
3. Stronger untested axis variants exist:
   - `top_tier_mirror_policy` (v3.5.1-style aggressive snipe) — `BASELINE_OPP_TIER=1` route shipped but never A/B'd vs anchor.
   - `mlp_validated_policy` (trained 3-MLP shot filter) — `BASELINE_OPP_MODEL=mlp` route built today AM, A/B'd against the regressed leaf_pv_2p baseline, never vs the current anchor.
   - Both are bigger structural changes than swapping `prod/d` for `1/d`.

## What this falsifies (preview-grade)

The simplest opp-axis variant ("just pick the closest target") is
directionally weaker than ROI-greedy in this paired-seed test. The
underlying assumption — "live opps act more like 'grab what's close'
than 'compute ROI'" — is not supported by n=5 evidence here. The
two seeds where nearest won (2083, 1649) might cluster in specific
archetypes worth inspecting if we want to retest a hybrid (e.g.
nearest-when-close-else-ROI), but the simple variant alone is
preview-falsified.

## Artifacts

- Code: `lib/opp_model.py::nearest_opp_policy`, dispatch in
  `agents/baseline/chooser.py:50-57`.
- Tests: `tests/test_opp_model_nearest.py` (5/5 green).
- Baked bundle: `submissions/baseline_pv_eta_nearest_opp.py` (gitignored;
  copy of anchor with `_select_opp_policy()` returning
  `nearest_opp_policy` unconditionally).
- Logs (gitignored): `logs/ab_lite_greedy_vs_v7_0_n5.log`,
  `logs/ab_nearest_vs_v7_0_n5.log`,
  `logs/ab_lite_greedy_vs_nearest_asymmetric_n5.log`.
- Plan (off-tree): `/root/.claude/plans/run-5-seeds-for-compiled-dream.md`.

## What could come next on the opp-model axis

- `top_tier_mirror_policy` baked vs anchor, n=5 paired-seed.
  Larger behavioral diff, ladder-realistic.
- `mlp_validated_policy` baked vs anchor, n=5 paired-seed.
  Already-built code; bench cost (5-10× lite_greedy) might push past
  the 1000ms env cap and need throttling.
- Hybrid nearest-when-close (e.g. `if d < D_THRESHOLD: nearest else
  ROI`) — only worth pursuing if the seed-2083/1649 archetype reads
  argue for it.
