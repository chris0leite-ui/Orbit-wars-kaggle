# 2026-06-02 — ML on champion base: hypothesis, design rationale, deferrals

_PI second-brain entry (Rule 35). Session `claude/champion-ml-graft-majestic-storm`._

## The hypothesis under test

Two ML components built on `hqNVM` had failed badly on the live ladder:
`baseline_pv_eta_vh_dist_composite` settled at μ=526, `baseline_pv_eta_vh_dist_slotres` at
μ=688. Both were the same ML stack — distilled-ladder Tier-2 opp predictor +
B.3 value-head additive — sitting on `baseline_pv_eta` (a learned-reward-tuned
chooser that lacks joint coalitions and the neutral-grab early tilt).

The asymmetric isolated A/Bs on hqNVM had said: Tier-2 alone is +14 μ, VH alone
is +5-10 μ. Yet shipped together they killed the agent. The reconciliation:
**ML refines the candidates the chooser considers, and a chooser making bad
candidate-class decisions can't be rescued by refinement.** Pv_eta over-allocated
to defense (87% of emits per the slotres probe) because its value function
mis-prioritised; Tier-2 sharpened the opponent simulation but the chooser was
already pointing at the wrong moves.

Hypothesis-under-test this session: **graft the same ML pieces onto a chooser
that is already making competent class-of-decision choices, and the per-decision
refinement should net positive.** The strongest such chooser on 00JzI is
`launch_rules_universal` (historical peak μ=1183.7, settled this week at 1110.6).

## Why two wrappers, not one

The plan agent's critique surfaced that bundling Tier-2 and VH into one wrapper
couples two independent lift sources. If the wrapper underperforms, we can't
attribute which piece failed. Two wrappers = one independent lever per
submission slot = cleaner Rule 42 risk posture. **First submit: Tier-2 only.
Second submit (next session): Tier-2 + VH.** This matches the "single-variable
eviction" discipline.

A side benefit: the VH retrain takes 3-4 h CPU. Tier-2 alone gives us a 24-h
read on whether the broader ML-on-champion hypothesis holds. If Tier-2 doesn't
transfer, we skip the retrain entirely. **If it does, we pay the cost knowing
it'll pay back.** Compute is gated by evidence.

## Why VH stayed dormant (λ=0)

The shipped VH model was trained on `baseline_pv_eta` self-play. Its CRN-paired
labels encode "ship-delta over K=10 turns given the *pv_eta* chooser's
downstream play." Joint-aggressive coalition candidates that
`baseline_joint_aggr_*` fires were not in its training distribution. Activating
it at λ=1.0 would systematically under-rate those candidates. The choice was
between:
- Ship VH at λ=1.0 untrained — measurable but probably negative lift.
- Ship VH at λ=0 — single-variable A/B for Tier-2 + makes the second submit
  the natural pivot for VH-on-correct-data.

The latter wins on two grounds: cleaner attribution, and the VH retrain becomes
the **biggest single unrealized lift on the table** for the next session.

## The joint-path patch I deliberately deferred

Block B (the per-candidate VH additive term) was inserted only into the solo
scoring path of `score_candidate_v4`. The joint path at
`score_candidate_v4_joint` (lines 1250 and 1414) does NOT get the same patch.
Reason: `vh_featurize_prerank` (in `agents/baseline/_value_head.py`) builds
its feature dict keyed on solo prerank rows. Joint candidates have legs with
*different* `(src_id, tgt_id, ships, angle, wait_N)` keys, so the cache lookup
would miss and `vh_predict_one` would return 0.0 for every joint candidate.

Three options for properly handling joints:
1. **Sum-of-legs** — apply VH to each leg of the joint and sum. Over-weights
   joints relative to solos (each leg gets its own full VH credit).
2. **Average-of-legs** — divide the sum by 2. Magnitudes match solos.
3. **Primary-leg-only** — apply VH using one leg's features. Biased toward
   that leg's situation.

For this session, the cleanest call was: no joint patch, because **the wrapper
ships at VH_LAMBDA=0 anyway**, so the joint-path code path runs dead. The
asymmetry only matters when VH activates. Documented in code comments and
flagged in HANDOVER as a hard-prerequisite for any VH-enabled submit.

The harder underlying question: `vh_featurize_prerank` itself doesn't know
about joints because joints are CONSTRUCTED INSIDE the chooser, not in
the proposer's prerank. To VH-evaluate them properly the featurizer needs
joint enumeration too — and that's a non-trivial refactor of the prerank
data model. The "two options" listed in HANDOVER (extend featurizer vs
on-the-fly inside chooser) are both engineering tradeoffs of the underlying
data-flow question.

## Why I extended `DEFAULT_LIB_ORDER` but not via wholesale script copy

The plan agent flagged "extend `DEFAULT_LIB_ORDER` to include the three new
modules" — easy to forget when cherry-picking from hqNVM. The composite
bundler had its own `required` list, but `bundle_agent.py` (the generic
bundler that runs by default) wouldn't have inlined the new modules if I
hadn't edited its `DEFAULT_LIB_ORDER`. I chose this small edit over copying
hqNVM's full build script because:
- The edit is 10 lines, well-localised, and documents what the new modules
  are for.
- The composite bundler is the right entry point for any ML-laden agent
  going forward; future ML-composite wrappers don't need their own build
  script.

## Decisions I'd revisit if I had more time

- **Probe size.** `--max-seeds 16` got auto-bumped to 32 by `--geometry-panel`,
  not the 128 I initially expected. At n=32, Wilson [0.483, 0.796] is wide
  enough that I'd want n=64 minimum for a real submit decision. We submitted
  on PI override; settling will resolve it either way, but the formal probe
  was thinner than ideal.
- **TLE max=1308 ms.** I diagnosed it as a 1-in-8000 outlier and moved on.
  A more thorough check would have run the per-turn timing instrument I
  wrote (`/tmp/per_turn_timing.py`) across 4-8 seeds to find which late-game
  states drive the spike. Skipped for time; flagged as a follow-up.
- **The Tier-2 disagreement rate.** Cheapest, most-informative diagnostic
  I didn't run: how often does `trained_logreg_policy` predict different
  emits than the `lite_greedy_policy` fallback would? If the answer is
  "rarely," much of the +14 μ "Tier-2 lift" is noise. 5-minute check that
  belongs in the next session.

## Open questions for the next session

- Does Tier-2's lift transfer when stacked on top of `BASELINE_ADAPTIVE_K=1`?
  (Adaptive-K changes the candidate distribution by widening early-game K;
  Tier-2 was distilled on top-ladder replays that may not match this
  distribution.)
- Is the joint-path VH problem solvable by extending the featurizer's input,
  or does it require the chooser to maintain its own joint-feature cache?
- What's the right tier-2 threshold for the champion-base chooser? 0.15
  was hand-picked for pv_eta. A 30-min A/B sweep across {0.10, 0.20, 0.25,
  0.30} would settle it.
