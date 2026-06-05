# HANDOVER.md — next-session brief

## Mode

**producer_plus migration track, 2nd lift shipped (opp-mirror + opp-aware
defense). The structural ceiling has been identified.** Next session
opens with reading the settled live μ for sub 53384340, then deciding
between the multi-tick opp_proj implementation and the value-head ML
path. Both are explicitly captured in
`knowledge-base/thoughts/2026-06-05-*.md` and locked for later sessions.

## Live status (after 2026-06-05 07:00 UTC submit)

- **Newest (#1):** `producer_plus_multi_opp_def_on.py`, sub **53384340**
  (2026-06-05 06:58 UTC), 241 790 B. Producer's engine + multi-size
  enumeration (Step 4) + Producer-mirror opp projection + opp-aware
  defensive shortlist augmentation (Approach A). **Live μ in TrueSkill
  warm-up: μ=947.7, 4/4 wins, 100% winrate.** Will continue climbing
  over ~24 h.
- **Backstop (#2):** `producer_plus_coalitions_on.py`, sub **53373322**,
  live μ ≈ 1262.6.
- **Evicted by 53384340:** `producer_plus_multi_size_on` (sub 53369848,
  μ = 1282.1).
- Older submissions visible via `kaggle competitions submissions
  orbit-wars`.

## What landed this session

1. **Producer-mirror opp model.** Replaced the inherited ROI-greedy
   projector with `predict_opp_launches_via_mirror` (calls Producer's own
   `plan_lite_waves` from each opp's seat). Massive accuracy lift over
   ROI-greedy (which modelled the wrong agent).
2. **roi_threshold renormalisation per turn.** `_score_do_nothing` helper
   computes do-nothing baseline against the opp_proj background; threshold
   shifts so "fire if ≥ 1.5 ships gained vs not firing" semantics survive
   the opp-aware scoring's compression.
3. **Opp-aware defensive shortlist (Approach A).** `friendly_flip_targets`
   augments its flip set from the opp_proj background; defensive lane now
   reacts to opp's predicted attacks, not only to fleets already in flight.
4. **Bundler vs source-tree import asymmetry fix.** First submit (sub
   53384019) ERRORed because `from agents.producer_plus.main import
   plan_lite_waves` does not exist in the bundled namespace. Resubmission
   53384340 passes `plan_fn` as a callback parameter; no cross-package
   import. **Pattern to remember for any future orbit_lite code that's
   destined for a bundle.**
5. **H bump env knob (`PRODUCER_PLUS_HORIZON_2P / _4P`)** committed but
   defaulted OFF. Empirical test at H=36 regressed to 2/16 wins -- horizon
   and opp-model depth must scale together (see postmortem).

## The structural defect identified (carry forward)

Diagnosed via validation game 78807326 (4P self-match, 500-step cap,
4-way tie). Detail in
`knowledge-base/thoughts/2026-06-05-cycle-stalemate-and-horizon-scaling.md`.

- The 18-tick horizon is shorter than typical exchange cycles (~50
  turns).
- Greedy fires on any positive-margin attack within H=18, even when the
  recapture leg falls just outside the horizon.
- Stockpiling on a high-prod planet ("wait 30 turns, then overpower")
  is invisible to the scorer.
- Bumping H alone REGRESSES (opp_proj only projects 1 turn; longer H
  exposes more static-opp time).
- Proper fix: multi-tick opp projection (project opp at ticks 0, 1, 2,
  ..., K, not just tick 0) -- enables the H bump.

## Locked for later sessions

- **ML next-steps**
  (`knowledge-base/thoughts/2026-06-05-ml-next-steps-locked.md`):
  value head ≥ opp policy distill ≥ defensive urgency learner ≥
  direct policy net. Existing `data/value_head/` infrastructure is
  decoupled from producer_plus; would need retraining on
  current-strategy self-play data.
- **Multi-tick opp_proj**: the proper fix for the stalemate. Enables
  H bump.
- **Defensive coalitions**: deferred until stalemate is broken.

## PI guidance for next session

- **No band-aids.** PI explicitly declined the cycle-detection patch.
  Pick from the principled fixes (multi-tick opp_proj or value head).
- **Watch the live μ settle.** Local 75 % vs producer projects to
  ~1310 μ but the previous multi_size 62.5 % local landed at ~1282 live.
  The lift this submission represents may be smaller on the ladder than
  locally.

## Next action

1. **Read live μ for sub 53384340** after TrueSkill warm-up (~24 h).
2. **If μ ≥ ~1300:** the opp-mirror + defense mechanism is the new
   anchor. Choose the next bet: multi-tick opp_proj (code-only) or value
   head (ML). Multi-tick opp_proj is the lower-risk first step --
   directly addresses the stalemate, no training data required.
3. **If μ < ~1300:** investigate the local-vs-live gap (opponent panel
   diversity? seat bias? renormalisation interacting with non-Producer
   opponents on the ladder?). The defensive shortlist may need a
   confidence gate.
4. **Submission budget for tomorrow:** 5/day. Use 0-2 max while gathering
   live evidence on 53384340.

## Files of note touched this session

- `agents/producer/orbit_lite/opp_projection.py` -- rewritten end-to-end
  (Producer-mirror, callback-driven).
- `agents/producer/orbit_lite/planner_core.py` -- `friendly_flip_targets`
  and `build_target_shortlist` take optional `background: LaunchSet`.
- `agents/producer_plus/main.py` -- run_turn calls opp-mirror,
  renormalises roi_threshold via `_score_do_nothing`, threads `background`
  into `plan_lite_waves` -> `build_target_shortlist`. `_config_for`
  reads horizon env override.
- `agents/producer_plus/producer_plus_multi_opp.py` -- new shim.
- `agents/producer_plus/producer_plus_kitchen_sink.py` -- new shim
  (kept for archaeology / diagnostic; not shipped).
- `scripts/bundle_producer_plus.py` -- `multi_opp_def` variant added.
- `audit/live-episodes/53384340/` -- validation replays pulled, summary
  archived.
- `audit/2026-06-05-postmortem-opp-mirror-defense-shipped.md` -- session
  postmortem.
- `knowledge-base/thoughts/2026-06-05-ml-next-steps-locked.md` -- ML
  opportunities catalogued.
- `knowledge-base/thoughts/2026-06-05-cycle-stalemate-and-horizon-scaling.md`
  -- structural defect documented.
- `state/MULTI_BRANCH.md` -- push-claim row for sub 53384340.

## Rules in force unchanged

- Rule 0: plain English with PI.
- Rule 1 + 12 + 42 + 45 + 46: submission discipline.
- Rule 38: env knobs default OFF, bit-identical to baseline.
- Rule 40: modelling-correctness over restriction-tuning. **PI reinforced
  this at session end: "no band-aids."**

## Open questions (carry to next session)

1. Multi-tick opp_proj vs value head -- which to do first?
2. What K (projection depth) for multi-tick opp_proj?
3. Does the cycle stalemate show up in 2P games or only 4P?
4. Does the local 75 % vs producer transfer to live μ ~1310, or does the
   diverse ladder regress to ~1290 (as multi_size did from 62.5 % local
   to μ=1282 live)?
