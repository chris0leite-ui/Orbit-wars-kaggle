# Postmortem — opp-mirror + opp-aware defense shipped (sub 53384340)

**Date:** 2026-06-05.
**Session branch:** `claude/champion-ml-graft-majestic-storm`.
**Live result:** sub 53384340 (`producer_plus_multi_opp_def_on.py`) settled in
TrueSkill warmup at μ=947.7 after 4/4 wins (100 % winrate, mix of 2P and 4P
public matches).

## What was shipped

The first opponent-aware mechanism on the producer_plus track.

1. **Producer-mirror opponent projection.** `predict_opp_launches_via_mirror`
   runs Producer's own planner from each opp seat with `background=None`
   (one-step best response). Replaced the inherited ROI-greedy projector
   from `lib/joint_solver/opp_projection.py` which modelled the wrong
   agent (ROI-greedy ≠ Producer's competitive-score-greedy).
2. **Per-turn roi_threshold renormalisation.** `_score_do_nothing` helper
   scores an empty-my-launches candidate against the opp_proj background;
   roi_threshold shifts by that amount so "fire if you gain ≥ 1.5 ships
   over not firing" semantics survive the opp-aware scoring's compression.
3. **Opp-aware defensive shortlist (Approach A).** `friendly_flip_targets`
   now augments its flip set from the opp_proj background -- a planet is
   marked as flipping at `ceil(eta)` if a valid opp launch in the background
   targets it (currently mine) with enough ships to crack the projected
   defender. Defensive lane reacts to opp's predicted attacks, not only
   to fleets already in flight.
4. **Bundler + callback decoupling.** `predict_opp_launches_via_mirror`
   takes `plan_fn` as a callback parameter instead of importing
   `agents.producer_plus.main` -- the latter does not exist in the bundled
   submission namespace, causing the original sub 53384019 to ERROR.

Local evidence at submit time: n=32 seat-balanced vs producer = 24/32 = 75 %,
Wilson 95 % CI [0.58, 0.87]. Wilson-lo 0.58 clears Rule 45 0.55 gate.
Per-seat: 10W/6L P0, 14W/2L P1 (defense lifts P1 strongly).

## What progressed the science

The session converged on three coupled findings:

1. **The static-opp scorer was the right diagnostic frame.** Opp_proj
   alone (no defense, no multi_size) was parity with no projection
   (11/16 vs 11/16). The lift came from composing opp_proj with multi_size
   and adding the defensive augmentation. The mechanism only earns its
   keep when it unlocks other parts of the planner.
2. **Coalitions are dormant in this regime.** Diagnostic trace at seed 7
   showed coalitions firing as the best candidate exactly once across 150
   `_greedy_select` calls, and the one time it did its score was below
   threshold. The kitchen sink (multi + coal + opp_proj) lost one win
   relative to multi + opp_proj. Coalitions stay OFF in production until
   we find the regime where they fit (likely defensive coalitions for
   reinforcement; deferred).
3. **The remaining defect is structural: 18-tick horizon vs ~50-tick
   cycle.** Validation game 78807326 (4P self-match) ran 500 steps in a
   perfect static-rotation stalemate. Each agent owns 2 planets at any
   tick, ship counts cycle 3 → 6 → ... reset, launches fire in pairs every
   ~11 turns. PI insight: the high-prod planet should stockpile until it
   has overwhelming force. Our scorer can't see this because (a) horizon
   too short, (b) roi_threshold fires on any positive-margin attack within
   18 ticks.

## What was attempted and rolled back

**H=36 horizon bump.** Hypothesis: longer scorer view → see recapture leg →
stockpile correctly. Result: 2/16 wins (Wilson [3.5 %, 36 %]) -- catastrophic
regression. Root cause: opp_proj projects only ONE turn of opp launches
(etas in [1, 8]). With H=36, the scorer sees opp doing nothing for ticks
9-36 -- 28 of 36 simulated ticks under the static-opp assumption we were
trying to fix. My candidates look artificially great over the longer
window; planner over-commits to attacks the real opponent will counter
outside the projection.

**Implication.** Bumping H alone amplifies the static-opp defect rather
than fixing it. The horizon bump only works in concert with multi-tick
opp_proj (extending opp_proj's prediction window from 1 turn to N turns).
That is the deferred work. Env knob (`PRODUCER_PLUS_HORIZON_2P/_4P`)
stays committed but defaults OFF; can be re-enabled once multi-tick
opp_proj lands.

## What is locked for later sessions

- **ML next-steps**
  (`knowledge-base/thoughts/2026-06-05-ml-next-steps-locked.md`):
  value head ≥ opp policy distill ≥ defensive urgency learner ≥ direct
  policy net. Existing `data/value_head/` and `data/opp_distill/` are
  decoupled from producer_plus; would need retraining on current-strategy
  self-play data.
- **Cycle-detection band-aid:** PI explicitly declined this path
  ("we will not use band-aids for these situations"). Skip.
- **Multi-tick opp projection:** the proper fix for the stalemate. Project
  opp's launches at game-ticks 0, 1, 2, ..., K, not just tick 0. Cost:
  recursive simulation + per-tick plan calls; ~K x current opp_proj cost.
  Enables the H bump that empirically regressed today.
- **Defensive coalitions:** revisit only after the stalemate is broken.
  The original "coalitions naturally emerge as defensive reinforcement
  candidates" hypothesis remains untested but is gated on multi-tick
  opp_proj first surfacing the heavy threats coalitions would address.

## Frictions surfaced

- **Bundler vs source-tree import asymmetry.** `from agents.producer_plus.main
  import plan_lite_waves` works in the source tree but breaks in the
  bundled namespace. Pattern: pass functions as callbacks across modules
  that will be inlined into a single bundle. The first submit (sub 53384019)
  ERRORed because of this. The lesson is general: every cross-package
  import in `orbit_lite/` is a future bundler risk. A pre-submit smoke
  via the bundled `submissions/*.py` (not just the source tree) would
  have caught this. Worth promoting as a Rule extension.
- **Env-var leak across producer_plus shims.** When `fast.py play X --vs Y`
  loads two producer_plus variants into the same parent process, the
  shim-set env vars cross-pollinate via `multiprocessing.Pool` fork
  inheritance. Direct head-to-head between variants is therefore unsafe;
  comparison must go through the vanilla `producer` baseline.
  Documented; no fix this session.

## Validation game 78807326 -- archived for future ref

Self-match (all 4 seats us). 500-step cap reached. 4-way tie. Each agent
cycled 2-planet ownership in lockstep. Launch cadence pair-every-11-turns
on the high-prod planets (P20-P23). Ship counts oscillate 3 -> 6 -> launch
-> 3. Replay at `audit/live-episodes/53384340/episode-78807326-replay.json`
(gitignored; pull with `python -m scripts.live_episode_summary 53384340
--pull`).

## Rules check at session end

- Rule 1: single submit per kaggle-submit invocation. ✅
- Rule 12: 5/day budget. Used 2 today (53384019 ERROR + 53384340 success).
- Rule 32: session-start `git fetch` ran. ✅
- Rule 35: PI thoughts archived (`knowledge-base/thoughts/`). ✅
- Rule 36: session-end second-brain entry (this file + the ML next-steps
  doc). ✅
- Rule 38: H knob defaults OFF, OFF parity bit-identical to before. ✅
- Rule 39: no session URLs in commits / artifacts. ✅
- Rule 42: push-claim board row added (`state/MULTI_BRANCH.md`). ✅
- Rule 45: Wilson-lo 0.58 cleared 0.55 gate. ✅
- Rule 46: bundle + tests + smoke green at submit time. ✅
