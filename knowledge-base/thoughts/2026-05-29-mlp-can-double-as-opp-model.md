# 2026-05-29 — PI insight: the trained validator MLP can double as the opponent model

Source: PI dialogue, 2026-05-29 PM, branch `claude/kaggle-submission-review-gZsCu`.

## Setup

Today's session implemented the opponent-policy rate-cap (top-K
launches per tick). All three values of K passed the v4_planner
smoke (ceiling effect) but the joint leader K=2 lost 10/16 against
the current live champion `baseline_leaf_pv_2p`. The cap regresses.

I drafted four alternative design directions: smarter single
heuristic; mirror-self predictor; ensemble of predictors; measure-
first calibration. I proposed (4) measure-first as the cheap next
step.

## The insight

PI asked: "the last submission decides based on a trained MLP — that
should be fast and suitable as a smart opp model, right?"

I had missed it because I was reading only this branch's
`submissions/` directory. The sibling branch
`claude/competition-objective-alignment-hqNVM` shipped sub 53131296
two days ago — a 3-MLP ensemble trained on 5366 labeled shots from
top-10 + midpack replays, classifying `P(launch will succeed)`. Its
existing job is filtering OUR-side candidate launches (reject if
`P < 0.30`).

PI's reframing: the same network, queried with the opponent's seat,
becomes a learned opp-model predictor. It is exactly the "smarter
single predictor" from the option list — except the smart rule is
*learned from the right corpus* rather than hand-coded.

## Why this is structurally correct

The K-cap was an attempt to calibrate the opponent model to "real
top-10 fire rate" using a fingerprint number (≈1.3 launches/tick).
We never verified that number, and we never asked "what features
predict whether a top-10 player actually fires?" The MLP's training
set answers exactly that question. Using it as the opp model means:

- The opp-model rule is no longer "fire if ROI > c1 OR top-K by ROI"
  but "fire iff `P(success | features) ≥ τ`."
- The threshold τ is a single tunable scalar on a calibrated
  probability, not on an uncalibrated ROI score.
- The opp behaviour automatically mirrors top-10 selectivity:
  garrison-at-launch averages, enemy-target fraction, etc. are
  implicit in the trained boundary.

## Why I didn't see it

Two reasons, both about how I read state:

1. **I checked the wrong directory for "the latest submission."** I
   looked at `submissions/*.py` sorted by mtime, found
   `baseline_leaf_pv_2p.py` (last build on THIS branch), and called
   it "the latest." The actual most-recent Kaggle submission came
   from a sibling branch — `kaggle competitions submissions
   orbit-wars` is the source of truth, not the filesystem.
2. **The sibling branch's MLP work was not in this branch's
   HANDOVER.** The PM3 ADDENDUM at the top of HANDOVER.md mentions
   the opp-model pivot but does not mention that a related learned
   predictor is already live. `state/MULTI_BRANCH.md` would have
   surfaced it if I had read it under Rule 44 before proposing
   variants. I read HANDOVER's "Item 3 — Opp-model spatial
   restriction" and went straight to building it.

Friction to file under `read-multi-branch-md-before-opp-model-design`.

## Implications for the rest of the comp

- The MLP is reusable. Two orthogonal uses on the same network:
  validate our shots (already in sub 53131296, ~+10pp directional
  local but inconclusive on live) AND model the opponent.
- The substrate cost is mostly already paid: corpus generated,
  weights trained, inline blob encoded, feature encoder symmetric
  in seat. Wiring is ~1-2 hours.
- If MLP-as-opp-model lifts, the natural follow-on is to layer the
  own-side validator filter on top of it — single network, both
  uses. That would unify two of the sibling branch's wins into one
  build on our branch.
- If MLP-as-opp-model does NOT lift, the result is itself
  informative: it tells us that calibrated-to-top-10 opponent
  modelling is not sufficient against the current μ ≥ 1100
  ladder, and pushes us toward distributional models (mixture of
  styles, ensemble of predictors).

## Open question for next session

The validator's μ=1086.1 is below its base predecessors (`pv_eta`
1157.2). Is that because the τ=0.30 own-side filter throws away
shots that *would have* succeeded on the live distribution but
not on the training corpus? If so, the opp-model use (which
queries the *same* classifier from the *opposite* seat) could
inherit a similar miscalibration.

This is the right diagnostic to run alongside the build: instrument
the MLP's per-emit `P(success)` on a small live-vs-self-play
trace, compare distributions, and see whether the classifier is
producing the same calibration on both sides.
