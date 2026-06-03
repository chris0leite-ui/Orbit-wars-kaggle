# Question — what's the right target for a learned head on this chooser?

**Date:** 2026-06-03
**Open:** yes

The current chooser computes a PV-discounted ship-delta integral over
10-30 turns via `chooser_trajectory.score_candidate_v4`. Any learned
head that targets a subset of this integral (e.g. K=10 ship-delta) is
trying to predict a *quantity the chooser already computes better*, and
testing today confirmed this is futile (6 falsifications;
`audit/2026-06-03-vh-axis-closure.md`).

For a learned head to add value, its target must carry information the
chooser CANNOT compute. Candidate targets, in rough order of
hypothesised information gain:

1. **End-of-game seat-share** (production-share at step 500). Trivially
   labelable from existing replays; high variance but high information.
   Reach-frontier doctrine (`knowledge-base/concepts/reach-frontier-doctrine.md`)
   argues this is the right objective; the rerank failure here would not
   apply because rolllouts can't see step 500.

2. **Opponent-response prior** (multinomial over opp next-N actions
   conditional on our candidate). Replaces/augments the hardcoded
   `lib.opp_model` policy. Overlaps with Tier-2 distillation work
   (currently parked).

3. **Capture-realisation probability conditional on aware opponent.**
   `predict_fleet_fate` gives a kinematic answer that ignores opponent
   reactions. A learned head could capture the conditional realisation
   rate. Narrow but high-leverage if opp-aware physics-waste is large.

4. **Opening-prior** (first-50-turns advantage scoring). Chooser's
   PV-discount has minimal data early game. Restricted scope; could
   be paired with the `chooser_trajectory` opening branch.

Q: Which of these is the right next bet given the deadline (2026-06-23)
and the closed VH-K10 axis? Or pivot entirely away from learned heads?

Recommendation: option 1 (end-of-game seat-share) is the most aligned
with the comp metric (TrueSkill on win/loss) but takes the longest to
get right. Option 2 is the cleanest extension of work already started.
Option 4 has the narrowest scope and best blast-radius if it doesn't
work.

Defer decision to next session opener; the current session closes the
existing VH axis without a new direction nominated.
