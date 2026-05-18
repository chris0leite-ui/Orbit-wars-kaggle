# PV term is structurally right but introduces calibration debt

Date: 2026-05-18 (claude/audit-workflow-performance-btjeK PM session)

## The observation

The chooser's leaf-value head adds a `(my_prod − opp_prod) × pv` term
to the base ship-delta. Without it, captures of equal-production
planets score Δ = 0 against an idle baseline (both owners produce
at the same rate; ship counts cancel out symmetrically). The sanity
oracle (`test_oracle_sanity_trivial_capture`) was designed to flag
exactly this failure mode and DID pass once the PV term was added.

But: three independent A/Bs vs the pre-fix bundle all regressed to
≈ 40% winrate (vs 50% baseline) regardless of whether the PV term
came with a per-fleet counterfactual credit (v1: 40.6%), alone (v2:
39.6%), or with rollout-defense as a "compensating" fix (option 5
v2: 39.6%).

## What this means

The PV term is **right** in the sense that it makes the value head
correctly distinguish leaf-states where we own a captured planet
from those where we don't. But the chooser's emit gate (`Δ > 0`) is
calibrated against the **pre-PV** scale of the base ship-delta. PV
adds ~100 value units per captured planet at leaf; chooser pre-fix
was emitting based on Δ swings of ~20-50 (ship counts only). With
PV active, every capture candidate gets +100 → emit-by-default →
over-emission → drained sources → losses.

This is the same class of issue as bug #15 v1's
"double-counting" — but the cause isn't double-counting per capture
(per-fleet credit only fires on in-flight at leaf, PV credits arrived
captures at leaf — disjoint regimes). The cause is **calibration
mismatch**: an upgraded value-head is being judged by a downstream
chooser that expects the old value-head's scale.

## The lesson

Adding a structurally-better signal to a value head is necessary but
not sufficient. The downstream policy (here, the chooser's emit
gate) must be re-calibrated against the new signal scale. Without
that, the structurally-better signal becomes a structurally-worse
agent.

This is a general pattern. Whenever you bolt a new term onto a
value function inside an existing greedy/threshold policy:

1. **Measure the term's magnitude** on real game states (not just
   synthetic oracles). What's `+ PV` adding to a typical leaf
   evaluation? Is it 5% of the base, 50%, 500%?

2. **Re-fit the policy's threshold** to the new value-head's scale.
   For the chooser, this would mean either: (a) shifting `Δ > 0`
   to `Δ > shift`, where `shift` = average PV contribution of an
   uninteresting capture; or (b) normalizing the leaf-value head
   so PV is in the same range as the base.

3. **A/B before declaring victory on the oracle**. Sanity oracles
   detect structural bugs (capture credit = 0 is structurally wrong)
   but they can't detect calibration debt (capture credit too big
   is structurally right but tactically wrong).

## Practical decision for next session

Option A: disable PV in production via the `_COMPOSITE_PV_ENABLED`
kill-switch. Sanity oracle reverts to xfail; chooser returns to its
pre-#15 ~50% baseline. The PV-term-in-base remains in code; can be
re-enabled later if we want to invest in the chooser-recalibration
work.

Option B: invest in chooser recalibration. Identify the gate that
needs adjusting; sweep over thresholds; A/B at each. Probably 2-3
session-days of work for a 5-15μ improvement IF it works.

For now (next session), I'd recommend A. The PV term's diagnostic
value (sanity oracle passes) is real but the cost (40% winrate) is
not worth carrying in production. We can revisit B once the bug #14
follow-ups (source-defense penalty at leaf, etc.) are tried — those
may be cheaper alternatives that bypass the calibration debt.

## Pointers

- `lib/value_heads.py:160-200` — composite_capture_value v2, PV term.
- `lib/value_heads.py:177` — `_COMPOSITE_PV_ENABLED` kill-switch.
- `audit/2026-05-18-postmortem-bug-15-v2-and-bug-14-option-5.md`
  — full postmortem with A/B numbers.
- `audit/friction.md` `tag: wrong-root-cause-from-symptom-similarity`
  — what we should have done earlier (run the kill-switch FIRST).
