# Early-capture-opportunity gate — proactive-garrison axis decision (2026-06-02)

PI question: "early garrison ships look wasted — has the drain idea been tried
thoroughly, and might it work now though it failed earlier?"

## What was re-examined
- H1 `drain_idle_rear` (2026-05-18): n=32 → 11/32 = 34.4%, Wilson-lo 0.204, FAIL
  (+ 1528 ms over budget). Relocated rear idle ships to *closer owned* planets →
  pure defense cost, no capture-EV.
- Spatial-leaf value head: failed same root cause (commit b5f5296).
- Both are the SAME axis: force-emit reserves. The PI's idea (early expansion
  tempo) was argued to be different (capture-oriented, early-only) → built a
  fresh data gate instead of arguing.

## Gate: `scripts/early_capture_gap.py` (steps 1-20, HORIZON=25, 32 champion games)
Counts DECLINED AFFORDABLE CAPTURES per early step: an uncaptured neutral a single
friendly source already has enough idle ships to take, reachable in ≤25 turns, not
covered by an inbound friendly fleet — left idle instead of launched.

| cohort | declined/step (cover-aware) | max_idle_source | nearest_neutral_cost |
|---|---|---|---|
| ALL (32) | 2.22 | 13.6 | 30.1 |
| WINS (16) | 2.32 | 14.0 | 29.0 |
| LOSSES (16) | 2.13 | 13.1 | 31.2 |

### Per-step trace (one game) overturned two priors
Maps have **many neutrals with a wide cost spread** (9,10,15,21,29,44,46,86), not
uniform ~44. **Cheap reachable neutrals exist from turn 1.** Champion opens with one
planet at 11 ships, accumulates 11→13→15→18 over steps 1-8 while a **cost-10 neutral
sits 6 turns away**, then launches (~step 8-11). So the early hold is:
- NOT "idle army wasted" (idle is small, 11-18 ships), and
- NOT pure "can't-afford stockpiling" (the H1-audit framing) — cheap captures ARE
  affordable from turn 1.
It is a **~7-10 turn delay on the first cheap expansion it could already afford.**

## Mechanism (code-confirmed)
Not a keep/reserve floor — `_keep_floor_from_threat` returns ~0 when the enemy is
far, so `max_sendable ≈ budget`; the proposer DOES emit the cheap-capture column.
The delay is a **deliberate CHOOSER decision: a wait-band variant out-scores
fire-now in chooser Δ, so the chooser reserves the source and waits** (documented at
`proposer.py:323-329`). I.e. the value function prefers a later/bigger capture over
the immediate cheap one. Making the opening eager = an **early-game fire-now value
bump / wait penalty** = a VALUE-HEAD / chooser calibration change.

## Verdict: POSITIVE-but-non-discriminating → NOT a reflexive build
1. **Rule 41 confound:** wins (2.32) and losses (2.13) show the SAME early-delay
   rate. If slow expansion cost us games, losses would show MORE, not less. The
   delay does not discriminate outcomes in these replays. (Caveat: replays are
   champion-vs-champion-derived → both sides delay symmetrically; the metric cannot
   detect a *universal* inefficiency a genuinely eager opponent would punish. The
   only way to settle that is an eager-variant A/B vs champion.)
2. **Rule 44 closed-adjacent:** the fix lives in the value-head / chooser-Δ axis,
   which is in Closed Tracks (spatial-leaf + value-head aggregators falsified).
3. STAGNANT_DRAIN (the existing OFF flag) tests the REAR-DRAIN axis (H1's cousin),
   NOT this early-timing finding — so it is the wrong cheap probe for this result.

**Recommendation:** the premise is partly real (the opening is ~7-10 turns slower
than it could be), but two independent discipline signals say it is a low-expected-
value build and the loss driver is elsewhere (consistent with 2026-06-01
"conversion not volume" + "lose the step-50-100 expansion race in CERTAIN maps").
Decision (build narrow early-fire-now lever vs pivot to conversion gap vs drop) is
PI's per the plan gate — surfaced with the stack A/B result.
