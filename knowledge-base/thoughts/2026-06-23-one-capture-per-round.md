# 2026-06-23 — one high-confidence capture per round (final-day lever)

> Session record (Rule 36). Plain English (Rule 0). PI directive: a strategy that
> "commits to one high confidence capture attack per round at most, while
> regrouping still happens." Final day (deadline 23:59 UTC).

## Context found on entry (load-bearing)
- The assigned branch (`competition-strategy-review-hjdr6z`) was STALE — equal to
  `origin/main`, missing ~4 days of work. The real latest agent lived on
  `claude/submission-strategy-review-r48nve`. Merged r48nve into hjdr6z (favoring
  r48nve) before doing anything. **Lesson: on session start, check the live
  submissions against the branch — the newest code may be on another branch.**
- Live ladder reality (settled μ): **pure offense stack ~1112–1139** (sub 53906150,
  the four native levers: NATIVE_LEAF/REINFORCE/CONCENTRATE/OFFENSE) >
  neutral-margin ~1035 > 4P reserve package ~1044. So the 4P reserve+front-cap
  package the PI liked *by eye* actually settled BELOW pure offense on the ladder.
  The proven best is pure offense.

## The lever (PI's spec, implemented)
`LR_ONE_CAPTURE` (default OFF, byte-identical OFF path). When ON, commit **at most
one OFFENSIVE capture per round** — the single highest-confidence one, since
candidates are already value-ordered by the native flip-hazard leaf — while
DEFENSIVE regroup/reinforcement stays uncapped. Both 2P and 4P (PI chose the strict,
both-formats reading after the snowball-regression risk was surfaced).

Enforced at three layers (the first attempt — capping only the greedy commit loop —
did NOT change the emitted move, because the 2-ply pick fell back to the producer's
multi-target floor; Rule 38 caught this):
1. greedy commit loop caps committed offensive captures to `LR_MAX_CAPTURES` (=1);
2. the producer-floor move is trimmed (`_cap_emit`) where it enters the 2-ply menu,
   so what the leaf scores == what we emit;
3. a final `_cap_emit` backstop on the emitted action (keep all non-offensive
   launches + the single highest-ship offensive target) guards every code path.

Shipped config = pure proven offense + `LR_ONE_CAPTURE`; neutral-margin and the 4P
reserve package are NOT baked.

## Evidence so far
- **Rule 46 smoke green**: bundle builds, `test_bundle` 10/10, 2P max 559 ms /
  4P 233 ms, 0 turns ≥ 1000 ms. `test_one_capture.py` 4/4.
- **Rule 38 mechanism check** (real game, replay obs): ON caps to ≤1 distinct
  offensive target every turn (0 multi-target turns vs ~20 OFF); regroup preserved.
- **n=1 before/after (same map, same opponents):**
  - 2P seed 6013 vs V2: OFF **loses** (blows 2.6:1 lead, share 34→29→24→0), ON
    **wins** (monotonic 29→48→77→95).
  - 4P seed 219030400 vs V2+Roman+konbu: OFF **loses** (collapses to 0 vs 5014), ON
    **wins** (comeback 27→59→74→96).
  Both are seeds the baseline LOST; both flip to wins with monotonic production
  growth (the winning signature). n=1 — illustrative, not proof.
- **n=32 2P A/B vs V2 running** as the regression gate. CAVEAT: local games vs the
  strong V2 have historically been inert to scatter-reduction (2026-06-19, n=40) —
  the upside, if any, is on the weak ladder we cannot simulate. The A/B's real job
  here is to catch a 2P regression, not to confirm upside.

## Open / next
- Read the n=32 2P A/B (regression gate). If no 2P regression, the qualitative
  before/after + the weak-ladder thesis support an experiment-slot submit.
- Submission plan (PI deciding after watching): protect proven offense in the kept
  pair; submit one-capture as an experiment; rolling-last-2 means order matters
  (Rule 42 eviction analysis: submitting one-capture would evict offense ~1112
  unless offense is re-banked first).
- Code: `agents/least_resistance/main.py` (`_one_capture`, `_cap_emit`, commit-loop
  cap, producer-floor trim, agent wrapper). Tools: `scripts/ab_one_capture.py`,
  `scripts/probe_one_capture.py`. Test: `tests/test_one_capture.py`.
