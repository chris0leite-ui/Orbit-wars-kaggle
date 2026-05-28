# 2026-05-28 PM3 — macro-layer null result, opp-model pivot

## What happened this session

1. PI diagnosed (PM2 handover) that the agent is a per-move local
   optimizer with no macro layer — three symptoms: fleet sizes too
   small, rear planets don't mobilise, no early expansion.
2. Designed the "smart and simple" macro: a 4-state mission planner
   (EXPAND / STOCKPILE / STRIKE / DEFEND) layered on top of the live
   PV_ETA + LEAF_PV_2P stack, gated 2P-only, default OFF for byte
   parity. Closed-form geometry exploits two engine facts:
   - `omega > 0` always — forward direction is fixed across games.
   - Diagonal homes mean direct chord home→opp's-home crosses the sun;
     the two laterals are the only sun-free attack platforms.
3. Built and shipped (commits `1640792`, `faa8401`). Single-game trace
   at seed 7 caught two bugs and validated the state machine fires:
   lateral captured by step 32, bundled strikes of 70-200 ships at
   opp's home. 10 unit tests + bundle parity green.
4. Two A/Bs vs no-macro control (same env stack):
   - First (wallclock-confounded): 13/26 = 50.0%
   - Second (clean, partial): 24/46 = 52.2%
   No measurable lift at default knobs. Calibration sweep killed by
   container reclaim before STRIKE_MARGIN variants ran.
5. PI conclusion: no clear lift = pivot to opp-model.

## Why the macro didn't lift

**The chooser was already doing this.** PV_ETA + LEAF_PV_2P (both
shipped earlier today) taught the per-candidate scorer to value
forward, bundled, opp-targeting launches via:
- PV_ETA's `γ^(wait + eta)` discount: heavy time-discount on slow
  captures → bias toward fast (close, big-fleet) captures.
- LEAF_PV_2P: 2P leaf production-PV term → reinstates value of
  captures that produce ships for many turns post-capture.

The macro added structural constraints (commit to one lateral; reserve
stockpile from chooser; one bundled strike at threshold) but the
constraints were redundant — the chooser's existing scoring already
preferred those candidates.

PI's PM2 diagnosis (per-move local optimizer) was directionally right
when written, but the SAME-DAY PV_ETA + LEAF_PV_2P shipments
materially closed that gap. The remaining gap is a different problem.

## What I'm carrying out of session

- **The macro stays opt-in (BASELINE_MACRO=0 default).** No live
  ladder effect. Code is dormant; either revert or leave for a
  future use case (e.g. if we identify a chooser-orthogonal lever
  the macro could enforce).
- **Opp-model is the next frontier.** Specifically `lite_greedy_policy`
  inside the rollout. PI's "opponents from everywhere" diagnosis
  applies HERE, not at the action-sequence layer. The opp model is
  pessimistic about counter-attacks because it treats every opp planet
  as a viable counter-source, regardless of geometry. Restricting to
  planets where `eta(opp_planet → our_target) < safe_horizon` is a
  modeling fix (Rule 40 priority).
- **The diagnostic before building Item 3:** instrument the chooser
  to log per-candidate `lite_greedy_policy` predicted opp ships vs
  realised opp ships at the post-game position. If predictor
  systematically over-estimates (e.g. > 30% mean over-shoot on a
  10-game sample), opp-model spatial restriction has a strong prior.
  Cost ~1-2 hr to build the instrumentation; cheap calibration step
  before the ~1 session of Item 3 work.

## Decision-quality observations

- **Wallclock confound was self-inflicted.** I knew `actTimeout: 1.0`
  was in comp-context.md; I set `WALLCLOCK_MS=2000` from habit
  (PM2 used 2000 to pin n_aff, but PM2 was a diagnostic where TIMEOUT
  was acceptable). Should have applied Rule 45b's spirit to A/Bs that
  inform calibration, not just to submit-strength decisions.
- **Background compute near session-end is a known anti-pattern.**
  Rule 32 exists because containers reclaim; my "launch a 2.5h sweep
  in background" decision lost ~2h of intended compute. Should have
  chunked or foreground-run.
- **Macro design was sound; A/B was honest.** Not every reasonable
  design produces lift. The null result is informative — it tells us
  the chooser has internalised forward-bundled-launch valuation, so
  the next lever is upstream of the chooser, not parallel to it.

## What would change my mind on the opp-model pivot

If the instrumentation step shows `lite_greedy_policy` predictions are
WITHIN 10% of realised opp ships, then the opp-model is well-calibrated
and Item 3 won't lift. In that case, the remaining levers are:
- Item 4 (commit-to-hold sizing): replace "min-ships-to-capture under
  best-case opp response" with "min-ships-to-hold for K turns."
- 4P parity work: PV_ETA / LEAF_PV_2P may be gated 2P-only; check
  whether the 4P branch has lift left.
- Drop the chooser axis entirely and explore the proposer (candidate
  pool widening, multi-turn coordinated strikes).
