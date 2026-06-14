# The 4-player axis: two defensive levers tried, both point at disengagement

*2026-06-14, continuation of the reinforcement-floor session. Records a long
observation-driven arc whose value is the eliminations, not a shipped win.*

This builds directly on `2026-06-14-reinforcement-floor-observed-not-assumed.md`
(the V2 reinforcement-term work) and reconciles it with the team's prior 4P
diagnosis in `audit/2026-06-13-kingmaker-tax.md` and
`audit/2026-06-12-garval-first-live-read.md` (which live on the producer-plus
branch).

## The reconciliation that reframed everything

I had been treating the 4P loss as "whoever reinforces more wins" and built a
capture-floor reinforcement term to make us decline contested captures. Reading
the team's own measured diagnosis stopped that cold:

1. **The 4P loss is not target-choice.** Re-weighting our objective toward the
   weak rival "does almost nothing" — even a steep inverse still points us at
   the leader on half our launches. *You cannot re-price your way out of a war
   you are already in.* The leader-fights are **positionally forced**: the
   strongest rival is our neighbour and is already attacking us.

2. **The loss is positional / midgame collapse.** We get pinned in a border war
   with our strongest neighbour, the other two compound past both of us, and we
   get carved after our ~step-60 peak.

So my reinforcement term was doubly misaligned: it raises our capture floors
(declines MORE), and it raises them most on the dense leader — amplifying the
exact dilution that loses 4P games. The +16pp it showed vs a *balanced* 3×V2
pool is a homogeneous-pool artifact (no single snowball to dilute against); real
games have a leader. Dropped it. (A field-behaviour probe also killed the
"reinforcement" framing outright: vs V2 and vs V1 the enemy holds the SAME
garrison fraction (0.64 vs 0.63) and reinforces the SAME amount — V2 just
*attacks* 1.7× harder. The discriminator was aggression, not reinforcement.)

## Lever tried: the threat-window extension (the team's recommended-next)

The garrison-value deficit (defensive: should we pre-reinforce an own planet
whose local balance-of-force is negative) **fires too late**. Live 4P collapse
probe: the deficit IS detected (planet 22, deficit=82) but no friendly source is
within the planning horizon to rescue it — the half-weight deficit only turns
positive once the enemy is already too close. The audit's fix: assess the threat
over a longer lookahead `W > K` so the alarm rings while rescue is still
feasible (sends still arrive within K).

Built it (`PRODUCER_PLUS_GARRISON_VALUE_W`, gated, default-off, byte-identical at
W=K). Two hard lessons:

- **Rule 38 earned its keep.** First build was a silent no-op: a defensive cap
  clamped W back to K because the distance cache only holds K+1 tick slices. The
  outcome was byte-identical (13/13, 8/8) and looked like a clean null — but the
  instrumented mechanism check showed `W=13` in the "W=24" build. The fix never
  applied. (The static reach path extrapolates and needs no cache slices; the
  cap was bogus.)

- **The naive extension is structurally wrong in two ways.** Once actually
  active: (a) detection is still gated at K — `_append_deficit_targets`, which
  decides which planets become defensive candidates, computes its deficit over
  K, so early-threatened planets never enter the pool; (b) the bonus's
  `max`-over-W *inflates* the deficit magnitude (more enemy mass is "reachable"
  over 24 ticks), making it uncoverable by a single send — credits actually
  DROPPED 45→32. A correct version must **decouple detection from sizing**
  (widen the appender, keep the size coverable), which fights the 0.5 weight +
  rival-cap that exist precisely to stop the agent turtling.

Outcome (n=32 paired): +2 wins vs producer (40.6→46.9%), +1 vs V2 (25→28.1%).
Positive in both pools but firmly within noise. A ~6pp effect needs n≈200 to
confirm — impractical in homogeneous pools that may not even reproduce the
positional collapse. Preserved gated (`vetorf4p_sync_garval_w24`, commit
5c4e8de on `claude/pp-4p-v2-reinforce`); a future ladder slot could settle it,
but it is not a confident ship.

## Where both levers point: disengagement

Both defensive patches (decline-more, defend-earlier) are on the **wrong side**
of a positional collapse. The audit's own deeper read says the *winner's*
behaviour is **disengagement**: stop reinforcing a losing forced border war,
redirect that force to expand into weak/neutral territory (the winner feeds the
weak, rank 2.66; we feed the strong neighbour, rank 1.76). This matches "garval
showed no live 4P lift despite firing" — defending the pinned planets harder
doesn't change the collapse; *leaving* the war does.

Disengagement is a structural behavioural change (a "we're losing this border
war → withdraw and expand elsewhere" detector + redirection), explicitly flagged
in the audit as needing PI design discussion before any build. That is the
recommended next direction for the 4P axis — not another floor/deficit tweak.

## The transferable lesson (again, sharper)

Two sessions running, every *floor/deficit* lever on the 4P problem has been
either misaligned (reinforcement term) or structurally cornered (threat-window
fights the anti-turtle guards). The measured loss mode is positional, and the
winner's edge is *behavioural disengagement*, not better defensive accounting.
Stop tuning the capture/garrison floors for 4P; the next real swing is
disengagement, and it needs a design conversation, not a constant.
